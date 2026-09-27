//! Marker / MarkerArray renderer (shapes + billboard text, with ns/id state, lifetime, DELETE / DELETEALL).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::RichText;
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};

use crate::decode::value::Value;
use crate::render::{
    Label, PointBatchBuilder, RenderStatus, Renderer, SceneBatch, SizeSpec, TfContext, Vertex,
    extract_pose, push_arrow_mesh, push_prism, push_triangle,
};
use crate::tf::buffer::TimeNs;
use crate::theme;

/// Longitude divisions of the UV sphere.
const SPHERE_SLICES: usize = 16;
/// Latitude divisions of the UV sphere.
const SPHERE_STACKS: usize = 8;
/// Circumference divisions of the cylinder.
const CYLINDER_SEGMENTS: usize = 16;
/// Minimum line width [m] for LINE_* markers, guarding against a zero-width prism.
const LINE_WIDTH_MIN: f32 = 0.001;
/// Cone head length / shaft length used to map an arrow's total length back to push_arrow_mesh's shaft argument.
const ARROW_TOTAL_OVER_SHAFT: f32 = 1.3;

/// Marker type (visualization_msgs/Marker constants; unsupported types are tracked but not drawn).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerKind {
    Arrow,
    Cube,
    Sphere,
    Cylinder,
    LineStrip,
    LineList,
    CubeList,
    SphereList,
    Points,
    Text,
    TriangleList,
    Unsupported,
}

impl MarkerKind {
    fn from_type(t: i32) -> Self {
        match t {
            0 => MarkerKind::Arrow,
            1 => MarkerKind::Cube,
            2 => MarkerKind::Sphere,
            3 => MarkerKind::Cylinder,
            4 => MarkerKind::LineStrip,
            5 => MarkerKind::LineList,
            6 => MarkerKind::CubeList,
            7 => MarkerKind::SphereList,
            8 => MarkerKind::Points,
            9 => MarkerKind::Text,
            11 => MarkerKind::TriangleList,
            _ => MarkerKind::Unsupported,
        }
    }
}

/// Marker action (visualization_msgs/Marker constants; value 1 is deprecated and ignored).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerAction {
    AddModify,
    Delete,
    DeleteAll,
    Ignore,
}

impl MarkerAction {
    fn from_action(a: i32) -> Self {
        match a {
            0 => MarkerAction::AddModify,
            2 => MarkerAction::Delete,
            3 => MarkerAction::DeleteAll,
            _ => MarkerAction::Ignore,
        }
    }
}

/// One live marker, keyed by (ns, id). Geometry is stored in the marker's own frame and baked to fixed at scene() time.
#[derive(Debug, Clone)]
struct MarkerEntry {
    kind: MarkerKind,
    pose: Isometry3<f64>,
    scale: [f64; 3],
    color: [f32; 4],
    points: Vec<Point3<f64>>,
    colors: Vec<[f32; 4]>,
    text: String,
    frame_id: String,
    stamp: TimeNs,
    received_at: Instant,
    lifetime: Option<Duration>,
}

#[derive(Default)]
pub struct MarkerRenderer {
    markers: BTreeMap<(String, i32), MarkerEntry>,
    /// Namespaces hidden by the user (persisted).
    ns_hidden: BTreeSet<String>,
    /// Top-level decode failure of the most recent message.
    parse_error: Option<String>,
    /// Unsupported-type marker count from the last bake (shown in settings_ui).
    skipped_unsupported: usize,
    /// TF-unresolvable marker count from the last bake (shown in settings_ui).
    skipped_tf: usize,
    /// Representative frame of the last TF failure (used for the TfUnavailable status).
    unresolved_frame: Option<String>,
    fixed_frame: String,
    bake_dirty: bool,
    baked: Vec<SceneBatch>,
    generation: u64,
}

impl Renderer for MarkerRenderer {
    fn on_message(&mut self, value: &Value) {
        self.parse_error = None;
        // MarkerArray carries a `markers` array; a bare Marker does not, so decode either shape here.
        if let Some(Value::Array(markers)) = value.get("markers") {
            for marker in markers {
                if let Err(e) = self.ingest(marker) {
                    self.parse_error = Some(e);
                    return;
                }
            }
        } else if let Err(e) = self.ingest(value) {
            self.parse_error = Some(e);
        }
    }

    fn scene(&mut self, tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
        if let Some(error) = &self.parse_error {
            return Err(RenderStatus::InvalidMessage(error.clone()));
        }
        self.expire_lifetimes();
        if self.markers.is_empty() {
            return Err(RenderStatus::NoData);
        }
        if self.bake_dirty || self.fixed_frame != tf.fixed_frame {
            self.fixed_frame = tf.fixed_frame.to_owned();
            self.rebake(tf);
        }
        if !self.baked.is_empty() {
            return Ok(self.baked.clone());
        }
        // Nothing drawable: surface TF failure only when it was the reason, otherwise an empty (non-error) scene.
        if self.skipped_tf > 0 {
            return Err(RenderStatus::TfUnavailable {
                frame: self.unresolved_frame.clone().unwrap_or_default(),
            });
        }
        Ok(Vec::new())
    }

    /// Markers persist until deleted or expired, so a jump in playback would otherwise leave a skipped span's markers on screen.
    fn reset(&mut self) {
        self.markers.clear();
        self.parse_error = None;
        self.skipped_unsupported = 0;
        self.skipped_tf = 0;
        self.unresolved_frame = None;
        self.bake_dirty = true;
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        let namespaces: BTreeSet<String> =
            self.markers.keys().map(|(ns, _)| ns.clone()).collect();
        let mut changed = false;
        for ns in &namespaces {
            let mut visible = !self.ns_hidden.contains(ns);
            let label = if ns.is_empty() { "(default)" } else { ns.as_str() };
            if ui.checkbox(&mut visible, label).changed() {
                if visible {
                    self.ns_hidden.remove(ns);
                } else {
                    self.ns_hidden.insert(ns.clone());
                }
                changed = true;
            }
        }
        ui.label(RichText::new(format!("markers: {}", self.markers.len())).color(p.text_muted));
        if self.skipped_unsupported > 0 {
            ui.label(
                RichText::new(format!("unsupported: {}", self.skipped_unsupported))
                    .color(p.status_warn),
            );
        }
        if self.skipped_tf > 0 {
            ui.label(
                RichText::new(format!("tf-unresolved: {}", self.skipped_tf)).color(p.status_warn),
            );
        }
        if changed {
            self.bake_dirty = true;
        }
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(MarkerSettings {
            hidden_ns: self.ns_hidden.iter().cloned().collect(),
        })
        .ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(s) = value.clone().try_into::<MarkerSettings>() {
            self.ns_hidden = s.hidden_ns.into_iter().collect();
            self.bake_dirty = true;
        }
    }
}

impl MarkerRenderer {
    /// Apply one Marker's action (add/modify/delete/delete-all) to the (ns, id) store.
    fn ingest(&mut self, marker: &Value) -> Result<(), String> {
        let action = MarkerAction::from_action(get_i32(marker, "action")?);
        let ns = get_string(marker, "ns")?.to_owned();
        match action {
            MarkerAction::AddModify => {
                let id = get_i32(marker, "id")?;
                let entry = parse_marker(marker)?;
                self.markers.insert((ns, id), entry);
                self.bake_dirty = true;
            }
            MarkerAction::Delete => {
                let id = get_i32(marker, "id")?;
                if self.markers.remove(&(ns, id)).is_some() {
                    self.bake_dirty = true;
                }
            }
            MarkerAction::DeleteAll => {
                // RViz clears every namespace on DELETEALL (ns is ignored).
                if !self.markers.is_empty() {
                    self.markers.clear();
                    self.bake_dirty = true;
                }
            }
            MarkerAction::Ignore => {}
        }
        Ok(())
    }

    /// Drop markers whose lifetime has elapsed; mark dirty if any were removed.
    fn expire_lifetimes(&mut self) {
        let now = Instant::now();
        let before = self.markers.len();
        self.markers.retain(|_, entry| match entry.lifetime {
            Some(lifetime) => now.duration_since(entry.received_at) < lifetime,
            None => true,
        });
        if self.markers.len() != before {
            self.bake_dirty = true;
        }
    }

    /// Rebuild the baked mesh + label batches in the fixed frame, updating skip counters.
    fn rebake(&mut self, tf: &TfContext<'_>) {
        let mut mesh = Vec::new();
        let mut labels = Vec::new();
        self.skipped_unsupported = 0;
        self.skipped_tf = 0;
        self.unresolved_frame = None;
        self.generation += 1;
        let mut point_batches = Vec::new();
        for ((ns, _), entry) in &self.markers {
            if self.ns_hidden.contains(ns) {
                continue;
            }
            if entry.kind == MarkerKind::Unsupported {
                self.skipped_unsupported += 1;
                continue;
            }
            // alpha == 0 means fully transparent; RViz skips drawing it (no warning).
            if entry.color[3] <= 0.0 {
                continue;
            }
            let Some(frame_iso) = tf.resolve_at(&entry.frame_id, entry.stamp) else {
                self.skipped_tf += 1;
                self.unresolved_frame.get_or_insert_with(|| entry.frame_id.clone());
                continue;
            };
            let full = (frame_iso * entry.pose).cast::<f32>();
            if entry.kind == MarkerKind::Points {
                if let Some(batch) = bake_points(entry, &full, self.generation) {
                    point_batches.push(batch);
                }
                continue;
            }
            bake_marker(entry, &full, &mut mesh, &mut labels);
        }
        let mut batches = Vec::new();
        if !mesh.is_empty() {
            batches.push(SceneBatch::mesh(Arc::new(mesh), self.generation));
        }
        if !labels.is_empty() {
            batches.push(SceneBatch::labels(Arc::new(labels), self.generation));
        }
        batches.extend(point_batches);
        self.baked = batches;
        // Keep retrying while some frame stays unresolved (viewport repaints ~60fps when items exist).
        self.bake_dirty = self.skipped_tf > 0;
    }
}

/// Persistence DTO (only the hidden-namespace set is user state; markers themselves are live data).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct MarkerSettings {
    hidden_ns: Vec<String>,
}

/// Bake one marker's geometry into the shared mesh/label buffers (positions already in fixed frame via full).
fn bake_marker(
    entry: &MarkerEntry,
    full: &Isometry3<f32>,
    mesh: &mut Vec<Vertex>,
    labels: &mut Vec<Label>,
) {
    let color = linear_rgba(entry.color);
    match entry.kind {
        MarkerKind::Arrow => bake_arrow(entry, full, color, mesh),
        MarkerKind::Cube => push_cube(mesh, full, scale_f32(entry.scale), color),
        MarkerKind::Sphere => push_sphere(mesh, full, scale_f32(entry.scale), color),
        MarkerKind::Cylinder => push_cylinder(mesh, full, scale_f32(entry.scale), color),
        MarkerKind::LineStrip => bake_lines(entry, full, color, false, mesh),
        MarkerKind::LineList => bake_lines(entry, full, color, true, mesh),
        MarkerKind::CubeList => bake_shape_list(entry, full, color, mesh, push_cube),
        MarkerKind::SphereList => bake_shape_list(entry, full, color, mesh, push_sphere),
        MarkerKind::TriangleList => bake_triangles(entry, full, color, mesh),
        // Handled in rebake as its own point batch, not a mesh.
        MarkerKind::Points => {}
        MarkerKind::Text => {
            let p = full.translation.vector;
            labels.push(Label {
                position: [p.x, p.y, p.z],
                text: entry.text.clone(),
                color,
                height_m: entry.scale[2].max(0.0) as f32,
            });
        }
        MarkerKind::Unsupported => {}
    }
}

/// Bake an ARROW marker: two-point form uses start->end, otherwise the pose points +X with length = scale.x.
fn bake_arrow(entry: &MarkerEntry, full: &Isometry3<f32>, color: [f32; 4], mesh: &mut Vec<Vertex>) {
    if entry.points.len() >= 2 {
        let start = full * entry.points[0].cast::<f32>();
        let end = full * entry.points[1].cast::<f32>();
        let dir = end - start;
        let len = dir.norm();
        if len < f32::EPSILON {
            return;
        }
        let rotation = UnitQuaternion::rotation_between(&Vector3::x(), &dir)
            .unwrap_or_else(|| UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f32::consts::PI));
        let iso = Isometry3::from_parts(Translation3::from(start.coords), rotation);
        push_arrow_mesh(mesh, &iso, len / ARROW_TOTAL_OVER_SHAFT, color);
    } else {
        let len = entry.scale[0].max(0.0) as f32;
        if len < f32::EPSILON {
            return;
        }
        push_arrow_mesh(mesh, full, len / ARROW_TOTAL_OVER_SHAFT, color);
    }
}

/// Bake LINE_STRIP (consecutive pairs) or LINE_LIST (disjoint pairs) as world-width prisms; per-point colors use the segment start.
fn bake_lines(
    entry: &MarkerEntry,
    full: &Isometry3<f32>,
    default_color: [f32; 4],
    list: bool,
    mesh: &mut Vec<Vertex>,
) {
    let half_width = ((entry.scale[0] as f32) * 0.5).max(LINE_WIDTH_MIN);
    let step = if list { 2 } else { 1 };
    let mut i = 0;
    while i + 1 < entry.points.len() {
        let a = full * entry.points[i].cast::<f32>();
        let b = full * entry.points[i + 1].cast::<f32>();
        let color = entry
            .colors
            .get(i)
            .map(|c| linear_rgba(*c))
            .unwrap_or(default_color);
        push_prism(mesh, a, b, color, half_width);
        i += step;
    }
}

/// Per-point color for *_LIST markers: colors[i] if present (RViz uses it), else the marker's color. Returns linear RGBA.
fn point_color(entry: &MarkerEntry, i: usize, default_color: [f32; 4]) -> [f32; 4] {
    entry
        .colors
        .get(i)
        .map(|c| linear_rgba(*c))
        .unwrap_or(default_color)
}

/// Draws one primitive (cube/sphere) into the mesh at the given pose with a per-axis scale and color.
type ShapeFn = fn(&mut Vec<Vertex>, &Isometry3<f32>, [f32; 3], [f32; 4]);

/// Bake CUBE_LIST / SPHERE_LIST: draw one shape (scale = marker scale) at each point, oriented by the marker pose.
fn bake_shape_list(
    entry: &MarkerEntry,
    full: &Isometry3<f32>,
    default_color: [f32; 4],
    mesh: &mut Vec<Vertex>,
    draw: ShapeFn,
) {
    let scale = scale_f32(entry.scale);
    for (i, p) in entry.points.iter().enumerate() {
        let iso = full * Translation3::from(p.cast::<f32>().coords);
        draw(mesh, &iso, scale, point_color(entry, i, default_color));
    }
}

/// Bake TRIANGLE_LIST: consecutive point triples form triangles; per-triangle color uses the first vertex's colors[i].
fn bake_triangles(
    entry: &MarkerEntry,
    full: &Isometry3<f32>,
    default_color: [f32; 4],
    mesh: &mut Vec<Vertex>,
) {
    let mut i = 0;
    while i + 2 < entry.points.len() {
        let tri = [
            full * entry.points[i].cast::<f32>(),
            full * entry.points[i + 1].cast::<f32>(),
            full * entry.points[i + 2].cast::<f32>(),
        ];
        push_triangle(mesh, tri, point_color(entry, i, default_color));
        i += 3;
    }
}

/// Bake POINTS as a world-fixed square point batch (edge = scale.x), pose = fixed_from_frame; None if there are no points.
fn bake_points(entry: &MarkerEntry, full: &Isometry3<f32>, generation: u64) -> Option<SceneBatch> {
    if entry.points.is_empty() {
        return None;
    }
    let default_color = linear_rgba(entry.color);
    let mut builder = PointBatchBuilder::with_capacity(entry.points.len());
    for (i, p) in entry.points.iter().enumerate() {
        let c = point_color(entry, i, default_color);
        builder.push([p.x as f32, p.y as f32, p.z as f32], linear_rgba8(c));
    }
    let size = (entry.scale[0] as f32).max(LINE_WIDTH_MIN);
    Some(SceneBatch::points(
        builder.build(),
        generation,
        full,
        SizeSpec::Meters(size),
    ))
}

/// Encode a linear RGBA float color as the 8-bit linear RGBA the point batch expects.
fn linear_rgba8(c: [f32; 4]) -> [u8; 4] {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    [b(c[0]), b(c[1]), b(c[2]), b(c[3])]
}

/// Append a box centered at the pose origin with per-axis full-length edges (RViz CUBE scale = full size).
fn push_cube(mesh: &mut Vec<Vertex>, iso: &Isometry3<f32>, scale: [f32; 3], color: [f32; 4]) {
    let h = [scale[0] * 0.5, scale[1] * 0.5, scale[2] * 0.5];
    let corner = |sx: f32, sy: f32, sz: f32| iso * Point3::new(sx * h[0], sy * h[1], sz * h[2]);
    let faces = [
        [(1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0)],
        [(-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0)],
        [(-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, -1.0)],
        [(1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, -1.0, -1.0)],
        [(-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)],
        [(-1.0, 1.0, -1.0), (1.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)],
    ];
    for f in faces {
        let p: Vec<Point3<f32>> = f.iter().map(|(x, y, z)| corner(*x, *y, *z)).collect();
        push_triangle(mesh, [p[0], p[1], p[2]], color);
        push_triangle(mesh, [p[0], p[2], p[3]], color);
    }
}

/// Append a UV sphere (an ellipsoid, since scale is per-axis diameter) centered at the pose origin.
fn push_sphere(mesh: &mut Vec<Vertex>, iso: &Isometry3<f32>, scale: [f32; 3], color: [f32; 4]) {
    let r = [scale[0] * 0.5, scale[1] * 0.5, scale[2] * 0.5];
    let vertex = |stack: usize, slice: usize| {
        let theta = std::f32::consts::PI * stack as f32 / SPHERE_STACKS as f32;
        let phi = std::f32::consts::TAU * slice as f32 / SPHERE_SLICES as f32;
        iso * Point3::new(
            r[0] * theta.sin() * phi.cos(),
            r[1] * theta.sin() * phi.sin(),
            r[2] * theta.cos(),
        )
    };
    for stack in 0..SPHERE_STACKS {
        for slice in 0..SPHERE_SLICES {
            let a = vertex(stack, slice);
            let b = vertex(stack + 1, slice);
            let c = vertex(stack + 1, slice + 1);
            let d = vertex(stack, slice + 1);
            push_triangle(mesh, [a, b, c], color);
            push_triangle(mesh, [a, c, d], color);
        }
    }
}

/// Append a Z-aligned cylinder (scale.x/y = diameter, scale.z = height) centered at the pose origin, with end caps.
fn push_cylinder(mesh: &mut Vec<Vertex>, iso: &Isometry3<f32>, scale: [f32; 3], color: [f32; 4]) {
    let (rx, ry, hz) = (scale[0] * 0.5, scale[1] * 0.5, scale[2] * 0.5);
    let rim = |slice: usize, z: f32| {
        let phi = std::f32::consts::TAU * slice as f32 / CYLINDER_SEGMENTS as f32;
        iso * Point3::new(rx * phi.cos(), ry * phi.sin(), z)
    };
    let top_center = iso * Point3::new(0.0, 0.0, hz);
    let bottom_center = iso * Point3::new(0.0, 0.0, -hz);
    for slice in 0..CYLINDER_SEGMENTS {
        let b0 = rim(slice, -hz);
        let b1 = rim(slice + 1, -hz);
        let t0 = rim(slice, hz);
        let t1 = rim(slice + 1, hz);
        push_triangle(mesh, [b0, b1, t1], color);
        push_triangle(mesh, [b0, t1, t0], color);
        push_triangle(mesh, [top_center, t0, t1], color);
        push_triangle(mesh, [bottom_center, b1, b0], color);
    }
}

/// Interpret ColorRGBA (0..1 sRGB display values) as a linear RGBA for the sRGB target; alpha passes through.
fn linear_rgba(rgba: [f32; 4]) -> [f32; 4] {
    let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    let col = egui::Color32::from_rgb(byte(rgba[0]), byte(rgba[1]), byte(rgba[2]));
    let l = theme::to_linear_rgba(col);
    [l[0], l[1], l[2], rgba[3].clamp(0.0, 1.0)]
}

fn scale_f32(scale: [f64; 3]) -> [f32; 3] {
    [scale[0] as f32, scale[1] as f32, scale[2] as f32]
}

/// Parse an AddModify Marker into an entry (pose/scale/color required; points/colors/text/lifetime optional).
fn parse_marker(marker: &Value) -> Result<MarkerEntry, String> {
    let kind = MarkerKind::from_type(get_i32(marker, "type")?);
    let pose = marker
        .get("pose")
        .ok_or("missing field `pose`")
        .and_then(|p| extract_pose(p).map_err(|_| "invalid field `pose`"))?;
    let scale = extract_vec3(marker.get("scale").ok_or("missing field `scale`")?)?;
    let color = extract_color(marker.get("color").ok_or("missing field `color`")?)?;
    let text = get_string(marker, "text").unwrap_or("").to_owned();
    let frame_id = marker
        .get("header")
        .and_then(|h| h.get("frame_id"))
        .and_then(|f| match f {
            Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or("missing field `header.frame_id`")?;
    let stamp = header_stamp(marker);
    let points = extract_points(marker.get("points"));
    let colors = extract_colors(marker.get("colors"));
    let lifetime = extract_lifetime(marker.get("lifetime"));
    Ok(MarkerEntry {
        kind,
        pose,
        scale,
        color,
        points,
        colors,
        text,
        frame_id,
        stamp,
        received_at: Instant::now(),
        lifetime,
    })
}

fn get_i32(v: &Value, field: &str) -> Result<i32, String> {
    match v.get(field) {
        Some(Value::I32(x)) => Ok(*x),
        _ => Err(format!("missing field `{field}` (int32)")),
    }
}

fn get_string<'a>(v: &'a Value, field: &str) -> Result<&'a str, String> {
    match v.get(field) {
        Some(Value::String(s)) => Ok(s),
        _ => Err(format!("missing field `{field}` (string)")),
    }
}

fn as_f64(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::F64(x)) => Some(*x),
        Some(Value::F32(x)) => Some(*x as f64),
        _ => None,
    }
}

fn as_f32(v: Option<&Value>) -> Option<f32> {
    match v {
        Some(Value::F32(x)) => Some(*x),
        Some(Value::F64(x)) => Some(*x as f32),
        _ => None,
    }
}

fn extract_vec3(v: &Value) -> Result<[f64; 3], String> {
    Ok([
        as_f64(v.get("x")).ok_or("scale.x")?,
        as_f64(v.get("y")).ok_or("scale.y")?,
        as_f64(v.get("z")).ok_or("scale.z")?,
    ])
}

fn extract_color(v: &Value) -> Result<[f32; 4], String> {
    Ok([
        as_f32(v.get("r")).ok_or("color.r")?,
        as_f32(v.get("g")).ok_or("color.g")?,
        as_f32(v.get("b")).ok_or("color.b")?,
        as_f32(v.get("a")).ok_or("color.a")?,
    ])
}

fn extract_points(v: Option<&Value>) -> Vec<Point3<f64>> {
    let Some(Value::Array(items)) = v else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|p| {
            Some(Point3::new(
                as_f64(p.get("x"))?,
                as_f64(p.get("y"))?,
                as_f64(p.get("z"))?,
            ))
        })
        .collect()
}

fn extract_colors(v: Option<&Value>) -> Vec<[f32; 4]> {
    let Some(Value::Array(items)) = v else {
        return Vec::new();
    };
    items.iter().filter_map(|c| extract_color(c).ok()).collect()
}

/// Read builtin_interfaces/Duration; a zero duration means "forever" (None).
fn extract_lifetime(v: Option<&Value>) -> Option<Duration> {
    let v = v?;
    let sec = match v.get("sec") {
        Some(Value::I32(s)) => *s,
        _ => return None,
    };
    let nanosec = match v.get("nanosec") {
        Some(Value::U32(n)) => *n,
        _ => return None,
    };
    if sec <= 0 && nanosec == 0 {
        return None;
    }
    Some(Duration::new(sec.max(0) as u64, nanosec))
}

fn header_stamp(marker: &Value) -> TimeNs {
    let stamp = marker.get("header").and_then(|h| h.get("stamp"));
    match (
        stamp.and_then(|s| s.get("sec")),
        stamp.and_then(|s| s.get("nanosec")),
    ) {
        (Some(Value::I32(sec)), Some(Value::U32(nanosec))) => {
            *sec as i64 * 1_000_000_000 + *nanosec as i64
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{BatchData, prism_vertex_count};
    use crate::tf::buffer::{TfBuffer, TfTransform, tf_update};

    fn tf_identity() -> TfBuffer {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![TfTransform {
                parent: "map".to_owned(),
                child: "base".to_owned(),
                stamp: 0,
                transform: Isometry3::identity(),
            }], true));
        buffer
    }

    fn ctx<'a>(buffer: &'a TfBuffer) -> TfContext<'a> {
        TfContext {
            buffer,
            fixed_frame: "map",
        }
    }

    fn f32v(x: f64, y: f64, z: f64) -> Value {
        Value::Struct(vec![
            ("x".to_owned(), Value::F64(x)),
            ("y".to_owned(), Value::F64(y)),
            ("z".to_owned(), Value::F64(z)),
        ])
    }

    fn color(r: f32, g: f32, b: f32, a: f32) -> Value {
        Value::Struct(vec![
            ("r".to_owned(), Value::F32(r)),
            ("g".to_owned(), Value::F32(g)),
            ("b".to_owned(), Value::F32(b)),
            ("a".to_owned(), Value::F32(a)),
        ])
    }

    fn identity_pose() -> Value {
        Value::Struct(vec![
            ("position".to_owned(), f32v(0.0, 0.0, 0.0)),
            (
                "orientation".to_owned(),
                Value::Struct(vec![
                    ("x".to_owned(), Value::F64(0.0)),
                    ("y".to_owned(), Value::F64(0.0)),
                    ("z".to_owned(), Value::F64(0.0)),
                    ("w".to_owned(), Value::F64(1.0)),
                ]),
            ),
        ])
    }

    fn header(frame: &str) -> Value {
        Value::Struct(vec![
            (
                "stamp".to_owned(),
                Value::Struct(vec![
                    ("sec".to_owned(), Value::I32(0)),
                    ("nanosec".to_owned(), Value::U32(0)),
                ]),
            ),
            ("frame_id".to_owned(), Value::String(frame.to_owned())),
        ])
    }

    /// Builds a Marker Value; extra field overrides are appended last (Value::get returns the first match, so prepend to override).
    fn marker(ns: &str, id: i32, kind: i32, action: i32, extra: Vec<(&str, Value)>) -> Value {
        let mut fields = vec![
            ("header".to_owned(), header("base")),
            ("ns".to_owned(), Value::String(ns.to_owned())),
            ("id".to_owned(), Value::I32(id)),
            ("type".to_owned(), Value::I32(kind)),
            ("action".to_owned(), Value::I32(action)),
            ("pose".to_owned(), identity_pose()),
            ("scale".to_owned(), f32v(1.0, 1.0, 1.0)),
            ("color".to_owned(), color(1.0, 0.0, 0.0, 1.0)),
            ("text".to_owned(), Value::String(String::new())),
        ];
        for (k, v) in extra {
            fields.insert(0, (k.to_owned(), v));
        }
        Value::Struct(fields)
    }

    fn mesh_len(batches: &[SceneBatch]) -> usize {
        batches
            .iter()
            .find_map(|b| match &b.data {
                BatchData::Mesh(v) => Some(v.len()),
                _ => None,
            })
            .unwrap_or(0)
    }

    #[test]
    fn reset_drops_the_markers_but_keeps_hidden_namespaces() {
        let mut r = MarkerRenderer::default();
        r.on_message(&marker("a", 0, 1, 0, vec![]));
        r.on_message(&marker("b", 1, 1, 0, vec![]));
        r.ns_hidden.insert("b".to_owned());
        assert_eq!(r.markers.len(), 2);
        r.reset();
        // Markers from a span playback skipped would otherwise stay on screen and be read as current.
        assert!(r.markers.is_empty());
        assert!(r.bake_dirty);
        // The user's per-namespace visibility choices are settings, not history, so they survive.
        assert!(r.ns_hidden.contains("b"));
    }

    #[test]
    fn upsert_keeps_one_entry_per_ns_id() {
        let mut r = MarkerRenderer::default();
        r.on_message(&marker("a", 0, 1, 0, vec![]));
        r.on_message(&marker("a", 0, 1, 0, vec![]));
        assert_eq!(r.markers.len(), 1);
        r.on_message(&marker("b", 0, 1, 0, vec![]));
        r.on_message(&marker("a", 1, 1, 0, vec![]));
        assert_eq!(r.markers.len(), 3);
    }

    #[test]
    fn delete_and_delete_all() {
        let mut r = MarkerRenderer::default();
        r.on_message(&marker("a", 0, 1, 0, vec![]));
        r.on_message(&marker("a", 1, 1, 0, vec![]));
        r.on_message(&marker("a", 0, 1, 2, vec![]));
        assert_eq!(r.markers.len(), 1);
        r.on_message(&marker("b", 5, 1, 0, vec![]));
        r.on_message(&marker("", 0, 1, 3, vec![]));
        assert!(r.markers.is_empty());
    }

    #[test]
    fn marker_array_and_single_marker_both_parse() {
        let mut r = MarkerRenderer::default();
        r.on_message(&marker("a", 0, 1, 0, vec![]));
        assert_eq!(r.markers.len(), 1);
        let array = Value::Struct(vec![(
            "markers".to_owned(),
            Value::Array(vec![marker("b", 0, 1, 0, vec![]), marker("b", 1, 1, 0, vec![])]),
        )]);
        r.on_message(&array);
        assert_eq!(r.markers.len(), 3);
    }

    #[test]
    fn lifetime_expiry_removes_marker() {
        let mut r = MarkerRenderer::default();
        r.on_message(&marker(
            "a",
            0,
            1,
            0,
            vec![(
                "lifetime",
                Value::Struct(vec![
                    ("sec".to_owned(), Value::I32(0)),
                    ("nanosec".to_owned(), Value::U32(1)),
                ]),
            )],
        ));
        assert_eq!(r.markers.len(), 1);
        let entry = r.markers.values_mut().next().unwrap();
        entry.received_at = Instant::now() - Duration::from_secs(1);
        r.expire_lifetimes();
        assert!(r.markers.is_empty());
    }

    #[test]
    fn cube_bakes_36_vertices_and_ns_toggle_hides_it() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        r.on_message(&marker("a", 0, 1, 0, vec![]));
        let batches = r.scene(&ctx(&buffer)).expect("baked");
        assert_eq!(mesh_len(&batches), 36);
        r.ns_hidden.insert("a".to_owned());
        r.bake_dirty = true;
        let batches = r.scene(&ctx(&buffer)).unwrap_or_default();
        assert_eq!(mesh_len(&batches), 0);
    }

    #[test]
    fn sphere_and_cylinder_vertex_counts() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        r.on_message(&marker("s", 0, 2, 0, vec![]));
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), SPHERE_STACKS * SPHERE_SLICES * 6);
        let mut r = MarkerRenderer::default();
        r.on_message(&marker("c", 0, 3, 0, vec![]));
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), CYLINDER_SEGMENTS * 12);
    }

    #[test]
    fn line_strip_makes_n_minus_1_segments() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        let points = Value::Array(vec![
            f32v(0.0, 0.0, 0.0),
            f32v(1.0, 0.0, 0.0),
            f32v(1.0, 1.0, 0.0),
        ]);
        r.on_message(&marker("l", 0, 4, 0, vec![("points", points)]));
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), 2 * prism_vertex_count());
    }

    #[test]
    fn line_list_makes_pairs() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        let points = Value::Array(vec![
            f32v(0.0, 0.0, 0.0),
            f32v(1.0, 0.0, 0.0),
            f32v(2.0, 0.0, 0.0),
            f32v(3.0, 0.0, 0.0),
        ]);
        r.on_message(&marker("l", 0, 5, 0, vec![("points", points)]));
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), 2 * prism_vertex_count());
    }

    #[test]
    fn cube_list_bakes_one_cube_per_point() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        let points = Value::Array(vec![f32v(0.0, 0.0, 0.0), f32v(2.0, 0.0, 0.0)]);
        r.on_message(&marker("cl", 0, 6, 0, vec![("points", points)]));
        // One cube = 36 vertices; two points = 72.
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), 72);
        assert_eq!(r.skipped_unsupported, 0);
    }

    #[test]
    fn sphere_list_bakes_one_sphere_per_point() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        let points = Value::Array(vec![f32v(0.0, 0.0, 0.0), f32v(1.0, 0.0, 0.0)]);
        r.on_message(&marker("sl", 0, 7, 0, vec![("points", points)]));
        assert_eq!(
            mesh_len(&r.scene(&ctx(&buffer)).unwrap()),
            2 * SPHERE_STACKS * SPHERE_SLICES * 6
        );
    }

    #[test]
    fn triangle_list_bakes_triples() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        let points = Value::Array(vec![
            f32v(0.0, 0.0, 0.0),
            f32v(1.0, 0.0, 0.0),
            f32v(0.0, 1.0, 0.0),
            // A trailing incomplete triple is ignored.
            f32v(2.0, 0.0, 0.0),
        ]);
        r.on_message(&marker("t", 0, 11, 0, vec![("points", points)]));
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), 3);
    }

    #[test]
    fn points_bakes_a_point_batch() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        let points = Value::Array(vec![
            f32v(0.0, 0.0, 0.0),
            f32v(1.0, 0.0, 0.0),
            f32v(2.0, 0.0, 0.0),
        ]);
        r.on_message(&marker("p", 0, 8, 0, vec![("points", points)]));
        let batches = r.scene(&ctx(&buffer)).expect("baked");
        let count = batches.iter().find_map(|b| match &b.data {
            BatchData::Points(pb) => Some(pb.count),
            _ => None,
        });
        assert_eq!(count, Some(3));
        assert_eq!(r.skipped_unsupported, 0);
    }

    #[test]
    fn arrow_points_form_and_pose_form() {
        use crate::render::arrow_mesh_vertex_count;
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        r.on_message(&marker("a", 0, 0, 0, vec![]));
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), arrow_mesh_vertex_count());
        let mut r = MarkerRenderer::default();
        let points = Value::Array(vec![f32v(0.0, 0.0, 0.0), f32v(2.0, 0.0, 0.0)]);
        r.on_message(&marker("a", 0, 0, 0, vec![("points", points)]));
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), arrow_mesh_vertex_count());
    }

    #[test]
    fn text_marker_goes_to_labels_baked_in_fixed_frame() {
        let mut r = MarkerRenderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![TfTransform {
                parent: "map".to_owned(),
                child: "base".to_owned(),
                stamp: 0,
                transform: Isometry3::from_parts(
                    Translation3::new(5.0, 0.0, 0.0),
                    UnitQuaternion::identity(),
                ),
            }], true));
        r.on_message(&marker(
            "t",
            0,
            9,
            0,
            vec![
                ("text", Value::String("robot_1".to_owned())),
                ("scale", f32v(0.0, 0.0, 0.5)),
            ],
        ));
        let batches = r.scene(&ctx(&buffer)).expect("baked");
        let labels = batches
            .iter()
            .find_map(|b| match &b.data {
                BatchData::Labels(l) => Some(l.clone()),
                _ => None,
            })
            .expect("labels batch");
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].text, "robot_1");
        assert_eq!(labels[0].position, [5.0, 0.0, 0.0]);
        assert!((labels[0].height_m - 0.5).abs() < 1e-6);
    }

    #[test]
    fn transparent_marker_is_skipped() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        r.on_message(&marker("a", 0, 1, 0, vec![("color", color(1.0, 0.0, 0.0, 0.0))]));
        assert_eq!(mesh_len(&r.scene(&ctx(&buffer)).unwrap()), 0);
    }

    #[test]
    fn unsupported_type_counts_but_does_not_error() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        // type 10 = MESH_RESOURCE, still unsupported.
        r.on_message(&marker("a", 0, 10, 0, vec![]));
        let batches = r.scene(&ctx(&buffer)).expect("empty ok");
        assert!(batches.is_empty());
        assert_eq!(r.skipped_unsupported, 1);
    }

    #[test]
    fn tf_unresolved_reports_status() {
        let mut r = MarkerRenderer::default();
        let buffer = TfBuffer::new();
        r.on_message(&marker("a", 0, 1, 0, vec![]));
        assert_eq!(
            r.scene(&ctx(&buffer)).unwrap_err(),
            RenderStatus::TfUnavailable {
                frame: "base".to_owned()
            }
        );
    }

    #[test]
    fn status_transitions_no_data_and_invalid() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        assert_eq!(r.scene(&ctx(&buffer)).unwrap_err(), RenderStatus::NoData);
        r.on_message(&Value::Struct(vec![("ns".to_owned(), Value::String("x".to_owned()))]));
        assert!(matches!(
            r.scene(&ctx(&buffer)),
            Err(RenderStatus::InvalidMessage(_))
        ));
    }

    #[test]
    fn per_vertex_colors_apply_to_line_segments() {
        let mut r = MarkerRenderer::default();
        let buffer = tf_identity();
        let points = Value::Array(vec![f32v(0.0, 0.0, 0.0), f32v(1.0, 0.0, 0.0)]);
        let colors = Value::Array(vec![color(0.0, 1.0, 0.0, 1.0), color(0.0, 0.0, 1.0, 1.0)]);
        r.on_message(&marker("l", 0, 4, 0, vec![("points", points), ("colors", colors)]));
        let batches = r.scene(&ctx(&buffer)).unwrap();
        let BatchData::Mesh(verts) = &batches[0].data else {
            panic!("expected mesh");
        };
        // The single segment uses colors[0] (green): green channel present, red/blue stay zero (shading only scales).
        assert!(verts.iter().any(|v| v.color[1] > 0.0));
        assert!(verts.iter().all(|v| v.color[0] == 0.0 && v.color[2] == 0.0));
    }

    #[test]
    fn settings_roundtrip_hidden_ns() {
        let mut r = MarkerRenderer::default();
        r.ns_hidden.insert("hidden".to_owned());
        let value = r.settings().expect("has settings");
        let mut restored = MarkerRenderer::default();
        restored.apply_settings(&value);
        assert!(restored.ns_hidden.contains("hidden"));
    }
}
