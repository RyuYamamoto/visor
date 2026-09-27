//! Path renderer (draws poses as a LINE_STRIP-like polyline; optionally overlays axis/arrow markers per pose).

use std::sync::Arc;

use egui::{Color32, RichText};
use nalgebra::{Isometry3, Point3};
use serde::{Deserialize, Serialize};

use crate::decode::value::Value;
use crate::render::{
    LINE_WIDTH_PX_DEFAULT, RenderStatus, Renderer, SceneBatch, SizeSpec, TfContext, Vertex,
    arrow_mesh_vertex_count, extract_header, extract_pose, push_arrow_mesh, push_triad,
};
use crate::theme;

/// Default pose marker size [m] (Axes axis length / Arrows shaft length).
const MARKER_SIZE_DEFAULT: f32 = 0.3;

/// Default world-fixed line width [m], used by the Billboards style (RViz Path's default).
const LINE_WIDTH_M_DEFAULT: f32 = 0.03;

/// How the polyline is drawn (RViz Path's Line Style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum PathLineStyle {
    /// One-pixel wide lines, ignoring Line width.
    #[default]
    Lines,
    /// Camera-facing ribbons of Line width meters (world-fixed, so they shrink as the camera pulls back).
    Billboards,
}

impl PathLineStyle {
    const ALL: [PathLineStyle; 2] = [PathLineStyle::Lines, PathLineStyle::Billboards];

    fn label(self) -> &'static str {
        match self {
            PathLineStyle::Lines => "Lines",
            PathLineStyle::Billboards => "Billboards",
        }
    }
}

/// Marker type overlaid on each pose (RViz Path's Pose Style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PathPoseStyle {
    None,
    Axes,
    Arrows,
}

impl PathPoseStyle {
    const ALL: [PathPoseStyle; 3] = [
        PathPoseStyle::None,
        PathPoseStyle::Axes,
        PathPoseStyle::Arrows,
    ];

    fn label(self) -> &'static str {
        match self {
            PathPoseStyle::None => "None",
            PathPoseStyle::Axes => "Axes",
            PathPoseStyle::Arrows => "Arrows",
        }
    }
}

/// Pose list extracted from a Path message (relative to header.frame_id; position + orientation).
#[derive(Debug)]
struct ExtractedPath {
    frame_id: String,
    stamp: crate::tf::buffer::TimeNs,
    poses: Vec<Isometry3<f64>>,
}

pub struct PathRenderer {
    path: Option<ExtractedPath>,
    parse_error: Option<String>,
    color: Color32,
    /// Opacity multiplied into lines/arrows/axes (RViz Path's Alpha).
    alpha: f32,
    /// Polyline style (thin lines or world-width ribbons).
    line_style: PathLineStyle,
    /// Ribbon width [m], used by the Billboards style.
    line_width: f32,
    /// Marker type per pose.
    pose_style: PathPoseStyle,
    /// Marker size [m].
    marker_size: f32,
    /// frame->fixed transform fixed at message arrival; batches are baked into the fixed frame.
    pose: Option<Isometry3<f32>>,
    pose_fixed: String,
    pose_dirty: bool,
    /// Baked batches (line + optional markers); Arc-shared, so per-frame clones are cheap.
    baked: Vec<SceneBatch>,
    generation: u64,
    /// Set on new message/settings change; the next scene() rebakes.
    bake_dirty: bool,
}

impl Default for PathRenderer {
    fn default() -> Self {
        Self {
            path: None,
            parse_error: None,
            color: theme::PATH_DEFAULT,
            alpha: 1.0,
            line_style: PathLineStyle::Lines,
            line_width: LINE_WIDTH_M_DEFAULT,
            pose_style: PathPoseStyle::None,
            marker_size: MARKER_SIZE_DEFAULT,
            pose: None,
            pose_fixed: String::new(),
            pose_dirty: false,
            baked: Vec::new(),
            generation: 0,
            bake_dirty: false,
        }
    }
}

impl PathRenderer {
    /// Width spec handed to the line batch: one pixel, or the ribbon width in meters.
    fn line_size(&self) -> SizeSpec {
        match self.line_style {
            PathLineStyle::Lines => SizeSpec::Pixels(LINE_WIDTH_PX_DEFAULT),
            PathLineStyle::Billboards => SizeSpec::Meters(self.line_width),
        }
    }
}

impl Renderer for PathRenderer {
    fn on_message(&mut self, value: &Value) {
        match extract_path(value) {
            Ok(path) => {
                self.path = Some(path);
                self.parse_error = None;
                self.pose_dirty = true;
            }
            Err(e) => self.parse_error = Some(e),
        }
    }

    fn scene(&mut self, tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
        if let Some(error) = &self.parse_error {
            return Err(RenderStatus::InvalidMessage(error.clone()));
        }
        let Some(path) = &self.path else {
            return Err(RenderStatus::NoData);
        };
        // Empty Path draws nothing (nav2 may publish empty paths; not an error).
        if path.poses.is_empty() {
            return Ok(vec![]);
        }
        if self.pose_dirty || self.pose_fixed != tf.fixed_frame {
            match tf.resolve_at(&path.frame_id, path.stamp) {
                Some(iso) => {
                    self.pose = Some(iso.cast::<f32>());
                    self.pose_fixed = tf.fixed_frame.to_owned();
                    self.pose_dirty = false;
                    self.bake_dirty = true;
                }
                None => {
                    if self.pose_fixed != tf.fixed_frame {
                        self.pose = None;
                        self.baked = Vec::new();
                        self.pose_fixed = tf.fixed_frame.to_owned();
                        self.pose_dirty = true;
                    }
                    // Keep emitting the old bake until TF catches up to the stamp (same convention as laser_scan).
                    if self.pose.is_none() {
                        return Err(RenderStatus::TfUnavailable {
                            frame: path.frame_id.clone(),
                        });
                    }
                }
            }
        }
        let Some(pose) = &self.pose else {
            return Err(RenderStatus::TfUnavailable {
                frame: path.frame_id.clone(),
            });
        };
        if !self.pose_dirty && self.bake_dirty {
            self.generation += 1;
            self.baked = bake_batches(
                path,
                pose,
                theme::to_linear_rgba(self.color),
                self.alpha,
                self.line_size(),
                self.pose_style,
                self.marker_size,
                self.generation,
            );
            self.bake_dirty = false;
        }
        Ok(self.baked.clone())
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Line style").color(p.text_muted));
            egui::ComboBox::from_id_salt("path_line_style")
                .selected_text(self.line_style.label())
                .show_ui(ui, |ui| {
                    for style in PathLineStyle::ALL {
                        changed |= ui
                            .selectable_value(&mut self.line_style, style, style.label())
                            .changed();
                    }
                });
            // A one-pixel line has no meaningful width, so the value is offered only for ribbons (as RViz does).
            if self.line_style == PathLineStyle::Billboards {
                ui.label(RichText::new("Line width (m)").color(p.text_muted));
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.line_width)
                            .range(0.002..=1.0)
                            .speed(0.005),
                    )
                    .changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Color").color(p.text_muted));
            changed |= ui.color_edit_button_srgba(&mut self.color).changed();
            ui.label(RichText::new("Alpha").color(p.text_muted));
            changed |= ui
                .add(egui::Slider::new(&mut self.alpha, 0.0..=1.0))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Pose markers").color(p.text_muted));
            egui::ComboBox::from_id_salt("path_pose_style")
                .selected_text(self.pose_style.label())
                .show_ui(ui, |ui| {
                    for style in PathPoseStyle::ALL {
                        changed |= ui
                            .selectable_value(&mut self.pose_style, style, style.label())
                            .changed();
                    }
                });
            if self.pose_style != PathPoseStyle::None {
                ui.label(RichText::new("Size (m)").color(p.text_muted));
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.marker_size)
                            .range(0.02..=5.0)
                            .speed(0.02),
                    )
                    .changed();
            }
        });
        if let Some(path) = &self.path {
            ui.label(RichText::new(format!("poses: {}", path.poses.len())).color(p.text_muted));
        }
        if changed {
            self.bake_dirty = true;
        }
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(PathSettings {
            color: self.color,
            alpha: self.alpha,
            line_style: self.line_style,
            line_width: self.line_width,
            pose_style: self.pose_style,
            marker_size: self.marker_size,
        })
        .ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(s) = value.clone().try_into::<PathSettings>() {
            self.color = s.color;
            self.alpha = s.alpha;
            self.line_style = s.line_style;
            self.line_width = s.line_width;
            self.pose_style = s.pose_style;
            self.marker_size = s.marker_size;
            self.bake_dirty = true;
        }
    }
}

/// Persistence DTO for PathRenderer's user-editable settings (mirrors what settings_ui touches).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct PathSettings {
    #[serde(with = "crate::config::color_hex")]
    color: Color32,
    alpha: f32,
    line_style: PathLineStyle,
    line_width: f32,
    pose_style: PathPoseStyle,
    marker_size: f32,
}

impl Default for PathSettings {
    fn default() -> Self {
        Self {
            color: theme::PATH_DEFAULT,
            alpha: 1.0,
            line_style: PathLineStyle::Lines,
            line_width: LINE_WIDTH_M_DEFAULT,
            pose_style: PathPoseStyle::None,
            marker_size: MARKER_SIZE_DEFAULT,
        }
    }
}

/// Build batches baking the line strip + optional pose markers into the fixed frame.
#[allow(clippy::too_many_arguments)]
fn bake_batches(
    path: &ExtractedPath,
    pose: &Isometry3<f32>,
    color: [f32; 4],
    alpha: f32,
    line_size: SizeSpec,
    style: PathPoseStyle,
    marker_size: f32,
    generation: u64,
) -> Vec<SceneBatch> {
    let line_color = [color[0], color[1], color[2], color[3] * alpha];
    let mut batches = Vec::new();
    if path.poses.len() >= 2 {
        let positions: Vec<[f32; 3]> = path
            .poses
            .iter()
            .map(|p| {
                let t = p.translation.vector;
                [t.x as f32, t.y as f32, t.z as f32]
            })
            .collect();
        let strip = bake_strip(&positions, pose, line_color);
        batches.push(match line_size {
            // A ribbon joins its corners; the one-pixel style has no joins to worry about.
            SizeSpec::Meters(_) => SceneBatch::ribbon(Arc::new(strip), generation, line_size),
            SizeSpec::Pixels(_) => {
                SceneBatch::lines_sized(Arc::new(strip_to_line_list(&strip)), generation, line_size)
            }
        });
    }
    match style {
        PathPoseStyle::None => {}
        PathPoseStyle::Axes => {
            let mut verts = Vec::with_capacity(path.poses.len() * 6);
            for p in &path.poses {
                let world = pose * p.cast::<f32>();
                push_triad(&mut verts, &world, marker_size, alpha);
            }
            if !verts.is_empty() {
                batches.push(SceneBatch::lines(Arc::new(verts), generation));
            }
        }
        PathPoseStyle::Arrows => {
            let mut verts = Vec::with_capacity(path.poses.len() * arrow_mesh_vertex_count());
            for p in &path.poses {
                let world = pose * p.cast::<f32>();
                push_arrow_mesh(&mut verts, &world, marker_size, line_color);
            }
            if !verts.is_empty() {
                batches.push(SceneBatch::mesh(Arc::new(verts), generation));
            }
        }
    }
    batches
}

/// Bake the polyline points into the fixed frame, in order (what a ribbon batch takes).
fn bake_strip(positions: &[[f32; 3]], pose: &Isometry3<f32>, color: [f32; 4]) -> Vec<Vertex> {
    positions
        .iter()
        .map(|p| {
            let baked = pose * Point3::new(p[0], p[1], p[2]);
            Vertex {
                position: [baked.x, baked.y, baked.z],
                color,
            }
        })
        .collect()
}

/// Expand a polyline into the segment pairs a LineList batch expects (N-1 segments x 2 vertices).
fn strip_to_line_list(strip: &[Vertex]) -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(strip.len().saturating_sub(1) * 2);
    for pair in strip.windows(2) {
        vertices.extend_from_slice(pair);
    }
    vertices
}

/// Extract the pose list from a Path Value (per-PoseStamped headers are ignored, matching RViz).
fn extract_path(value: &Value) -> Result<ExtractedPath, String> {
    let (frame_id, stamp) = extract_header(value)?;
    let Some(Value::Array(poses)) = value.get("poses") else {
        return Err("missing field `poses` (PoseStamped[])".to_owned());
    };
    let mut out = Vec::with_capacity(poses.len());
    for (i, item) in poses.iter().enumerate() {
        let Some(pose_value) = item.get("pose") else {
            return Err(format!("poses[{i}] has no `pose`"));
        };
        out.push(extract_pose(pose_value).map_err(|e| format!("poses[{i}]: {e}"))?);
    }
    Ok(ExtractedPath {
        frame_id,
        stamp,
        poses: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::BatchData;
    use crate::tf::buffer::{TfBuffer, TfTransform, tf_update};

    #[test]
    fn settings_roundtrip_via_trait() {
        let r = PathRenderer {
            color: Color32::from_rgb(4, 5, 6),
            alpha: 0.25,
            line_style: PathLineStyle::Billboards,
            line_width: 0.07,
            pose_style: PathPoseStyle::Arrows,
            marker_size: 0.75,
            ..Default::default()
        };
        let value = r.settings().expect("path has settings");
        let mut restored = PathRenderer::default();
        restored.apply_settings(&value);
        assert_eq!(restored.color, r.color);
        assert_eq!(restored.alpha, r.alpha);
        assert_eq!(restored.line_style, r.line_style);
        assert_eq!(restored.line_width, r.line_width);
        assert_eq!(restored.pose_style, r.pose_style);
        assert_eq!(restored.marker_size, r.marker_size);
    }
    use nalgebra::{Translation3, UnitQuaternion};

    fn pose_stamped(x: f64, y: f64, z: f64) -> Value {
        Value::Struct(vec![
            ("header".to_owned(), Value::Struct(vec![])),
            (
                "pose".to_owned(),
                Value::Struct(vec![
                    (
                        "position".to_owned(),
                        Value::Struct(vec![
                            ("x".to_owned(), Value::F64(x)),
                            ("y".to_owned(), Value::F64(y)),
                            ("z".to_owned(), Value::F64(z)),
                        ]),
                    ),
                    (
                        "orientation".to_owned(),
                        Value::Struct(vec![
                            ("x".to_owned(), Value::F64(0.0)),
                            ("y".to_owned(), Value::F64(0.0)),
                            ("z".to_owned(), Value::F64(0.0)),
                            ("w".to_owned(), Value::F64(1.0)),
                        ]),
                    ),
                ]),
            ),
        ])
    }

    fn path_value(frame_id: &str, poses: Vec<Value>) -> Value {
        Value::Struct(vec![
            (
                "header".to_owned(),
                Value::Struct(vec![
                    (
                        "stamp".to_owned(),
                        Value::Struct(vec![
                            ("sec".to_owned(), Value::I32(0)),
                            ("nanosec".to_owned(), Value::U32(0)),
                        ]),
                    ),
                    ("frame_id".to_owned(), Value::String(frame_id.to_owned())),
                ]),
            ),
            ("poses".to_owned(), Value::Array(poses)),
        ])
    }

    fn tf_with(x: f64) -> TfBuffer {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![TfTransform {
                parent: "map".to_owned(),
                child: "odom".to_owned(),
                stamp: 1_000,
                transform: Isometry3::from_parts(
                    Translation3::new(x, 0.0, 0.0),
                    UnitQuaternion::identity(),
                ),
            }], false));
        buffer
    }

    #[test]
    fn extracts_positions_from_poses() {
        let value = path_value(
            "odom",
            vec![pose_stamped(1.0, 2.0, 3.0), pose_stamped(-4.0, 5.0, 0.5)],
        );
        let path = extract_path(&value).expect("valid path");
        assert_eq!(path.frame_id, "odom");
        let translations: Vec<[f64; 3]> = path
            .poses
            .iter()
            .map(|p| {
                let t = p.translation.vector;
                [t.x, t.y, t.z]
            })
            .collect();
        assert_eq!(translations, vec![[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5]]);
    }

    #[test]
    fn pose_markers_add_axes_or_arrow_batch_and_alpha_applies() {
        use crate::render::arrow_mesh_vertex_count;
        let mut renderer = PathRenderer::default();
        let buffer = tf_with(0.0);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        renderer.on_message(&path_value(
            "odom",
            vec![pose_stamped(0.0, 0.0, 0.0), pose_stamped(1.0, 0.0, 0.0)],
        ));
        // Default (None) is a line batch only.
        assert_eq!(renderer.scene(&tf).expect("baked").len(), 1);
        // Axes: line batch + a triad per pose (2 poses x 3 lines x 2 vertices = 12).
        renderer.pose_style = PathPoseStyle::Axes;
        renderer.bake_dirty = true;
        let batches = renderer.scene(&tf).expect("axes");
        assert_eq!(batches.len(), 2);
        let BatchData::Lines(axes) = &batches[1].data else {
            panic!("expected lines batch for axes");
        };
        assert_eq!(axes.len(), 2 * 6);
        // Arrows: the second is a Mesh batch (arrows for 2 poses).
        renderer.pose_style = PathPoseStyle::Arrows;
        renderer.bake_dirty = true;
        let batches = renderer.scene(&tf).expect("arrows");
        let BatchData::Mesh(arrows) = &batches[1].data else {
            panic!("expected mesh batch for arrows");
        };
        assert_eq!(arrows.len(), 2 * arrow_mesh_vertex_count());
        // Alpha applies to the line color.
        renderer.pose_style = PathPoseStyle::None;
        renderer.alpha = 0.5;
        renderer.bake_dirty = true;
        let batches = renderer.scene(&tf).expect("alpha");
        let BatchData::Lines(line) = &batches[0].data else {
            panic!("expected lines batch");
        };
        let expected = theme::to_linear_rgba(theme::PATH_DEFAULT)[3] * 0.5;
        assert!((line[0].color[3] - expected).abs() < 1e-6);
    }

    #[test]
    fn extract_rejects_missing_poses_and_malformed_pose() {
        let value = Value::Struct(vec![(
            "header".to_owned(),
            Value::Struct(vec![
                (
                    "stamp".to_owned(),
                    Value::Struct(vec![
                        ("sec".to_owned(), Value::I32(0)),
                        ("nanosec".to_owned(), Value::U32(0)),
                    ]),
                ),
                ("frame_id".to_owned(), Value::String("odom".to_owned())),
            ]),
        )]);
        assert!(extract_path(&value).is_err());
        let err = extract_path(&path_value("odom", vec![Value::Struct(vec![])]))
            .expect_err("malformed pose");
        assert!(err.contains("poses[0]"), "err={err}");
    }

    #[test]
    fn empty_and_single_pose_paths_draw_nothing_without_error() {
        let mut renderer = PathRenderer::default();
        let buffer = tf_with(0.0);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        renderer.on_message(&path_value("odom", vec![]));
        assert_eq!(renderer.scene(&tf).expect("empty ok").len(), 0);
        renderer.on_message(&path_value("odom", vec![pose_stamped(1.0, 0.0, 0.0)]));
        assert_eq!(renderer.scene(&tf).expect("single ok").len(), 0);
    }

    #[test]
    fn scene_expands_strip_to_line_list_baked_into_fixed_frame() {
        let mut renderer = PathRenderer::default();
        let buffer = tf_with(10.0);
        renderer.on_message(&path_value(
            "odom",
            vec![
                pose_stamped(0.0, 0.0, 0.0),
                pose_stamped(1.0, 0.0, 0.0),
                pose_stamped(1.0, 2.0, 0.0),
            ],
        ));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let batches = renderer.scene(&tf).expect("baked");
        let BatchData::Lines(vertices) = &batches[0].data else {
            panic!("expected lines batch");
        };
        // 3 points -> 2 segments x 2 vertices; the middle point appears twice as a segment endpoint.
        assert_eq!(vertices.len(), 4);
        assert_eq!(vertices[0].position, [10.0, 0.0, 0.0]);
        assert_eq!(vertices[1].position, [11.0, 0.0, 0.0]);
        assert_eq!(vertices[2].position, [11.0, 0.0, 0.0]);
        assert_eq!(vertices[3].position, [11.0, 2.0, 0.0]);
        assert!(
            vertices
                .iter()
                .all(|v| v.color == theme::to_linear_rgba(theme::PATH_DEFAULT))
        );
    }

    #[test]
    fn color_change_rebakes_and_bumps_generation() {
        let mut renderer = PathRenderer::default();
        let buffer = tf_with(0.0);
        renderer.on_message(&path_value(
            "odom",
            vec![pose_stamped(0.0, 0.0, 0.0), pose_stamped(1.0, 0.0, 0.0)],
        ));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let gen1 = renderer.scene(&tf).expect("baked")[0].generation;
        // Re-scene without a settings change keeps the generation (no per-frame rebake).
        assert_eq!(renderer.scene(&tf).expect("cached")[0].generation, gen1);
        renderer.color = Color32::from_rgb(1, 2, 3);
        renderer.bake_dirty = true;
        let b2 = renderer.scene(&tf).expect("rebaked");
        assert!(b2[0].generation > gen1);
        let BatchData::Lines(vertices) = &b2[0].data else {
            panic!("expected lines batch");
        };
        assert_eq!(
            vertices[0].color,
            theme::to_linear_rgba(Color32::from_rgb(1, 2, 3))
        );
    }

    #[test]
    fn line_style_selects_pixel_or_meter_width() {
        let mut renderer = PathRenderer::default();
        let buffer = tf_with(0.0);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        renderer.on_message(&path_value(
            "odom",
            vec![pose_stamped(0.0, 0.0, 0.0), pose_stamped(1.0, 0.0, 0.0)],
        ));
        // Default is a one-pixel line, so existing configs keep the look they had.
        let batches = renderer.scene(&tf).expect("baked");
        assert_eq!(batches[0].size, SizeSpec::Pixels(LINE_WIDTH_PX_DEFAULT));
        renderer.line_style = PathLineStyle::Billboards;
        renderer.line_width = 0.08;
        renderer.bake_dirty = true;
        let batches = renderer.scene(&tf).expect("baked");
        assert_eq!(batches[0].size, SizeSpec::Meters(0.08));
        // Billboards go out as a ribbon so the corners are mitered instead of notched.
        let BatchData::Ribbon(points) = &batches[0].data else {
            panic!("expected a ribbon batch");
        };
        assert_eq!(points.len(), 2);
        // Pose markers stay thin regardless of the polyline width.
        renderer.pose_style = PathPoseStyle::Axes;
        renderer.bake_dirty = true;
        let batches = renderer.scene(&tf).expect("baked");
        assert_eq!(batches[1].size, SizeSpec::Pixels(LINE_WIDTH_PX_DEFAULT));
    }

    #[test]
    fn renderer_status_transitions() {
        let mut renderer = PathRenderer::default();
        let buffer = TfBuffer::new();
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        assert!(matches!(renderer.scene(&tf), Err(RenderStatus::NoData)));
        renderer.on_message(&Value::Struct(vec![]));
        assert!(matches!(
            renderer.scene(&tf),
            Err(RenderStatus::InvalidMessage(_))
        ));
        renderer.on_message(&path_value(
            "odom",
            vec![pose_stamped(0.0, 0.0, 0.0), pose_stamped(1.0, 0.0, 0.0)],
        ));
        assert_eq!(
            renderer.scene(&tf).unwrap_err(),
            RenderStatus::TfUnavailable {
                frame: "odom".to_owned()
            }
        );
    }
}
