//! Renderer trait and its shared vocabulary. Registration lives in plugin::registry.

pub mod camera;
pub mod renderers;
pub mod viewport;

use std::sync::Arc;

use nalgebra::{Isometry3, Matrix4, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::decode::value::Value;
use crate::tf::buffer::{TfBuffer, TimeNs};
use crate::theme;

/// One line vertex (position + linear RGBA). Shared vocabulary of trait, viewport, and renderers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// GPU layout of one point: position f32x3 (12B) + packed RGBA8 (4B) = 16B, LE.
pub const POINT_STRIDE: usize = 16;

/// Point batch with finalized GPU layout (bytes.len() == count * POINT_STRIDE). Arc-shared so scene() only bumps the refcount.
#[derive(Debug, Clone)]
pub struct PointBatch {
    pub bytes: Arc<Vec<u8>>,
    pub count: u32,
}

/// GPU layout of one posed-mesh vertex: position f32x3 (12B) + normal f32x3 (12B) + packed RGBA8 (4B) = 28B, LE.
pub const MESH_STRIDE: usize = 28;

/// Posed-mesh vertex batch with finalized GPU layout (bytes.len() == count * MESH_STRIDE). Arc-shared so scene() only bumps the refcount.
#[derive(Debug, Clone)]
pub struct MeshBatch {
    pub bytes: Arc<Vec<u8>>,
    pub count: u32,
}

/// Max 2D texture dimension [texel] (max_texture_dimension_2d of egui-wgpu 0.35's default `wgpu::Limits`).
pub const MAX_TEXTURE_DIM: u32 = 8192;

/// Occupancy color scheme (RViz Color Scheme vocabulary + this viewer's theme). Each scheme's palette is cached by `palette()`, so switching between them re-uploads nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccupancyScheme {
    /// RViz map palette (0=white -> 100=black, -1=teal).
    Map,
    /// RViz costmap palette (0=transparent, blue->red, 99=cyan, 100=magenta).
    Costmap,
    /// RViz raw palette (value used directly as luminance).
    Raw,
    /// This viewer's theme (dark background + cyan glow).
    Viewer,
}

impl OccupancyScheme {
    pub const ALL: [OccupancyScheme; 4] = [
        OccupancyScheme::Map,
        OccupancyScheme::Costmap,
        OccupancyScheme::Raw,
        OccupancyScheme::Viewer,
    ];

    pub fn label(self) -> &'static str {
        match self {
            OccupancyScheme::Map => "Map",
            OccupancyScheme::Costmap => "Costmap",
            OccupancyScheme::Raw => "Raw",
            OccupancyScheme::Viewer => "Viewer",
        }
    }

    /// This scheme's palette, cached process-wide so the pointer stays stable and the LUT reaches the GPU once.
    pub fn palette(self) -> GridPalette {
        static CACHE: std::sync::OnceLock<[GridPalette; OccupancyScheme::ALL.len()]> =
            std::sync::OnceLock::new();
        let cache =
            CACHE.get_or_init(|| OccupancyScheme::ALL.map(|s| GridPalette::new(occupancy_lut(s))));
        cache[self.index()].clone()
    }

    /// Fixed index into the cached palette array.
    fn index(self) -> usize {
        match self {
            OccupancyScheme::Map => 0,
            OccupancyScheme::Costmap => 1,
            OccupancyScheme::Raw => 2,
            OccupancyScheme::Viewer => 3,
        }
    }
}

/// 256-entry cell-value -> RGBA lookup table for the occupancy tile shader; Arc-shared so scene() only bumps the refcount and the viewport can skip the upload by pointer identity.
#[derive(Debug, Clone)]
pub struct GridPalette(pub Arc<[[f32; 4]; 256]>);

impl GridPalette {
    /// Wrap a freshly built table; keep the result around, since rebuilding it every frame re-uploads 4 KiB every frame.
    pub fn new(colors: [[f32; 4]; 256]) -> Self {
        Self(Arc::new(colors))
    }

    /// Same table by identity; this is what lets the viewport skip the upload.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// One occupancy-grid texture, applied to a quad on the local XY plane. The viewport manages the GPU resources.
#[derive(Debug, Clone)]
pub struct GridTexture {
    /// R8 texels (len == width * height, raw occupancy bytes). Arc-shared so scene() only bumps the refcount.
    pub pixels: Arc<Vec<u8>>,
    /// Width [texel].
    pub width: u32,
    /// Height [texel].
    pub height: u32,
    /// Quad's local XY size [m] (= resolution * width / height).
    pub size_m: [f32; 2],
    /// Overall tile opacity. Applied via the batch uniform only, so changing it keeps the generation unchanged.
    pub alpha: f32,
    /// Cell-value palette. Uploaded as a per-batch uniform, so changing it keeps the generation unchanged.
    pub palette: GridPalette,
    /// Draw as background with depth disabled, before everything else (RViz Map's Draw Behind). Generation-invariant.
    pub draw_behind: bool,
}

/// Batch contents. Adding a variant = adding a primitive kind (extension point); the viewport's match enforces exhaustiveness.
#[derive(Debug, Clone)]
pub enum BatchData {
    /// LineList, baked in fixed-frame coordinates (model matrix unused).
    Lines(Arc<Vec<Vertex>>),
    /// Open polyline (consecutive points are joined), baked in fixed-frame coordinates. Drawn as a ribbon with mitered joins, so a curve reads as one continuous band; width comes from `size`.
    Ribbon(Arc<Vec<Vertex>>),
    /// Point quads in local coordinates, transformed to fixed frame by the model matrix.
    Points(PointBatch),
    /// Textured occupancy-grid quad in local coordinates (model matrix = fixed_from_frame * origin).
    TexturedQuad(GridTexture),
    /// TriangleList mesh, baked in fixed-frame coordinates with shading in vertex color (model matrix unused).
    Mesh(Arc<Vec<Vertex>>),
    /// TriangleList mesh in local coordinates (position + normal), placed by the rigid model matrix and shaded in the shader.
    PosedMesh(MeshBatch),
    /// Billboard text labels, positioned in fixed-frame coordinates. Drawn by the viewport's painter, not the GPU callback (model matrix unused).
    Labels(Arc<Vec<Label>>),
}

/// One billboard text label, baked in fixed-frame coordinates. The viewport projects it to screen and draws it with egui's painter.
#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    /// Anchor position in fixed-frame coordinates.
    pub position: [f32; 3],
    pub text: String,
    /// Linear RGBA (alpha applied as text opacity).
    pub color: [f32; 4],
    /// Text height in world meters (RViz Marker scale.z); the viewport projects it to a pixel font size.
    pub height_m: f32,
}

/// Size spec shared by point quads (RViz Points / Squares) and line width (RViz Lines / Billboards).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeSpec {
    /// World-fixed size (quad edge / line width [m]); shrinks on screen as the camera pulls back.
    Meters(f32),
    /// Screen-fixed size (quad edge / line width [px], constant apparent size).
    Pixels(f32),
}

/// Line width [px] of the default one-pixel line, matching RViz's Lines style.
pub const LINE_WIDTH_PX_DEFAULT: f32 = 1.0;

/// Draw batch returned by scene(). If generation matches the previous frame the viewport skips the GPU upload.
#[derive(Debug, Clone)]
pub struct SceneBatch {
    pub data: BatchData,
    /// Monotonically bumped when data changes (++ on re-bake).
    pub generation: u64,
    /// Points / PosedMesh: local->fixed-frame model matrix (unused for Lines; pass identity).
    pub model: Matrix4<f32>,
    /// Points: quad size. Lines: line width (Pixels = screen-fixed 1px-style, Meters = world-fixed like RViz Billboards). Unused by the other variants.
    pub size: SizeSpec,
    /// PosedMesh: overall opacity, applied via the batch uniform (unused by the other variants; pass 1.0).
    pub alpha: f32,
}

impl SceneBatch {
    /// Line batch baked in fixed-frame coordinates, one pixel wide.
    pub fn lines(vertices: Arc<Vec<Vertex>>, generation: u64) -> Self {
        Self::lines_sized(
            vertices,
            generation,
            SizeSpec::Pixels(LINE_WIDTH_PX_DEFAULT),
        )
    }

    /// Polyline ribbon of `width`: consecutive points are joined with mitered corners (what RViz Billboards / jsk BillboardLine look like). Pass the points in order, not as segment pairs.
    pub fn ribbon(points: Arc<Vec<Vertex>>, generation: u64, width: SizeSpec) -> Self {
        Self {
            data: BatchData::Ribbon(points),
            generation,
            model: Matrix4::identity(),
            size: width,
            alpha: 1.0,
        }
    }

    /// Line batch with an explicit width: Meters expands each segment on its own (no joins; use `ribbon` for a polyline), Pixels keeps them screen-fixed.
    pub fn lines_sized(vertices: Arc<Vec<Vertex>>, generation: u64, width: SizeSpec) -> Self {
        Self {
            data: BatchData::Lines(vertices),
            generation,
            model: Matrix4::identity(),
            size: width,
            alpha: 1.0,
        }
    }

    /// Triangle mesh baked in fixed-frame coordinates (shading baked into vertex color).
    pub fn mesh(vertices: Arc<Vec<Vertex>>, generation: u64) -> Self {
        Self {
            data: BatchData::Mesh(vertices),
            generation,
            model: Matrix4::identity(),
            size: SizeSpec::Meters(0.0),
            alpha: 1.0,
        }
    }

    /// Billboard text labels baked in fixed-frame coordinates (drawn by the viewport's painter; no GPU resources).
    pub fn labels(labels: Arc<Vec<Label>>, generation: u64) -> Self {
        Self {
            data: BatchData::Labels(labels),
            generation,
            model: Matrix4::identity(),
            size: SizeSpec::Meters(0.0),
            alpha: 1.0,
        }
    }

    /// Textured quad in local coordinates (pose = fixed_from_frame * origin; follows via uniform update only).
    pub fn textured_quad(texture: GridTexture, generation: u64, pose: &Isometry3<f32>) -> Self {
        Self {
            data: BatchData::TexturedQuad(texture),
            generation,
            model: pose.to_homogeneous(),
            size: SizeSpec::Meters(0.0),
            alpha: 1.0,
        }
    }

    /// Point batch in local coordinates (pose passed to GPU every frame as a per-batch uniform).
    pub fn points(
        batch: PointBatch,
        generation: u64,
        pose: &Isometry3<f32>,
        size: SizeSpec,
    ) -> Self {
        Self {
            data: BatchData::Points(batch),
            generation,
            model: pose.to_homogeneous(),
            size,
            alpha: 1.0,
        }
    }

    /// Posed mesh in local coordinates: pose (rigid) and alpha go to the GPU every frame as a per-batch uniform, so TF/opacity changes need no re-bake.
    pub fn posed_mesh(
        batch: MeshBatch,
        generation: u64,
        pose: &Isometry3<f32>,
        alpha: f32,
    ) -> Self {
        Self {
            data: BatchData::PosedMesh(batch),
            generation,
            model: pose.to_homogeneous(),
            size: SizeSpec::Meters(0.0),
            alpha,
        }
    }
}

/// Point display style (subset of RViz's Style enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointStyle {
    /// Screen-fixed size points (Size in px).
    Points,
    /// World-fixed size billboard quads (Size in m).
    Squares,
}

impl PointStyle {
    fn label(self) -> &'static str {
        match self {
            PointStyle::Points => "Points",
            PointStyle::Squares => "Squares",
        }
    }
}

/// Shared Style dropdown + size settings (aligns UI and conventions across LaserScan / PointCloud2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PointStyleSettings {
    pub style: PointStyle,
    /// Edge for Squares [m].
    pub size_m: f32,
    /// Edge for Points [px].
    pub size_px: f32,
}

impl Default for PointStyleSettings {
    fn default() -> Self {
        Self::new(0.05)
    }
}

impl PointStyleSettings {
    pub fn new(size_m: f32) -> Self {
        Self {
            style: PointStyle::Points,
            size_m,
            size_px: 3.0,
        }
    }

    /// SceneBatch size for the current style (applied via uniform only, so no re-bake needed).
    pub fn size(&self) -> SizeSpec {
        match self.style {
            PointStyle::Points => SizeSpec::Pixels(self.size_px),
            PointStyle::Squares => SizeSpec::Meters(self.size_m),
        }
    }

    /// Style ComboBox + style-specific size DragValue (equivalent to RViz Style / Size (m|Pixels)).
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Style").color(p.text_muted));
            egui::ComboBox::from_id_salt("point_style")
                .selected_text(self.style.label())
                .show_ui(ui, |ui| {
                    for style in [PointStyle::Points, PointStyle::Squares] {
                        ui.selectable_value(&mut self.style, style, style.label());
                    }
                });
            match self.style {
                PointStyle::Points => {
                    ui.label(egui::RichText::new("Size (px)").color(p.text_muted));
                    ui.add(
                        egui::DragValue::new(&mut self.size_px)
                            .range(1.0..=20.0)
                            .speed(0.1),
                    );
                }
                PointStyle::Squares => {
                    ui.label(egui::RichText::new("Size (m)").color(p.text_muted));
                    ui.add(
                        egui::DragValue::new(&mut self.size_m)
                            .range(0.001..=1.0)
                            .speed(0.005)
                            .suffix(" m"),
                    );
                }
            }
        });
    }
}

/// Type-safe point-batch builder. Confines the 16B LE layout here so renderers never hand-pack raw bytes.
#[derive(Debug, Default)]
pub struct PointBatchBuilder {
    bytes: Vec<u8>,
    count: u32,
}

impl PointBatchBuilder {
    pub fn with_capacity(points: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(points * POINT_STRIDE),
            count: 0,
        }
    }

    /// Push one point (position in local coordinates, rgba as linear-encoded 8-bit).
    pub fn push(&mut self, position: [f32; 3], rgba: [u8; 4]) {
        for v in position {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.bytes.extend_from_slice(&rgba);
        self.count += 1;
    }

    pub fn build(self) -> PointBatch {
        PointBatch {
            bytes: Arc::new(self.bytes),
            count: self.count,
        }
    }
}

/// Type-safe posed-mesh builder. Confines the 28B LE layout (position + normal + RGBA8) here so renderers never hand-pack raw bytes.
#[derive(Debug, Default)]
pub struct MeshBatchBuilder {
    bytes: Vec<u8>,
    count: u32,
}

impl MeshBatchBuilder {
    pub fn with_capacity(vertices: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(vertices * MESH_STRIDE),
            count: 0,
        }
    }

    /// Push one vertex: position and normal in local coordinates (the model matrix is rigid, so the shader rotates the normal as-is), rgba as linear-encoded 8-bit.
    pub fn push_vertex(&mut self, position: [f32; 3], normal: [f32; 3], rgba: [u8; 4]) {
        for v in position.iter().chain(normal.iter()) {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.bytes.extend_from_slice(&rgba);
        self.count += 1;
    }

    /// Push one triangle, deriving the flat face normal (sugar for procedurally generated geometry that carries no normals).
    pub fn push_triangle(&mut self, tri: [Point3<f32>; 3], rgba: [u8; 4]) {
        let edge = (tri[1] - tri[0]).cross(&(tri[2] - tri[0]));
        let normal = if edge.norm() > f32::EPSILON {
            edge.normalize()
        } else {
            Vector3::z()
        };
        for p in tri {
            self.push_vertex([p.x, p.y, p.z], [normal.x, normal.y, normal.z], rgba);
        }
    }

    /// Vertex count pushed so far (lets callers size a following batch without a build).
    pub fn len(&self) -> u32 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn build(self) -> MeshBatch {
        MeshBatch {
            bytes: Arc::new(self.bytes),
            count: self.count,
        }
    }
}

/// Evaluate the intensity colormap: clamp t to [0,1], linearly interpolate theme::POINT_COLORMAP, return linear RGBA8.
pub fn colormap(t: f32) -> [u8; 4] {
    static STOPS: std::sync::OnceLock<Vec<[f32; 4]>> = std::sync::OnceLock::new();
    let stops = STOPS.get_or_init(|| {
        theme::POINT_COLORMAP
            .iter()
            .map(|c| theme::to_linear_rgba(*c))
            .collect()
    });
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let scaled = t * (stops.len() - 1) as f32;
    let index = (scaled as usize).min(stops.len() - 2);
    let frac = scaled - index as f32;
    let (a, b) = (&stops[index], &stops[index + 1]);
    std::array::from_fn(|i| {
        let v = a[i] + (b[i] - a[i]) * frac;
        (v * 255.0).round().clamp(0.0, 255.0) as u8
    })
}

/// 256-entry occupancy->RGBA LUT (u8 index = int8 bit pattern used directly). Reaches the GPU as the `GridPalette` a `GridTexture` carries.
pub fn occupancy_lut(scheme: OccupancyScheme) -> [[f32; 4]; 256] {
    // RViz palettes use plain 1/255 normalization; the non-sRGB swapchain makes on-screen values match RViz.
    let rviz = |palette: fn(u8) -> [u8; 4]| -> [[f32; 4]; 256] {
        std::array::from_fn(|i| palette(i as u8).map(|v| v as f32 / 255.0))
    };
    match scheme {
        OccupancyScheme::Map => rviz(theme::rviz_map_palette),
        OccupancyScheme::Costmap => rviz(theme::rviz_costmap_palette),
        OccupancyScheme::Raw => rviz(theme::rviz_raw_palette),
        OccupancyScheme::Viewer => viewer_lut(),
    }
}

/// Theme color scheme: interpolate free (pale cyan) -> occupied (bright cyan); unknown is pale purple.
fn viewer_lut() -> [[f32; 4]; 256] {
    let rgba = |color: egui::Color32, alpha: f32| -> [f32; 4] {
        let c = theme::to_linear_rgba(color);
        [c[0], c[1], c[2], alpha]
    };
    let free = rgba(theme::MAP_FREE, theme::MAP_FREE_ALPHA);
    let occupied = rgba(theme::MAP_OCCUPIED, theme::MAP_OCCUPIED_ALPHA);
    let unknown = rgba(theme::MAP_UNKNOWN, theme::MAP_UNKNOWN_ALPHA);
    std::array::from_fn(|i| match i {
        // 0..=100: occupancy probability (color and alpha both linearly interpolated).
        0..=100 => {
            let t = i as f32 / 100.0;
            std::array::from_fn(|c| free[c] + (occupied[c] - free[c]) * t)
        }
        // 101..=127: out-of-range positive values clamp to occupied.
        101..=127 => occupied,
        // 128..=255: int8 -128..-1, all treated as unknown (per the 255 = -1 = unknown convention).
        _ => unknown,
    })
}

/// Intensity colormap min/max settings (auto + manual override). Shared UI and conventions across LaserScan / PointCloud2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IntensityScale {
    pub auto: bool,
    pub min: f32,
    pub max: f32,
}

impl Default for IntensityScale {
    fn default() -> Self {
        Self {
            auto: true,
            min: 0.0,
            max: 1.0,
        }
    }
}

impl IntensityScale {
    /// Range actually used: measured min/max from the message when auto, otherwise the configured values.
    pub fn resolve(&self, measured: Option<(f32, f32)>) -> (f32, f32) {
        if self.auto {
            measured.unwrap_or((0.0, 0.0))
        } else {
            (self.min, self.max)
        }
    }

    /// Shared range-settings UI (returns true on change = re-pack needed).
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut self.auto, "Auto range").changed();
            ui.add_enabled_ui(!self.auto, |ui| {
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.min)
                            .speed(0.1)
                            .prefix("min "),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.max)
                            .speed(0.1)
                            .prefix("max "),
                    )
                    .changed();
            });
        });
        changed
    }
}

/// Normalize an intensity value over the range and sample the colormap (min == max uses t = 0 to avoid division by zero).
pub fn intensity_color(range: (f32, f32), value: f32) -> [u8; 4] {
    let t = if range.1 > range.0 {
        (value - range.0) / (range.1 - range.0)
    } else {
        0.0
    };
    colormap(t)
}

/// Circumference segments of the 3D arrow mesh (enough for smooth cylinder/cone even when flat). Shared by Odometry / Path.
const ARROW_SEGMENTS: usize = 20;
/// Cone head length / shaft length (RViz default head 0.3 / shaft 1.0).
const ARROW_HEAD_LEN_RATIO: f32 = 0.3;
/// Shaft radius / shaft length (RViz default 0.05).
const ARROW_SHAFT_RADIUS_RATIO: f32 = 0.05;
/// Head radius / shaft length (RViz default 0.1).
const ARROW_HEAD_RADIUS_RATIO: f32 = 0.1;

/// Vertex count of one arrow mesh (TriangleList). Used for capacity reservation and tests.
pub fn arrow_mesh_vertex_count() -> usize {
    ARROW_SEGMENTS * 5 * 3
}

/// Bake a solid +X-facing 3D arrow (cylinder shaft + cone head) into fixed frame via iso, shading vertex color from vertex normals.
pub fn push_arrow_mesh(
    out: &mut Vec<Vertex>,
    iso: &Isometry3<f32>,
    shaft_len: f32,
    color: [f32; 4],
) {
    let head_len = ARROW_HEAD_LEN_RATIO * shaft_len;
    let shaft_r = ARROW_SHAFT_RADIUS_RATIO * shaft_len;
    let head_r = ARROW_HEAD_RADIUS_RATIO * shaft_len;
    let n = ARROW_SEGMENTS;
    let angle = |i: usize| std::f32::consts::TAU * i as f32 / n as f32;
    let ring = |x: f32, r: f32| -> Vec<Point3<f32>> {
        (0..n)
            .map(|i| {
                let a = angle(i);
                Point3::new(x, r * a.cos(), r * a.sin())
            })
            .collect()
    };
    let radial = |i: usize| {
        let a = angle(i);
        Vector3::new(0.0, a.cos(), a.sin())
    };
    let cone = |i: usize| {
        let a = angle(i);
        Vector3::new(head_r, head_len * a.cos(), head_len * a.sin()).normalize()
    };
    let back = Vector3::new(-1.0, 0.0, 0.0);
    let base = ring(0.0, shaft_r);
    let top = ring(shaft_len, shaft_r);
    let head_base = ring(shaft_len, head_r);
    let c_base = Point3::new(0.0, 0.0, 0.0);
    let c_head = Point3::new(shaft_len, 0.0, 0.0);
    let tip = Point3::new(shaft_len + head_len, 0.0, 0.0);
    let mut tris: Vec<[(Point3<f32>, Vector3<f32>); 3]> = Vec::with_capacity(n * 5);
    for i in 0..n {
        let j = (i + 1) % n;
        tris.push([(c_base, back), (base[j], back), (base[i], back)]);
        tris.push([
            (base[i], radial(i)),
            (base[j], radial(j)),
            (top[j], radial(j)),
        ]);
        tris.push([
            (base[i], radial(i)),
            (top[j], radial(j)),
            (top[i], radial(i)),
        ]);
        tris.push([(c_head, back), (head_base[i], back), (head_base[j], back)]);
        let tip_n = (cone(i) + cone(j)).normalize();
        tris.push([
            (head_base[i], cone(i)),
            (head_base[j], cone(j)),
            (tip, tip_n),
        ]);
    }
    for t in &tris {
        for (p, normal) in t {
            let wp = iso * p;
            let wn = iso.rotation * normal;
            out.push(Vertex {
                position: [wp.x, wp.y, wp.z],
                color: shade_color(color, wn),
            });
        }
    }
}

/// Append an axis triad of length len (X red / Y green / Z blue, alpha applied) at pose iso: 3 segments, 6 vertices.
pub fn push_triad(out: &mut Vec<Vertex>, iso: &Isometry3<f32>, len: f32, alpha: f32) {
    let origin = iso * Point3::origin();
    for (axis, color32) in [
        (Vector3::x(), theme::AXIS_X),
        (Vector3::y(), theme::AXIS_Y),
        (Vector3::z(), theme::AXIS_Z),
    ] {
        let c = theme::to_linear_rgba(color32);
        let color = [c[0], c[1], c[2], c[3] * alpha];
        let tip = origin + (iso.rotation * axis) * len;
        out.push(Vertex {
            position: [origin.x, origin.y, origin.z],
            color,
        });
        out.push(Vertex {
            position: [tip.x, tip.y, tip.z],
            color,
        });
    }
}

/// Cylinder side segment count when extruding a line into a rod (higher is smoother but adds vertices). Shared by TF thick lines and Marker LINE_*.
const CYLINDER_SEGMENTS: usize = 12;

/// Directional light for CPU-baked mesh shading; same theme constant mesh.wgsl uses, so baked and shader-lit meshes look alike.
fn prism_light() -> Vector3<f32> {
    Vector3::from(theme::MESH_LIGHT_DIR).normalize()
}

/// Apply directional lighting for a face normal to color; per the convention of baking shading into vertex color.
fn shade_color(color: [f32; 4], normal: Vector3<f32>) -> [f32; 4] {
    let shade = theme::MESH_AMBIENT + theme::MESH_DIFFUSE * normal.dot(&prism_light()).max(0.0);
    [
        color[0] * shade,
        color[1] * shade,
        color[2] * shade,
        color[3],
    ]
}

/// Vertex count of one prism (TriangleList). Used for capacity reservation and tests.
pub fn prism_vertex_count() -> usize {
    CYLINDER_SEGMENTS * 12
}

/// Extrude segment a->b into a cylinder of radius half_width (polygonal prism + simple lighting) and append it to out (fixed-frame baked).
pub fn push_prism(
    out: &mut Vec<Vertex>,
    a: Point3<f32>,
    b: Point3<f32>,
    color: [f32; 4],
    half_width: f32,
) {
    let axis = b - a;
    let len = axis.norm();
    if len < f32::EPSILON {
        return;
    }
    let dir = axis / len;
    let reference = if dir.z.abs() < 0.9 {
        Vector3::z()
    } else {
        Vector3::x()
    };
    let u = dir.cross(&reference).normalize();
    let v = dir.cross(&u).normalize();
    let n = CYLINDER_SEGMENTS;
    let radial = |i: usize| {
        let angle = std::f32::consts::TAU * i as f32 / n as f32;
        u * angle.cos() + v * angle.sin()
    };
    let mut push = |p: Point3<f32>, normal: Vector3<f32>| {
        out.push(Vertex {
            position: [p.x, p.y, p.z],
            color: shade_color(color, normal),
        });
    };
    for i in 0..n {
        let j = (i + 1) % n;
        let (ni, nj) = (radial(i), radial(j));
        let (bi, bj) = (a + ni * half_width, a + nj * half_width);
        let (ti, tj) = (b + ni * half_width, b + nj * half_width);
        push(bi, ni);
        push(bj, nj);
        push(tj, nj);
        push(bi, ni);
        push(tj, nj);
        push(ti, ni);
        push(a, -dir);
        push(a + nj * half_width, -dir);
        push(a + ni * half_width, -dir);
        push(b, dir);
        push(b + ni * half_width, dir);
        push(b + nj * half_width, dir);
    }
}

/// Append one fixed-frame triangle to out, shading its geometric normal with the shared light (stays correct under non-uniform scale).
pub fn push_triangle(out: &mut Vec<Vertex>, tri: [Point3<f32>; 3], color: [f32; 4]) {
    let edge = (tri[1] - tri[0]).cross(&(tri[2] - tri[0]));
    let normal = if edge.norm() > f32::EPSILON {
        edge.normalize()
    } else {
        Vector3::z()
    };
    for p in tri {
        out.push(Vertex {
            position: [p.x, p.y, p.z],
            color: shade_color(color, normal),
        });
    }
}

/// Extract an Isometry3 from a geometry_msgs/Pose Value (quaternion is normalized). Shared by three renderers.
pub fn extract_pose(value: &Value) -> Result<Isometry3<f64>, String> {
    let get = |parent: &Value, ctx: &str, name: &str| -> Result<f64, String> {
        match parent.get(name) {
            Some(Value::F64(v)) => Ok(*v),
            _ => Err(format!("missing field `{ctx}.{name}` (float64)")),
        }
    };
    let Some(position) = value.get("position") else {
        return Err("missing field `position`".to_owned());
    };
    let Some(orientation) = value.get("orientation") else {
        return Err("missing field `orientation`".to_owned());
    };
    let translation = Translation3::new(
        get(position, "position", "x")?,
        get(position, "position", "y")?,
        get(position, "position", "z")?,
    );
    // Quaternion::new takes (w, x, y, z); from_quaternion normalizes it.
    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(
        get(orientation, "orientation", "w")?,
        get(orientation, "orientation", "x")?,
        get(orientation, "orientation", "y")?,
        get(orientation, "orientation", "z")?,
    ));
    Ok(Isometry3::from_parts(translation, rotation))
}

/// Extract header.frame_id and header.stamp (nanoseconds). Common preprocessing for stamped messages.
pub fn extract_header(value: &Value) -> Result<(String, TimeNs), String> {
    let frame_id = match value.get("header").and_then(|h| h.get("frame_id")) {
        Some(Value::String(s)) => s.clone(),
        _ => return Err("missing field `header.frame_id`".to_owned()),
    };
    let stamp_value = value.get("header").and_then(|h| h.get("stamp"));
    let stamp = match (
        stamp_value.and_then(|s| s.get("sec")),
        stamp_value.and_then(|s| s.get("nanosec")),
    ) {
        (Some(Value::I32(sec)), Some(Value::U32(nanosec))) => {
            *sec as i64 * 1_000_000_000 + *nanosec as i64
        }
        _ => return Err("missing field `header.stamp`".to_owned()),
    };
    Ok((frame_id, stamp))
}

/// Display-item id (monotonic sequence in add order). Also used to map GPU buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DisplayItemId(pub u64);

/// TF resolution context (built every frame on the UI thread; same data path as viewport's tf_scene).
pub struct TfContext<'a> {
    pub buffer: &'a TfBuffer,
    pub fixed_frame: &'a str,
}

impl TfContext<'_> {
    /// frame_id -> fixed-frame transform (lookup_transform_latest; failure returns None = skip drawing).
    pub fn resolve(&self, frame_id: &str) -> Option<Isometry3<f64>> {
        self.buffer
            .lookup_transform_latest(self.fixed_frame, frame_id)
            .ok()
    }

    /// resolve() plus the TF time it resolved at (inner None = static-only path); use the pair together so a time-based decision matches the pose drawn.
    pub fn resolve_stamped(&self, frame_id: &str) -> Option<(Isometry3<f64>, Option<TimeNs>)> {
        self.buffer
            .lookup_transform_latest_stamped(self.fixed_frame, frame_id)
            .ok()
    }

    /// Transform at the message time (header.stamp) so baking stays aligned even while rotating (stamp 0 means latest, per tf2).
    pub fn resolve_at(&self, frame_id: &str, stamp: TimeNs) -> Option<Isometry3<f64>> {
        if stamp == 0 {
            return self.resolve(frame_id);
        }
        self.buffer
            .lookup_transform(self.fixed_frame, frame_id, stamp)
            .ok()
    }
}

/// Why scene() cannot draw (shown as status in the Displays panel).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderStatus {
    /// No message received yet.
    NoData,
    /// Message shape differs from expected (type version mismatch or decode failure).
    InvalidMessage(String),
    /// Transform to fixed frame cannot be resolved.
    TfUnavailable { frame: String },
    /// A non-topic source is not loaded, or has nothing drawable (not an error; the renderer supplies the wording).
    NoSource(String),
    /// A non-topic source failed to load (missing file, parse error).
    SourceError(String),
}

/// Native file-open request raised from settings_ui. The renderer cannot open the dialog itself (rfd's xdg-portal backend needs comm's async runtime), so app.rs runs it and hands back the choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRequest {
    /// Filter label shown in the dialog (e.g. "URDF").
    pub filter_name: &'static str,
    /// Accepted extensions without the dot (e.g. `["urdf", "xml"]`).
    pub extensions: &'static [&'static str],
    /// Directory to open in (e.g. the currently loaded file's parent); None leaves it to the dialog.
    pub start_dir: Option<std::path::PathBuf>,
}

/// Companion subscription spec: subscribe to a sibling topic (base name + suffix) as ros_type (e.g. OccupancyGridUpdate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Companion {
    /// Suffix appended to the base topic name (e.g. `_updates`).
    pub suffix: &'static str,
    /// Companion topic's ROS-format type name (compared against TopicRow::ros_type and used to decode).
    pub ros_type: &'static str,
}

/// Common renderer interface (object safe). A wgpu-agnostic pure data producer; the viewport manages all GPU state.
pub trait Renderer {
    /// Ingest a decoded message (decoding already done on the tokio side; only light extraction here).
    fn on_message(&mut self, value: &Value);
    /// Return draw batches. Called every frame, but Arc-shared data + generation number uploads to GPU only on change.
    fn scene(&mut self, tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus>;
    /// Take in results of background work (e.g. mesh files loaded off-thread); true means more is outstanding, so app.rs schedules another frame. scene() cannot do this because it is skipped while the fixed frame is unset or the item is hidden.
    fn poll(&mut self) -> bool {
        false
    }
    /// Drop state accumulated over time because playback jumped (seek / loop / a new bag); keep settings, GPU resources and loaded models. Default no-op.
    fn reset(&mut self) {}
    /// Per-item settings panel (drawn in the expanded region of the Displays panel).
    fn settings_ui(&mut self, ui: &mut egui::Ui);
    /// Companion topic to subscribe beyond the base topic, e.g. incremental updates (default none).
    fn companion(&self) -> Option<Companion> {
        None
    }
    /// Ingest a decoded companion-topic message (default no-op).
    fn on_companion(&mut self, _value: &Value) {}
    /// Return current per-item settings as an opaque config value (None if the renderer has no settings).
    fn settings(&self) -> Option<toml::Value> {
        None
    }
    /// Apply saved settings (unknown/missing items fall back to each field's default; default no-op).
    fn apply_settings(&mut self, _value: &toml::Value) {}
    /// Why a settings-supplied source (file etc.) failed to load, for the notice shown after a config restore (default none).
    fn source_error(&self) -> Option<String> {
        None
    }
    /// Take a file-open request raised by settings_ui; app.rs opens the native dialog and reports back via on_file_picked (default none).
    fn take_file_request(&mut self) -> Option<FileRequest> {
        None
    }
    /// Receive the path chosen for a file request (not called when the dialog is cancelled; default no-op).
    fn on_file_picked(&mut self, _path: &std::path::Path) {}
}

/// Extracted draw data for one item (gathered from scene() on the UI thread and moved into the Callback).
pub struct ItemScene {
    pub id: DisplayItemId,
    pub batches: Vec<SceneBatch>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_batch_builder_packs_16_byte_little_endian_layout() {
        let mut builder = PointBatchBuilder::with_capacity(2);
        builder.push([1.0, 2.0, 3.0], [10, 20, 30, 255]);
        builder.push([-1.5, 0.0, 4.0], [0, 0, 0, 0]);
        let batch = builder.build();
        assert_eq!(batch.count, 2);
        assert_eq!(batch.bytes.len(), 2 * POINT_STRIDE);
        assert_eq!(&batch.bytes[0..4], &1.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[4..8], &2.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[8..12], &3.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[12..16], &[10, 20, 30, 255]);
        assert_eq!(&batch.bytes[16..20], &(-1.5_f32).to_le_bytes());
        assert_eq!(&batch.bytes[28..32], &[0, 0, 0, 0]);
    }

    #[test]
    fn mesh_batch_builder_packs_28_byte_little_endian_layout() {
        let mut builder = MeshBatchBuilder::with_capacity(2);
        builder.push_vertex([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [10, 20, 30, 255]);
        builder.push_vertex([-1.5, 0.0, 4.0], [-1.0, 0.0, 0.0], [0, 0, 0, 0]);
        assert_eq!(builder.len(), 2);
        let batch = builder.build();
        assert_eq!(batch.count, 2);
        assert_eq!(batch.bytes.len(), 2 * MESH_STRIDE);
        assert_eq!(&batch.bytes[0..4], &1.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[4..8], &2.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[8..12], &3.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[12..16], &0.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[16..20], &0.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[20..24], &1.0_f32.to_le_bytes());
        assert_eq!(&batch.bytes[24..28], &[10, 20, 30, 255]);
        // Second vertex starts exactly one stride later.
        assert_eq!(&batch.bytes[28..32], &(-1.5_f32).to_le_bytes());
        assert_eq!(&batch.bytes[40..44], &(-1.0_f32).to_le_bytes());
        assert_eq!(&batch.bytes[52..56], &[0, 0, 0, 0]);
        assert!(MeshBatchBuilder::default().is_empty());
    }

    #[test]
    fn mesh_batch_builder_push_triangle_derives_the_face_normal() {
        let mut builder = MeshBatchBuilder::default();
        // CCW triangle on the XY plane -> +Z normal for all 3 vertices.
        builder.push_triangle(
            [
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            [255, 255, 255, 255],
        );
        let batch = builder.build();
        assert_eq!(batch.count, 3);
        for i in 0..3 {
            let at = i * MESH_STRIDE + 12;
            let normal: [f32; 3] = std::array::from_fn(|c| {
                f32::from_le_bytes(
                    batch.bytes[at + c * 4..at + (c + 1) * 4]
                        .try_into()
                        .expect("4 bytes"),
                )
            });
            assert_eq!(normal, [0.0, 0.0, 1.0], "vertex {i}");
        }
        // Degenerate (zero-area) triangles fall back to +Z instead of emitting NaN normals.
        let mut degenerate = MeshBatchBuilder::default();
        degenerate.push_triangle([Point3::origin(); 3], [0, 0, 0, 0]);
        let batch = degenerate.build();
        assert_eq!(&batch.bytes[20..24], &1.0_f32.to_le_bytes());
    }

    #[test]
    fn posed_mesh_batch_carries_pose_and_alpha_without_touching_the_vertices() {
        let mut builder = MeshBatchBuilder::default();
        builder.push_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [255, 255, 255, 255]);
        let pose =
            Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), UnitQuaternion::identity());
        let batch = SceneBatch::posed_mesh(builder.build(), 7, &pose, 0.5);
        assert_eq!(batch.generation, 7);
        assert_eq!(batch.alpha, 0.5);
        assert_eq!(batch.model, pose.to_homogeneous());
        let BatchData::PosedMesh(mesh) = &batch.data else {
            panic!("expected PosedMesh");
        };
        // Vertices stay in local coordinates; the pose only travels in the batch uniform.
        assert_eq!(&mesh.bytes[0..4], &1.0_f32.to_le_bytes());
        // The other constructors leave alpha fully opaque.
        assert_eq!(SceneBatch::lines(Arc::new(Vec::new()), 0).alpha, 1.0);
        assert_eq!(SceneBatch::mesh(Arc::new(Vec::new()), 0).alpha, 1.0);
    }

    #[test]
    fn colormap_clamps_and_hits_end_stops() {
        let first = theme::to_linear_rgba8(theme::POINT_COLORMAP[0]);
        let last = theme::to_linear_rgba8(*theme::POINT_COLORMAP.last().unwrap());
        assert_eq!(colormap(0.0), first);
        assert_eq!(colormap(-5.0), first);
        assert_eq!(colormap(1.0), last);
        assert_eq!(colormap(7.0), last);
        assert_eq!(colormap(f32::NAN), first);
    }

    #[test]
    fn colormap_interpolates_between_stops() {
        // Midpoint of stop 0 and stop 1 (4 stops -> t = 1/6 is the center of segment 0).
        let mid = colormap(1.0 / 6.0);
        let a = theme::to_linear_rgba(theme::POINT_COLORMAP[0]);
        let b = theme::to_linear_rgba(theme::POINT_COLORMAP[1]);
        for i in 0..4 {
            let expected = ((a[i] + b[i]) * 0.5 * 255.0).round() as i32;
            assert!((mid[i] as i32 - expected).abs() <= 1, "channel {i}");
        }
    }

    #[test]
    fn occupancy_lut_viewer_maps_free_occupied_unknown_and_clamps() {
        let lut = occupancy_lut(OccupancyScheme::Viewer);
        let rgba = |color: egui::Color32, alpha: f32| -> [f32; 4] {
            let c = theme::to_linear_rgba(color);
            [c[0], c[1], c[2], alpha]
        };
        let free = rgba(theme::MAP_FREE, theme::MAP_FREE_ALPHA);
        let occupied = rgba(theme::MAP_OCCUPIED, theme::MAP_OCCUPIED_ALPHA);
        let unknown = rgba(theme::MAP_UNKNOWN, theme::MAP_UNKNOWN_ALPHA);
        assert_eq!(lut[0], free);
        assert_eq!(lut[100], occupied);
        // Mid value 50 is the midpoint in both color and alpha.
        for c in 0..4 {
            let expected = (free[c] + occupied[c]) * 0.5;
            assert!((lut[50][c] - expected).abs() < 1e-6, "channel {c}");
        }
        // 101..=127 clamp to occupied; 128 (-128)..=255 (-1) are unknown.
        assert_eq!(lut[101], occupied);
        assert_eq!(lut[127], occupied);
        assert_eq!(lut[128], unknown);
        assert_eq!(lut[255], unknown);
        // free / unknown are semi-transparent (grid shows through the tile); occupied is opaque.
        assert!(lut[0][3] < 0.5 && lut[255][3] < 0.5 && lut[100][3] == 1.0);
    }

    #[test]
    fn occupancy_lut_map_scheme_matches_rviz_palette() {
        let lut = occupancy_lut(OccupancyScheme::Map);
        // 0=white / 100=black / 50=gray (matches RViz down to the 255 - 255*50/100 = 128 integer math).
        assert_eq!(lut[0], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(lut[100], [0.0, 0.0, 0.0, 1.0]);
        let g = 128.0 / 255.0;
        assert_eq!(lut[50], [g, g, g, 1.0]);
        // Out-of-range positive=green / 128..=254=red->yellow / 255 (-1)=teal.
        assert_eq!(lut[101], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(lut[128], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(lut[254], [1.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            lut[255],
            [
                0x70 as f32 / 255.0,
                0x89 as f32 / 255.0,
                0x86 as f32 / 255.0,
                1.0
            ]
        );
    }

    #[test]
    fn occupancy_lut_costmap_scheme_matches_rviz_palette() {
        let lut = occupancy_lut(OccupancyScheme::Costmap);
        // 0 is fully transparent (free is not drawn as a tile).
        assert_eq!(lut[0], [0.0, 0.0, 0.0, 0.0]);
        // 1..=98 is a blue->red gradient (v = 255*i/100).
        assert_eq!(lut[1], [2.0 / 255.0, 0.0, 253.0 / 255.0, 1.0]);
        assert_eq!(lut[98], [249.0 / 255.0, 0.0, 6.0 / 255.0, 1.0]);
        // 99=cyan (inscribed) / 100=magenta (lethal).
        assert_eq!(lut[99], [0.0, 1.0, 1.0, 1.0]);
        assert_eq!(lut[100], [1.0, 0.0, 1.0, 1.0]);
        // raw scheme uses the value directly as luminance.
        let raw = occupancy_lut(OccupancyScheme::Raw);
        assert_eq!(raw[0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(raw[255], [1.0, 1.0, 1.0, 1.0]);
        let g = 128.0 / 255.0;
        assert_eq!(raw[128], [g, g, g, 1.0]);
    }

    #[test]
    fn scheme_palettes_match_the_lut_and_keep_one_shared_allocation() {
        for scheme in OccupancyScheme::ALL {
            let palette = scheme.palette();
            assert_eq!(*palette.0, occupancy_lut(scheme), "{scheme:?}");
            assert!(palette.ptr_eq(&scheme.palette()), "{scheme:?}");
        }
        assert!(
            !OccupancyScheme::Map
                .palette()
                .ptr_eq(&OccupancyScheme::Raw.palette())
        );
        assert!(!GridPalette::new([[0.0; 4]; 256]).ptr_eq(&GridPalette::new([[0.0; 4]; 256])));
    }

    #[test]
    fn intensity_color_normalizes_and_survives_degenerate_range() {
        assert_eq!(intensity_color((0.0, 10.0), 5.0), colormap(0.5));
        assert_eq!(intensity_color((0.0, 10.0), 20.0), colormap(1.0));
        assert_eq!(intensity_color((0.0, 10.0), -3.0), colormap(0.0));
        // min == max (all points same intensity) uses t = 0, no division by zero.
        assert_eq!(intensity_color((5.0, 5.0), 5.0), colormap(0.0));
    }

    #[test]
    fn push_prism_emits_cylinder_geometry_within_radius() {
        let mut out = Vec::new();
        push_prism(
            &mut out,
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            [1.0, 0.0, 0.0, 1.0],
            0.05,
        );
        assert_eq!(out.len(), prism_vertex_count());
        assert!(
            out.iter()
                .all(|v| v.position[0] >= -1e-6 && v.position[0] <= 1.0 + 1e-6)
        );
        assert!(
            out.iter()
                .all(|v| v.position[1].abs() <= 0.05 + 1e-6 && v.position[2].abs() <= 0.05 + 1e-6)
        );
    }

    #[test]
    fn push_prism_bakes_directional_shading_into_vertex_color() {
        let mut out = Vec::new();
        push_prism(
            &mut out,
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            [1.0, 1.0, 1.0, 1.0],
            0.05,
        );
        let reds: Vec<f32> = out.iter().map(|v| v.color[0]).collect();
        let min = reds.iter().cloned().fold(f32::MAX, f32::min);
        let max = reds.iter().cloned().fold(f32::MIN, f32::max);
        assert!(min < max, "faces should be shaded differently (not flat)");
        assert!(out.iter().all(|v| v.color[3] == 1.0));
    }

    #[test]
    fn push_prism_skips_zero_length_segment() {
        let mut out = Vec::new();
        push_prism(&mut out, Point3::origin(), Point3::origin(), [1.0; 4], 0.05);
        assert!(out.is_empty());
    }

    #[test]
    fn extract_pose_reads_position_and_normalized_quaternion() {
        let value = Value::Struct(vec![
            (
                "position".to_owned(),
                Value::Struct(vec![
                    ("x".to_owned(), Value::F64(1.0)),
                    ("y".to_owned(), Value::F64(2.0)),
                    ("z".to_owned(), Value::F64(3.0)),
                ]),
            ),
            (
                "orientation".to_owned(),
                Value::Struct(vec![
                    ("x".to_owned(), Value::F64(0.0)),
                    ("y".to_owned(), Value::F64(0.0)),
                    ("z".to_owned(), Value::F64(2.0)),
                    ("w".to_owned(), Value::F64(0.0)),
                ]),
            ),
        ]);
        let pose = extract_pose(&value).expect("valid pose");
        assert_eq!(pose.translation.vector.x, 1.0);
        assert_eq!(pose.translation.vector.z, 3.0);
        // (w=0, z=2) normalizes to a 180-degree yaw.
        assert!((pose.rotation.quaternion().norm() - 1.0).abs() < 1e-12);
        let rotated = pose.rotation * nalgebra::Vector3::x();
        assert!((rotated.x + 1.0).abs() < 1e-12);
        assert!(extract_pose(&Value::Struct(vec![])).is_err());
    }

    #[test]
    fn resolve_stamped_returns_pose_with_its_tf_time() {
        use crate::tf::buffer::{TfTransform, tf_update};
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(
            vec![TfTransform {
                parent: "map".to_owned(),
                child: "base_link".to_owned(),
                stamp: 7_000,
                transform: Isometry3::translation(1.0, 0.0, 0.0),
            }],
            false,
        ));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let (pose, stamp) = tf.resolve_stamped("base_link").expect("resolvable");
        assert_eq!(stamp, Some(7_000));
        assert_eq!(pose.translation.vector.x, 1.0);
        // The fixed frame itself resolves to identity with no time (no edge is walked).
        assert_eq!(
            tf.resolve_stamped("map"),
            Some((Isometry3::identity(), None))
        );
        assert_eq!(tf.resolve_stamped("nope"), None);
    }

    #[test]
    fn extract_header_reads_frame_and_stamp() {
        let value = Value::Struct(vec![(
            "header".to_owned(),
            Value::Struct(vec![
                (
                    "stamp".to_owned(),
                    Value::Struct(vec![
                        ("sec".to_owned(), Value::I32(2)),
                        ("nanosec".to_owned(), Value::U32(500)),
                    ]),
                ),
                ("frame_id".to_owned(), Value::String("laser".to_owned())),
            ]),
        )]);
        assert_eq!(
            extract_header(&value),
            Ok(("laser".to_owned(), 2_000_000_500))
        );
        assert!(extract_header(&Value::Struct(vec![])).is_err());
    }
}
