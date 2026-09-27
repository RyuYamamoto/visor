//! 3D viewport (wgpu setup, camera, grid, axes) via egui-wgpu Callback.

use std::collections::{HashMap, HashSet};

use egui_wgpu::{Callback, CallbackResources, CallbackTrait, RenderState, ScreenDescriptor};
use nalgebra::{Isometry3, Matrix4, Point3, Vector3};
use wgpu::util::DeviceExt;

use crate::render::camera::{ViewCameras, ViewType};
use crate::render::{
    BatchData, DisplayItemId, GridPalette, GridTexture, ItemScene, Label, MESH_STRIDE,
    POINT_STRIDE, SceneBatch, push_prism,
};
use crate::tf::buffer::TfBuffer;
use crate::theme;

pub use crate::render::Vertex;

/// Grid half-extent [m]; larger than the visible area because the outer rings fade out.
const GRID_HALF_EXTENT: f32 = 25.0;
/// Grid spacing [m].
const GRID_STEP: f32 = 1.0;
/// Radial distance [m] where the grid starts dimming, and where it is fully gone.
const GRID_FADE_START: f32 = 6.0;
const GRID_FADE_END: f32 = 24.0;
/// Grid Z offset [m] to avoid z-fighting with ground axis/TF lines; visually negligible.
const GRID_Z: f32 = -0.001;
/// Fixed-frame origin triad length [m] (matches RViz Axes default).
const ORIGIN_AXIS_LEN: f32 = 1.0;
/// Default TF frame triad length [m] (adjustable from the Frames panel).
const TF_AXIS_LEN_DEFAULT: f32 = 0.3;
/// Default TF axis/link line width [m] (adjustable from the Frames panel).
const TF_LINE_WIDTH_DEFAULT: f32 = 0.02;
/// Minimum line width [m] when extruding lines to prisms, guarding against degeneracy/NaN at 0.
const TF_LINE_WIDTH_MIN: f32 = 0.001;
/// Initial capacity [byte] of the TF axis dynamic vertex buffer (96 vertices; doubles when exceeded).
const TF_BUFFER_INITIAL_BYTES: usize = 96 * VERTEX_STRIDE as usize;
/// Initial capacity [byte] of display-item buffers (fits a few hundred LaserScan points; 100k-point clouds reach it via a few doublings).
const ITEM_BUFFER_INITIAL_BYTES: usize = 1024 * VERTEX_STRIDE as usize;
/// Frame-name label font size.
const FRAME_LABEL_FONT_SIZE: f32 = 11.0;
/// Label offset [pt] from the frame origin (down-right so it clears the triad).
const FRAME_LABEL_OFFSET: egui::Vec2 = egui::vec2(6.0, 3.0);
/// Inner padding of a frame-label chip.
const FRAME_LABEL_PADDING: egui::Vec2 = egui::vec2(4.0, 2.0);
/// Corner radius of a frame-label chip.
const FRAME_LABEL_CORNER: u8 = 3;
/// Chip slots around a frame origin; labels take one by name order so the arrangement is camera-independent.
const FRAME_LABEL_SLOTS: usize = 8;
/// Extra vertical gap [pt] between the stacked chip slots.
const FRAME_LABEL_SLOT_GAP: f32 = 4.0;
/// Inset [pt] of the on-screen instruments (scale bar, orientation gizmo) from the view edges.
const INSTRUMENT_MARGIN: f32 = 12.0;
/// Preferred and minimum on-screen width [pt] of the scale bar.
const SCALE_BAR_TARGET_PX: f32 = 96.0;
const SCALE_BAR_MIN_PX: f32 = 36.0;
/// End-tick height [pt] of the scale bar.
const SCALE_BAR_TICK: f32 = 4.0;
/// Arm length [pt] of the orientation gizmo, and its axis label size.
const GIZMO_RADIUS: f32 = 22.0;
const GIZMO_LABEL_FONT_SIZE: f32 = 9.0;
/// Min/max font size [pt] for TEXT_VIEW_FACING markers (projected world height is clamped so name tags stay readable).
const MARKER_TEXT_MIN_PX: f32 = 8.0;
const MARKER_TEXT_MAX_PX: f32 = 96.0;

const VERTEX_STRIDE: u64 = (3 + 4) * 4;

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

/// ribbon.wgsl instance layout: four bindings of the same buffer, each one Vertex record apart (prev / p0 / p1 / next).
const RIBBON_ATTRIBUTES: [[wgpu::VertexAttribute; 2]; 4] = [
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
    wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32x4],
    wgpu::vertex_attr_array![4 => Float32x3, 5 => Float32x4],
    wgpu::vertex_attr_array![6 => Float32x3, 7 => Float32x4],
];

/// Serialize a polyline for ribbon.wgsl: the first and last point are duplicated so every segment has a neighbour on both sides.
fn ribbon_bytes(points: &[Vertex]) -> Vec<u8> {
    let mut out = Vec::with_capacity((points.len() + 2) * VERTEX_STRIDE as usize);
    let padded = points
        .first()
        .into_iter()
        .chain(points.iter())
        .chain(points.last());
    for vertex in padded {
        for value in vertex.position {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in vertex.color {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// wide_lines.wgsl instance layout: one segment = two consecutive Vertex records (p0/c0/p1/c1), step_mode Instance.
const LINE_INSTANCE_STRIDE: u64 = VERTEX_STRIDE * 2;

const LINE_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32x3, 3 => Float32x4];

/// points.wgsl vertex attributes (16B/point: position + packed RGBA8; step_mode Instance).
const POINT_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32];

/// mesh.wgsl vertex attributes (28B/vertex: local position + local normal + packed RGBA8; step_mode Vertex).
const MESH_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4];

/// Offscreen scene target: float, so the glow blur has headroom and no banding in dark gradients.
const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// MSAA sample count for the scene pass, used when the adapter supports it for SCENE_FORMAT.
const MSAA_SAMPLES: u32 = 4;
/// The glow is built at 1/N resolution: cheaper and gives a wider falloff.
const BLOOM_DIVISOR: u32 = 2;
/// Luminance above which a pixel feeds the glow, and how much blurred light the composite adds back.
const BLOOM_THRESHOLD: f32 = 0.32;
const BLOOM_INTENSITY: f32 = 0.75;
/// post.wgsl group(0) uniform size (texel vec2 + threshold + intensity).
const POST_UNIFORM_SIZE: u64 = 16;
/// composite.wgsl group(0) uniform size (intensity + padding).
const COMPOSITE_UNIFORM_SIZE: u64 = 16;

/// points.wgsl group(0) uniform size (view_proj 64B + cam_right 16B + cam_up 16B + viewport 16B).
const POINTS_FRAME_UNIFORM_SIZE: u64 = 112;
/// mesh.wgsl group(0) uniform size (view_proj 64B + light_dir/ambient 16B + light_color/diffuse 16B).
const MESH_FRAME_UNIFORM_SIZE: u64 = 96;
/// points.wgsl / mesh.wgsl group(1) uniform size (model 64B + 2 batch scalars + padding).
const BATCH_UNIFORM_SIZE: u64 = 80;
/// occupancy.wgsl group(1) uniform size (model 64B + size_m 8B + alpha 4B + padding 4B).
const GRID_BATCH_UNIFORM_SIZE: u64 = 80;
/// occupancy.wgsl LUT uniform size (vec4<f32> x 256; written once at init).
const OCCUPANCY_LUT_SIZE: u64 = 4096;

fn vertex_bytes(vertices: &[Vertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vertices.len() * VERTEX_STRIDE as usize);
    for vertex in vertices {
        for value in vertex.position.iter().chain(vertex.color.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn write_mat4(out: &mut [u8], matrix: &Matrix4<f32>) {
    for (i, value) in matrix.as_slice().iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&value.to_le_bytes());
    }
}

/// Per-frame uniform for points.wgsl group(0): view_proj + camera basis + viewport physical pixel size.
fn points_frame_uniform_bytes(
    view_proj: &Matrix4<f32>,
    cam_right: &Vector3<f32>,
    cam_up: &Vector3<f32>,
    viewport_px: [f32; 2],
) -> [u8; POINTS_FRAME_UNIFORM_SIZE as usize] {
    let mut bytes = [0u8; POINTS_FRAME_UNIFORM_SIZE as usize];
    write_mat4(&mut bytes[0..64], view_proj);
    for (offset, v) in [(64, cam_right), (80, cam_up)] {
        for (i, value) in [v.x, v.y, v.z].iter().enumerate() {
            bytes[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes[96..100].copy_from_slice(&viewport_px[0].to_le_bytes());
    bytes[100..104].copy_from_slice(&viewport_px[1].to_le_bytes());
    bytes
}

/// Per-batch uniform for points.wgsl / wide_lines.wgsl group(1): model + half_size + size_mode. 80B/batch, written every frame (independent of vertex count).
fn batch_uniform_bytes(batch: &SceneBatch) -> [u8; BATCH_UNIFORM_SIZE as usize] {
    size_uniform_bytes(&batch.model, batch.size)
}

fn size_uniform_bytes(
    model: &Matrix4<f32>,
    size: crate::render::SizeSpec,
) -> [u8; BATCH_UNIFORM_SIZE as usize] {
    let mut bytes = [0u8; BATCH_UNIFORM_SIZE as usize];
    write_mat4(&mut bytes[0..64], model);
    let (half_size, size_mode) = match size {
        crate::render::SizeSpec::Meters(side) => (side * 0.5, 0u32),
        crate::render::SizeSpec::Pixels(side) => (side * 0.5, 1u32),
    };
    bytes[64..68].copy_from_slice(&half_size.to_le_bytes());
    bytes[68..72].copy_from_slice(&size_mode.to_le_bytes());
    bytes
}

/// Per-frame uniform for mesh.wgsl group(0): view_proj + world light direction/color and the ambient/diffuse coefficients from theme.
fn mesh_frame_uniform_bytes(view_proj: &Matrix4<f32>) -> [u8; MESH_FRAME_UNIFORM_SIZE as usize] {
    let mut bytes = [0u8; MESH_FRAME_UNIFORM_SIZE as usize];
    write_mat4(&mut bytes[0..64], view_proj);
    let light_color = theme::to_linear_rgba(theme::MESH_LIGHT_COLOR);
    let values = [
        theme::MESH_LIGHT_DIR[0],
        theme::MESH_LIGHT_DIR[1],
        theme::MESH_LIGHT_DIR[2],
        theme::MESH_AMBIENT,
        light_color[0],
        light_color[1],
        light_color[2],
        theme::MESH_DIFFUSE,
    ];
    for (i, value) in values.iter().enumerate() {
        let at = 64 + i * 4;
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Per-batch uniform for mesh.wgsl group(1): model + alpha. 80B/batch, written every frame so pose/opacity changes need no re-bake.
fn mesh_batch_uniform_bytes(batch: &SceneBatch) -> [u8; BATCH_UNIFORM_SIZE as usize] {
    let mut bytes = [0u8; BATCH_UNIFORM_SIZE as usize];
    write_mat4(&mut bytes[0..64], &batch.model);
    bytes[64..68].copy_from_slice(&batch.alpha.to_le_bytes());
    bytes
}

/// Per-batch uniform for occupancy.wgsl group(1): model + size_m + alpha. 80B/batch, written every frame.
fn grid_batch_uniform_bytes(
    batch: &SceneBatch,
    grid: &GridTexture,
) -> [u8; GRID_BATCH_UNIFORM_SIZE as usize] {
    let mut bytes = [0u8; GRID_BATCH_UNIFORM_SIZE as usize];
    write_mat4(&mut bytes[0..64], &batch.model);
    bytes[64..68].copy_from_slice(&grid.size_m[0].to_le_bytes());
    bytes[68..72].copy_from_slice(&grid.size_m[1].to_le_bytes());
    bytes[72..76].copy_from_slice(&grid.alpha.to_le_bytes());
    bytes
}

/// Flatten a grid palette to little-endian f32 for the uniform buffer.
fn palette_bytes(palette: &GridPalette) -> Vec<u8> {
    palette
        .0
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// Alpha multiplier that dissolves the grid radially, so it fades into the background instead of ending in a hard square.
fn grid_fade(x: f32, y: f32) -> f32 {
    let distance = (x * x + y * y).sqrt();
    let t = ((distance - GRID_FADE_START) / (GRID_FADE_END - GRID_FADE_START)).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// Grid lines on the XY plane (Z=0, center lines in GRID_3D_MAJOR), split per cell so the radial fade follows each line.
fn grid_vertices() -> Vec<Vertex> {
    let line_count = (GRID_HALF_EXTENT / GRID_STEP) as i32;
    let mut vertices = Vec::with_capacity(((line_count * 2 + 1) * line_count * 8) as usize);
    for i in -line_count..=line_count {
        let coord = i as f32 * GRID_STEP;
        let base = if i == 0 {
            theme::to_linear_rgba(theme::GRID_3D_MAJOR)
        } else {
            theme::to_linear_rgba(theme::GRID_3D)
        };
        for j in -line_count..line_count {
            let from = j as f32 * GRID_STEP;
            let to = (j + 1) as f32 * GRID_STEP;
            for [(x0, y0), (x1, y1)] in [[(from, coord), (to, coord)], [(coord, from), (coord, to)]]
            {
                for (x, y) in [(x0, y0), (x1, y1)] {
                    let color = [base[0], base[1], base[2], base[3] * grid_fade(x, y)];
                    vertices.push(Vertex {
                        position: [x, y, GRID_Z],
                        color,
                    });
                }
            }
        }
    }
    vertices
}

/// One axis line segment (start, end, linear RGBA color).
type AxisSegment = (Point3<f32>, Point3<f32>, [f32; 4]);

/// Build the 3 center segments of a triad (X red / Y green / Z blue) of length len at pose iso; kept separate from thickness for testability.
fn axis_segments(iso: &Isometry3<f32>, len: f32) -> [AxisSegment; 3] {
    let origin = iso * Point3::origin();
    let axes = [
        (Vector3::x(), theme::AXIS_X),
        (Vector3::y(), theme::AXIS_Y),
        (Vector3::z(), theme::AXIS_Z),
    ];
    std::array::from_fn(|i| {
        let (axis, color32) = &axes[i];
        let tip = origin + (iso.rotation * axis) * len;
        (origin, tip, theme::to_linear_rgba(*color32))
    })
}

/// TF draw data extracted relative to the fixed frame (pure data built on the UI thread).
struct TfScene {
    /// Frames for triads and labels (fixed itself is excluded, as it overlaps the origin triad).
    axes: Vec<(String, Isometry3<f64>)>,
    /// Parent->child connection lines (only edges resolvable at both ends against fixed).
    links: Vec<(Point3<f64>, Point3<f64>)>,
}

/// Skip frames that fail lookup (disconnected, extrapolation, etc.) and hidden frames.
fn tf_scene(buffer: &TfBuffer, fixed: &str, hidden: &HashSet<String>) -> TfScene {
    let mut poses: HashMap<String, Isometry3<f64>> = HashMap::new();
    poses.insert(fixed.to_owned(), Isometry3::identity());
    for frame in buffer.frame_names() {
        if frame != fixed
            && let Ok(iso) = buffer.lookup_transform_latest(fixed, &frame)
        {
            poses.insert(frame, iso);
        }
    }
    // Attribute links to the child frame: drop if the child is hidden, but use a hidden parent's position as an endpoint.
    let links = buffer
        .frames()
        .iter()
        .filter(|edge| !hidden.contains(edge.name))
        .filter_map(|edge| {
            let parent = poses.get(edge.parent)?;
            let child = poses.get(edge.name)?;
            Some((parent * Point3::origin(), child * Point3::origin()))
        })
        .collect();
    let mut axes: Vec<(String, Isometry3<f64>)> = poses
        .into_iter()
        .filter(|(name, _)| name != fixed && !hidden.contains(name))
        .collect();
    axes.sort_by(|a, b| a.0.cmp(&b.0));
    TfScene { axes, links }
}

fn tf_vertices(scene: &TfScene, axis_len: f32, line_width: f32) -> Vec<Vertex> {
    let half_width = (line_width * 0.5).max(TF_LINE_WIDTH_MIN);
    let mut vertices: Vec<Vertex> = Vec::new();
    for (_, pose) in &scene.axes {
        for (a, b, color) in axis_segments(&pose.cast::<f32>(), axis_len) {
            push_prism(&mut vertices, a, b, color, half_width);
        }
    }
    vertices
}

/// Parent->child link segments as LineList pairs; drawn by the thin antialiased line pipeline so they never compete with the axis triads.
fn tf_link_vertices(scene: &TfScene, show_links: bool) -> Vec<Vertex> {
    if !show_links {
        return Vec::new();
    }
    let color = theme::to_linear_rgba(theme::TF_LINK);
    scene
        .links
        .iter()
        .flat_map(|(parent, child)| {
            [parent.cast::<f32>(), child.cast::<f32>()].map(|p| Vertex {
                position: [p.x, p.y, p.z],
                color,
            })
        })
        .collect()
}

/// Project a world point to NDC (x, y); None if behind the camera.
fn project_to_ndc(view_proj: &Matrix4<f32>, position: &Point3<f32>) -> Option<(f32, f32)> {
    let clip = view_proj * position.to_homogeneous();
    if clip.w <= 0.0 {
        return None;
    }
    Some((clip.x / clip.w, clip.y / clip.w))
}

/// NDC plus the perspective depth `w` (1.0 under an orthographic projection), used to decide which chip wins an overlap.
fn project_to_ndc_depth(
    view_proj: &Matrix4<f32>,
    position: &Point3<f32>,
) -> Option<(f32, f32, f32)> {
    let clip = view_proj * position.to_homogeneous();
    if clip.w <= 0.0 {
        return None;
    }
    Some((clip.x / clip.w, clip.y / clip.w, clip.w))
}

/// Maps NDC to a screen position inside `rect`.
fn ndc_to_screen(rect: egui::Rect, ndc_x: f32, ndc_y: f32) -> egui::Pos2 {
    egui::pos2(
        rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
        rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
    )
}

/// Nice scale-bar lengths [m]; the one whose on-screen width lands closest to SCALE_BAR_TARGET_PX is used.
const SCALE_BAR_STEPS_M: [f32; 12] = [
    0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
];

/// Picks the scale-bar length and its on-screen width [px] from the measured pixels per meter.
fn scale_bar_choice(px_per_m: f32) -> Option<(f32, f32)> {
    if !px_per_m.is_finite() || px_per_m <= 0.0 {
        return None;
    }
    SCALE_BAR_STEPS_M
        .iter()
        .map(|&m| (m, m * px_per_m))
        .filter(|(_, px)| *px >= SCALE_BAR_MIN_PX)
        .min_by(|a, b| {
            (a.1 - SCALE_BAR_TARGET_PX)
                .abs()
                .total_cmp(&(b.1 - SCALE_BAR_TARGET_PX).abs())
        })
}

/// Bottom-left scale bar: measures pixels per meter at the view's focus point, then labels a round distance.
fn draw_scale_bar(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &Matrix4<f32>,
    focus: Point3<f32>,
    cam_right: &Vector3<f32>,
) {
    let (Some(a), Some(b)) = (
        project_to_ndc(view_proj, &focus),
        project_to_ndc(view_proj, &(focus + cam_right)),
    ) else {
        return;
    };
    let px_per_m = (ndc_to_screen(rect, a.0, a.1) - ndc_to_screen(rect, b.0, b.1)).length();
    let Some((meters, width)) = scale_bar_choice(px_per_m) else {
        return;
    };
    let left = rect.left() + INSTRUMENT_MARGIN;
    let y = rect.bottom() - INSTRUMENT_MARGIN;
    let stroke = egui::Stroke::new(1.0, theme::INSTRUMENT);
    painter.line_segment([egui::pos2(left, y), egui::pos2(left + width, y)], stroke);
    for x in [left, left + width] {
        painter.line_segment(
            [egui::pos2(x, y - SCALE_BAR_TICK), egui::pos2(x, y)],
            stroke,
        );
    }
    let label = if meters < 1.0 {
        format!("{:.1} m", meters)
    } else {
        format!("{meters:.0} m")
    };
    painter.text(
        egui::pos2(left, y - SCALE_BAR_TICK - 2.0),
        egui::Align2::LEFT_BOTTOM,
        label,
        egui::FontId::monospace(FRAME_LABEL_FONT_SIZE),
        theme::INSTRUMENT,
    );
}

/// Bottom-right orientation gizmo: the world axes projected with the camera basis, far axes drawn first.
fn draw_orientation_gizmo(
    painter: &egui::Painter,
    rect: egui::Rect,
    cam_right: &Vector3<f32>,
    cam_up: &Vector3<f32>,
) {
    let center = egui::pos2(
        rect.right() - INSTRUMENT_MARGIN - GIZMO_RADIUS,
        rect.bottom() - INSTRUMENT_MARGIN - GIZMO_RADIUS,
    );
    // Screen depth of a world axis: positive means it points away from the viewer.
    let forward = cam_right.cross(cam_up);
    let mut axes = [
        (Vector3::x(), theme::AXIS_X, "X"),
        (Vector3::y(), theme::AXIS_Y, "Y"),
        (Vector3::z(), theme::AXIS_Z, "Z"),
    ];
    axes.sort_by(|a, b| b.0.dot(&forward).total_cmp(&a.0.dot(&forward)));
    painter.circle_filled(center, 2.0, theme::INSTRUMENT);
    for (axis, color, name) in axes {
        let tip = center + egui::vec2(axis.dot(cam_right), -axis.dot(cam_up)) * GIZMO_RADIUS;
        painter.line_segment([center, tip], egui::Stroke::new(1.5, color));
        painter.text(
            tip,
            egui::Align2::CENTER_CENTER,
            name,
            egui::FontId::monospace(GIZMO_LABEL_FONT_SIZE),
            color,
        );
    }
}

/// Chip offsets around a frame origin: right/left of it, then progressively further above and below.
fn label_chip_offsets(chip: egui::Vec2) -> [egui::Vec2; FRAME_LABEL_SLOTS] {
    let (right, left) = (FRAME_LABEL_OFFSET.x, -(chip.x + FRAME_LABEL_OFFSET.x));
    let below = FRAME_LABEL_OFFSET.y;
    let above = -(chip.y + FRAME_LABEL_OFFSET.y);
    let step = chip.y + FRAME_LABEL_SLOT_GAP;
    [
        egui::vec2(right, below),
        egui::vec2(right, above),
        egui::vec2(left, below),
        egui::vec2(left, above),
        egui::vec2(right, below + step),
        egui::vec2(right, above - step),
        egui::vec2(left, below + step),
        egui::vec2(left, above - step),
    ]
}

/// Draws frame names as chips in fixed slots (name order), with a leader line whenever a chip sits off its frame.
fn draw_frame_labels(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &Matrix4<f32>,
    labels: &[(String, Point3<f32>, bool)],
) {
    let font = egui::FontId::proportional(FRAME_LABEL_FONT_SIZE);
    let mut placed: Vec<(String, egui::Pos2, f32, egui::Galley, bool)> = Vec::new();
    for (name, position, is_fixed) in labels {
        let Some((ndc_x, ndc_y, depth)) = project_to_ndc_depth(view_proj, position) else {
            continue;
        };
        placed.push((
            name.clone(),
            ndc_to_screen(rect, ndc_x, ndc_y),
            depth,
            painter
                .layout_no_wrap(name.clone(), font.clone(), theme::LABEL_TEXT)
                .as_ref()
                .clone(),
            *is_fixed,
        ));
    }
    // Slot comes from the name order alone, so a chip never moves as the camera does.
    placed.sort_by(|a, b| a.0.cmp(&b.0));
    let mut chips: Vec<(egui::Rect, usize, f32, egui::Galley, bool)> = placed
        .into_iter()
        .enumerate()
        .map(|(index, (_, anchor, depth, galley, is_fixed))| {
            let chip_size = galley.size() + FRAME_LABEL_PADDING * 2.0;
            let slot = index % FRAME_LABEL_SLOTS;
            let chip =
                egui::Rect::from_min_size(anchor + label_chip_offsets(chip_size)[slot], chip_size);
            (chip, slot, depth, galley, is_fixed)
        })
        .collect();
    // Overlaps hide the farther chip instead of moving it: only visibility changes, never position.
    chips.sort_by(|a, b| a.2.total_cmp(&b.2));
    let mut taken: Vec<egui::Rect> = Vec::with_capacity(chips.len());
    for (chip, slot, _, galley, is_fixed) in chips {
        if taken.iter().any(|t| t.intersects(chip)) {
            continue;
        }
        taken.push(chip);
        let anchor = chip.min - label_chip_offsets(chip.size())[slot];
        // A chip sitting off its frame gets a leader back to the origin, so the name still reads as belonging to it.
        if slot != 0 {
            let target = egui::pos2(
                anchor.x.clamp(chip.left(), chip.right()),
                anchor.y.clamp(chip.top(), chip.bottom()),
            );
            painter.line_segment(
                [anchor, target],
                egui::Stroke::new(1.0, theme::LABEL_LEADER),
            );
        }
        painter.rect(
            chip,
            FRAME_LABEL_CORNER,
            theme::LABEL_CHIP_BG,
            egui::Stroke::new(1.0, theme::LABEL_CHIP_BORDER),
            egui::StrokeKind::Inside,
        );
        let text_color = if is_fixed {
            theme::ACCENT_CYAN
        } else {
            theme::LABEL_TEXT
        };
        painter.text(
            chip.min + FRAME_LABEL_PADDING,
            egui::Align2::LEFT_TOP,
            galley.text(),
            font.clone(),
            text_color,
        );
    }
}

/// Dynamic buffer that doubles capacity on demand (byte-length managed); handles both 28B line vertices and 16B point instances.
struct GrowableBuffer {
    buffer: wgpu::Buffer,
    capacity_bytes: usize,
    label: &'static str,
}

impl GrowableBuffer {
    fn new(device: &wgpu::Device, label: &'static str, capacity_bytes: usize) -> Self {
        Self {
            buffer: Self::create(device, label, capacity_bytes),
            capacity_bytes,
            label,
        }
    }

    fn create(device: &wgpu::Device, label: &'static str, capacity_bytes: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity_bytes as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Double capacity only when it falls short, then write_buffer.
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if bytes.len() > self.capacity_bytes {
            let mut capacity = self.capacity_bytes;
            while capacity < bytes.len() {
                capacity *= 2;
            }
            self.buffer = Self::create(device, self.label, capacity);
            self.capacity_bytes = capacity;
        }
        queue.write_buffer(&self.buffer, 0, bytes);
    }

    fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(..)
    }

    /// Slice starting `offset` bytes in; ribbons bind the same buffer several times at one-point offsets.
    fn slice_from(&self, offset: u64) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(offset..)
    }
}

/// Offscreen render targets for one viewport size: the scene (optionally multisampled) plus the glow ping-pong pair.
struct Offscreen {
    size: [u32; 2],
    /// MSAA color target; None when the adapter offers no MSAA for SCENE_FORMAT and the scene is drawn directly.
    msaa: Option<wgpu::TextureView>,
    scene: wgpu::TextureView,
    depth: wgpu::TextureView,
    bloom: [wgpu::TextureView; 2],
    bright_src: wgpu::BindGroup,
    blur_h_src: wgpu::BindGroup,
    blur_v_src: wgpu::BindGroup,
    composite_src: wgpu::BindGroup,
}

/// Creates a color/depth texture view of the given size, format and sample count.
fn target_view(
    device: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Offscreen {
    fn new(device: &wgpu::Device, resources: &SceneResources, size: [u32; 2]) -> Self {
        let bloom_size = [
            (size[0] / BLOOM_DIVISOR).max(1),
            (size[1] / BLOOM_DIVISOR).max(1),
        ];
        let scene = target_view(device, "viewport scene", size, SCENE_FORMAT, 1);
        let msaa = (resources.sample_count > 1).then(|| {
            target_view(
                device,
                "viewport scene msaa",
                size,
                SCENE_FORMAT,
                resources.sample_count,
            )
        });
        let depth = target_view(
            device,
            "viewport depth",
            size,
            wgpu::TextureFormat::Depth32Float,
            resources.sample_count,
        );
        let bloom = [
            target_view(device, "viewport bloom a", bloom_size, SCENE_FORMAT, 1),
            target_view(device, "viewport bloom b", bloom_size, SCENE_FORMAT, 1),
        ];
        let post_src = |label: &str, source: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.post_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&resources.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: resources.post_uniform.as_entire_binding(),
                    },
                ],
            })
        };
        let bright_src = post_src("viewport bright source", &scene);
        let blur_h_src = post_src("viewport blur h source", &bloom[0]);
        let blur_v_src = post_src("viewport blur v source", &bloom[1]);
        let composite_src = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport composite source"),
            layout: &resources.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.composite_uniform.as_entire_binding(),
                },
            ],
        });
        Self {
            size,
            msaa,
            scene,
            depth,
            bloom,
            bright_src,
            blur_h_src,
            blur_v_src,
            composite_src,
        }
    }

    /// Texel size of the bloom textures, for the blur's sampling step.
    fn bloom_texel(&self) -> [f32; 2] {
        let bloom_size = [
            (self.size[0] / BLOOM_DIVISOR).max(1) as f32,
            (self.size[1] / BLOOM_DIVISOR).max(1) as f32,
        ];
        [1.0 / bloom_size[0], 1.0 / bloom_size[1]]
    }
}

/// GPU resources created once at startup (kept in egui-wgpu CallbackResources, keyed by type).
struct SceneResources {
    /// MSAA sample count actually in use for the scene pass (1 when unsupported).
    sample_count: u32,
    /// Offscreen scene/glow targets, rebuilt when the viewport size changes.
    offscreen: Option<Offscreen>,
    post_layout: wgpu::BindGroupLayout,
    bright_pipeline: wgpu::RenderPipeline,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    composite_layout: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    post_uniform: wgpu::Buffer,
    composite_uniform: wgpu::Buffer,
    /// Line segments as screen-space quads (1 segment = 1 instance); consumes the points frame uniform.
    line_pipeline: wgpu::RenderPipeline,
    /// Ground grid variant: density fade, and no depth write so it cannot punch holes into the map above it.
    grid_line_pipeline: wgpu::RenderPipeline,
    /// Polyline variant: mitered joins, so a curve is one continuous band (4 bindings of the same buffer).
    ribbon_pipeline: wgpu::RenderPipeline,
    mesh_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    static_buffer: wgpu::Buffer,
    /// Vertex count of the grid portion at the start of the static buffer (origin triad follows).
    grid_vertex_count: u32,
    static_vertex_count: u32,
    tf_buffer: GrowableBuffer,
    /// TF parent->child links as line-segment pairs (thin screen-space quads).
    tf_link_buffer: GrowableBuffer,
    points_pipeline: wgpu::RenderPipeline,
    points_frame_buffer: wgpu::Buffer,
    points_frame_bind_group: wgpu::BindGroup,
    /// Kept to build per-batch uniform (group(1)) bind groups during prepare; shared by Points, PosedMesh and Lines (same 80B layout).
    batch_layout: wgpu::BindGroupLayout,
    /// group(1) for the viewport's own one-pixel lines (grid, origin triad, TF links); written once, never changes.
    line_batch_params: BatchParams,
    posed_mesh_pipeline: wgpu::RenderPipeline,
    mesh_frame_buffer: wgpu::Buffer,
    mesh_frame_bind_group: wgpu::BindGroup,
    grid_pipeline: wgpu::RenderPipeline,
    /// Draw Behind tile pipeline with depth test disabled (Always); used to draw as background first.
    grid_pipeline_behind: wgpu::RenderPipeline,
    /// group(0): view_proj (shares uniform_buffer); the palette is per batch, so one bind group serves every tile.
    grid_frame_bind_group: wgpu::BindGroup,
    /// Kept to build TexturedQuad batch bind groups (uniform + texture view) during prepare.
    grid_batch_layout: wgpu::BindGroupLayout,
}

/// GPU backing for the per-batch uniform (model + half_size); Points batches only.
struct BatchParams {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl BatchParams {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport batch params"),
            size: BATCH_UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport batch params"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { buffer, bind_group }
    }
}

/// Texture backing for a TexturedQuad batch (same-size resends use write_texture; recreated only on size change).
struct GridTextureResources {
    texture: wgpu::Texture,
    buffer: wgpu::Buffer,
    lut_buffer: wgpu::Buffer,
    /// Palette last written to lut_buffer; compared by pointer so an unchanged one skips the 4 KiB upload.
    palette: GridPalette,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

impl GridTextureResources {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, grid: &GridTexture) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport occupancy cells"),
            size: wgpu::Extent3d {
                width: grid.width,
                height: grid.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport occupancy batch params"),
            size: GRID_BATCH_UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lut_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewport occupancy lut"),
            contents: &palette_bytes(&grid.palette),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport occupancy batch"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lut_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            texture,
            buffer,
            lut_buffer,
            palette: grid.palette.clone(),
            bind_group,
            size: (grid.width, grid.height),
        }
    }
}

/// Resident GPU resources for one batch; no re-upload while uploaded_generation matches scene()'s generation.
struct BatchResources {
    buffer: GrowableBuffer,
    uploaded_generation: Option<u64>,
    /// Uploaded vertex count (Lines) or instance count (Points); 1 for a transferred TexturedQuad.
    count: u32,
    params: Option<BatchParams>,
    texture: Option<GridTextureResources>,
}

/// GPU buffer store for display items, keyed by (item ID, batch index); resides in CallbackResources.
#[derive(Default)]
struct ItemResources {
    batches: HashMap<(DisplayItemId, usize), BatchResources>,
}

/// Create GPU resources and register them into CallbackResources; called once from ViewerApp::new.
pub fn init(render_state: &RenderState) {
    // Our pipelines write colors straight to the target, so the conversion has to know whether it encodes sRGB.
    let target_format = render_state.target_format;
    theme::set_target_srgb(target_format.is_srgb());
    let sample_count = if render_state
        .adapter
        .get_texture_format_features(SCENE_FORMAT)
        .flags
        .sample_count_supported(MSAA_SAMPLES)
    {
        MSAA_SAMPLES
    } else {
        1
    };
    eprintln!("scene target: {SCENE_FORMAT:?} with {sample_count}x MSAA");
    let scene_multisample = wgpu::MultisampleState {
        count: sample_count,
        ..Default::default()
    };
    eprintln!(
        "wgpu target format: {target_format:?} (srgb: {})",
        target_format.is_srgb()
    );
    let device = &render_state.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewport lines"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lines.wgsl").into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewport uniforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(64),
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewport lines"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    // Mesh pipeline: vertex-color passthrough, TriangleList (shading baked into vertex colors). Must match NativeOptions depth_buffer: 32 / multisampling: 0, else wgpu validation errors.
    let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewport mesh"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: VERTEX_STRIDE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &VERTEX_ATTRIBUTES,
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: scene_multisample,
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport view_proj"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewport uniforms"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let mut static_vertices = grid_vertices();
    let grid_vertex_count = static_vertices.len() as u32;
    for (a, b, color) in axis_segments(&Isometry3::identity(), ORIGIN_AXIS_LEN) {
        for point in [a, b] {
            static_vertices.push(Vertex {
                position: [point.x, point.y, point.z],
                color,
            });
        }
    }
    let static_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewport static vertices"),
        contents: &vertex_bytes(&static_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let points_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewport points"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/points.wgsl").into()),
    });
    let points_frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewport points frame"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(POINTS_FRAME_UNIFORM_SIZE),
            },
            count: None,
        }],
    });
    let batch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewport points batch"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(BATCH_UNIFORM_SIZE),
            },
            count: None,
        }],
    });
    let points_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewport points"),
        bind_group_layouts: &[Some(&points_frame_layout), Some(&batch_layout)],
        immediate_size: 0,
    });
    // depth / blend / multisample match lines so depth stays consistent within one render pass.
    let points_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewport points"),
        layout: Some(&points_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &points_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: POINT_STRIDE as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &POINT_ATTRIBUTES,
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: scene_multisample,
        fragment: Some(wgpu::FragmentState {
            module: &points_shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    // Line segments expand to screen-space quads (analytic edge AA), so they share the points frame uniform for the viewport size.
    let wide_line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewport wide lines"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wide_lines.wgsl").into()),
    });
    let wide_line_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport wide lines"),
            bind_group_layouts: &[Some(&points_frame_layout), Some(&batch_layout)],
            immediate_size: 0,
        });
    let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewport wide lines"),
        layout: Some(&wide_line_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &wide_line_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: LINE_INSTANCE_STRIDE,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &LINE_INSTANCE_ATTRIBUTES,
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: scene_multisample,
        fragment: Some(wgpu::FragmentState {
            module: &wide_line_shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    // Ground grid: same quads, but no depth write — screen-space expansion offsets its depth from the ground plane at grazing angles, which would punch holes into the map lying 1mm above it.
    let grid_line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewport grid lines"),
        layout: Some(&wide_line_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &wide_line_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: LINE_INSTANCE_STRIDE,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &LINE_INSTANCE_ATTRIBUTES,
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: scene_multisample,
        fragment: Some(wgpu::FragmentState {
            module: &wide_line_shader,
            entry_point: Some("fs_grid"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    // Polyline ribbons: same frame/batch uniforms as the wide lines, but each instance also sees its neighbours so the joins can be mitered.
    let ribbon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewport ribbon"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ribbon.wgsl").into()),
    });
    let ribbon_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewport ribbon"),
        layout: Some(&wide_line_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &ribbon_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &std::array::from_fn::<_, 4, _>(|slot| wgpu::VertexBufferLayout {
                array_stride: VERTEX_STRIDE,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &RIBBON_ATTRIBUTES[slot],
            }),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: scene_multisample,
        fragment: Some(wgpu::FragmentState {
            module: &ribbon_shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let points_frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport points frame"),
        size: POINTS_FRAME_UNIFORM_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let points_frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewport points frame"),
        layout: &points_frame_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: points_frame_buffer.as_entire_binding(),
        }],
    });

    // The viewport's own lines (grid, origin triad, TF links) are always one pixel wide, so their group(1) is written once here.
    let line_batch_params = BatchParams::new(device, &batch_layout);
    render_state.queue.write_buffer(
        &line_batch_params.buffer,
        0,
        &size_uniform_bytes(
            &Matrix4::identity(),
            crate::render::SizeSpec::Pixels(crate::render::LINE_WIDTH_PX_DEFAULT),
        ),
    );
    let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewport posed mesh"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
    });
    let mesh_frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewport posed mesh frame"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            // Visible to both stages: view_proj is read in vertex, the light terms in fragment.
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(MESH_FRAME_UNIFORM_SIZE),
            },
            count: None,
        }],
    });
    let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewport posed mesh"),
        bind_group_layouts: &[Some(&mesh_frame_layout), Some(&batch_layout)],
        immediate_size: 0,
    });
    // depth / blend / multisample match lines so depth stays consistent within one render pass.
    let posed_mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewport posed mesh"),
        layout: Some(&mesh_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &mesh_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: MESH_STRIDE as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &MESH_ATTRIBUTES,
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: scene_multisample,
        fragment: Some(wgpu::FragmentState {
            module: &mesh_shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let mesh_frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport posed mesh frame"),
        size: MESH_FRAME_UNIFORM_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mesh_frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewport posed mesh frame"),
        layout: &mesh_frame_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: mesh_frame_buffer.as_entire_binding(),
        }],
    });
    let occupancy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewport occupancy"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/occupancy.wgsl").into()),
    });
    let grid_frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewport occupancy frame"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(64),
            },
            count: None,
        }],
    });
    let grid_batch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewport occupancy batch"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                // Visible to both stages: size_m is read in vertex, alpha in fragment.
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(GRID_BATCH_UNIFORM_SIZE),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    // Declare the LUT's full length so validation catches layout mismatches early.
                    min_binding_size: wgpu::BufferSize::new(OCCUPANCY_LUT_SIZE),
                },
                count: None,
            },
        ],
    });
    let grid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewport occupancy"),
        bind_group_layouts: &[Some(&grid_frame_layout), Some(&grid_batch_layout)],
        immediate_size: 0,
    });
    // Translucent tiles never write depth (so points/lines behind survive); a separate Always-compare version draws Draw Behind first as background.
    let make_grid_pipeline = |depth_compare: wgpu::CompareFunction| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport occupancy"),
            layout: Some(&grid_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &occupancy_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(depth_compare),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: scene_multisample,
            fragment: Some(wgpu::FragmentState {
                module: &occupancy_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let grid_pipeline = make_grid_pipeline(wgpu::CompareFunction::LessEqual);
    let grid_pipeline_behind = make_grid_pipeline(wgpu::CompareFunction::Always);
    let grid_frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viewport occupancy frame"),
        layout: &grid_frame_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    // Post-processing: bright-pass and separable blur read one texture; the composite reads scene + glow.
    let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewport post"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post.wgsl").into()),
    });
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewport composite"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("viewport post sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    let uniform_entry = |binding: u32, size: u64| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    };
    let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewport post"),
        entries: &[
            texture_entry(0),
            sampler_entry(1),
            uniform_entry(2, POST_UNIFORM_SIZE),
        ],
    });
    let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewport composite"),
        entries: &[
            texture_entry(0),
            texture_entry(1),
            sampler_entry(2),
            uniform_entry(3, COMPOSITE_UNIFORM_SIZE),
        ],
    });
    let fullscreen_pipeline = |label: &str,
                               layout: &wgpu::BindGroupLayout,
                               shader: &wgpu::ShaderModule,
                               entry: &str,
                               format: wgpu::TextureFormat,
                               depth: bool| {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            // Only the composite runs inside egui's pass, which carries a depth attachment it must match.
            depth_stencil: depth.then(|| wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let bright_pipeline = fullscreen_pipeline(
        "viewport bright",
        &post_layout,
        &post_shader,
        "fs_bright",
        SCENE_FORMAT,
        false,
    );
    let blur_h_pipeline = fullscreen_pipeline(
        "viewport blur h",
        &post_layout,
        &post_shader,
        "fs_blur_h",
        SCENE_FORMAT,
        false,
    );
    let blur_v_pipeline = fullscreen_pipeline(
        "viewport blur v",
        &post_layout,
        &post_shader,
        "fs_blur_v",
        SCENE_FORMAT,
        false,
    );
    let composite_pipeline = fullscreen_pipeline(
        "viewport composite",
        &composite_layout,
        &composite_shader,
        "fs_main",
        target_format,
        true,
    );
    let post_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport post params"),
        size: POST_UNIFORM_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let composite_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport composite params"),
        size: COMPOSITE_UNIFORM_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_state
        .queue
        .write_buffer(&composite_uniform, 0, &BLOOM_INTENSITY.to_le_bytes());
    let mut resources = render_state.renderer.write();
    resources.callback_resources.insert(SceneResources {
        sample_count,
        offscreen: None,
        post_layout,
        bright_pipeline,
        blur_h_pipeline,
        blur_v_pipeline,
        composite_layout,
        composite_pipeline,
        sampler,
        post_uniform,
        composite_uniform,
        line_pipeline,
        grid_line_pipeline,
        ribbon_pipeline,
        mesh_pipeline,
        bind_group,
        uniform_buffer,
        static_buffer,
        grid_vertex_count,
        static_vertex_count: static_vertices.len() as u32,
        tf_buffer: GrowableBuffer::new(device, "viewport tf vertices", TF_BUFFER_INITIAL_BYTES),
        tf_link_buffer: GrowableBuffer::new(device, "viewport tf links", TF_BUFFER_INITIAL_BYTES),
        points_pipeline,
        points_frame_buffer,
        points_frame_bind_group,
        batch_layout,
        line_batch_params,
        posed_mesh_pipeline,
        mesh_frame_buffer,
        mesh_frame_bind_group,
        grid_pipeline,
        grid_pipeline_behind,
        grid_frame_bind_group,
        grid_batch_layout,
    });
    resources
        .callback_resources
        .insert(ItemResources::default());
}

/// Per-frame throwaway draw data; UI-thread-extracted values moved into prepare/paint.
struct ViewportCallback {
    view_proj: Matrix4<f32>,
    cam_right: Vector3<f32>,
    cam_up: Vector3<f32>,
    /// Viewport physical pixel size, for converting Points screen-fixed sizes.
    viewport_px: [f32; 2],
    tf_vertices: Vec<Vertex>,
    tf_link_vertices: Vec<Vertex>,
    /// Whether to draw the origin triad (false if the fixed frame is hidden; the grid always draws).
    show_origin: bool,
    /// Extracted batches of visible items (app.rs collects from renderer.scene(); data is Arc-shared).
    items: Vec<ItemScene>,
    /// All live item IDs (including hidden/errored), used to retain GPU buffers of deleted items.
    live_ids: Vec<DisplayItemId>,
}

impl CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(resources) = callback_resources.get_mut::<SceneResources>() else {
            return Vec::new();
        };
        let mut uniform_bytes = [0u8; 64];
        write_mat4(&mut uniform_bytes, &self.view_proj);
        queue.write_buffer(&resources.uniform_buffer, 0, &uniform_bytes);
        queue.write_buffer(
            &resources.points_frame_buffer,
            0,
            &points_frame_uniform_bytes(
                &self.view_proj,
                &self.cam_right,
                &self.cam_up,
                self.viewport_px,
            ),
        );
        queue.write_buffer(
            &resources.mesh_frame_buffer,
            0,
            &mesh_frame_uniform_bytes(&self.view_proj),
        );
        resources
            .tf_buffer
            .write(device, queue, &vertex_bytes(&self.tf_vertices));
        resources
            .tf_link_buffer
            .write(device, queue, &vertex_bytes(&self.tf_link_vertices));
        let batch_layout = resources.batch_layout.clone();
        let grid_batch_layout = resources.grid_batch_layout.clone();
        if let Some(items) = callback_resources.get_mut::<ItemResources>() {
            // Drop deleted items and vanished batch slots (keep hidden/errored batches).
            let batch_counts: HashMap<DisplayItemId, usize> = self
                .items
                .iter()
                .map(|scene| (scene.id, scene.batches.len()))
                .collect();
            items.batches.retain(|(id, index), _| {
                self.live_ids.contains(id) && batch_counts.get(id).is_none_or(|count| index < count)
            });
            for scene in &self.items {
                for (index, batch) in scene.batches.iter().enumerate() {
                    let entry =
                        items
                            .batches
                            .entry((scene.id, index))
                            .or_insert_with(|| BatchResources {
                                buffer: GrowableBuffer::new(
                                    device,
                                    "viewport display item",
                                    ITEM_BUFFER_INITIAL_BYTES,
                                ),
                                uploaded_generation: None,
                                count: 0,
                                params: None,
                                texture: None,
                            });
                    // Generation gate: skip point-count-proportional transfer/reserialization on unchanged frames.
                    let stale = entry.uploaded_generation != Some(batch.generation);
                    match &batch.data {
                        BatchData::Lines(vertices) => {
                            if stale {
                                entry.buffer.write(device, queue, &vertex_bytes(vertices));
                                entry.count = vertices.len() as u32;
                            }
                            // half_size / size_mode are a fixed 80B, written every frame so a width change needs no rebake.
                            let params = entry
                                .params
                                .get_or_insert_with(|| BatchParams::new(device, &batch_layout));
                            queue.write_buffer(&params.buffer, 0, &batch_uniform_bytes(batch));
                        }
                        BatchData::Ribbon(points) => {
                            if stale {
                                entry.buffer.write(device, queue, &ribbon_bytes(points));
                                entry.count = points.len() as u32;
                            }
                            let params = entry
                                .params
                                .get_or_insert_with(|| BatchParams::new(device, &batch_layout));
                            queue.write_buffer(&params.buffer, 0, &batch_uniform_bytes(batch));
                        }
                        BatchData::Mesh(vertices) => {
                            if stale {
                                entry.buffer.write(device, queue, &vertex_bytes(vertices));
                                entry.count = vertices.len() as u32;
                            }
                        }
                        BatchData::Points(points) => {
                            if stale {
                                entry.buffer.write(device, queue, &points.bytes);
                                entry.count = points.count;
                            }
                            // model / half_size are a fixed 80B, written every frame so TF/settings changes apply without rebaking.
                            let params = entry
                                .params
                                .get_or_insert_with(|| BatchParams::new(device, &batch_layout));
                            queue.write_buffer(&params.buffer, 0, &batch_uniform_bytes(batch));
                        }
                        BatchData::PosedMesh(mesh) => {
                            if stale {
                                entry.buffer.write(device, queue, &mesh.bytes);
                                entry.count = mesh.count;
                            }
                            // model / alpha are a fixed 80B (same group(1) layout as Points), written every frame so pose changes apply without rebaking.
                            let params = entry
                                .params
                                .get_or_insert_with(|| BatchParams::new(device, &batch_layout));
                            queue.write_buffer(&params.buffer, 0, &mesh_batch_uniform_bytes(batch));
                        }
                        BatchData::TexturedQuad(grid) => {
                            // Recreate the texture only on size change; same-size resends overwrite via write_texture (handles SLAM full-map resends).
                            let recreate = entry
                                .texture
                                .as_ref()
                                .is_none_or(|t| t.size != (grid.width, grid.height));
                            if recreate {
                                entry.texture = Some(GridTextureResources::new(
                                    device,
                                    &grid_batch_layout,
                                    grid,
                                ));
                            }
                            let tex = entry.texture.as_mut().expect("created above");
                            // Palettes are Arc-shared and rebuilt only when settings change, so this is a pointer compare.
                            if !recreate && !tex.palette.ptr_eq(&grid.palette) {
                                queue.write_buffer(
                                    &tex.lut_buffer,
                                    0,
                                    &palette_bytes(&grid.palette),
                                );
                                tex.palette = grid.palette.clone();
                            }
                            if stale || recreate {
                                queue.write_texture(
                                    tex.texture.as_image_copy(),
                                    &grid.pixels,
                                    wgpu::TexelCopyBufferLayout {
                                        offset: 0,
                                        // write_texture needs no 256B alignment (unlike encoder copies); pass tightly packed bytes.
                                        bytes_per_row: Some(grid.width),
                                        rows_per_image: None,
                                    },
                                    wgpu::Extent3d {
                                        width: grid.width,
                                        height: grid.height,
                                        depth_or_array_layers: 1,
                                    },
                                );
                                entry.count = 1;
                            }
                            // model / size_m / alpha are a fixed 80B, written every frame so TF/alpha changes apply without re-transfer.
                            queue.write_buffer(
                                &tex.buffer,
                                0,
                                &grid_batch_uniform_bytes(batch, grid),
                            );
                        }
                        // Labels are drawn by the painter in show(), not the GPU callback; keep no resources here.
                        BatchData::Labels(_) => {}
                    }
                    entry.uploaded_generation = Some(batch.generation);
                }
            }
        }
        let Some(resources) = callback_resources.get_mut::<SceneResources>() else {
            return Vec::new();
        };
        let size = [
            (self.viewport_px[0].round() as u32).max(1),
            (self.viewport_px[1].round() as u32).max(1),
        ];
        if resources.offscreen.as_ref().is_none_or(|off| off.size != size) {
            resources.offscreen = Some(Offscreen::new(device, resources, size));
            let off = resources.offscreen.as_ref().expect("offscreen just created");
            let texel = off.bloom_texel();
            let mut params = [0u8; POST_UNIFORM_SIZE as usize];
            params[0..4].copy_from_slice(&texel[0].to_le_bytes());
            params[4..8].copy_from_slice(&texel[1].to_le_bytes());
            params[8..12].copy_from_slice(&BLOOM_THRESHOLD.to_le_bytes());
            params[12..16].copy_from_slice(&BLOOM_INTENSITY.to_le_bytes());
            queue.write_buffer(&resources.post_uniform, 0, &params);
        }
        let resources = callback_resources
            .get::<SceneResources>()
            .expect("scene resources present");
        let items_res = callback_resources.get::<ItemResources>();
        let Some(off) = &resources.offscreen else {
            return Vec::new();
        };
        let clear = theme::to_linear_rgba(theme::VIEWPORT_BG);
        {
            let (view, resolve_target) = match &off.msaa {
                Some(msaa) => (msaa, Some(&off.scene)),
                None => (&off.scene, None),
            };
            let mut scene_pass = egui_encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewport scene"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: clear[0] as f64,
                                g: clear[1] as f64,
                                b: clear[2] as f64,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &off.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            draw_scene(&mut scene_pass, resources, items_res, self);
        }
        // Bright-pass then a separable blur, ping-ponging between the two half-res glow textures.
        for (label, pipeline, source, target) in [
            ("viewport bright", &resources.bright_pipeline, &off.bright_src, &off.bloom[0]),
            ("viewport blur h", &resources.blur_h_pipeline, &off.blur_h_src, &off.bloom[1]),
            ("viewport blur v", &resources.blur_v_pipeline, &off.blur_v_src, &off.bloom[0]),
        ] {
            let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, source, &[]);
            pass.draw(0..3, 0..1);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        let Some(resources) = callback_resources.get::<SceneResources>() else {
            return;
        };
        let Some(off) = &resources.offscreen else {
            return;
        };
        render_pass.set_pipeline(&resources.composite_pipeline);
        render_pass.set_bind_group(0, &off.composite_src, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Issues every scene draw for one frame; shared so the same code can target either an offscreen pass or egui's.
fn draw_scene(
    render_pass: &mut wgpu::RenderPass<'static>,
    resources: &SceneResources,
    items_res: Option<&ItemResources>,
    callback: &ViewportCallback,
) {
    let draw_tiles = |render_pass: &mut wgpu::RenderPass<'static>,
                      items: &ItemResources,
                      behind: bool,
                      pipeline: &wgpu::RenderPipeline| {
        for scene in &callback.items {
            for (index, batch) in scene.batches.iter().enumerate() {
                let BatchData::TexturedQuad(grid) = &batch.data else {
                    continue;
                };
                if grid.draw_behind != behind {
                    continue;
                }
                let Some(entry) = items.batches.get(&(scene.id, index)) else {
                    continue;
                };
                let Some(tex) = &entry.texture else {
                    continue;
                };
                if entry.count == 0 {
                    continue;
                }
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &resources.grid_frame_bind_group, &[]);
                render_pass.set_bind_group(1, &tex.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
        }
    };
    // Draw Behind maps render first with depth disabled, as background (grid, TF, other maps, points/lines appear on top).
    if let Some(items) = items_res {
        draw_tiles(render_pass, items, true, &resources.grid_pipeline_behind);
    }
    render_pass.set_bind_group(0, &resources.points_frame_bind_group, &[]);
    render_pass.set_bind_group(1, &resources.line_batch_params.bind_group, &[]);
    render_pass.set_vertex_buffer(0, resources.static_buffer.slice(..));
    let grid_instances = resources.grid_vertex_count / 2;
    render_pass.set_pipeline(&resources.grid_line_pipeline);
    render_pass.draw(0..4, 0..grid_instances);
    // The origin triad follows the grid in the same buffer, but keeps depth writes and no density fade.
    if callback.show_origin {
        render_pass.set_pipeline(&resources.line_pipeline);
        render_pass.draw(0..4, grid_instances..resources.static_vertex_count / 2);
    }
    let tf_count = callback.tf_vertices.len() as u32;
    if tf_count > 0 {
        render_pass.set_pipeline(&resources.mesh_pipeline);
        // The line pipeline above binds the points frame uniform, so rebind the mesh pipeline's own group 0.
        render_pass.set_bind_group(0, &resources.bind_group, &[]);
        render_pass.set_vertex_buffer(0, resources.tf_buffer.slice());
        render_pass.draw(0..tf_count, 0..1);
    }
    // Display items draw in the same render pass and depth buffer, staying depth-consistent with existing content.
    if let Some(items) = items_res {
        // Normal (non-Draw-Behind) tiles use depth test on, depth write off; points/lines composite on top afterwards.
        draw_tiles(render_pass, items, false, &resources.grid_pipeline);
    }
    // Links draw after the map: their quads carry endpoint depth, which could otherwise reject coplanar map fragments.
    let tf_link_count = callback.tf_link_vertices.len() as u32;
    if tf_link_count > 0 {
        render_pass.set_pipeline(&resources.line_pipeline);
        render_pass.set_bind_group(0, &resources.points_frame_bind_group, &[]);
        render_pass.set_bind_group(1, &resources.line_batch_params.bind_group, &[]);
        render_pass.set_vertex_buffer(0, resources.tf_link_buffer.slice());
        render_pass.draw(0..4, 0..tf_link_count / 2);
    }
    if let Some(items) = items_res {
        for scene in &callback.items {
            for (index, batch) in scene.batches.iter().enumerate() {
                let Some(entry) = items.batches.get(&(scene.id, index)) else {
                    continue;
                };
                if entry.count == 0 {
                    continue;
                }
                match &batch.data {
                    BatchData::Mesh(_) => {
                        render_pass.set_pipeline(&resources.mesh_pipeline);
                        render_pass.set_bind_group(0, &resources.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, entry.buffer.slice());
                        render_pass.draw(0..entry.count, 0..1);
                    }
                    BatchData::Lines(_) => {
                        let Some(params) = &entry.params else {
                            continue;
                        };
                        render_pass.set_pipeline(&resources.line_pipeline);
                        render_pass.set_bind_group(0, &resources.points_frame_bind_group, &[]);
                        render_pass.set_bind_group(1, &params.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, entry.buffer.slice());
                        render_pass.draw(0..4, 0..entry.count / 2);
                    }
                    BatchData::Ribbon(_) => {
                        let Some(params) = &entry.params else {
                            continue;
                        };
                        if entry.count < 2 {
                            continue;
                        }
                        render_pass.set_pipeline(&resources.ribbon_pipeline);
                        render_pass.set_bind_group(0, &resources.points_frame_bind_group, &[]);
                        render_pass.set_bind_group(1, &params.bind_group, &[]);
                        // Four bindings one point apart give each segment its neighbours.
                        for slot in 0..4u32 {
                            let offset = u64::from(slot) * VERTEX_STRIDE;
                            render_pass.set_vertex_buffer(slot, entry.buffer.slice_from(offset));
                        }
                        render_pass.draw(0..4, 0..entry.count - 1);
                    }
                    BatchData::Points(_) => {
                        let Some(params) = &entry.params else {
                            continue;
                        };
                        render_pass.set_pipeline(&resources.points_pipeline);
                        render_pass.set_bind_group(0, &resources.points_frame_bind_group, &[]);
                        render_pass.set_bind_group(1, &params.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, entry.buffer.slice());
                        render_pass.draw(0..4, 0..entry.count);
                    }
                    BatchData::PosedMesh(_) => {
                        let Some(params) = &entry.params else {
                            continue;
                        };
                        render_pass.set_pipeline(&resources.posed_mesh_pipeline);
                        render_pass.set_bind_group(0, &resources.mesh_frame_bind_group, &[]);
                        render_pass.set_bind_group(1, &params.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, entry.buffer.slice());
                        render_pass.draw(0..entry.count, 0..1);
                    }
                    // TexturedQuad is drawn by draw_tiles above; Labels are drawn by the painter in show().
                    BatchData::TexturedQuad(_) | BatchData::Labels(_) => {}
                }
            }
        }
    }
}

/// UI-side state of the View3d tab (camera and display toggles persist across tab closes).
pub struct ViewportState {
    pub cameras: ViewCameras,
    pub gpu_ready: bool,
    pub show_names: bool,
    pub show_links: bool,
    /// TF triad length [m], for visibility tuning in dense TF trees.
    pub tf_axis_len: f32,
    /// TF axis/link line width [m] (RViz-style world-space geometry width).
    pub tf_line_width: f32,
    /// Frames to hide (empty by default = show all; new frames show automatically).
    pub hidden_frames: HashSet<String>,
    /// Target Frame to follow (None = no follow, world-fixed camera).
    pub target_frame: Option<String>,
    /// Most recent successful fixed->target offset, kept when a lookup momentarily fails (no snap to origin).
    last_follow_offset: Vector3<f32>,
}

impl ViewportState {
    pub fn new(gpu_ready: bool) -> Self {
        Self {
            cameras: ViewCameras::default(),
            gpu_ready,
            show_names: true,
            show_links: true,
            tf_axis_len: TF_AXIS_LEN_DEFAULT,
            tf_line_width: TF_LINE_WIDTH_DEFAULT,
            hidden_frames: HashSet::new(),
            target_frame: None,
            last_follow_offset: Vector3::zeros(),
        }
    }
}

/// Top-left overlay: view-type combo, Target Frame combo, and Zero (reset current view). Foreground layer captures its own clicks.
fn show_overlay(ui: &egui::Ui, state: &mut ViewportState, rect: egui::Rect, tf_buffer: &TfBuffer) {
    let frame_names = tf_buffer.frame_names();
    egui::Area::new(ui.id().with("view_overlay"))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.left_top() + egui::vec2(8.0, 8.0))
        .show(ui.ctx(), |ui| {
            let p = theme::ui::palette();
            egui::Frame::default()
                .fill(p.overlay_bg)
                .stroke(egui::Stroke::new(1.0, p.border))
                .corner_radius(6)
                .inner_margin(egui::Margin::symmetric(8, 5))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("VIEW").small().color(p.text_muted));
                        egui::ComboBox::from_id_salt("view_type")
                            .selected_text(state.cameras.view_type.label())
                            .show_ui(ui, |ui| {
                                for view_type in ViewType::ALL {
                                    ui.selectable_value(
                                        &mut state.cameras.view_type,
                                        view_type,
                                        view_type.label(),
                                    );
                                }
                            });
                        ui.separator();
                        ui.label(egui::RichText::new("TARGET").small().color(p.text_muted));
                        egui::ComboBox::from_id_salt("target_frame")
                            .selected_text(state.target_frame.as_deref().unwrap_or("<none>"))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut state.target_frame, None, "<none>");
                                for name in &frame_names {
                                    ui.selectable_value(
                                        &mut state.target_frame,
                                        Some(name.clone()),
                                        name,
                                    );
                                }
                            });
                        ui.separator();
                        if ui.button("Zero").clicked() {
                            state.cameras.reset_current();
                        }
                    });
                });
        });
}

/// Draw the View3d tab (input handling -> TF extraction -> Callback). items are pre-extracted by app.rs from renderer.scene().
pub fn show(
    ui: &mut egui::Ui,
    state: &mut ViewportState,
    tf_buffer: &TfBuffer,
    fixed_frame: Option<&str>,
    items: Vec<ItemScene>,
    live_ids: Vec<DisplayItemId>,
) {
    if !state.gpu_ready {
        ui.colored_label(
            theme::ui::palette().status_error,
            "3D view unavailable: wgpu render state was not initialized",
        );
        return;
    }
    let (rect, response) =
        ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
    if rect.width() < 1.0 || rect.height() < 1.0 {
        return;
    }
    let shift = ui.input(|i| i.modifiers.shift);
    let delta = response.drag_delta();
    let secondary_drag = response.dragged_by(egui::PointerButton::Middle)
        || (shift && response.dragged_by(egui::PointerButton::Primary));
    let primary_drag = !shift && response.dragged_by(egui::PointerButton::Primary);
    let (scroll_y, zoom) = if response.hovered() {
        ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()))
    } else {
        (0.0, 1.0)
    };
    match state.cameras.view_type {
        ViewType::Orbit => {
            let cam = &mut state.cameras.orbit;
            if secondary_drag {
                cam.pan(delta.x, delta.y, rect.height());
            } else if primary_drag {
                cam.rotate(delta.x, delta.y);
            }
            // Prevent double-applying where pinch also emits scroll (zoom_delta takes priority).
            if zoom != 1.0 {
                cam.zoom_factor(zoom);
            } else if scroll_y != 0.0 {
                cam.zoom_scroll(scroll_y);
            }
        }
        ViewType::TopDownOrtho => {
            let cam = &mut state.cameras.topdown;
            // RViz FixedOrientationOrtho mapping: left = rotate about Z, middle/Shift+left = pan.
            if secondary_drag {
                cam.pan(delta.x, delta.y, rect.height());
            } else if primary_drag {
                cam.rotate_z(delta.x);
            }
            if zoom != 1.0 {
                cam.zoom_factor(zoom);
            } else if scroll_y != 0.0 {
                cam.zoom_scroll(scroll_y);
            }
        }
        ViewType::Fps => {
            let cam = &mut state.cameras.fps;
            if secondary_drag {
                cam.pan(delta.x, delta.y, rect.height());
            } else if primary_drag {
                cam.look(delta.x, delta.y);
            }
            if scroll_y != 0.0 {
                cam.dolly(scroll_y);
            }
        }
    }
    // Target Frame follow: translate the camera by the fixed->target offset (keep last on lookup failure; no origin snap).
    let follow = match (fixed_frame, state.target_frame.as_deref()) {
        (Some(fixed), Some(target)) if fixed != target => {
            if let Ok(iso) = tf_buffer.lookup_transform_latest(fixed, target) {
                state.last_follow_offset = iso.translation.vector.cast::<f32>();
            }
            state.last_follow_offset
        }
        _ => Vector3::zeros(),
    };
    let view_proj = state
        .cameras
        .view_proj(rect.width() / rect.height(), follow);
    show_overlay(ui, state, rect, tf_buffer);
    let scene = fixed_frame.map(|fixed| tf_scene(tf_buffer, fixed, &state.hidden_frames));
    let tf_vertices = scene
        .as_ref()
        .map(|scene| tf_vertices(scene, state.tf_axis_len, state.tf_line_width))
        .unwrap_or_default();
    let tf_link_vertices = scene
        .as_ref()
        .map(|scene| tf_link_vertices(scene, state.show_links))
        .unwrap_or_default();
    let fixed_visible = fixed_frame.is_none_or(|fixed| !state.hidden_frames.contains(fixed));
    let (cam_right, cam_up) = state.cameras.basis();
    let pixels_per_point = ui.ctx().pixels_per_point();
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme::VIEWPORT_BG);
    // Collect marker text labels before `items` moves into the callback; they are drawn by the painter, not the GPU.
    let marker_labels: Vec<Label> = items
        .iter()
        .flat_map(|scene| scene.batches.iter())
        .filter_map(|batch| match &batch.data {
            BatchData::Labels(labels) => Some(labels.iter().cloned()),
            _ => None,
        })
        .flatten()
        .collect();
    painter.add(Callback::new_paint_callback(
        rect,
        ViewportCallback {
            view_proj,
            cam_right,
            cam_up,
            viewport_px: [
                rect.width() * pixels_per_point,
                rect.height() * pixels_per_point,
            ],
            tf_vertices,
            tf_link_vertices,
            show_origin: fixed_visible,
            items,
            live_ids,
        },
    ));
    if state.show_names
        && let (Some(scene), Some(fixed)) = (&scene, fixed_frame)
    {
        let labels: Vec<(String, Point3<f32>, bool)> = fixed_visible
            .then(|| (fixed.to_owned(), Point3::origin(), true))
            .into_iter()
            .chain(scene.axes.iter().map(|(name, pose)| {
                let p = pose.translation.vector.cast::<f32>();
                (name.clone(), Point3::from(p), false)
            }))
            .collect();
        draw_frame_labels(&painter, rect, &view_proj, &labels);
    }
    draw_scale_bar(
        &painter,
        rect,
        &view_proj,
        state.cameras.focus_point(follow),
        &cam_right,
    );
    draw_orientation_gizmo(&painter, rect, &cam_right, &cam_up);
    // TEXT_VIEW_FACING markers: project the anchor, size the font from the world height, draw as a billboard.
    for label in &marker_labels {
        let position = Point3::new(label.position[0], label.position[1], label.position[2]);
        let Some((ndc_x, ndc_y)) = project_to_ndc(&view_proj, &position) else {
            continue;
        };
        // Project a point one world-height above (along camera up) to get the on-screen text height.
        let top = position + cam_up * label.height_m;
        let px = match project_to_ndc(&view_proj, &top) {
            Some((_, top_ndc_y)) => (ndc_y - top_ndc_y).abs() * 0.5 * rect.height(),
            None => MARKER_TEXT_MIN_PX,
        }
        .clamp(MARKER_TEXT_MIN_PX, MARKER_TEXT_MAX_PX);
        let screen = egui::pos2(
            rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
            rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
        );
        let c = label.color;
        let color = egui::Color32::from(egui::Rgba::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]));
        painter.text(
            screen,
            egui::Align2::CENTER_CENTER,
            &label.text,
            egui::FontId::proportional(px),
            color,
        );
    }
    // No fixed-rate repaint here: the view is event-driven (egui repaints on input; comm notifies on message/TF arrival).
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tf::buffer::{TfTransform, tf_update};
    use nalgebra::{Translation3, UnitQuaternion};

    #[test]
    fn label_slots_are_distinct_and_only_the_first_sits_on_the_frame() {
        let chip = egui::vec2(80.0, 16.0);
        let offsets = label_chip_offsets(chip);
        // Slot 0 is the plain "just right of the origin" spot, so a lone label needs no leader.
        assert_eq!(offsets[0], FRAME_LABEL_OFFSET);
        // Every slot is a different place, so consecutive names in a cluster never stack up.
        for (i, a) in offsets.iter().enumerate() {
            for b in offsets.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
        // Left-hand slots clear the origin by the chip width, right-hand ones sit just off it.
        assert!(offsets[2].x <= -chip.x);
        assert!(offsets[1].y <= -chip.y);
        // Assignment depends only on the name-sorted index, never on the camera.
        assert_eq!(0 % FRAME_LABEL_SLOTS, 0);
        assert_eq!(FRAME_LABEL_SLOTS % FRAME_LABEL_SLOTS, 0);
    }

    #[test]
    fn scale_bar_picks_a_round_length_near_the_target_width() {
        // 48 px/m: 2 m lands at 96 px, exactly the target width.
        let (meters, width) = scale_bar_choice(48.0).unwrap();
        assert_eq!(meters, 2.0);
        assert!((width - SCALE_BAR_TARGET_PX).abs() < 1e-3);
        // Zoomed far out, a large round distance is used instead of a sub-pixel bar.
        let (meters, width) = scale_bar_choice(0.5).unwrap();
        assert!(meters >= 100.0, "meters = {meters}");
        assert!(width >= SCALE_BAR_MIN_PX);
        // Degenerate scales produce no bar rather than a wrong one.
        assert!(scale_bar_choice(0.0).is_none());
        assert!(scale_bar_choice(f32::NAN).is_none());
    }

    #[test]
    fn shader_grid_cell_matches_the_rust_grid_step() {
        // The density fade needs the cell size, which the shader cannot import; keep the two literals in sync.
        let source = include_str!("shaders/wide_lines.wgsl");
        let line = source
            .lines()
            .find(|line| line.contains("const GRID_CELL_M"))
            .expect("wide_lines.wgsl declares GRID_CELL_M");
        let value: f32 = line
            .split('=')
            .nth(1)
            .and_then(|rhs| rhs.trim().trim_end_matches(';').parse().ok())
            .expect("GRID_CELL_M is a plain float literal");
        assert_eq!(value, GRID_STEP);
    }

    #[test]
    fn grid_covers_the_extent_split_per_cell() {
        let vertices = grid_vertices();
        let line_count = (GRID_HALF_EXTENT / GRID_STEP) as usize;
        // One segment = 2 vertices; every row is split per cell in both directions.
        assert_eq!(vertices.len(), (2 * line_count + 1) * line_count * 8);
        assert_eq!(vertices.len() % 2, 0);
        for vertex in &vertices {
            assert!(vertex.position[0].abs() <= GRID_HALF_EXTENT);
            assert!(vertex.position[1].abs() <= GRID_HALF_EXTENT);
            assert_eq!(vertex.position[2], GRID_Z);
        }
        let major = theme::to_linear_rgba(theme::GRID_3D_MAJOR);
        assert!(
            vertices
                .iter()
                .any(|v| v.position[1] == 0.0 && v.color[..3] == major[..3])
        );
    }

    #[test]
    fn grid_alpha_fades_out_with_radius() {
        let vertices = grid_vertices();
        assert!(
            vertices
                .iter()
                .any(|v| v.position == [0.0, 0.0, GRID_Z] && v.color[3] == 1.0)
        );
        assert!(
            vertices
                .iter()
                .filter(|v| v.position[0].hypot(v.position[1]) >= GRID_FADE_END)
                .all(|v| v.color[3] == 0.0)
        );
        assert_eq!(grid_fade(0.0, 0.0), 1.0);
        assert_eq!(grid_fade(GRID_FADE_START, 0.0), 1.0);
        assert_eq!(grid_fade(GRID_FADE_END, 0.0), 0.0);
        assert!(grid_fade(8.0, 0.0) > grid_fade(12.0, 0.0));
        assert!(grid_fade(12.0, 0.0) > 0.0);
    }

    #[test]
    fn identity_triad_spans_unit_axes() {
        let segments = axis_segments(&Isometry3::identity(), 1.0);
        assert_eq!(segments[0].0, Point3::origin());
        assert_eq!(segments[0].1, Point3::new(1.0, 0.0, 0.0));
        assert_eq!(segments[1].1, Point3::new(0.0, 1.0, 0.0));
        assert_eq!(segments[2].1, Point3::new(0.0, 0.0, 1.0));
        assert_eq!(segments[0].2, theme::to_linear_rgba(theme::AXIS_X));
        assert_eq!(segments[1].2, theme::to_linear_rgba(theme::AXIS_Y));
        assert_eq!(segments[2].2, theme::to_linear_rgba(theme::AXIS_Z));
    }

    #[test]
    fn translated_triad_offsets_all_endpoints() {
        let iso = Isometry3::from_parts(
            Translation3::new(1.0, 2.0, 3.0),
            nalgebra::UnitQuaternion::identity(),
        );
        let segments = axis_segments(&iso, 0.3);
        assert_eq!(segments[0].0, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(segments[0].1, Point3::new(1.3, 2.0, 3.0));
        assert_eq!(segments[1].1, Point3::new(1.0, 2.3, 3.0));
        assert_eq!(segments[2].1, Point3::new(1.0, 2.0, 3.3));
    }

    #[test]
    fn vertex_bytes_serializes_little_endian_with_expected_stride() {
        let vertex = Vertex {
            position: [1.0, 2.0, 3.0],
            color: [0.5, 0.25, 0.125, 1.0],
        };
        let bytes = vertex_bytes(&[vertex]);
        assert_eq!(bytes.len(), VERTEX_STRIDE as usize);
        assert_eq!(&bytes[0..4], &1.0_f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0.5_f32.to_le_bytes());
        assert_eq!(&bytes[24..28], &1.0_f32.to_le_bytes());
    }

    fn read_f32(bytes: &[u8], at: usize) -> f32 {
        f32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
    }

    #[test]
    fn mesh_batch_uniform_packs_model_then_alpha() {
        let pose =
            Isometry3::from_parts(Translation3::new(1.0, 2.0, 3.0), UnitQuaternion::identity());
        let batch = crate::render::SceneBatch::posed_mesh(
            crate::render::MeshBatchBuilder::default().build(),
            0,
            &pose,
            0.25,
        );
        let bytes = mesh_batch_uniform_bytes(&batch);
        assert_eq!(bytes.len(), BATCH_UNIFORM_SIZE as usize);
        // Column-major mat4: the translation column sits at floats 12..15.
        assert_eq!(read_f32(&bytes, 48), 1.0);
        assert_eq!(read_f32(&bytes, 52), 2.0);
        assert_eq!(read_f32(&bytes, 56), 3.0);
        assert_eq!(read_f32(&bytes, 60), 1.0);
        assert_eq!(read_f32(&bytes, 64), 0.25);
        // Remaining padding stays zeroed.
        assert!(bytes[68..].iter().all(|b| *b == 0));
    }

    #[test]
    fn mesh_frame_uniform_packs_view_proj_and_theme_light_terms() {
        let view_proj = Matrix4::identity();
        let bytes = mesh_frame_uniform_bytes(&view_proj);
        assert_eq!(bytes.len(), MESH_FRAME_UNIFORM_SIZE as usize);
        assert_eq!(&bytes[0..64], &{
            let mut expected = [0u8; 64];
            write_mat4(&mut expected, &view_proj);
            expected
        });
        // light_dir.xyz then ambient in w.
        for (i, expected) in theme::MESH_LIGHT_DIR.iter().enumerate() {
            assert_eq!(read_f32(&bytes, 64 + i * 4), *expected);
        }
        assert_eq!(read_f32(&bytes, 76), theme::MESH_AMBIENT);
        // light_color.rgb (linear) then diffuse in w.
        let light_color = theme::to_linear_rgba(theme::MESH_LIGHT_COLOR);
        for (i, expected) in light_color.iter().take(3).enumerate() {
            assert_eq!(read_f32(&bytes, 80 + i * 4), *expected);
        }
        assert_eq!(read_f32(&bytes, 92), theme::MESH_DIFFUSE);
    }

    fn static_tf(parent: &str, child: &str, xyz: (f64, f64, f64)) -> TfTransform {
        TfTransform {
            parent: parent.to_owned(),
            child: child.to_owned(),
            stamp: 0,
            transform: Isometry3::from_parts(
                Translation3::new(xyz.0, xyz.1, xyz.2),
                UnitQuaternion::identity(),
            ),
        }
    }

    #[test]
    fn tf_scene_skips_fixed_frame_and_disconnected_frames() {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![
                static_tf("map", "base_link", (1.0, 0.0, 0.0)),
                static_tf("island", "leaf", (5.0, 0.0, 0.0)),
            ], true));
        let scene = tf_scene(&buffer, "map", &HashSet::new());
        assert_eq!(scene.axes.len(), 1);
        assert_eq!(scene.axes[0].0, "base_link");
        assert_eq!(scene.axes[0].1.translation.vector.x, 1.0);
        let empty = tf_scene(&buffer, "unknown_frame", &HashSet::new());
        assert!(empty.axes.is_empty());
        assert!(empty.links.is_empty());
    }

    #[test]
    fn tf_scene_links_connect_parent_to_child_including_fixed_origin() {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![
                static_tf("map", "odom", (1.0, 0.0, 0.0)),
                static_tf("odom", "base_link", (0.0, 1.0, 0.0)),
            ], true));
        let scene = tf_scene(&buffer, "map", &HashSet::new());
        assert_eq!(scene.axes.len(), 2);
        let mut links = scene.links.clone();
        links.sort_by(|a, b| a.1.x.total_cmp(&b.1.x).then(a.1.y.total_cmp(&b.1.y)));
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].0, Point3::origin());
        assert_eq!(links[0].1, Point3::new(1.0, 0.0, 0.0));
        assert_eq!(links[1].0, Point3::new(1.0, 0.0, 0.0));
        assert_eq!(links[1].1, Point3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn tf_axes_are_prisms_and_links_are_separate_thin_segments() {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![
                static_tf("map", "odom", (1.0, 0.0, 0.0)),
                static_tf("odom", "base_link", (0.0, 1.0, 0.0)),
            ], true));
        let per_line = crate::render::prism_vertex_count();
        let scene = tf_scene(&buffer, "map", &HashSet::new());
        // Only the 2 triads (3 axes each) are prisms; links never inflate the axis mesh.
        assert_eq!(tf_vertices(&scene, 0.3, 0.02).len(), 6 * per_line);
        // Links are LineList pairs drawn by the thin line pipeline, and only when enabled.
        assert!(tf_link_vertices(&scene, false).is_empty());
        let links = tf_link_vertices(&scene, true);
        assert_eq!(links.len(), 2 * scene.links.len());
        let link_color = theme::to_linear_rgba(theme::TF_LINK);
        assert!(links.iter().all(|v| v.color == link_color));
        // Each pair is exactly one scene link's parent -> child endpoints (order follows scene.links).
        for (i, (parent, child)) in scene.links.iter().enumerate() {
            let to_array = |p: &Point3<f64>| [p.x as f32, p.y as f32, p.z as f32];
            assert_eq!(links[2 * i].position, to_array(parent));
            assert_eq!(links[2 * i + 1].position, to_array(child));
        }
        // Axis-length parameter reaches the endpoints (odom's +X tip at x = 1 + len).
        let long_axes = tf_vertices(&scene, 0.5, 0.02);
        assert!(long_axes.iter().any(|v| (v.position[0] - 1.5).abs() < 1e-6));
    }

    #[test]
    fn tf_scene_hidden_frames_drop_axes_and_child_links_but_keep_parent_endpoints() {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![
                static_tf("map", "odom", (1.0, 0.0, 0.0)),
                static_tf("odom", "base_link", (0.0, 1.0, 0.0)),
            ], true));
        let hidden = HashSet::from(["odom".to_owned()]);
        let scene = tf_scene(&buffer, "map", &hidden);
        // odom's axis disappears; base_link remains.
        assert_eq!(scene.axes.len(), 1);
        assert_eq!(scene.axes[0].0, "base_link");
        // map->odom (child hidden) disappears; odom->base_link (parent hidden) remains as an endpoint.
        assert_eq!(scene.links.len(), 1);
        assert_eq!(scene.links[0].0, Point3::new(1.0, 0.0, 0.0));
        assert_eq!(scene.links[0].1, Point3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn project_to_ndc_maps_target_to_center_and_culls_behind_camera() {
        let camera = crate::render::camera::OrbitCamera::default();
        let view_proj = camera.view_proj(16.0 / 9.0, Vector3::zeros());
        let (x, y) = project_to_ndc(&view_proj, &Point3::origin()).unwrap();
        assert!(x.abs() < 1e-5 && y.abs() < 1e-5);
        let eye = camera.eye();
        let behind = eye + (eye - camera.target);
        assert!(project_to_ndc(&view_proj, &behind).is_none());
    }
}
