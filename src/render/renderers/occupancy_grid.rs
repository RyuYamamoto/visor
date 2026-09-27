//! OccupancyGrid renderer (occupancy values as an R8Uint texture, drawn as a translucent tile per the origin Pose).

use std::sync::Arc;

use egui::RichText;
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};

use crate::decode::value::Value;
use crate::render::{
    Companion, GridTexture, MAX_TEXTURE_DIM, OccupancyScheme, RenderStatus, Renderer, SceneBatch,
    TfContext, extract_header, extract_pose,
};
use crate::theme;

/// Default tile opacity (matches RViz Map's alpha default).
const ALPHA_DEFAULT: f32 = 0.7;

/// Companion topic where nav2 costmap streams diffs (`<topic>_updates` = map_msgs/OccupancyGridUpdate).
const UPDATE_COMPANION: Companion = Companion {
    suffix: "_updates",
    ros_type: "map_msgs/msg/OccupancyGridUpdate",
};

/// One map extracted from an OccupancyGrid message.
#[derive(Debug)]
struct ExtractedGrid {
    frame_id: String,
    stamp: crate::tf::buffer::TimeNs,
    /// Cell side length [m/cell].
    resolution: f32,
    width: u32,
    height: u32,
    /// World Pose of cell (0,0)'s bottom-left corner (includes quaternion = supports yawed maps).
    origin: Isometry3<f64>,
    /// Occupancy bytes (raw int8 bit pattern, mapped directly to texture texels).
    pixels: Arc<Vec<u8>>,
}

pub struct OccupancyGridRenderer {
    grid: Option<ExtractedGrid>,
    parse_error: Option<String>,
    /// Composed placement fixed_from_frame x origin (follows via uniform only, no texture re-transfer).
    pose: Option<Isometry3<f32>>,
    pose_fixed: String,
    pose_dirty: bool,
    /// Bumped only on accepting a new message; alpha/pose/scheme apply via uniform, so generation stays.
    generation: u64,
    alpha: f32,
    scheme: OccupancyScheme,
    draw_behind: bool,
}

impl Default for OccupancyGridRenderer {
    fn default() -> Self {
        Self {
            grid: None,
            parse_error: None,
            pose: None,
            pose_fixed: String::new(),
            pose_dirty: false,
            generation: 0,
            alpha: ALPHA_DEFAULT,
            // Default to this viewer's cyan-glow palette; the RViz Map/Costmap/Raw palettes stay selectable.
            scheme: OccupancyScheme::Viewer,
            // Off, matching RViz's Draw Behind default (draw with normal depth/add order).
            draw_behind: false,
        }
    }
}

impl Renderer for OccupancyGridRenderer {
    fn on_message(&mut self, value: &Value) {
        match extract_grid(value) {
            Ok(grid) => {
                self.grid = Some(grid);
                self.generation += 1;
                self.parse_error = None;
                self.pose_dirty = true;
            }
            Err(e) => self.parse_error = Some(e),
        }
    }

    fn companion(&self) -> Option<Companion> {
        Some(UPDATE_COMPANION)
    }

    fn on_companion(&mut self, value: &Value) {
        let update = match extract_update(value) {
            Ok(update) => update,
            Err(e) => {
                eprintln!("visor: occupancy update decode failed: {e}");
                return;
            }
        };
        // Discard if the full map is not yet received or the rect is out of range; never corrupt the existing map.
        let Some(grid) = &mut self.grid else { return };
        match apply_update(grid, &update) {
            Ok(()) => self.generation += 1,
            Err(e) => eprintln!("visor: occupancy update ignored: {e}"),
        }
    }

    fn scene(&mut self, tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
        if let Some(error) = &self.parse_error {
            return Err(RenderStatus::InvalidMessage(error.clone()));
        }
        let Some(grid) = &self.grid else {
            return Err(RenderStatus::NoData);
        };
        if self.pose_dirty || self.pose_fixed != tf.fixed_frame {
            // Static maps may carry a stale stamp and never resend, so fall back to latest.
            let resolved = tf
                .resolve_at(&grid.frame_id, grid.stamp)
                .or_else(|| tf.resolve(&grid.frame_id));
            match resolved {
                Some(iso) => {
                    self.pose = Some((iso * grid.origin).cast::<f32>());
                    self.pose_fixed = tf.fixed_frame.to_owned();
                    self.pose_dirty = false;
                }
                None => {
                    if self.pose_fixed != tf.fixed_frame {
                        self.pose = None;
                        self.pose_fixed = tf.fixed_frame.to_owned();
                        self.pose_dirty = true;
                    }
                    if self.pose.is_none() {
                        return Err(RenderStatus::TfUnavailable {
                            frame: grid.frame_id.clone(),
                        });
                    }
                }
            }
        }
        let Some(pose) = &self.pose else {
            return Err(RenderStatus::TfUnavailable {
                frame: grid.frame_id.clone(),
            });
        };
        Ok(vec![SceneBatch::textured_quad(
            GridTexture {
                pixels: grid.pixels.clone(),
                width: grid.width,
                height: grid.height,
                size_m: [
                    grid.resolution * grid.width as f32,
                    grid.resolution * grid.height as f32,
                ],
                alpha: self.alpha,
                palette: self.scheme.palette(),
                draw_behind: self.draw_behind,
            },
            self.generation,
            pose,
        )])
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Color scheme").color(p.text_muted));
            egui::ComboBox::from_id_salt("occupancy_scheme")
                .selected_text(self.scheme.label())
                .show_ui(ui, |ui| {
                    for scheme in OccupancyScheme::ALL {
                        ui.selectable_value(&mut self.scheme, scheme, scheme.label());
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Alpha").color(p.text_muted));
            ui.add(egui::Slider::new(&mut self.alpha, 0.0..=1.0));
        });
        ui.checkbox(&mut self.draw_behind, "Draw behind");
        if let Some(grid) = &self.grid {
            ui.label(
                RichText::new(format!(
                    "{} x {} @ {} m/cell",
                    grid.width, grid.height, grid.resolution
                ))
                .color(p.text_muted),
            );
        }
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(OccupancyGridSettings {
            alpha: self.alpha,
            scheme: self.scheme,
            draw_behind: self.draw_behind,
        })
        .ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(s) = value.clone().try_into::<OccupancyGridSettings>() {
            self.alpha = s.alpha;
            self.scheme = s.scheme;
            self.draw_behind = s.draw_behind;
        }
    }
}

/// Persistence DTO for OccupancyGridRenderer's user-editable settings (uniform-equivalent values scene() reads each frame).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct OccupancyGridSettings {
    alpha: f32,
    scheme: OccupancyScheme,
    draw_behind: bool,
}

impl Default for OccupancyGridSettings {
    fn default() -> Self {
        Self {
            alpha: ALPHA_DEFAULT,
            scheme: OccupancyScheme::Viewer,
            draw_behind: false,
        }
    }
}

fn get_u32(value: &Value, ctx: &str, field: &str) -> Result<u32, String> {
    match value.get(field) {
        Some(Value::U32(v)) => Ok(*v),
        _ => Err(format!("missing field `{ctx}.{field}` (uint32)")),
    }
}

/// Extract a map from an OccupancyGrid Value (reject inconsistent data with diagnostics; never panic).
fn extract_grid(value: &Value) -> Result<ExtractedGrid, String> {
    let (frame_id, stamp) = extract_header(value)?;
    let Some(info) = value.get("info") else {
        return Err("missing field `info` (MapMetaData)".to_owned());
    };
    let resolution = match info.get("resolution") {
        Some(Value::F32(v)) => *v,
        _ => return Err("missing field `info.resolution` (float32)".to_owned()),
    };
    let width = get_u32(info, "info", "width")?;
    let height = get_u32(info, "info", "height")?;
    let Some(origin_value) = info.get("origin") else {
        return Err("missing field `info.origin` (Pose)".to_owned());
    };
    let origin = extract_pose(origin_value)?;
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(format!("resolution must be > 0 (got {resolution})"));
    }
    if width == 0 || height == 0 {
        return Err(format!("map {width}x{height} has a zero dimension"));
    }
    if width > MAX_TEXTURE_DIM || height > MAX_TEXTURE_DIM {
        return Err(format!(
            "map {width}x{height} exceeds texture limit {MAX_TEXTURE_DIM}"
        ));
    }
    let Some(Value::Bytes(data)) = value.get("data") else {
        return Err("missing field `data` (int8[])".to_owned());
    };
    let cells = width as usize * height as usize;
    if data.len() != cells {
        return Err(format!(
            "data length {} does not match {width}x{height} cells",
            data.len()
        ));
    }
    Ok(ExtractedGrid {
        frame_id,
        stamp,
        resolution,
        width,
        height,
        origin,
        pixels: Arc::new(data.clone()),
    })
}

/// Diff rectangle extracted from OccupancyGridUpdate (row-major, [x, x+width) x [y, y+height)).
#[derive(Debug)]
struct ExtractedUpdate {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    data: Vec<u8>,
}

fn get_i32(value: &Value, field: &str) -> Result<i32, String> {
    match value.get(field) {
        Some(Value::I32(v)) => Ok(*v),
        _ => Err(format!("missing field `{field}` (int32)")),
    }
}

/// Extract a diff rectangle from an OccupancyGridUpdate Value (reject inconsistencies with diagnostics).
fn extract_update(value: &Value) -> Result<ExtractedUpdate, String> {
    let x = get_i32(value, "x")?;
    let y = get_i32(value, "y")?;
    let width = get_u32(value, "OccupancyGridUpdate", "width")?;
    let height = get_u32(value, "OccupancyGridUpdate", "height")?;
    let Some(Value::Bytes(data)) = value.get("data") else {
        return Err("missing field `data` (int8[])".to_owned());
    };
    let cells = width as usize * height as usize;
    if data.len() != cells {
        return Err(format!(
            "update data length {} does not match {width}x{height} cells",
            data.len()
        ));
    }
    Ok(ExtractedUpdate {
        x,
        y,
        width,
        height,
        data: data.clone(),
    })
}

/// Overwrite the held grid with the diff rectangle (reject out-of-range/negative offsets to avoid corrupting the map).
fn apply_update(grid: &mut ExtractedGrid, update: &ExtractedUpdate) -> Result<(), String> {
    if update.x < 0 || update.y < 0 {
        return Err(format!("negative offset ({}, {})", update.x, update.y));
    }
    let (ox, oy) = (update.x as u32, update.y as u32);
    if ox + update.width > grid.width || oy + update.height > grid.height {
        return Err(format!(
            "region {}x{} at ({ox}, {oy}) exceeds map {}x{}",
            update.width, update.height, grid.width, grid.height
        ));
    }
    let pixels = Arc::make_mut(&mut grid.pixels);
    for row in 0..update.height {
        let src = (row * update.width) as usize;
        let dst = ((oy + row) * grid.width + ox) as usize;
        pixels[dst..dst + update.width as usize]
            .copy_from_slice(&update.data[src..src + update.width as usize]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::BatchData;
    use crate::tf::buffer::{TfBuffer, TfTransform, tf_update};

    #[test]
    fn settings_roundtrip_via_trait() {
        let r = OccupancyGridRenderer {
            alpha: 0.5,
            scheme: OccupancyScheme::Costmap,
            draw_behind: true,
            ..Default::default()
        };
        let value = r.settings().expect("occupancy grid has settings");
        let mut restored = OccupancyGridRenderer::default();
        restored.apply_settings(&value);
        assert_eq!(restored.alpha, r.alpha);
        assert_eq!(restored.scheme, r.scheme);
        assert_eq!(restored.draw_behind, r.draw_behind);
    }
    use nalgebra::{Translation3, UnitQuaternion, Vector3};

    fn pose_value(x: f64, y: f64, yaw: f64) -> Value {
        let q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
        Value::Struct(vec![
            (
                "position".to_owned(),
                Value::Struct(vec![
                    ("x".to_owned(), Value::F64(x)),
                    ("y".to_owned(), Value::F64(y)),
                    ("z".to_owned(), Value::F64(0.0)),
                ]),
            ),
            (
                "orientation".to_owned(),
                Value::Struct(vec![
                    ("x".to_owned(), Value::F64(q.i)),
                    ("y".to_owned(), Value::F64(q.j)),
                    ("z".to_owned(), Value::F64(q.k)),
                    ("w".to_owned(), Value::F64(q.w)),
                ]),
            ),
        ])
    }

    fn header_value(stamp: i64) -> Value {
        Value::Struct(vec![
            (
                "stamp".to_owned(),
                Value::Struct(vec![
                    ("sec".to_owned(), Value::I32((stamp / 1_000_000_000) as i32)),
                    (
                        "nanosec".to_owned(),
                        Value::U32((stamp % 1_000_000_000) as u32),
                    ),
                ]),
            ),
            ("frame_id".to_owned(), Value::String("map".to_owned())),
        ])
    }

    fn grid_value_stamped(
        stamp: i64,
        width: u32,
        height: u32,
        resolution: f32,
        origin: Value,
        data: Vec<u8>,
    ) -> Value {
        Value::Struct(vec![
            ("header".to_owned(), header_value(stamp)),
            (
                "info".to_owned(),
                Value::Struct(vec![
                    (
                        "map_load_time".to_owned(),
                        Value::Struct(vec![
                            ("sec".to_owned(), Value::I32(0)),
                            ("nanosec".to_owned(), Value::U32(0)),
                        ]),
                    ),
                    ("resolution".to_owned(), Value::F32(resolution)),
                    ("width".to_owned(), Value::U32(width)),
                    ("height".to_owned(), Value::U32(height)),
                    ("origin".to_owned(), origin),
                ]),
            ),
            ("data".to_owned(), Value::Bytes(data)),
        ])
    }

    fn grid_value(width: u32, height: u32, resolution: f32, origin: Value, data: Vec<u8>) -> Value {
        grid_value_stamped(0, width, height, resolution, origin, data)
    }

    #[test]
    fn extracts_metadata_and_row_major_pixels() {
        // 4x3 known pattern: cell (x, y) = x + y*4 (row-major, x varies first).
        let data: Vec<u8> = (0..12).collect();
        let value = grid_value(4, 3, 0.5, pose_value(1.0, 2.0, 0.0), data.clone());
        let grid = extract_grid(&value).expect("valid grid");
        assert_eq!(grid.frame_id, "map");
        assert_eq!((grid.width, grid.height), (4, 3));
        assert_eq!(grid.resolution, 0.5);
        assert_eq!(grid.origin.translation.vector.x, 1.0);
        // Texels follow data order (row 0 = y=0, ascending x within a row).
        assert_eq!(*grid.pixels, data);
        assert_eq!(grid.pixels[1 + 2 * 4], 9);
    }

    #[test]
    fn extract_rejects_inconsistent_data_with_diagnostics() {
        let err = extract_grid(&grid_value(4, 3, 0.5, pose_value(0.0, 0.0, 0.0), vec![0; 11]))
            .expect_err("length mismatch");
        assert_eq!(err, "data length 11 does not match 4x3 cells");
        let err = extract_grid(&grid_value(4, 3, 0.0, pose_value(0.0, 0.0, 0.0), vec![0; 12]))
            .expect_err("zero resolution");
        assert_eq!(err, "resolution must be > 0 (got 0)");
        let err = extract_grid(&grid_value(0, 3, 0.5, pose_value(0.0, 0.0, 0.0), vec![]))
            .expect_err("zero width");
        assert_eq!(err, "map 0x3 has a zero dimension");
        let err = extract_grid(&grid_value(
            MAX_TEXTURE_DIM + 1,
            2,
            0.5,
            pose_value(0.0, 0.0, 0.0),
            vec![],
        ))
        .expect_err("over limit");
        assert_eq!(err, "map 8193x2 exceeds texture limit 8192");
        assert!(extract_grid(&Value::Struct(vec![])).is_err());
    }

    fn map_tf(x: f64, stamp: i64) -> TfTransform {
        TfTransform {
            parent: "odom".to_owned(),
            child: "map".to_owned(),
            stamp,
            transform: Isometry3::from_parts(
                Translation3::new(x, 0.0, 0.0),
                UnitQuaternion::identity(),
            ),
        }
    }

    fn tf_with(transforms: Vec<TfTransform>) -> TfBuffer {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(transforms, false));
        buffer
    }

    #[test]
    fn scene_composes_origin_into_model_and_sizes_quad() {
        let mut renderer = OccupancyGridRenderer::default();
        let buffer = tf_with(vec![map_tf(10.0, 1_000)]);
        renderer.on_message(&grid_value(
            4,
            2,
            0.5,
            pose_value(1.0, 2.0, 0.0),
            vec![0; 8],
        ));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "odom",
        };
        let batches = renderer.scene(&tf).expect("baked");
        let BatchData::TexturedQuad(grid) = &batches[0].data else {
            panic!("expected textured quad");
        };
        assert_eq!((grid.width, grid.height), (4, 2));
        assert_eq!(grid.size_m, [2.0, 1.0]);
        assert_eq!(grid.alpha, ALPHA_DEFAULT);
        // Default scheme is the viewer's own palette (RViz palettes remain selectable).
        assert!(grid.palette.ptr_eq(&OccupancyScheme::Viewer.palette()));
        // model = fixed_from_map (x=10) × origin (x=1, y=2)
        assert!((batches[0].model[(0, 3)] - 11.0).abs() < 1e-5);
        assert!((batches[0].model[(1, 3)] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn scene_applies_origin_yaw_rotation() {
        let mut renderer = OccupancyGridRenderer::default();
        let buffer = tf_with(vec![map_tf(0.0, 1_000)]);
        renderer.on_message(&grid_value(
            2,
            2,
            1.0,
            pose_value(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            vec![0; 4],
        ));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "odom",
        };
        let batches = renderer.scene(&tf).expect("baked");
        // Local +X (cell x direction) points to fixed's +Y.
        assert!(batches[0].model[(0, 0)].abs() < 1e-5);
        assert!((batches[0].model[(1, 0)] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn generation_bumps_only_on_new_message() {
        let mut renderer = OccupancyGridRenderer::default();
        let buffer = tf_with(vec![map_tf(0.0, 1_000)]);
        let value = grid_value(2, 2, 1.0, pose_value(0.0, 0.0, 0.0), vec![0; 4]);
        renderer.on_message(&value);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "odom",
        };
        let gen1 = renderer.scene(&tf).expect("baked")[0].generation;
        // Alpha/scheme change and re-scene keep the generation (no texture re-transfer = generation gate).
        renderer.alpha = 0.3;
        renderer.scheme = OccupancyScheme::Costmap;
        let b2 = renderer.scene(&tf).expect("same generation");
        assert_eq!(b2[0].generation, gen1);
        let BatchData::TexturedQuad(grid) = &b2[0].data else {
            panic!("expected textured quad");
        };
        assert_eq!(grid.alpha, 0.3);
        assert!(grid.palette.ptr_eq(&OccupancyScheme::Costmap.palette()));
        // A resend (new message) bumps the generation -> viewport overwrites via same-size write_texture.
        renderer.on_message(&value);
        let b3 = renderer.scene(&tf).expect("rebaked");
        assert!(b3[0].generation > gen1);
    }

    fn update_value(x: i32, y: i32, width: u32, height: u32, data: Vec<u8>) -> Value {
        Value::Struct(vec![
            ("header".to_owned(), header_value(0)),
            ("x".to_owned(), Value::I32(x)),
            ("y".to_owned(), Value::I32(y)),
            ("width".to_owned(), Value::U32(width)),
            ("height".to_owned(), Value::U32(height)),
            ("data".to_owned(), Value::Bytes(data)),
        ])
    }

    #[test]
    fn companion_is_occupancy_grid_update() {
        let renderer = OccupancyGridRenderer::default();
        let companion = renderer.companion().expect("has companion");
        assert_eq!(companion.suffix, "_updates");
        assert_eq!(companion.ros_type, "map_msgs/msg/OccupancyGridUpdate");
    }

    #[test]
    fn on_companion_patches_region_and_bumps_generation() {
        let mut renderer = OccupancyGridRenderer::default();
        let buffer = tf_with(vec![map_tf(0.0, 1_000)]);
        // 4x3 all-zero full map -> generation 1.
        renderer.on_message(&grid_value(4, 3, 1.0, pose_value(0.0, 0.0, 0.0), vec![0; 12]));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "odom",
        };
        let gen1 = renderer.scene(&tf).expect("baked")[0].generation;
        // Overwrite a 2x2 rectangle at (1,1).
        renderer.on_companion(&update_value(1, 1, 2, 2, vec![10, 20, 30, 40]));
        let batch = renderer.scene(&tf).expect("baked");
        assert!(batch[0].generation > gen1);
        let grid = renderer.grid.as_ref().unwrap();
        // Row y=1: [_, 10, 20, _]; row y=2: [_, 30, 40, _] (row-major, width=4).
        assert_eq!(grid.pixels[5], 10);
        assert_eq!(grid.pixels[6], 20);
        assert_eq!(grid.pixels[9], 30);
        assert_eq!(grid.pixels[10], 40);
        // Outside the rectangle is unchanged.
        assert_eq!(grid.pixels[0], 0);
        assert_eq!(grid.pixels[11], 0);
    }

    #[test]
    fn on_companion_rejects_out_of_range_without_touching_map() {
        let mut renderer = OccupancyGridRenderer::default();
        renderer.on_message(&grid_value(4, 3, 1.0, pose_value(0.0, 0.0, 0.0), vec![7; 12]));
        // Out of range (x+width > 4) is discarded; the map is unchanged.
        renderer.on_companion(&update_value(3, 0, 2, 1, vec![1, 2]));
        assert!(renderer.grid.as_ref().unwrap().pixels.iter().all(|&v| v == 7));
        // With no full map received, the diff is ignored (no panic).
        let mut empty = OccupancyGridRenderer::default();
        empty.on_companion(&update_value(0, 0, 1, 1, vec![5]));
        assert!(empty.grid.is_none());
    }

    #[test]
    fn extract_update_rejects_length_mismatch() {
        let err = extract_update(&update_value(0, 0, 2, 2, vec![0; 3])).expect_err("mismatch");
        assert_eq!(err, "update data length 3 does not match 2x2 cells");
    }

    #[test]
    fn draw_behind_defaults_off_and_propagates_without_bumping_generation() {
        let mut renderer = OccupancyGridRenderer::default();
        assert!(!renderer.draw_behind);
        let buffer = tf_with(vec![map_tf(0.0, 1_000)]);
        renderer.on_message(&grid_value(2, 2, 1.0, pose_value(0.0, 0.0, 0.0), vec![0; 4]));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "odom",
        };
        let gen0 = renderer.scene(&tf).expect("baked")[0].generation;
        renderer.draw_behind = true;
        let batch = renderer.scene(&tf).expect("same generation");
        assert_eq!(batch[0].generation, gen0);
        let BatchData::TexturedQuad(grid) = &batch[0].data else {
            panic!("expected textured quad");
        };
        assert!(grid.draw_behind);
    }

    #[test]
    fn stale_stamp_falls_back_to_latest_transform() {
        let mut renderer = OccupancyGridRenderer::default();
        // TF buffer holds only 10s-11s. Map stamp 1s is outside the interpolation range -> latest fallback.
        let buffer = tf_with(vec![
            map_tf(5.0, 10_000_000_000),
            map_tf(5.0, 11_000_000_000),
        ]);
        renderer.on_message(&grid_value_stamped(
            1_000_000_000,
            2,
            2,
            1.0,
            pose_value(0.0, 0.0, 0.0),
            vec![0; 4],
        ));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "odom",
        };
        let batches = renderer.scene(&tf).expect("latest fallback");
        assert!((batches[0].model[(0, 3)] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn renderer_status_transitions() {
        let mut renderer = OccupancyGridRenderer::default();
        let buffer = TfBuffer::new();
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "odom",
        };
        assert!(matches!(renderer.scene(&tf), Err(RenderStatus::NoData)));
        renderer.on_message(&Value::Struct(vec![]));
        assert!(matches!(
            renderer.scene(&tf),
            Err(RenderStatus::InvalidMessage(_))
        ));
        renderer.on_message(&grid_value(
            2,
            2,
            1.0,
            pose_value(0.0, 0.0, 0.0),
            vec![0; 4],
        ));
        assert_eq!(
            renderer.scene(&tf).unwrap_err(),
            RenderStatus::TfUnavailable {
                frame: "map".to_owned()
            }
        );
    }
}
