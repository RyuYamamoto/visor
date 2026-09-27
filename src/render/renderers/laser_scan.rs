//! LaserScan renderer (polar -> point cloud; Points/Squares styles and intensity coloring).

use egui::{Color32, RichText};
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};

use crate::decode::value::Value;
use crate::render::{
    IntensityScale, PointBatch, PointBatchBuilder, PointStyleSettings, RenderStatus, Renderer,
    SceneBatch, TfContext, extract_header, intensity_color,
};
use crate::theme;

/// Default side length [m] of the Squares style.
const SQUARE_SIZE_DEFAULT: f32 = 0.05;

/// Point coloring mode (Intensity only for messages with an intensities field; else falls back to Flat).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ColorMode {
    Flat,
    Intensity,
}

impl ColorMode {
    fn label(self) -> &'static str {
        match self {
            ColorMode::Flat => "Flat",
            ColorMode::Intensity => "Intensity",
        }
    }
}

/// Local-coordinate points extracted from a LaserScan message.
#[derive(Debug)]
struct ScanPoints {
    /// header.frame_id (laser frame name).
    frame_id: String,
    /// header.stamp used for TF resolution (0 = latest).
    stamp: crate::tf::buffer::TimeNs,
    /// Points on the laser XY plane (expanded via angle_min + i * angle_increment).
    points: Vec<[f32; 2]>,
    /// Intensity per accepted point (None if intensities is empty or length-mismatched; optional in LaserScan.msg).
    intensities: Option<Vec<f32>>,
    /// (min, max) of accepted intensities, computed during extraction for auto-range.
    intensity_range: Option<(f32, f32)>,
}

pub struct LaserScanRenderer {
    scan: Option<ScanPoints>,
    parse_error: Option<String>,
    style: PointStyleSettings,
    color: Color32,
    color_mode: ColorMode,
    intensity: IntensityScale,
    /// Fallback notice (e.g. missing field) in Intensity mode; not an error.
    mode_warning: Option<String>,
    /// laser->fixed transform fixed at message arrival; like RViz, points stay put until the next message.
    pose: Option<Isometry3<f32>>,
    /// Fixed frame at the time pose was resolved; a change triggers re-resolution.
    pose_fixed: String,
    /// Set on new message arrival; the next scene() re-resolves pose.
    pose_dirty: bool,
    baked: Option<PointBatch>,
    /// Bumped whenever baked changes; used by viewport's transfer-skip check.
    generation: u64,
    /// Set on point-data/color changes; the next scene() rebuilds the batch (style/size apply via uniform only).
    bake_dirty: bool,
}

impl Default for LaserScanRenderer {
    fn default() -> Self {
        Self {
            scan: None,
            parse_error: None,
            style: PointStyleSettings::new(SQUARE_SIZE_DEFAULT),
            color: theme::POINT_FLAT_DEFAULT,
            color_mode: ColorMode::Flat,
            intensity: IntensityScale::default(),
            mode_warning: None,
            pose: None,
            pose_fixed: String::new(),
            pose_dirty: false,
            baked: None,
            generation: 0,
            bake_dirty: false,
        }
    }
}

impl Renderer for LaserScanRenderer {
    fn on_message(&mut self, value: &Value) {
        match extract_scan(value) {
            Ok(scan) => {
                self.scan = Some(scan);
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
        let Some(scan) = &self.scan else {
            return Err(RenderStatus::NoData);
        };
        // Resolve once at stamp: re-resolving latest every frame would drag a stale scan and jitter while moving.
        if self.pose_dirty || self.pose_fixed != tf.fixed_frame {
            match tf.resolve_at(&scan.frame_id, scan.stamp) {
                Some(iso) => {
                    self.pose = Some(iso.cast::<f32>());
                    self.pose_fixed = tf.fixed_frame.to_owned();
                    self.pose_dirty = false;
                    self.bake_dirty = true;
                }
                None => {
                    // On a fixed-frame switch that fails to resolve, do not keep the old-basis rendering.
                    if self.pose_fixed != tf.fixed_frame {
                        self.pose = None;
                        self.baked = None;
                        self.pose_fixed = tf.fixed_frame.to_owned();
                        self.pose_dirty = true;
                    }
                    // TF likely just lags the stamp, so keep emitting the old bake and retry next frame.
                    if self.pose.is_none() {
                        return Err(RenderStatus::TfUnavailable {
                            frame: scan.frame_id.clone(),
                        });
                    }
                }
            }
        }
        let Some(pose) = &self.pose else {
            return Err(RenderStatus::TfUnavailable {
                frame: scan.frame_id.clone(),
            });
        };
        if self.bake_dirty {
            let (colors, warning) = point_colors(
                scan,
                self.color_mode,
                theme::to_linear_rgba8(self.color),
                &self.intensity,
            );
            self.mode_warning = warning;
            let mut builder = PointBatchBuilder::with_capacity(scan.points.len());
            for (point, rgba) in scan.points.iter().zip(&colors) {
                builder.push([point[0], point[1], 0.0], *rgba);
            }
            self.baked = Some(builder.build());
            self.generation += 1;
            self.bake_dirty = false;
        }
        match self.baked.as_ref() {
            Some(batch) => Ok(vec![SceneBatch::points(
                batch.clone(),
                self.generation,
                pose,
                self.style.size(),
            )]),
            None => Err(RenderStatus::NoData),
        }
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        // Style/size apply via uniform only, so no rebake is needed.
        self.style.ui(ui);
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Color").color(p.text_muted));
            egui::ComboBox::from_id_salt("color_mode")
                .selected_text(self.color_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [ColorMode::Flat, ColorMode::Intensity] {
                        changed |= ui
                            .selectable_value(&mut self.color_mode, mode, mode.label())
                            .changed();
                    }
                });
            if self.color_mode == ColorMode::Flat {
                changed |= ui.color_edit_button_srgba(&mut self.color).changed();
            }
        });
        if self.color_mode == ColorMode::Intensity {
            changed |= self.intensity.ui(ui);
        }
        if let Some(warning) = &self.mode_warning {
            ui.colored_label(p.status_warn, warning);
        }
        if let Some(scan) = &self.scan {
            ui.label(RichText::new(format!("points: {}", scan.points.len())).color(p.text_muted));
        }
        if changed {
            self.bake_dirty = true;
        }
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(LaserScanSettings {
            style: self.style.clone(),
            color: self.color,
            color_mode: self.color_mode,
            intensity: self.intensity.clone(),
        })
        .ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(s) = value.clone().try_into::<LaserScanSettings>() {
            self.style = s.style;
            self.color = s.color;
            self.color_mode = s.color_mode;
            self.intensity = s.intensity;
            self.bake_dirty = true;
        }
    }
}

/// Persistence DTO for LaserScanRenderer's user-editable settings (fields mirror what settings_ui touches).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct LaserScanSettings {
    style: PointStyleSettings,
    #[serde(with = "crate::config::color_hex")]
    color: Color32,
    color_mode: ColorMode,
    intensity: IntensityScale,
}

impl Default for LaserScanSettings {
    fn default() -> Self {
        Self {
            style: PointStyleSettings::new(SQUARE_SIZE_DEFAULT),
            color: theme::POINT_FLAT_DEFAULT,
            color_mode: ColorMode::Flat,
            intensity: IntensityScale::default(),
        }
    }
}

/// Build the per-point color list per settings (Intensity mode without intensities falls back to Flat + a notice).
fn point_colors(
    scan: &ScanPoints,
    mode: ColorMode,
    flat: [u8; 4],
    scale: &IntensityScale,
) -> (Vec<[u8; 4]>, Option<String>) {
    match mode {
        ColorMode::Flat => (vec![flat; scan.points.len()], None),
        ColorMode::Intensity => match &scan.intensities {
            Some(intensities) => {
                let range = scale.resolve(scan.intensity_range);
                (
                    intensities
                        .iter()
                        .map(|v| intensity_color(range, *v))
                        .collect(),
                    None,
                )
            }
            None => (
                vec![flat; scan.points.len()],
                Some("intensities not available — using flat color".to_owned()),
            ),
        },
    }
}

fn get_f32(value: &Value, field: &str) -> Result<f32, String> {
    match value.get(field) {
        Some(Value::F32(v)) => Ok(*v),
        _ => Err(format!("missing field `{field}` (float32)")),
    }
}

/// Extract local points from a LaserScan Value (skip non-finite values and those outside [range_min, range_max]).
fn extract_scan(value: &Value) -> Result<ScanPoints, String> {
    let (frame_id, stamp) = extract_header(value)?;
    let angle_min = get_f32(value, "angle_min")?;
    let angle_increment = get_f32(value, "angle_increment")?;
    let range_min = get_f32(value, "range_min")?;
    let range_max = get_f32(value, "range_max")?;
    let Some(Value::Array(ranges)) = value.get("ranges") else {
        return Err("missing field `ranges` (float32[])".to_owned());
    };
    // intensities is optional; empty or length-mismatched is treated as absent, not an error.
    let raw_intensities = match value.get("intensities") {
        Some(Value::Array(items)) if items.len() == ranges.len() => Some(items),
        _ => None,
    };
    let mut points = Vec::with_capacity(ranges.len());
    let mut intensities = raw_intensities.map(|_| Vec::with_capacity(ranges.len()));
    let mut intensity_range: Option<(f32, f32)> = None;
    for (i, item) in ranges.iter().enumerate() {
        let Value::F32(range) = item else {
            return Err(format!("ranges[{i}] is not float32"));
        };
        if !range.is_finite() || *range < range_min || *range > range_max {
            continue;
        }
        let angle = angle_min + i as f32 * angle_increment;
        points.push([range * angle.cos(), range * angle.sin()]);
        if let (Some(out), Some(raw)) = (&mut intensities, raw_intensities) {
            let Value::F32(v) = &raw[i] else {
                return Err(format!("intensities[{i}] is not float32"));
            };
            let v = *v;
            out.push(v);
            intensity_range = Some(match intensity_range {
                Some((lo, hi)) => (lo.min(v), hi.max(v)),
                None => (v, v),
            });
        }
    }
    Ok(ScanPoints {
        frame_id,
        stamp,
        points,
        intensities,
        intensity_range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{BatchData, POINT_STRIDE, PointStyle, SizeSpec, colormap};

    #[test]
    fn settings_roundtrip_via_trait() {
        let r = LaserScanRenderer {
            color: Color32::from_rgb(1, 2, 3),
            color_mode: ColorMode::Intensity,
            style: PointStyleSettings {
                style: PointStyle::Squares,
                size_m: 0.25,
                size_px: 6.0,
            },
            intensity: IntensityScale {
                auto: false,
                min: -1.0,
                max: 5.0,
            },
            ..Default::default()
        };
        let value = r.settings().expect("laser scan has settings");
        let mut restored = LaserScanRenderer::default();
        restored.apply_settings(&value);
        assert_eq!(restored.color, r.color);
        assert_eq!(restored.color_mode, r.color_mode);
        assert_eq!(restored.style, r.style);
        assert_eq!(restored.intensity, r.intensity);
    }
    use nalgebra::{Translation3, UnitQuaternion};

    fn scan_value(frame_id: &str, ranges: Vec<Value>) -> Value {
        scan_value_stamped(frame_id, 0, ranges)
    }

    fn scan_value_stamped(frame_id: &str, stamp: i64, ranges: Vec<Value>) -> Value {
        scan_value_full(frame_id, stamp, ranges, vec![])
    }

    fn scan_value_full(
        frame_id: &str,
        stamp: i64,
        ranges: Vec<Value>,
        intensities: Vec<Value>,
    ) -> Value {
        Value::Struct(vec![
            (
                "header".to_owned(),
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
                    ("frame_id".to_owned(), Value::String(frame_id.to_owned())),
                ]),
            ),
            ("angle_min".to_owned(), Value::F32(0.0)),
            (
                "angle_increment".to_owned(),
                Value::F32(std::f32::consts::FRAC_PI_2),
            ),
            ("range_min".to_owned(), Value::F32(0.1)),
            ("range_max".to_owned(), Value::F32(10.0)),
            ("ranges".to_owned(), Value::Array(ranges)),
            ("intensities".to_owned(), Value::Array(intensities)),
        ])
    }

    /// Read a point instance's local coordinates from the batch.
    fn point_position(batch: &SceneBatch, index: usize) -> [f32; 3] {
        let BatchData::Points(points) = &batch.data else {
            panic!("expected points batch");
        };
        let base = index * POINT_STRIDE;
        std::array::from_fn(|i| {
            f32::from_le_bytes(
                points.bytes[base + i * 4..base + (i + 1) * 4]
                    .try_into()
                    .unwrap(),
            )
        })
    }

    /// Read a point instance's packed RGBA8 from the batch.
    fn point_color(batch: &SceneBatch, index: usize) -> [u8; 4] {
        let BatchData::Points(points) = &batch.data else {
            panic!("expected points batch");
        };
        let base = index * POINT_STRIDE + 12;
        points.bytes[base..base + 4].try_into().unwrap()
    }

    fn model_translation_x(batch: &SceneBatch) -> f32 {
        batch.model[(0, 3)]
    }

    #[test]
    fn extract_expands_angles_into_local_points() {
        let value = scan_value("laser", vec![Value::F32(2.0), Value::F32(3.0)]);
        let scan = extract_scan(&value).expect("valid scan");
        assert_eq!(scan.frame_id, "laser");
        assert_eq!(scan.points.len(), 2);
        // i=0: angle 0 -> (2, 0); i=1: angle pi/2 -> (0, 3).
        assert!((scan.points[0][0] - 2.0).abs() < 1e-5);
        assert!(scan.points[0][1].abs() < 1e-5);
        assert!(scan.points[1][0].abs() < 1e-5);
        assert!((scan.points[1][1] - 3.0).abs() < 1e-5);
        assert!(scan.intensities.is_none());
    }

    #[test]
    fn extract_filters_invalid_and_out_of_range_values() {
        let value = scan_value(
            "laser",
            vec![
                Value::F32(f32::INFINITY),
                Value::F32(f32::NAN),
                Value::F32(0.05),
                Value::F32(11.0),
                Value::F32(1.0),
            ],
        );
        let scan = extract_scan(&value).expect("valid scan");
        assert_eq!(scan.points.len(), 1);
        // Only i=4 survives (angle 2pi = +X direction, distance 1.0).
        assert!((scan.points[0][0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn extract_aligns_intensities_with_accepted_points() {
        let value = scan_value_full(
            "laser",
            0,
            vec![Value::F32(f32::NAN), Value::F32(2.0), Value::F32(3.0)],
            vec![Value::F32(9.0), Value::F32(10.0), Value::F32(50.0)],
        );
        let scan = extract_scan(&value).expect("valid scan");
        assert_eq!(scan.points.len(), 2);
        // The NaN-dropped i=0 intensity 9.0 is excluded.
        assert_eq!(scan.intensities, Some(vec![10.0, 50.0]));
        assert_eq!(scan.intensity_range, Some((10.0, 50.0)));
    }

    #[test]
    fn extract_treats_length_mismatch_intensities_as_absent() {
        let value = scan_value_full(
            "laser",
            0,
            vec![Value::F32(2.0), Value::F32(3.0)],
            vec![Value::F32(1.0)],
        );
        let scan = extract_scan(&value).expect("valid scan");
        assert!(scan.intensities.is_none());
        assert!(scan.intensity_range.is_none());
    }

    #[test]
    fn extract_reports_missing_ranges() {
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
                ("frame_id".to_owned(), Value::String("l".to_owned())),
            ]),
        )]);
        assert!(extract_scan(&value).is_err());
    }

    #[test]
    fn extract_reports_non_f32_range_element() {
        let value = scan_value("laser", vec![Value::F64(2.0)]);
        let err = extract_scan(&value).expect_err("f64 element must be rejected");
        assert!(err.contains("ranges[0]"), "err={err}");
    }

    fn dyn_tf(parent: &str, child: &str, x: f64, stamp: i64) -> crate::tf::buffer::TfTransform {
        crate::tf::buffer::TfTransform {
            parent: parent.to_owned(),
            child: child.to_owned(),
            stamp,
            transform: Isometry3::from_parts(
                Translation3::new(x, 0.0, 0.0),
                UnitQuaternion::identity(),
            ),
        }
    }

    #[test]
    fn scan_stays_fixed_until_next_message() {
        use crate::tf::buffer::{TfBuffer, tf_update};
        let mut renderer = LaserScanRenderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![dyn_tf("map", "laser", 1.0, 1_000)], false));
        renderer.on_message(&scan_value("laser", vec![Value::F32(2.0)]));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let b1 = renderer.scene(&tf).expect("resolvable");
        // Local coord (2, 0, 0) + model matrix (arrival-time pose x=1).
        assert_eq!(point_position(&b1[0], 0), [2.0, 0.0, 0.0]);
        assert!((model_translation_x(&b1[0]) - 1.0).abs() < 1e-5);
        // Robot motion (TF update) alone does not move pose (RViz-style fix-at-arrival).
        buffer.insert(&tf_update(vec![dyn_tf("map", "laser", 5.0, 2_000)], false));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let b2 = renderer.scene(&tf).expect("still baked");
        assert!((model_translation_x(&b2[0]) - 1.0).abs() < 1e-5);
        assert_eq!(b1[0].generation, b2[0].generation);
        // A new message arrival applies the new TF.
        renderer.on_message(&scan_value("laser", vec![Value::F32(2.0)]));
        let b3 = renderer.scene(&tf).expect("rebaked");
        assert!((model_translation_x(&b3[0]) - 5.0).abs() < 1e-5);
        assert!(b3[0].generation > b2[0].generation);
    }

    #[test]
    fn scan_bakes_at_message_stamp_with_interpolation() {
        use crate::tf::buffer::{TfBuffer, tf_update};
        let mut renderer = LaserScanRenderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![
                dyn_tf("map", "laser", 1.0, 1_000),
                dyn_tf("map", "laser", 5.0, 2_000),
            ], false));
        // stamp=1500 is the midpoint of two samples -> interpolates to x=3 (not latest x=5).
        renderer.on_message(&scan_value_stamped("laser", 1_500, vec![Value::F32(2.0)]));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let b = renderer.scene(&tf).expect("interpolated");
        assert!((model_translation_x(&b[0]) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn scan_keeps_previous_bake_while_tf_lags_behind_stamp() {
        use crate::tf::buffer::{TfBuffer, tf_update};
        let mut renderer = LaserScanRenderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![dyn_tf("map", "laser", 1.0, 1_000)], false));
        renderer.on_message(&scan_value_stamped("laser", 1_000, vec![Value::F32(2.0)]));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let b1 = renderer.scene(&tf).expect("baked");
        assert!((model_translation_x(&b1[0]) - 1.0).abs() < 1e-5);
        // stamp is ahead of TF -> keep emitting the old bake (no disappear, no error).
        renderer.on_message(&scan_value_stamped("laser", 3_000, vec![Value::F32(2.0)]));
        let b2 = renderer.scene(&tf).expect("previous bake kept");
        assert!((model_translation_x(&b2[0]) - 1.0).abs() < 1e-5);
        // Once TF catches up, it bakes at the stamp's pose.
        buffer.insert(&tf_update(vec![dyn_tf("map", "laser", 5.0, 3_000)], false));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let b3 = renderer.scene(&tf).expect("rebaked at stamp");
        assert!((model_translation_x(&b3[0]) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn fixed_frame_switch_rebakes_in_new_basis() {
        use crate::tf::buffer::{TfBuffer, tf_update};
        let mut renderer = LaserScanRenderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![
                dyn_tf("map", "odom", 1.0, 1_000),
                dyn_tf("odom", "laser", 1.0, 1_000),
            ], false));
        renderer.on_message(&scan_value("laser", vec![Value::F32(2.0)]));
        let map_tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let b_map = renderer.scene(&map_tf).expect("map basis");
        assert!((model_translation_x(&b_map[0]) - 2.0).abs() < 1e-5);
        // A fixed-frame switch re-resolves even without a new message.
        let odom_tf = TfContext {
            buffer: &buffer,
            fixed_frame: "odom",
        };
        let b_odom = renderer.scene(&odom_tf).expect("odom basis");
        assert!((model_translation_x(&b_odom[0]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn style_selects_pixel_or_meter_point_size_without_rebake() {
        use crate::tf::buffer::{TfBuffer, tf_update};
        let mut renderer = LaserScanRenderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![dyn_tf("map", "laser", 0.0, 1_000)], false));
        renderer.on_message(&scan_value("laser", vec![Value::F32(2.0)]));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        // Default is Points (screen-fixed [px]).
        let b1 = renderer.scene(&tf).expect("baked");
        assert_eq!(b1[0].size, SizeSpec::Pixels(3.0));
        // Switch to Squares (world-fixed [m]): no rebake (generation unchanged), only the size spec changes.
        renderer.style.style = PointStyle::Squares;
        let b2 = renderer.scene(&tf).expect("same bake");
        assert_eq!(b2[0].size, SizeSpec::Meters(SQUARE_SIZE_DEFAULT));
        assert_eq!(b1[0].generation, b2[0].generation);
    }

    #[test]
    fn intensity_mode_colors_points_by_colormap() {
        use crate::tf::buffer::{TfBuffer, tf_update};
        let mut renderer = LaserScanRenderer {
            color_mode: ColorMode::Intensity,
            ..Default::default()
        };
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![dyn_tf("map", "laser", 0.0, 1_000)], false));
        renderer.on_message(&scan_value_full(
            "laser",
            0,
            vec![Value::F32(1.0), Value::F32(2.0)],
            vec![Value::F32(10.0), Value::F32(50.0)],
        ));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let batches = renderer.scene(&tf).expect("baked");
        // Auto-range (10, 50): min -> colormap(0), max -> colormap(1).
        assert_eq!(point_color(&batches[0], 0), colormap(0.0));
        assert_eq!(point_color(&batches[0], 1), colormap(1.0));
        assert!(renderer.mode_warning.is_none());
    }

    #[test]
    fn intensity_mode_falls_back_to_flat_when_absent() {
        use crate::tf::buffer::{TfBuffer, tf_update};
        let mut renderer = LaserScanRenderer {
            color_mode: ColorMode::Intensity,
            ..Default::default()
        };
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![dyn_tf("map", "laser", 0.0, 1_000)], false));
        renderer.on_message(&scan_value("laser", vec![Value::F32(1.0)]));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let batches = renderer.scene(&tf).expect("baked with fallback");
        assert_eq!(
            point_color(&batches[0], 0),
            theme::to_linear_rgba8(theme::POINT_FLAT_DEFAULT)
        );
        assert!(renderer.mode_warning.is_some());
    }

    #[test]
    fn renderer_status_transitions_from_no_data_to_invalid() {
        let mut renderer = LaserScanRenderer::default();
        let buffer = crate::tf::buffer::TfBuffer::new();
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
        // Valid message but no TF registered -> TfUnavailable.
        renderer.on_message(&scan_value("laser", vec![Value::F32(1.0)]));
        assert_eq!(
            renderer.scene(&tf).unwrap_err(),
            RenderStatus::TfUnavailable {
                frame: "laser".to_owned()
            }
        );
    }
}
