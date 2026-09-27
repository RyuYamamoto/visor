//! PointCloud2 renderer (parses PointField offset/datatype to extract xyz/intensity/rgb from data, drawn as point quads).

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
const SQUARE_SIZE_DEFAULT: f32 = 0.02;

const DT_INT8: u8 = 1;
const DT_UINT8: u8 = 2;
const DT_INT16: u8 = 3;
const DT_UINT16: u8 = 4;
const DT_INT32: u8 = 5;
const DT_UINT32: u8 = 6;
const DT_FLOAT32: u8 = 7;
const DT_FLOAT64: u8 = 8;

/// Point coloring mode (Intensity/Rgb fall back to Flat for messages lacking the field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ColorMode {
    Flat,
    Intensity,
    Rgb,
}

impl ColorMode {
    fn label(self) -> &'static str {
        match self {
            ColorMode::Flat => "Flat",
            ColorMode::Intensity => "Intensity",
            ColorMode::Rgb => "RGB",
        }
    }
}

/// Read location of one field resolved from fields.
#[derive(Debug, Clone, Copy)]
struct FieldRef {
    offset: usize,
    datatype: u8,
}

/// rgb / rgba field (4 bytes read as u32; has_alpha only when the name is rgba).
#[derive(Debug, Clone, Copy)]
struct ColorField {
    offset: usize,
    has_alpha: bool,
}

/// Traversal layout of data resolved from the message.
#[derive(Debug)]
struct CloudLayout {
    x: FieldRef,
    y: FieldRef,
    z: FieldRef,
    intensity: Option<FieldRef>,
    color: Option<ColorField>,
    width: usize,
    height: usize,
    point_step: usize,
    /// Effective row stride (floored at width x point_step to handle publishers with row_step = 0).
    row_step: usize,
    big_endian: bool,
}

/// Extracted cloud (non-finite filtered, local-coordinate SoA); color changes repack without re-parsing data.
#[derive(Debug)]
struct ExtractedCloud {
    frame_id: String,
    stamp: crate::tf::buffer::TimeNs,
    positions: Vec<[f32; 3]>,
    intensity: Option<Vec<f32>>,
    /// RGBA left as sRGB (passed through an sRGB->linear LUT at pack time).
    rgba: Option<Vec<[u8; 4]>>,
    /// (min, max) of intensity, computed during the extraction pass.
    intensity_range: Option<(f32, f32)>,
}

pub struct PointCloud2Renderer {
    cloud: Option<ExtractedCloud>,
    /// Bumped each time a new message is extracted; diffed against packed_seq to decide whether to repack.
    cloud_seq: u64,
    packed_seq: u64,
    batch: Option<PointBatch>,
    generation: u64,
    parse_error: Option<String>,
    pose: Option<Isometry3<f32>>,
    pose_fixed: String,
    pose_dirty: bool,
    /// Set on color-setting changes (repack from SoA only; no data re-parse).
    repack_dirty: bool,
    color_mode: ColorMode,
    flat_color: Color32,
    style: PointStyleSettings,
    intensity: IntensityScale,
    mode_warning: Option<String>,
}

impl Default for PointCloud2Renderer {
    fn default() -> Self {
        Self {
            cloud: None,
            cloud_seq: 0,
            packed_seq: 0,
            batch: None,
            generation: 0,
            parse_error: None,
            pose: None,
            pose_fixed: String::new(),
            pose_dirty: false,
            repack_dirty: false,
            color_mode: ColorMode::Flat,
            flat_color: theme::POINT_FLAT_DEFAULT,
            style: PointStyleSettings::new(SQUARE_SIZE_DEFAULT),
            intensity: IntensityScale::default(),
            mode_warning: None,
        }
    }
}

impl Renderer for PointCloud2Renderer {
    fn on_message(&mut self, value: &Value) {
        match extract_cloud(value) {
            Ok(cloud) => {
                self.cloud = Some(cloud);
                self.cloud_seq += 1;
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
        let Some(cloud) = &self.cloud else {
            return Err(RenderStatus::NoData);
        };
        // TF convention matches laser_scan (interpolated resolve at header.stamp, fix-at-arrival, keep old view while lagging).
        if self.pose_dirty || self.pose_fixed != tf.fixed_frame {
            match tf.resolve_at(&cloud.frame_id, cloud.stamp) {
                Some(iso) => {
                    self.pose = Some(iso.cast::<f32>());
                    self.pose_fixed = tf.fixed_frame.to_owned();
                    self.pose_dirty = false;
                }
                None => {
                    if self.pose_fixed != tf.fixed_frame {
                        self.pose = None;
                        self.batch = None;
                        self.pose_fixed = tf.fixed_frame.to_owned();
                        self.pose_dirty = true;
                        self.repack_dirty = true;
                    }
                    if self.pose.is_none() {
                        return Err(RenderStatus::TfUnavailable {
                            frame: cloud.frame_id.clone(),
                        });
                    }
                }
            }
        }
        let Some(pose) = &self.pose else {
            return Err(RenderStatus::TfUnavailable {
                frame: cloud.frame_id.clone(),
            });
        };
        // Until pose catches up to the new message's stamp, keep the old batch + old pose (avoid mixing a new cloud with an old pose).
        if !self.pose_dirty && (self.packed_seq != self.cloud_seq || self.repack_dirty) {
            let (batch, warning) = pack_cloud(
                cloud,
                self.color_mode,
                theme::to_linear_rgba8(self.flat_color),
                &self.intensity,
            );
            self.batch = Some(batch);
            self.mode_warning = warning;
            self.packed_seq = self.cloud_seq;
            self.repack_dirty = false;
            self.generation += 1;
        }
        match &self.batch {
            Some(batch) => Ok(vec![SceneBatch::points(
                batch.clone(),
                self.generation,
                pose,
                self.style.size(),
            )]),
            None => Err(RenderStatus::TfUnavailable {
                frame: cloud.frame_id.clone(),
            }),
        }
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        // Style/size apply via uniform only, so no repack is needed.
        self.style.ui(ui);
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Color").color(p.text_muted));
            egui::ComboBox::from_id_salt("color_mode")
                .selected_text(self.color_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [ColorMode::Flat, ColorMode::Intensity, ColorMode::Rgb] {
                        changed |= ui
                            .selectable_value(&mut self.color_mode, mode, mode.label())
                            .changed();
                    }
                });
            if self.color_mode == ColorMode::Flat {
                changed |= ui.color_edit_button_srgba(&mut self.flat_color).changed();
            }
        });
        if self.color_mode == ColorMode::Intensity {
            changed |= self.intensity.ui(ui);
        }
        if let Some(warning) = &self.mode_warning {
            ui.colored_label(p.status_warn, warning);
        }
        if let Some(cloud) = &self.cloud {
            ui.label(
                RichText::new(format!("points: {}", cloud.positions.len())).color(p.text_muted),
            );
        }
        if changed {
            self.repack_dirty = true;
        }
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(PointCloud2Settings {
            style: self.style.clone(),
            flat_color: self.flat_color,
            color_mode: self.color_mode,
            intensity: self.intensity.clone(),
        })
        .ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(s) = value.clone().try_into::<PointCloud2Settings>() {
            self.style = s.style;
            self.flat_color = s.flat_color;
            self.color_mode = s.color_mode;
            self.intensity = s.intensity;
            self.repack_dirty = true;
        }
    }
}

/// Persistence DTO for PointCloud2Renderer's user-editable settings (mirrors what settings_ui touches).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct PointCloud2Settings {
    style: PointStyleSettings,
    #[serde(with = "crate::config::color_hex")]
    flat_color: Color32,
    color_mode: ColorMode,
    intensity: IntensityScale,
}

impl Default for PointCloud2Settings {
    fn default() -> Self {
        Self {
            style: PointStyleSettings::new(SQUARE_SIZE_DEFAULT),
            flat_color: theme::POINT_FLAT_DEFAULT,
            color_mode: ColorMode::Flat,
            intensity: IntensityScale::default(),
        }
    }
}

fn datatype_size(datatype: u8) -> usize {
    match datatype {
        DT_INT8 | DT_UINT8 => 1,
        DT_INT16 | DT_UINT16 => 2,
        DT_INT32 | DT_UINT32 | DT_FLOAT32 => 4,
        DT_FLOAT64 => 8,
        _ => 0,
    }
}

fn get_u32(value: &Value, field: &str) -> Result<u32, String> {
    match value.get(field) {
        Some(Value::U32(v)) => Ok(*v),
        _ => Err(format!("missing field `{field}` (uint32)")),
    }
}

/// Resolve read locations from the fields array for x/y/z (required), intensity, and rgb/rgba (optional).
fn parse_layout(value: &Value) -> Result<CloudLayout, String> {
    let width = get_u32(value, "width")? as usize;
    let height = get_u32(value, "height")? as usize;
    let point_step = get_u32(value, "point_step")? as usize;
    if point_step == 0 {
        return Err("point_step is 0".to_owned());
    }
    let row_step = get_u32(value, "row_step")? as usize;
    let big_endian = match value.get("is_bigendian") {
        Some(Value::Bool(v)) => *v,
        _ => return Err("missing field `is_bigendian` (bool)".to_owned()),
    };
    let Some(Value::Array(fields)) = value.get("fields") else {
        return Err("missing field `fields` (PointField[])".to_owned());
    };
    let mut xyz: [Option<FieldRef>; 3] = [None; 3];
    let mut intensity = None;
    let mut color = None;
    for field in fields {
        let Some(Value::String(name)) = field.get("name") else {
            continue;
        };
        let offset = get_u32(field, "offset")? as usize;
        let datatype = match field.get("datatype") {
            Some(Value::U8(v)) => *v,
            _ => return Err(format!("field `{name}` has no datatype")),
        };
        let count = get_u32(field, "count")?;
        let fits = offset + datatype_size(datatype) <= point_step;
        match name.as_str() {
            axis @ ("x" | "y" | "z") => {
                if !matches!(datatype, DT_FLOAT32 | DT_FLOAT64) {
                    return Err(format!(
                        "field `{axis}` has datatype {datatype} (expected FLOAT32/FLOAT64)"
                    ));
                }
                if count != 1 {
                    return Err(format!("field `{axis}` has count {count} (expected 1)"));
                }
                if !fits {
                    return Err(format!(
                        "field `{axis}` at offset {offset} does not fit in point_step {point_step}"
                    ));
                }
                let index = (axis.as_bytes()[0] - b'x') as usize;
                xyz[index] = Some(FieldRef { offset, datatype });
            }
            // intensity accepts a wide range of widths (real data is often UINT16 reflectance); unsuitable is treated as absent, not an error.
            "intensity" => {
                if count == 1 && datatype_size(datatype) > 0 && fits {
                    intensity = Some(FieldRef { offset, datatype });
                }
            }
            // Accept both rgb (FLOAT32 packed) / rgba (UINT32) conventions plus non-standard UINT32 rgb (matching RViz).
            "rgb" | "rgba" => {
                if count == 1 && matches!(datatype, DT_FLOAT32 | DT_UINT32) && offset + 4 <= point_step {
                    color = Some(ColorField {
                        offset,
                        has_alpha: name == "rgba",
                    });
                }
            }
            _ => {}
        }
    }
    let [Some(x), Some(y), Some(z)] = xyz else {
        let missing = ["x", "y", "z"]
            .iter()
            .zip(&xyz)
            .filter(|(_, slot)| slot.is_none())
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("missing field(s) `{missing}` in fields"));
    };
    Ok(CloudLayout {
        x,
        y,
        z,
        intensity,
        color,
        width,
        height,
        point_step,
        row_step: row_step.max(width * point_step),
        big_endian,
    })
}

fn read_u32_at(data: &[u8], offset: usize, big_endian: bool) -> u32 {
    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
    if big_endian {
        u32::from_be_bytes(bytes)
    } else {
        u32::from_le_bytes(bytes)
    }
}

/// Read a numeric field as f32 (FLOAT64 narrowed, integers converted; range validated before the call).
fn read_scalar(data: &[u8], offset: usize, datatype: u8, big_endian: bool) -> f32 {
    macro_rules! read {
        ($ty:ty) => {{
            let bytes: [u8; size_of::<$ty>()] =
                data[offset..offset + size_of::<$ty>()].try_into().unwrap();
            if big_endian {
                <$ty>::from_be_bytes(bytes)
            } else {
                <$ty>::from_le_bytes(bytes)
            }
        }};
    }
    match datatype {
        DT_INT8 => read!(i8) as f32,
        DT_UINT8 => read!(u8) as f32,
        DT_INT16 => read!(i16) as f32,
        DT_UINT16 => read!(u16) as f32,
        DT_INT32 => read!(i32) as f32,
        DT_UINT32 => read!(u32) as f32,
        DT_FLOAT32 => read!(f32),
        DT_FLOAT64 => read!(f64) as f32,
        _ => f32::NAN,
    }
}

/// Extract SoA from a PointCloud2 Value (skip points with any non-finite x/y/z; supports is_dense=false).
fn extract_cloud(value: &Value) -> Result<ExtractedCloud, String> {
    let (frame_id, stamp) = extract_header(value)?;
    let layout = parse_layout(value)?;
    let Some(Value::Bytes(data)) = value.get("data") else {
        return Err("missing field `data` (uint8[])".to_owned());
    };
    if layout.width > 0 && layout.height > 0 {
        // The last row need not include row_step padding (require (height-1)*row_step + the actual point row).
        let required =
            (layout.height - 1) * layout.row_step + layout.width * layout.point_step;
        if data.len() < required {
            return Err(format!(
                "data too short: {} bytes (expected at least {required} for {}x{} points)",
                data.len(),
                layout.width,
                layout.height
            ));
        }
    }
    let capacity = layout.width * layout.height;
    let mut positions = Vec::with_capacity(capacity);
    let mut intensity = layout.intensity.map(|_| Vec::with_capacity(capacity));
    let mut rgba = layout.color.map(|_| Vec::with_capacity(capacity));
    let mut intensity_range: Option<(f32, f32)> = None;
    let be = layout.big_endian;
    for row in 0..layout.height {
        let row_base = row * layout.row_step;
        for col in 0..layout.width {
            let base = row_base + col * layout.point_step;
            let x = read_scalar(data, base + layout.x.offset, layout.x.datatype, be);
            let y = read_scalar(data, base + layout.y.offset, layout.y.datatype, be);
            let z = read_scalar(data, base + layout.z.offset, layout.z.datatype, be);
            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                continue;
            }
            positions.push([x, y, z]);
            if let (Some(out), Some(field)) = (&mut intensity, &layout.intensity) {
                let v = read_scalar(data, base + field.offset, field.datatype, be);
                out.push(v);
                if v.is_finite() {
                    intensity_range = Some(match intensity_range {
                        Some((lo, hi)) => (lo.min(v), hi.max(v)),
                        None => (v, v),
                    });
                }
            }
            if let (Some(out), Some(field)) = (&mut rgba, &layout.color) {
                let packed = read_u32_at(data, base + field.offset, be);
                let alpha = if field.has_alpha {
                    (packed >> 24) as u8
                } else {
                    255
                };
                out.push([
                    (packed >> 16) as u8,
                    (packed >> 8) as u8,
                    packed as u8,
                    alpha,
                ]);
            }
        }
    }
    Ok(ExtractedCloud {
        frame_id,
        stamp,
        positions,
        intensity,
        rgba,
        intensity_range,
    })
}

/// Pack the extracted SoA into 16B x N under the current color settings (missing field -> Flat fallback + a notice).
fn pack_cloud(
    cloud: &ExtractedCloud,
    mode: ColorMode,
    flat: [u8; 4],
    scale: &IntensityScale,
) -> (PointBatch, Option<String>) {
    let mut builder = PointBatchBuilder::with_capacity(cloud.positions.len());
    let mut warning = None;
    match mode {
        ColorMode::Flat => {
            for p in &cloud.positions {
                builder.push(*p, flat);
            }
        }
        ColorMode::Intensity => match &cloud.intensity {
            Some(values) => {
                let range = scale.resolve(cloud.intensity_range);
                for (p, v) in cloud.positions.iter().zip(values) {
                    builder.push(*p, intensity_color(range, *v));
                }
            }
            None => {
                warning = Some("intensity field not found — using flat color".to_owned());
                for p in &cloud.positions {
                    builder.push(*p, flat);
                }
            }
        },
        ColorMode::Rgb => match &cloud.rgba {
            Some(values) => {
                let lut = theme::srgb_to_linear_u8();
                for (p, c) in cloud.positions.iter().zip(values) {
                    builder.push(
                        *p,
                        [
                            lut[c[0] as usize],
                            lut[c[1] as usize],
                            lut[c[2] as usize],
                            c[3],
                        ],
                    );
                }
            }
            None => {
                warning = Some("rgb/rgba field not found — using flat color".to_owned());
                for p in &cloud.positions {
                    builder.push(*p, flat);
                }
            }
        },
    }
    (builder.build(), warning)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{BatchData, POINT_STRIDE, colormap};

    #[test]
    fn settings_roundtrip_via_trait() {
        let r = PointCloud2Renderer {
            flat_color: Color32::from_rgb(9, 8, 7),
            color_mode: ColorMode::Rgb,
            style: PointStyleSettings::new(0.5),
            intensity: IntensityScale {
                auto: false,
                min: 0.0,
                max: 4.0,
            },
            ..Default::default()
        };
        let value = r.settings().expect("point cloud has settings");
        let mut restored = PointCloud2Renderer::default();
        restored.apply_settings(&value);
        assert_eq!(restored.flat_color, r.flat_color);
        assert_eq!(restored.color_mode, r.color_mode);
        assert_eq!(restored.style, r.style);
        assert_eq!(restored.intensity, r.intensity);
    }
    use crate::tf::buffer::{TfBuffer, TfTransform, tf_update};
    use nalgebra::{Translation3, UnitQuaternion};

    fn field(name: &str, offset: u32, datatype: u8, count: u32) -> Value {
        Value::Struct(vec![
            ("name".to_owned(), Value::String(name.to_owned())),
            ("offset".to_owned(), Value::U32(offset)),
            ("datatype".to_owned(), Value::U8(datatype)),
            ("count".to_owned(), Value::U32(count)),
        ])
    }

    #[allow(clippy::too_many_arguments)]
    fn cloud_value(
        width: u32,
        height: u32,
        point_step: u32,
        row_step: u32,
        big_endian: bool,
        fields: Vec<Value>,
        data: Vec<u8>,
    ) -> Value {
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
                    ("frame_id".to_owned(), Value::String("cloud".to_owned())),
                ]),
            ),
            ("height".to_owned(), Value::U32(height)),
            ("width".to_owned(), Value::U32(width)),
            ("fields".to_owned(), Value::Array(fields)),
            ("is_bigendian".to_owned(), Value::Bool(big_endian)),
            ("point_step".to_owned(), Value::U32(point_step)),
            ("row_step".to_owned(), Value::U32(row_step)),
            ("data".to_owned(), Value::Bytes(data)),
            ("is_dense".to_owned(), Value::Bool(true)),
        ])
    }

    fn xyz_fields() -> Vec<Value> {
        vec![
            field("x", 0, DT_FLOAT32, 1),
            field("y", 4, DT_FLOAT32, 1),
            field("z", 8, DT_FLOAT32, 1),
        ]
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn extracts_xyz_only_cloud() {
        let data = f32_bytes(&[1.0, 2.0, 3.0, -4.0, 5.0, -6.0]);
        let value = cloud_value(2, 1, 12, 24, false, xyz_fields(), data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        assert_eq!(cloud.frame_id, "cloud");
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]);
        assert!(cloud.intensity.is_none());
        assert!(cloud.rgba.is_none());
    }

    #[test]
    fn extracts_intensity_with_auto_range() {
        let mut fields = xyz_fields();
        fields.push(field("intensity", 12, DT_UINT16, 1));
        let mut data = Vec::new();
        for (xyz, i) in [([0.0f32, 0.0, 0.0], 100u16), ([1.0, 0.0, 0.0], 500)] {
            data.extend(f32_bytes(&xyz));
            data.extend(i.to_le_bytes());
        }
        let value = cloud_value(2, 1, 14, 28, false, fields, data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        assert_eq!(cloud.intensity, Some(vec![100.0, 500.0]));
        assert_eq!(cloud.intensity_range, Some((100.0, 500.0)));
    }

    #[test]
    fn extracts_packed_rgb_float32() {
        let mut fields = xyz_fields();
        fields.push(field("rgb", 12, DT_FLOAT32, 1));
        let mut data = f32_bytes(&[0.0, 0.0, 0.0]);
        // PCL convention: 0x00RRGGBB stored as a FLOAT32 bit pattern ([b, g, r, 0] in LE).
        data.extend(0x0010_2030_u32.to_le_bytes());
        let value = cloud_value(1, 1, 16, 16, false, fields, data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        assert_eq!(cloud.rgba, Some(vec![[0x10, 0x20, 0x30, 255]]));
    }

    #[test]
    fn extracts_rgba_uint32_with_alpha() {
        let mut fields = xyz_fields();
        fields.push(field("rgba", 12, DT_UINT32, 1));
        let mut data = f32_bytes(&[0.0, 0.0, 0.0]);
        data.extend(0x8010_2030_u32.to_le_bytes());
        let value = cloud_value(1, 1, 16, 16, false, fields, data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        assert_eq!(cloud.rgba, Some(vec![[0x10, 0x20, 0x30, 0x80]]));
    }

    #[test]
    fn accepts_nonstandard_uint32_rgb_without_alpha() {
        let mut fields = xyz_fields();
        fields.push(field("rgb", 12, DT_UINT32, 1));
        let mut data = f32_bytes(&[0.0, 0.0, 0.0]);
        data.extend(0x8010_2030_u32.to_le_bytes());
        let value = cloud_value(1, 1, 16, 16, false, fields, data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        // Under the rgb name, the alpha byte is ignored and fixed at 255.
        assert_eq!(cloud.rgba, Some(vec![[0x10, 0x20, 0x30, 255]]));
    }

    #[test]
    fn follows_field_offsets_regardless_of_order() {
        // Order z, x, y with non-contiguous offsets (gaps for other fields).
        let fields = vec![
            field("z", 0, DT_FLOAT32, 1),
            field("x", 8, DT_FLOAT32, 1),
            field("y", 16, DT_FLOAT32, 1),
        ];
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(&3.0f32.to_le_bytes());
        data[8..12].copy_from_slice(&1.0f32.to_le_bytes());
        data[16..20].copy_from_slice(&2.0f32.to_le_bytes());
        let value = cloud_value(1, 1, 20, 20, false, fields, data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn narrows_float64_xyz() {
        let fields = vec![
            field("x", 0, DT_FLOAT64, 1),
            field("y", 8, DT_FLOAT64, 1),
            field("z", 16, DT_FLOAT64, 1),
        ];
        let data: Vec<u8> = [1.5f64, -2.5, 3.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let value = cloud_value(1, 1, 24, 24, false, fields, data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        assert_eq!(cloud.positions, vec![[1.5, -2.5, 3.5]]);
    }

    #[test]
    fn rejects_integer_xyz_with_diagnostic() {
        let fields = vec![
            field("x", 0, DT_INT32, 1),
            field("y", 4, DT_FLOAT32, 1),
            field("z", 8, DT_FLOAT32, 1),
        ];
        let value = cloud_value(1, 1, 12, 12, false, fields, vec![0; 12]);
        let err = extract_cloud(&value).expect_err("integer x must be rejected");
        assert!(err.contains("`x`") && err.contains("datatype 5"), "err={err}");
    }

    #[test]
    fn rejects_missing_xyz_with_field_names() {
        let fields = vec![field("x", 0, DT_FLOAT32, 1)];
        let value = cloud_value(1, 1, 12, 12, false, fields, vec![0; 12]);
        let err = extract_cloud(&value).expect_err("missing y/z");
        assert!(err.contains("y, z"), "err={err}");
    }

    #[test]
    fn rejects_multi_count_xyz() {
        let fields = vec![
            field("x", 0, DT_FLOAT32, 2),
            field("y", 8, DT_FLOAT32, 1),
            field("z", 12, DT_FLOAT32, 1),
        ];
        let value = cloud_value(1, 1, 16, 16, false, fields, vec![0; 16]);
        let err = extract_cloud(&value).expect_err("count 2 must be rejected");
        assert!(err.contains("count 2"), "err={err}");
    }

    #[test]
    fn rejects_zero_point_step() {
        let value = cloud_value(1, 1, 0, 0, false, xyz_fields(), vec![0; 12]);
        let err = extract_cloud(&value).expect_err("point_step 0");
        assert!(err.contains("point_step"), "err={err}");
    }

    #[test]
    fn rejects_short_data_with_expected_length() {
        let value = cloud_value(2, 1, 12, 24, false, xyz_fields(), vec![0; 12]);
        let err = extract_cloud(&value).expect_err("short data");
        assert!(err.contains("expected at least 24"), "err={err}");
    }

    #[test]
    fn reads_big_endian_clouds() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let value = cloud_value(1, 1, 12, 12, true, xyz_fields(), data);
        let cloud = extract_cloud(&value).expect("valid BE cloud");
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn honors_row_step_padding_in_organized_clouds() {
        // width=1, height=2, point_step=12, row_step=16 (4B padding at row end).
        let mut data = f32_bytes(&[1.0, 2.0, 3.0]);
        data.extend([0u8; 4]);
        data.extend(f32_bytes(&[4.0, 5.0, 6.0]));
        let value = cloud_value(1, 2, 12, 16, false, xyz_fields(), data);
        let cloud = extract_cloud(&value).expect("valid organized cloud");
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    }

    #[test]
    fn falls_back_to_packed_rows_when_row_step_is_zero() {
        let data = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let value = cloud_value(1, 2, 12, 0, false, xyz_fields(), data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        assert_eq!(cloud.positions.len(), 2);
    }

    #[test]
    fn skips_non_finite_points() {
        let data = f32_bytes(&[
            f32::NAN,
            0.0,
            0.0,
            1.0,
            f32::INFINITY,
            0.0,
            7.0,
            8.0,
            9.0,
        ]);
        let value = cloud_value(3, 1, 12, 36, false, xyz_fields(), data);
        let cloud = extract_cloud(&value).expect("valid cloud");
        assert_eq!(cloud.positions, vec![[7.0, 8.0, 9.0]]);
    }

    #[test]
    fn pack_flat_uses_uniform_color() {
        let cloud = ExtractedCloud {
            frame_id: "cloud".to_owned(),
            stamp: 0,
            positions: vec![[1.0, 2.0, 3.0]],
            intensity: None,
            rgba: None,
            intensity_range: None,
        };
        let (batch, warning) =
            pack_cloud(&cloud, ColorMode::Flat, [1, 2, 3, 4], &IntensityScale::default());
        assert!(warning.is_none());
        assert_eq!(batch.count, 1);
        assert_eq!(&batch.bytes[12..16], &[1, 2, 3, 4]);
    }

    #[test]
    fn pack_intensity_maps_range_ends_to_colormap_ends() {
        let cloud = ExtractedCloud {
            frame_id: "cloud".to_owned(),
            stamp: 0,
            positions: vec![[0.0; 3], [0.0; 3]],
            intensity: Some(vec![10.0, 50.0]),
            rgba: None,
            intensity_range: Some((10.0, 50.0)),
        };
        let (batch, warning) = pack_cloud(
            &cloud,
            ColorMode::Intensity,
            [0; 4],
            &IntensityScale::default(),
        );
        assert!(warning.is_none());
        assert_eq!(&batch.bytes[12..16], &colormap(0.0));
        assert_eq!(&batch.bytes[POINT_STRIDE + 12..POINT_STRIDE + 16], &colormap(1.0));
    }

    #[test]
    fn pack_rgb_applies_srgb_to_linear_lut() {
        let cloud = ExtractedCloud {
            frame_id: "cloud".to_owned(),
            stamp: 0,
            positions: vec![[0.0; 3]],
            intensity: None,
            rgba: Some(vec![[255, 128, 0, 200]]),
            intensity_range: None,
        };
        let (batch, warning) =
            pack_cloud(&cloud, ColorMode::Rgb, [0; 4], &IntensityScale::default());
        assert!(warning.is_none());
        let lut = theme::srgb_to_linear_u8();
        assert_eq!(
            &batch.bytes[12..16],
            &[lut[255], lut[128], lut[0], 200]
        );
    }

    #[test]
    fn pack_falls_back_to_flat_when_mode_field_missing() {
        let cloud = ExtractedCloud {
            frame_id: "cloud".to_owned(),
            stamp: 0,
            positions: vec![[0.0; 3]],
            intensity: None,
            rgba: None,
            intensity_range: None,
        };
        let (batch, warning) = pack_cloud(
            &cloud,
            ColorMode::Intensity,
            [9, 9, 9, 9],
            &IntensityScale::default(),
        );
        assert!(warning.is_some());
        assert_eq!(&batch.bytes[12..16], &[9, 9, 9, 9]);
        let (_, warning) = pack_cloud(&cloud, ColorMode::Rgb, [9; 4], &IntensityScale::default());
        assert!(warning.is_some());
    }

    fn dyn_tf(x: f64, stamp: i64) -> TfTransform {
        TfTransform {
            parent: "map".to_owned(),
            child: "cloud".to_owned(),
            stamp,
            transform: Isometry3::from_parts(
                Translation3::new(x, 0.0, 0.0),
                UnitQuaternion::identity(),
            ),
        }
    }

    fn stamped_cloud_value(stamp: i64, x: f32) -> Value {
        let mut value = cloud_value(1, 1, 12, 12, false, xyz_fields(), f32_bytes(&[x, 0.0, 0.0]));
        let Value::Struct(fields) = &mut value else {
            unreachable!();
        };
        fields[0].1 = Value::Struct(vec![
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
            ("frame_id".to_owned(), Value::String("cloud".to_owned())),
        ]);
        value
    }

    #[test]
    fn scene_returns_points_batch_with_stamp_pose_as_model() {
        let mut renderer = PointCloud2Renderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![dyn_tf(1.0, 1_000), dyn_tf(5.0, 2_000)], false));
        renderer.on_message(&stamped_cloud_value(1_500, 2.0));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let batches = renderer.scene(&tf).expect("baked");
        let BatchData::Points(points) = &batches[0].data else {
            panic!("expected points batch");
        };
        assert_eq!(points.count, 1);
        // Stored in local coords; the stamp=1500 interpolated pose (x=3) goes into the model matrix.
        assert_eq!(&points.bytes[0..4], &2.0f32.to_le_bytes());
        assert!((batches[0].model[(0, 3)] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn scene_keeps_old_batch_while_tf_lags_then_updates() {
        let mut renderer = PointCloud2Renderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![dyn_tf(1.0, 1_000)], false));
        renderer.on_message(&stamped_cloud_value(1_000, 2.0));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let b1 = renderer.scene(&tf).expect("baked");
        let gen1 = b1[0].generation;
        // Keep the old batch/pose until TF catches up to the new message's stamp (avoid mixing a new cloud with an old pose).
        renderer.on_message(&stamped_cloud_value(3_000, 7.0));
        let b2 = renderer.scene(&tf).expect("old batch kept");
        assert_eq!(b2[0].generation, gen1);
        let BatchData::Points(points) = &b2[0].data else {
            panic!("expected points batch");
        };
        assert_eq!(&points.bytes[0..4], &2.0f32.to_le_bytes());
        assert!((b2[0].model[(0, 3)] - 1.0).abs() < 1e-5);
        buffer.insert(&tf_update(vec![dyn_tf(5.0, 3_000)], false));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let b3 = renderer.scene(&tf).expect("rebaked");
        assert!(b3[0].generation > gen1);
        let BatchData::Points(points) = &b3[0].data else {
            panic!("expected points batch");
        };
        assert_eq!(&points.bytes[0..4], &7.0f32.to_le_bytes());
        assert!((b3[0].model[(0, 3)] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn settings_change_repacks_without_new_message() {
        let mut renderer = PointCloud2Renderer::default();
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![dyn_tf(0.0, 1_000)], false));
        renderer.on_message(&stamped_cloud_value(1_000, 2.0));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let gen1 = renderer.scene(&tf).expect("baked")[0].generation;
        renderer.flat_color = Color32::from_rgb(1, 2, 3);
        renderer.repack_dirty = true;
        let b2 = renderer.scene(&tf).expect("repacked");
        assert!(b2[0].generation > gen1);
        let BatchData::Points(points) = &b2[0].data else {
            panic!("expected points batch");
        };
        assert_eq!(
            &points.bytes[12..16],
            &theme::to_linear_rgba8(Color32::from_rgb(1, 2, 3))
        );
    }

    #[test]
    fn renderer_status_transitions() {
        let mut renderer = PointCloud2Renderer::default();
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
        renderer.on_message(&stamped_cloud_value(0, 1.0));
        assert_eq!(
            renderer.scene(&tf).unwrap_err(),
            RenderStatus::TfUnavailable {
                frame: "cloud".to_owned()
            }
        );
    }
}
