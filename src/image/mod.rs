//! 2D image display (sensor_msgs/Image + CompressedImage): a `View2d` whose frames go through egui's texture path, not wgpu.

pub mod codec;
pub mod convert;

use std::time::{Duration, Instant};

use egui::RichText;
use serde::{Deserialize, Serialize};

use crate::decode::value::Value;
use crate::plugin::registry::{Registrar, View2dDescriptor};
use crate::plugin::view2d::View2d;
use crate::render::{IntensityScale, RenderStatus};
use crate::theme;
use convert::{ConvertError, DepthInfo, RawImage};

/// ROS type names this view handles; registered as two builtin 2D display types.
const IMAGE_TYPES: [&str; 2] = ["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"];

/// Register the 2D display types visor ships with, through the same API a plugin uses.
pub fn register_builtin(reg: &mut Registrar<'_>) {
    for ros_type in IMAGE_TYPES {
        let label = short_label(ros_type);
        reg.view2d(View2dDescriptor::new(ros_type, label, move || {
            Box::new(ImageView::new(ros_type))
        }));
    }
}

/// Short label for the Displays title, mirroring a renderer descriptor's label.
fn short_label(ros_type: &str) -> &'static str {
    match ros_type {
        "sensor_msgs/msg/CompressedImage" => "CompressedImage",
        _ => "Image",
    }
}

/// Colormap applied to normalized depth (color images always use publisher pixels directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DepthColormap {
    /// RViz-like linear grayscale (near = dark, far = bright). Default.
    #[default]
    Grayscale,
    /// This viewer's theme gradient, interpolated in sRGB.
    Viewer,
}

impl DepthColormap {
    fn label(self) -> &'static str {
        match self {
            DepthColormap::Grayscale => "Grayscale",
            DepthColormap::Viewer => "Viewer",
        }
    }
}

/// Persisted per-item image settings (opaque toml under DisplayConfig::settings). `#[serde(default)]` keeps old configs valid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ImageSettings {
    /// Depth normalization range (Auto = measured min/max, Manual = fixed), reused from the 3D intensity UI.
    pub depth_range: IntensityScale,
    pub depth_colormap: DepthColormap,
}

/// Why the image cannot be shown (surfaced in the panel and mapped into the Displays status).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageStatus {
    /// No message received yet.
    NoData,
    /// Encoding/format recognized but not supported (Bayer/YUV/compressedDepth).
    Unsupported(String),
    /// Malformed message or codec failure.
    DecodeError(String),
}

/// A converted frame ready to upload (RGBA `ColorImage`) plus metadata for the header.
struct Frame {
    image: egui::ColorImage,
    width: usize,
    height: usize,
    encoding: String,
    depth: Option<DepthInfo>,
    frame_id: Option<String>,
    received_at: Instant,
}

/// Raw depth bytes kept so colormap/range changes recolor immediately without waiting for a new message.
struct DepthSource {
    data: Vec<u8>,
    width: usize,
    height: usize,
    encoding: String,
    is_bigendian: bool,
    step: usize,
}

/// Frame metadata for the panel header (returned by value so the panel doesn't hold a borrow across texture upload).
pub struct FrameMeta {
    pub width: usize,
    pub height: usize,
    pub encoding: String,
    pub depth: Option<DepthInfo>,
    pub frame_id: Option<String>,
    pub age: Duration,
}

/// One image topic's display state: latest converted frame, generation-gated texture, and settings.
pub struct ImageView {
    ros_type: &'static str,
    latest: Option<Frame>,
    /// Bumped on every new/recolored frame; the texture uploads only when its generation lags.
    generation: u64,
    texture: Option<(u64, egui::TextureHandle)>,
    settings: ImageSettings,
    status: Option<ImageStatus>,
    /// Present only while the latest frame is a depth image.
    depth_source: Option<DepthSource>,
}

impl ImageView {
    pub fn new(ros_type: &str) -> Self {
        let ros_type = IMAGE_TYPES
            .into_iter()
            .find(|t| *t == ros_type)
            .unwrap_or(IMAGE_TYPES[0]);
        Self {
            ros_type,
            latest: None,
            generation: 0,
            texture: None,
            settings: ImageSettings::default(),
            status: Some(ImageStatus::NoData),
            depth_source: None,
        }
    }

    fn is_compressed(&self) -> bool {
        self.ros_type == "sensor_msgs/msg/CompressedImage"
    }

    /// Ingest a decoded message on the UI thread; on failure keep the last good frame and record the status.
    fn ingest(&mut self, value: &Value) {
        let decoded = if self.is_compressed() {
            self.decode_compressed(value)
        } else {
            self.decode_raw(value)
        };
        match decoded {
            Ok(frame) => {
                self.latest = Some(frame);
                self.generation += 1;
                self.status = None;
            }
            Err(status) => self.status = Some(status),
        }
    }

    fn decode_raw(&mut self, value: &Value) -> Result<Frame, ImageStatus> {
        let width = u32_field(value, "width").ok_or_else(|| missing("width"))? as usize;
        let height = u32_field(value, "height").ok_or_else(|| missing("height"))? as usize;
        let encoding = string_field(value, "encoding").ok_or_else(|| missing("encoding"))?;
        let is_bigendian = u8_field(value, "is_bigendian").is_some_and(|v| v != 0);
        let step = u32_field(value, "step").ok_or_else(|| missing("step"))? as usize;
        let data = bytes_field(value, "data").ok_or_else(|| missing("data"))?;
        let raw = RawImage {
            width,
            height,
            encoding: &encoding,
            is_bigendian,
            step,
            data,
        };
        let converted = convert::convert_raw(&raw, &self.settings).map_err(status_from_convert)?;
        // Retain the raw bytes only for depth, so range/colormap edits recolor without a new message.
        self.depth_source = converted.depth.is_some().then(|| DepthSource {
            data: data.to_vec(),
            width,
            height,
            encoding: encoding.clone(),
            is_bigendian,
            step,
        });
        Ok(Frame {
            image: converted.image,
            width,
            height,
            encoding,
            depth: converted.depth,
            frame_id: frame_id(value),
            received_at: Instant::now(),
        })
    }

    fn decode_compressed(&mut self, value: &Value) -> Result<Frame, ImageStatus> {
        let format = string_field(value, "format").unwrap_or_default();
        let data = bytes_field(value, "data").ok_or_else(|| missing("data"))?;
        let image = codec::decode_compressed(&format, data).map_err(status_from_convert)?;
        let [width, height] = image.size;
        self.depth_source = None;
        Ok(Frame {
            image,
            width,
            height,
            encoding: format,
            depth: None,
            frame_id: frame_id(value),
            received_at: Instant::now(),
        })
    }

    /// Recolor the retained depth frame after a settings change (no-op for color/compressed frames).
    fn reconvert_depth(&mut self) {
        let Some(src) = &self.depth_source else {
            return;
        };
        let raw = RawImage {
            width: src.width,
            height: src.height,
            encoding: &src.encoding,
            is_bigendian: src.is_bigendian,
            step: src.step,
            data: &src.data,
        };
        if let Ok(converted) = convert::convert_raw(&raw, &self.settings)
            && let Some(frame) = &mut self.latest
        {
            frame.image = converted.image;
            frame.depth = converted.depth;
            self.generation += 1;
        }
    }

    /// Per-item settings panel (depth range + colormap). Only shown for depth encodings.
    fn draw_settings(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        let is_depth = self.depth_source.is_some();
        if !is_depth {
            ui.colored_label(p.text_muted, "no adjustable settings for this encoding");
            return;
        }
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Colormap").color(p.text_muted));
            egui::ComboBox::from_id_salt("image_depth_colormap")
                .selected_text(self.settings.depth_colormap.label())
                .show_ui(ui, |ui| {
                    for cmap in [DepthColormap::Grayscale, DepthColormap::Viewer] {
                        changed |= ui
                            .selectable_value(&mut self.settings.depth_colormap, cmap, cmap.label())
                            .changed();
                    }
                });
        });
        changed |= self.settings.depth_range.ui(ui);
        if changed {
            self.reconvert_depth();
        }
    }

    /// Map the image status onto the shared Displays status vocabulary.
    fn render_status(&self) -> Option<RenderStatus> {
        match &self.status {
            None => None,
            Some(ImageStatus::NoData) => Some(RenderStatus::NoData),
            Some(ImageStatus::Unsupported(s)) => {
                Some(RenderStatus::InvalidMessage(format!("unsupported: {s}")))
            }
            Some(ImageStatus::DecodeError(s)) => Some(RenderStatus::InvalidMessage(s.clone())),
        }
    }

    /// Codec-level status, kept separate from the shared Displays vocabulary because the panel words it differently.
    pub fn image_status(&self) -> Option<&ImageStatus> {
        self.status.as_ref()
    }

    /// Header metadata for the latest frame (None until a frame arrives).
    pub fn meta(&self) -> Option<FrameMeta> {
        self.latest.as_ref().map(|f| FrameMeta {
            width: f.width,
            height: f.height,
            encoding: f.encoding.clone(),
            depth: f.depth,
            frame_id: f.frame_id.clone(),
            age: f.received_at.elapsed(),
        })
    }

    /// The texture for the latest frame, (re)uploaded only when a new generation is pending.
    fn texture(&mut self, ctx: &egui::Context, name: &str) -> Option<&egui::TextureHandle> {
        let frame = self.latest.as_ref()?;
        let needs_upload = self
            .texture
            .as_ref()
            .map(|(g, _)| *g != self.generation)
            .unwrap_or(true);
        if needs_upload {
            let handle = ctx.load_texture(name, frame.image.clone(), egui::TextureOptions::LINEAR);
            self.texture = Some((self.generation, handle));
        }
        self.texture.as_ref().map(|(_, handle)| handle)
    }
}

impl View2d for ImageView {
    fn on_message(&mut self, value: &Value) {
        self.ingest(value);
    }

    /// Aspect-fit image with a topic/resolution/encoding/age header and status fallbacks.
    fn ui(&mut self, ui: &mut egui::Ui, topic: &str) {
        let p = theme::ui::palette();
        // Capture the true leaf width before the header: the horizontal header row does not wrap and inflates the content width, which would otherwise make the image overflow and clip.
        let leaf_width = ui.available_width();
        let meta = self.meta();
        ui.horizontal(|ui| {
            ui.label(RichText::new(topic).color(p.text_primary).strong());
            if let Some(m) = &meta {
                ui.label(RichText::new(format!("{}×{}", m.width, m.height)).color(p.text_muted));
                ui.label(RichText::new(&m.encoding).color(p.text_muted));
            }
        });
        if let Some(m) = &meta {
            let mut line = format!("received {:.1}s ago", m.age.as_secs_f32());
            if let Some(frame_id) = &m.frame_id {
                line = format!("frame: {frame_id}   {line}");
            }
            if let Some(depth) = &m.depth {
                line = format!(
                    "{line}   range {:.2}–{:.2} {}",
                    depth.range.0, depth.range.1, depth.unit
                );
            }
            ui.colored_label(p.text_muted, line);
        }
        // An error still leaves the last good frame visible below.
        match self.image_status() {
            Some(ImageStatus::NoData) => {
                ui.colored_label(p.text_muted, "waiting for messages…");
            }
            Some(ImageStatus::Unsupported(name)) => {
                ui.colored_label(p.status_warn, format!("unsupported encoding: {name}"));
            }
            Some(ImageStatus::DecodeError(error)) => {
                ui.colored_label(p.status_error, format!("decode error: {error}"));
            }
            None => {}
        }
        let ctx = ui.ctx().clone();
        if let Some(texture) = self.texture(&ctx, topic) {
            let size = egui::vec2(leaf_width, ui.available_height());
            ui.add_sized(
                size,
                egui::Image::from_texture(texture)
                    .maintain_aspect_ratio(true)
                    .shrink_to_fit(),
            );
        }
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        self.draw_settings(ui);
    }

    fn status(&self) -> Option<RenderStatus> {
        self.render_status()
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(&self.settings).ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(settings) = value.clone().try_into::<ImageSettings>() {
            self.settings = settings;
        }
    }
}

fn missing(field: &str) -> ImageStatus {
    ImageStatus::DecodeError(format!("missing field `{field}`"))
}

fn status_from_convert(error: ConvertError) -> ImageStatus {
    match error {
        ConvertError::UnsupportedEncoding(e) => ImageStatus::Unsupported(e),
        ConvertError::InvalidImage(e) => ImageStatus::DecodeError(e),
    }
}

fn frame_id(value: &Value) -> Option<String> {
    match value.get("header")?.get("frame_id")? {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

fn u32_field(value: &Value, name: &str) -> Option<u32> {
    match value.get(name)? {
        Value::U32(v) => Some(*v),
        _ => None,
    }
}

fn u8_field(value: &Value, name: &str) -> Option<u8> {
    match value.get(name)? {
        Value::U8(v) => Some(*v),
        _ => None,
    }
}

fn string_field(value: &Value, name: &str) -> Option<String> {
    match value.get(name)? {
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn bytes_field<'a>(value: &'a Value, name: &str) -> Option<&'a [u8]> {
    match value.get(name)? {
        Value::Bytes(b) => Some(b),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_message(encoding: &str, width: u32, height: u32, step: u32, data: Vec<u8>) -> Value {
        Value::Struct(vec![
            ("height".to_owned(), Value::U32(height)),
            ("width".to_owned(), Value::U32(width)),
            ("encoding".to_owned(), Value::String(encoding.to_owned())),
            ("is_bigendian".to_owned(), Value::U8(0)),
            ("step".to_owned(), Value::U32(step)),
            ("data".to_owned(), Value::Bytes(data)),
        ])
    }

    #[test]
    fn short_label_mirrors_the_registered_display_label() {
        assert_eq!(short_label("sensor_msgs/msg/Image"), "Image");
        assert_eq!(
            short_label("sensor_msgs/msg/CompressedImage"),
            "CompressedImage"
        );
    }

    #[test]
    fn on_message_updates_frame_and_bumps_generation() {
        let mut view = ImageView::new("sensor_msgs/msg/Image");
        assert_eq!(view.image_status(), Some(&ImageStatus::NoData));
        view.on_message(&image_message("rgb8", 1, 1, 3, vec![10, 20, 30]));
        assert_eq!(view.image_status(), None);
        assert_eq!(view.generation, 1);
        let meta = view.meta().unwrap();
        assert_eq!((meta.width, meta.height), (1, 1));
        assert!(meta.depth.is_none());
    }

    #[test]
    fn unsupported_encoding_keeps_status_and_no_frame() {
        let mut view = ImageView::new("sensor_msgs/msg/Image");
        view.on_message(&image_message("yuv422", 2, 1, 4, vec![0; 4]));
        assert!(matches!(
            view.image_status(),
            Some(ImageStatus::Unsupported(_))
        ));
        assert!(view.meta().is_none());
    }

    #[test]
    fn depth_message_records_range_and_recolors_on_settings_change() {
        let mut view = ImageView::new("sensor_msgs/msg/Image");
        // 16UC1: values 100 and 300 (little-endian), auto range.
        view.on_message(&image_message("16UC1", 2, 1, 4, vec![100, 0, 44, 1]));
        let meta = view.meta().unwrap();
        assert_eq!(meta.depth.unwrap().range, (100.0, 300.0));
        let gen_before = view.generation;
        view.settings.depth_colormap = DepthColormap::Viewer;
        view.reconvert_depth();
        assert_eq!(view.generation, gen_before + 1);
    }

    #[test]
    fn settings_round_trip_through_toml() {
        let mut view = ImageView::new("sensor_msgs/msg/Image");
        view.settings.depth_colormap = DepthColormap::Viewer;
        view.settings.depth_range = IntensityScale {
            auto: false,
            min: 0.5,
            max: 4.0,
        };
        let toml = View2d::settings(&view).unwrap();
        let mut restored = ImageView::new("sensor_msgs/msg/Image");
        View2d::apply_settings(&mut restored, &toml);
        assert_eq!(restored.settings, view.settings);
    }
}
