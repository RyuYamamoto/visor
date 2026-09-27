//! Raw `sensor_msgs/Image` -> egui `ColorImage` conversion (color / mono / depth), with `step` and endianness handling.
//!
//! Colors use egui's sRGB `Color32` semantics: publisher bytes go
//! straight into `ColorImage` without linearization, and depth/mono LUTs are generated in sRGB space,
//! unlike the 3D wgpu path which packs linear values.

use egui::{Color32, ColorImage};

use super::{DepthColormap, ImageSettings};
use crate::theme;

/// Borrowed view of a decoded `sensor_msgs/Image` message (fields already extracted from the CDR `Value`).
pub struct RawImage<'a> {
    pub width: usize,
    pub height: usize,
    pub encoding: &'a str,
    pub is_bigendian: bool,
    /// Full row length in bytes (may exceed width * channels when padded).
    pub step: usize,
    pub data: &'a [u8],
}

/// Why a raw image could not be turned into a `ColorImage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    /// The encoding string is recognized as a real format we do not support yet (Bayer/YUV/...).
    UnsupportedEncoding(String),
    /// The message is malformed for its declared encoding (short data, zero size, ...).
    InvalidImage(String),
}

impl ConvertError {
    /// Human-readable one-liner for the Displays status and the panel fallback.
    pub fn message(&self) -> String {
        match self {
            ConvertError::UnsupportedEncoding(e) => format!("unsupported encoding: {e}"),
            ConvertError::InvalidImage(e) => format!("invalid image: {e}"),
        }
    }
}

/// Depth-specific metadata surfaced in the panel header (raw-value range + unit convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthInfo {
    /// Range actually used for normalization, in the encoding's native unit.
    pub range: (f32, f32),
    /// Native unit of the raw values (16UC1 = mm, 32FC1 = m).
    pub unit: &'static str,
}

/// Successful conversion: the RGBA image plus optional depth metadata.
pub struct Converted {
    pub image: ColorImage,
    /// Present only for depth encodings (16UC1 / 32FC1).
    pub depth: Option<DepthInfo>,
}

/// Convert a raw image to a `ColorImage`, applying depth normalization/colormap from `settings` when relevant.
pub fn convert_raw(img: &RawImage<'_>, settings: &ImageSettings) -> Result<Converted, ConvertError> {
    if img.width == 0 || img.height == 0 {
        return Err(ConvertError::InvalidImage("zero-sized image".to_owned()));
    }
    match canonical_encoding(img.encoding) {
        "rgb8" => color(img, ChannelOrder::Rgb),
        "rgba8" => color(img, ChannelOrder::Rgba),
        "bgr8" => color(img, ChannelOrder::Bgr),
        "bgra8" => color(img, ChannelOrder::Bgra),
        "mono8" => mono8(img),
        "mono16" => mono16(img),
        "16UC1" => depth_u16(img, settings),
        "32FC1" => depth_f32(img, settings),
        other => Err(ConvertError::UnsupportedEncoding(other.to_owned())),
    }
}

/// Map OpenCV-style aliases to their sensor_msgs equivalents (rest pass through unchanged).
fn canonical_encoding(encoding: &str) -> &str {
    match encoding {
        "8UC1" => "mono8",
        "8UC3" => "bgr8",
        other => other,
    }
}

/// Byte order of an 8-bit color pixel.
#[derive(Clone, Copy)]
enum ChannelOrder {
    Rgb,
    Rgba,
    Bgr,
    Bgra,
}

impl ChannelOrder {
    fn channels(self) -> usize {
        match self {
            ChannelOrder::Rgb | ChannelOrder::Bgr => 3,
            ChannelOrder::Rgba | ChannelOrder::Bgra => 4,
        }
    }

    fn to_rgba(self, p: &[u8]) -> [u8; 4] {
        match self {
            ChannelOrder::Rgb => [p[0], p[1], p[2], 255],
            ChannelOrder::Rgba => [p[0], p[1], p[2], p[3]],
            ChannelOrder::Bgr => [p[2], p[1], p[0], 255],
            ChannelOrder::Bgra => [p[2], p[1], p[0], p[3]],
        }
    }
}

/// One image row's byte slice, honoring `step` (row stride) and reporting short data.
fn row_slice<'a>(img: &'a RawImage<'a>, row: usize, bytes_per_row: usize) -> Result<&'a [u8], ConvertError> {
    let start = row * img.step;
    img.data
        .get(start..start + bytes_per_row)
        .ok_or_else(|| {
            ConvertError::InvalidImage(format!(
                "data too short: need step*height = {}*{} bytes, got {}",
                img.step,
                img.height,
                img.data.len()
            ))
        })
}

fn color(img: &RawImage<'_>, order: ChannelOrder) -> Result<Converted, ConvertError> {
    let channels = order.channels();
    let mut rgba = Vec::with_capacity(img.width * img.height * 4);
    for row in 0..img.height {
        let line = row_slice(img, row, img.width * channels)?;
        for px in 0..img.width {
            let p = &line[px * channels..px * channels + channels];
            rgba.extend_from_slice(&order.to_rgba(p));
        }
    }
    Ok(Converted {
        image: ColorImage::from_rgba_unmultiplied([img.width, img.height], &rgba),
        depth: None,
    })
}

fn mono8(img: &RawImage<'_>) -> Result<Converted, ConvertError> {
    let mut rgba = Vec::with_capacity(img.width * img.height * 4);
    for row in 0..img.height {
        let line = row_slice(img, row, img.width)?;
        for &g in &line[..img.width] {
            rgba.extend_from_slice(&[g, g, g, 255]);
        }
    }
    Ok(Converted {
        image: ColorImage::from_rgba_unmultiplied([img.width, img.height], &rgba),
        depth: None,
    })
}

/// Read a u16 sample from a 2-byte pixel with the message's endianness.
fn read_u16(p: &[u8], big_endian: bool) -> u16 {
    let bytes = [p[0], p[1]];
    if big_endian {
        u16::from_be_bytes(bytes)
    } else {
        u16::from_le_bytes(bytes)
    }
}

fn mono16(img: &RawImage<'_>) -> Result<Converted, ConvertError> {
    // RViz shows mono16 as a linear full-range grayscale: normalize over the measured min/max.
    let mut values = Vec::with_capacity(img.width * img.height);
    for row in 0..img.height {
        let line = row_slice(img, row, img.width * 2)?;
        for px in 0..img.width {
            values.push(read_u16(&line[px * 2..px * 2 + 2], img.is_bigendian));
        }
    }
    let min = values.iter().copied().min().unwrap_or(0) as f32;
    let max = values.iter().copied().max().unwrap_or(0) as f32;
    let mut rgba = Vec::with_capacity(values.len() * 4);
    for v in values {
        let g = normalize(v as f32, (min, max));
        let g = (g * 255.0).round().clamp(0.0, 255.0) as u8;
        rgba.extend_from_slice(&[g, g, g, 255]);
    }
    Ok(Converted {
        image: ColorImage::from_rgba_unmultiplied([img.width, img.height], &rgba),
        depth: None,
    })
}

fn depth_u16(img: &RawImage<'_>, settings: &ImageSettings) -> Result<Converted, ConvertError> {
    let mut samples = Vec::with_capacity(img.width * img.height);
    for row in 0..img.height {
        let line = row_slice(img, row, img.width * 2)?;
        for px in 0..img.width {
            // 0 is the "no measurement" convention for uint16 depth.
            let raw = read_u16(&line[px * 2..px * 2 + 2], img.is_bigendian);
            samples.push((raw != 0).then_some(raw as f32));
        }
    }
    Ok(depth_image(img, &samples, settings, "mm"))
}

fn depth_f32(img: &RawImage<'_>, settings: &ImageSettings) -> Result<Converted, ConvertError> {
    let mut samples = Vec::with_capacity(img.width * img.height);
    for row in 0..img.height {
        let line = row_slice(img, row, img.width * 4)?;
        for px in 0..img.width {
            let b = &line[px * 4..px * 4 + 4];
            let bytes = [b[0], b[1], b[2], b[3]];
            let v = if img.is_bigendian {
                f32::from_be_bytes(bytes)
            } else {
                f32::from_le_bytes(bytes)
            };
            // NaN/inf and 0.0 are treated as "no measurement".
            samples.push((v.is_finite() && v != 0.0).then_some(v));
        }
    }
    Ok(depth_image(img, &samples, settings, "m"))
}

/// Colorize per-pixel valid/invalid depth samples with the settings' range and colormap.
fn depth_image(
    img: &RawImage<'_>,
    samples: &[Option<f32>],
    settings: &ImageSettings,
    unit: &'static str,
) -> Converted {
    let measured = measured_range(samples);
    let range = settings.depth_range.resolve(measured);
    let mut rgba = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        match sample {
            Some(v) => rgba.extend_from_slice(&depth_color(normalize(*v, range), settings.depth_colormap)),
            // Invalid values stay fully transparent so "no data" is visually distinct.
            None => rgba.extend_from_slice(&[0, 0, 0, 0]),
        }
    }
    Converted {
        image: ColorImage::from_rgba_unmultiplied([img.width, img.height], &rgba),
        depth: Some(DepthInfo { range, unit }),
    }
}

/// Measured (min, max) over valid samples, or None if there are none.
fn measured_range(samples: &[Option<f32>]) -> Option<(f32, f32)> {
    let mut it = samples.iter().flatten().copied();
    let first = it.next()?;
    let (mut min, mut max) = (first, first);
    for v in it {
        min = min.min(v);
        max = max.max(v);
    }
    Some((min, max))
}

/// Normalize a value into [0, 1] over `range` (degenerate range maps to 0 to avoid div-by-zero).
fn normalize(value: f32, range: (f32, f32)) -> f32 {
    if range.1 > range.0 {
        ((value - range.0) / (range.1 - range.0)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Depth colormap sample in sRGB Color32 bytes (near = dark, far = bright for grayscale).
fn depth_color(t: f32, colormap: DepthColormap) -> [u8; 4] {
    match colormap {
        DepthColormap::Grayscale => {
            let g = (t * 255.0).round().clamp(0.0, 255.0) as u8;
            [g, g, g, 255]
        }
        DepthColormap::Viewer => viewer_srgb(t),
    }
}

/// Interpolate the theme colormap stops directly in sRGB space (2D image color convention).
fn viewer_srgb(t: f32) -> [u8; 4] {
    let stops = &theme::POINT_COLORMAP;
    let t = t.clamp(0.0, 1.0);
    let scaled = t * (stops.len() - 1) as f32;
    let i = (scaled as usize).min(stops.len() - 2);
    let f = scaled - i as f32;
    let (a, b): (Color32, Color32) = (stops[i], stops[i + 1]);
    let lerp = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * f).round() as u8;
    [
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
        255,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::IntensityScale;

    fn settings(auto: bool, min: f32, max: f32, colormap: DepthColormap) -> ImageSettings {
        ImageSettings {
            depth_range: IntensityScale { auto, min, max },
            depth_colormap: colormap,
        }
    }

    /// First `n` pixels of the converted image as [r,g,b,a] arrays.
    fn head(c: &Converted, n: usize) -> Vec<[u8; 4]> {
        c.image.pixels.iter().take(n).map(|p| [p.r(), p.g(), p.b(), p.a()]).collect()
    }

    #[test]
    fn rgb8_maps_channels_directly() {
        let data = vec![10, 20, 30, 40, 50, 60];
        let img = RawImage { width: 2, height: 1, encoding: "rgb8", is_bigendian: false, step: 6, data: &data };
        let c = convert_raw(&img, &ImageSettings::default()).unwrap();
        assert_eq!(head(&c, 2), vec![[10, 20, 30, 255], [40, 50, 60, 255]]);
        assert!(c.depth.is_none());
    }

    #[test]
    fn bgr8_swaps_red_and_blue() {
        let data = vec![10, 20, 30];
        let img = RawImage { width: 1, height: 1, encoding: "bgr8", is_bigendian: false, step: 3, data: &data };
        let c = convert_raw(&img, &ImageSettings::default()).unwrap();
        assert_eq!(head(&c, 1), vec![[30, 20, 10, 255]]);
    }

    #[test]
    fn rgba8_and_bgra8_keep_alpha() {
        // egui stores premultiplied Color32, so compare against from_rgba_unmultiplied of the expected RGBA.
        let data = vec![10, 20, 30, 128];
        let rgba = RawImage { width: 1, height: 1, encoding: "rgba8", is_bigendian: false, step: 4, data: &data };
        assert_eq!(
            convert_raw(&rgba, &ImageSettings::default()).unwrap().image.pixels[0],
            Color32::from_rgba_unmultiplied(10, 20, 30, 128)
        );
        let bgra = RawImage { width: 1, height: 1, encoding: "bgra8", is_bigendian: false, step: 4, data: &data };
        assert_eq!(
            convert_raw(&bgra, &ImageSettings::default()).unwrap().image.pixels[0],
            Color32::from_rgba_unmultiplied(30, 20, 10, 128)
        );
    }

    #[test]
    fn mono8_replicates_luminance() {
        let data = vec![7, 200];
        let img = RawImage { width: 2, height: 1, encoding: "mono8", is_bigendian: false, step: 2, data: &data };
        assert_eq!(head(&convert_raw(&img, &ImageSettings::default()).unwrap(), 2), vec![[7, 7, 7, 255], [200, 200, 200, 255]]);
    }

    #[test]
    fn opencv_aliases_map_to_mono8_and_bgr8() {
        let mono = vec![42];
        let img = RawImage { width: 1, height: 1, encoding: "8UC1", is_bigendian: false, step: 1, data: &mono };
        assert_eq!(head(&convert_raw(&img, &ImageSettings::default()).unwrap(), 1), vec![[42, 42, 42, 255]]);
        let bgr = vec![10, 20, 30];
        let img = RawImage { width: 1, height: 1, encoding: "8UC3", is_bigendian: false, step: 3, data: &bgr };
        assert_eq!(head(&convert_raw(&img, &ImageSettings::default()).unwrap(), 1), vec![[30, 20, 10, 255]]);
    }

    #[test]
    fn step_padding_does_not_shift_rows() {
        // width 1, rgb8 (3 bytes/px) but step 4 (1 padding byte per row).
        let data = vec![1, 2, 3, 99, 4, 5, 6, 99];
        let img = RawImage { width: 1, height: 2, encoding: "rgb8", is_bigendian: false, step: 4, data: &data };
        let c = convert_raw(&img, &ImageSettings::default()).unwrap();
        assert_eq!(head(&c, 2), vec![[1, 2, 3, 255], [4, 5, 6, 255]]);
    }

    #[test]
    fn short_data_is_invalid() {
        let data = vec![1, 2];
        let img = RawImage { width: 1, height: 1, encoding: "rgb8", is_bigendian: false, step: 3, data: &data };
        assert!(matches!(convert_raw(&img, &ImageSettings::default()), Err(ConvertError::InvalidImage(_))));
    }

    #[test]
    fn mono16_honors_big_endian_and_normalizes_full_range() {
        // Two pixels: 0x0100 (256) and 0x0200 (512) in big-endian; min->0, max->255.
        let data = vec![0x01, 0x00, 0x02, 0x00];
        let img = RawImage { width: 2, height: 1, encoding: "mono16", is_bigendian: true, step: 4, data: &data };
        let c = convert_raw(&img, &ImageSettings::default()).unwrap();
        assert_eq!(head(&c, 2), vec![[0, 0, 0, 255], [255, 255, 255, 255]]);
    }

    #[test]
    fn depth_u16_excludes_zero_and_auto_ranges() {
        // Values: 0 (invalid), 100, 300 (little-endian). Auto range = (100, 300).
        let data = vec![0, 0, 100, 0, 44, 1];
        let img = RawImage { width: 3, height: 1, encoding: "16UC1", is_bigendian: false, step: 6, data: &data };
        let c = convert_raw(&img, &settings(true, 0.0, 1.0, DepthColormap::Grayscale)).unwrap();
        let px = head(&c, 3);
        assert_eq!(px[0], [0, 0, 0, 0]); // invalid -> transparent
        assert_eq!(px[1], [0, 0, 0, 255]); // min -> black
        assert_eq!(px[2], [255, 255, 255, 255]); // max -> white
        assert_eq!(c.depth.unwrap(), DepthInfo { range: (100.0, 300.0), unit: "mm" });
    }

    #[test]
    fn depth_f32_excludes_nan_inf_zero_and_honors_manual_range() {
        let mut data = Vec::new();
        for v in [f32::NAN, 0.0f32, 1.0f32, 2.0f32] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let img = RawImage { width: 4, height: 1, encoding: "32FC1", is_bigendian: false, step: 16, data: &data };
        // Manual range 0..2 m, grayscale.
        let c = convert_raw(&img, &settings(false, 0.0, 2.0, DepthColormap::Grayscale)).unwrap();
        let px = head(&c, 4);
        assert_eq!(px[0], [0, 0, 0, 0]); // NaN invalid
        assert_eq!(px[1], [0, 0, 0, 0]); // 0.0 invalid
        assert_eq!(px[2], [128, 128, 128, 255]); // 1.0 -> midpoint
        assert_eq!(px[3], [255, 255, 255, 255]); // 2.0 -> max
        assert_eq!(c.depth.unwrap(), DepthInfo { range: (0.0, 2.0), unit: "m" });
    }

    #[test]
    fn unsupported_encoding_reports_name() {
        let data = vec![0; 8];
        let img = RawImage { width: 2, height: 1, encoding: "yuv422", is_bigendian: false, step: 4, data: &data };
        match convert_raw(&img, &ImageSettings::default()) {
            Err(ConvertError::UnsupportedEncoding(name)) => assert_eq!(name, "yuv422"),
            _ => panic!("expected UnsupportedEncoding"),
        }
    }
}
