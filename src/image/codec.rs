//! CompressedImage decode: jpeg (zune-jpeg) / png (png crate) -> egui `ColorImage`.
//!
//! Only color jpeg/png are in scope. `compressedDepth` (16UC1/32FC1 with an RVL/PNG payload behind a
//! 12-byte header) is explicitly reported as unsupported so we never mis-decode it (see .plait/01-plan.md §6).

use egui::ColorImage;

use super::convert::ConvertError;

/// Detected compression codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    Jpeg,
    Png,
}

/// Decode a `sensor_msgs/CompressedImage` payload using its `format` string, falling back to magic bytes.
pub fn decode_compressed(format: &str, data: &[u8]) -> Result<ColorImage, ConvertError> {
    let lower = format.to_ascii_lowercase();
    if lower.contains("compresseddepth") {
        // Distinct from a generic "unknown format" so the panel can say depth compression isn't handled.
        return Err(ConvertError::UnsupportedEncoding("compressedDepth".to_owned()));
    }
    let codec = if lower.contains("jpeg") || lower.contains("jpg") {
        Codec::Jpeg
    } else if lower.contains("png") {
        Codec::Png
    } else {
        detect_magic(data)?
    };
    match codec {
        Codec::Jpeg => decode_jpeg(data),
        Codec::Png => decode_png(data),
    }
}

/// Fall back to the file signature when `format` is empty or unrecognized.
fn detect_magic(data: &[u8]) -> Result<Codec, ConvertError> {
    if data.starts_with(&[0xFF, 0xD8]) {
        Ok(Codec::Jpeg)
    } else if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        Ok(Codec::Png)
    } else {
        Err(ConvertError::UnsupportedEncoding("unknown compressed format".to_owned()))
    }
}

fn decode_jpeg(data: &[u8]) -> Result<ColorImage, ConvertError> {
    // zune-jpeg reads through its own cursor type (implements the reader trait).
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(data));
    let pixels = decoder
        .decode()
        .map_err(|e| ConvertError::InvalidImage(format!("jpeg decode failed: {e}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| ConvertError::InvalidImage("jpeg missing header info".to_owned()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    if w == 0 || h == 0 {
        return Err(ConvertError::InvalidImage("jpeg zero-sized image".to_owned()));
    }
    // zune-jpeg emits RGB by default; grayscale sources come back as 1 component, CMYK-ish as 4.
    let channels = pixels.len() / (w * h);
    let rgba = match channels {
        3 => to_rgba(&pixels, 3, |p| [p[0], p[1], p[2], 255]),
        1 => to_rgba(&pixels, 1, |p| [p[0], p[0], p[0], 255]),
        4 => to_rgba(&pixels, 4, |p| [p[0], p[1], p[2], p[3]]),
        n => return Err(ConvertError::InvalidImage(format!("jpeg unexpected {n} components"))),
    };
    Ok(ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

fn decode_png(data: &[u8]) -> Result<ColorImage, ConvertError> {
    // png::Decoder requires BufRead + Seek; wrap the byte slice in a Cursor.
    let mut decoder = png::Decoder::new(std::io::Cursor::new(data));
    // Expand palettes/low-bit grayscale to 8-bit and strip 16-bit down to 8-bit so output is one of the 4 8-bit types.
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder
        .read_info()
        .map_err(|e| ConvertError::InvalidImage(format!("png read failed: {e}")))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| ConvertError::InvalidImage("png output size unknown".to_owned()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| ConvertError::InvalidImage(format!("png decode failed: {e}")))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let valid = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Grayscale => to_rgba(valid, 1, |p| [p[0], p[0], p[0], 255]),
        png::ColorType::GrayscaleAlpha => to_rgba(valid, 2, |p| [p[0], p[0], p[0], p[1]]),
        png::ColorType::Rgb => to_rgba(valid, 3, |p| [p[0], p[1], p[2], 255]),
        png::ColorType::Rgba => to_rgba(valid, 4, |p| [p[0], p[1], p[2], p[3]]),
        other => return Err(ConvertError::InvalidImage(format!("png unexpected color type {other:?}"))),
    };
    Ok(ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

/// Expand tightly packed `channels`-byte pixels to RGBA8 via a per-pixel mapping.
fn to_rgba(pixels: &[u8], channels: usize, map: impl Fn(&[u8]) -> [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len() / channels * 4);
    for p in pixels.chunks_exact(channels) {
        out.extend_from_slice(&map(p));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal 1x1 opaque-red PNG (Rgb, 8-bit), produced by the `png` encoder.
    fn red_1x1_png() -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[255, 0, 0]).unwrap();
        }
        out
    }

    #[test]
    fn png_round_trips_a_single_pixel() {
        let img = decode_compressed("rgb8; png compressed rgb8", &red_1x1_png()).unwrap();
        assert_eq!(img.size, [1, 1]);
        let px = img.pixels[0];
        assert_eq!([px.r(), px.g(), px.b(), px.a()], [255, 0, 0, 255]);
    }

    #[test]
    fn png_is_detected_by_magic_when_format_is_empty() {
        let img = decode_compressed("", &red_1x1_png()).unwrap();
        assert_eq!(img.size, [1, 1]);
    }

    #[test]
    fn compressed_depth_is_reported_unsupported() {
        let err = decode_compressed("16UC1; compressedDepth png", &[0; 16]).unwrap_err();
        assert_eq!(err, ConvertError::UnsupportedEncoding("compressedDepth".to_owned()));
    }

    #[test]
    fn unknown_format_without_magic_is_unsupported() {
        let err = decode_compressed("", &[1, 2, 3, 4]).unwrap_err();
        assert!(matches!(err, ConvertError::UnsupportedEncoding(_)));
    }

    #[test]
    fn corrupt_jpeg_payload_is_invalid_not_a_panic() {
        let err = decode_compressed("rgb8; jpeg compressed bgr8", &[0xFF, 0xD8, 0x00, 0x01]).unwrap_err();
        assert!(matches!(err, ConvertError::InvalidImage(_)));
    }
}
