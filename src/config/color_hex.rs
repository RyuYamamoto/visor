//! serde conversion between `Color32` and `"#RRGGBBAA"` (round-trips premultiplied sRGBA 8-bit losslessly, even translucent).

use egui::Color32;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<S: Serializer>(color: &Color32, serializer: S) -> Result<S::Ok, S::Error> {
    let [r, g, b, a] = color.to_array();
    format!("#{r:02X}{g:02X}{b:02X}{a:02X}").serialize(serializer)
}

pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Color32, D::Error> {
    let text = String::deserialize(deserializer)?;
    parse(&text).map_err(serde::de::Error::custom)
}

/// Accepts both `"#RRGGBB"` (alpha defaulted to FF) and `"#RRGGBBAA"` (the `#` is optional).
fn parse(text: &str) -> Result<Color32, String> {
    let hex = text.strip_prefix('#').unwrap_or(text);
    let byte = |i: usize| {
        u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| format!("invalid color hex `{text}`"))
    };
    match hex.len() {
        6 => Ok(Color32::from_rgba_premultiplied(
            byte(0)?,
            byte(2)?,
            byte(4)?,
            255,
        )),
        8 => Ok(Color32::from_rgba_premultiplied(
            byte(0)?,
            byte(2)?,
            byte(4)?,
            byte(6)?,
        )),
        _ => Err(format!("color hex must be 6 or 8 digits, got `{text}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal wrapper to unit-test the `with` module (TOML top level must be a table).
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Wrap {
        #[serde(with = "super")]
        color: Color32,
    }

    fn roundtrip(color: Color32) -> Color32 {
        let text = toml::to_string(&Wrap { color }).unwrap();
        let back: Wrap = toml::from_str(&text).unwrap();
        back.color
    }

    #[test]
    fn opaque_color_roundtrips() {
        let color = Color32::from_rgba_premultiplied(0x12, 0x34, 0x56, 0xFF);
        assert_eq!(roundtrip(color), color);
    }

    #[test]
    fn translucent_color_roundtrips() {
        let color = Color32::from_rgba_premultiplied(10, 20, 30, 128);
        assert_eq!(roundtrip(color), color);
    }

    #[test]
    fn six_digit_hex_defaults_alpha_to_opaque() {
        assert_eq!(
            parse("#0A141E").unwrap(),
            Color32::from_rgba_premultiplied(10, 20, 30, 255)
        );
    }

    #[test]
    fn accepts_lowercase_and_missing_hash() {
        assert_eq!(parse("ff8800ff").unwrap(), parse("#FF8800FF").unwrap());
    }

    #[test]
    fn rejects_malformed_hex() {
        assert!(parse("#12").is_err());
        assert!(parse("#GGGGGG").is_err());
        assert!(parse("").is_err());
    }
}
