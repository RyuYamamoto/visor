//! Intermediate representation of decoded values (JSON-like value type).

/// A decoded CDR value. Structs keep field order via Vec so display order matches the .msg definition.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    /// Contiguous bytes for uint8[]/byte[]/char[]; per-element Values would multiply memory for Image/PointCloud2 data.
    Bytes(Vec<u8>),
    /// Any other array or sequence.
    Array(Vec<Value>),
    Struct(Vec<(String, Value)>),
}

impl Value {
    /// Look up a struct field by name (None for non-Struct values).
    pub fn get(&self, field: &str) -> Option<&Value> {
        match self {
            Value::Struct(fields) => fields.iter().find(|(n, _)| n == field).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// Truncation limits for `format_compact` (array elements, string chars, Bytes count).
#[derive(Debug, Clone)]
pub struct FormatLimits {
    pub array_preview: usize,
    pub string_preview: usize,
    pub bytes_preview: usize,
}

impl Default for FormatLimits {
    fn default() -> Self {
        Self {
            array_preview: 8,
            string_preview: 64,
            bytes_preview: 8,
        }
    }
}

/// Single-line JSON-like formatting, shared by probe and the initial raw_view.
pub fn format_compact(value: &Value, limits: &FormatLimits) -> String {
    let mut out = String::new();
    write_compact(value, limits, &mut out);
    out
}

fn write_compact(value: &Value, limits: &FormatLimits, out: &mut String) {
    match value {
        Value::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
        Value::U8(v) => out.push_str(&v.to_string()),
        Value::I8(v) => out.push_str(&v.to_string()),
        Value::U16(v) => out.push_str(&v.to_string()),
        Value::I16(v) => out.push_str(&v.to_string()),
        Value::U32(v) => out.push_str(&v.to_string()),
        Value::I32(v) => out.push_str(&v.to_string()),
        Value::U64(v) => out.push_str(&v.to_string()),
        Value::I64(v) => out.push_str(&v.to_string()),
        Value::F32(v) => out.push_str(&v.to_string()),
        Value::F64(v) => out.push_str(&v.to_string()),
        Value::String(s) => {
            out.push('"');
            // Truncate on char boundaries, not byte boundaries, to avoid invalid UTF-8.
            let shown: String = s.chars().take(limits.string_preview).collect();
            out.push_str(&shown);
            if s.chars().count() > limits.string_preview {
                out.push('…');
            }
            out.push('"');
        }
        Value::Bytes(bytes) => {
            out.push_str(&format!("bytes[{}", bytes.len()));
            if !bytes.is_empty() {
                out.push_str(": ");
                for (i, b) in bytes.iter().take(limits.bytes_preview).enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(&format!("{b:02x}"));
                }
                if bytes.len() > limits.bytes_preview {
                    out.push_str(" …");
                }
            }
            out.push(']');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().take(limits.array_preview).enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_compact(item, limits, out);
            }
            if items.len() > limits.array_preview {
                out.push_str(&format!(", …(+{})", items.len() - limits.array_preview));
            }
            out.push(']');
        }
        Value::Struct(fields) => {
            out.push('{');
            for (i, (name, v)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(name);
                out.push_str(": ");
                write_compact(v, limits, out);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_finds_struct_field() {
        let v = Value::Struct(vec![
            ("sec".to_owned(), Value::I32(7)),
            ("nanosec".to_owned(), Value::U32(500)),
        ]);
        assert_eq!(v.get("nanosec"), Some(&Value::U32(500)));
        assert_eq!(v.get("missing"), None);
        assert_eq!(Value::I32(1).get("sec"), None);
    }

    #[test]
    fn format_scalars_and_struct() {
        let v = Value::Struct(vec![
            ("ok".to_owned(), Value::Bool(true)),
            ("x".to_owned(), Value::F64(-2.5)),
            ("name".to_owned(), Value::String("map".to_owned())),
        ]);
        assert_eq!(
            format_compact(&v, &FormatLimits::default()),
            r#"{ok: true, x: -2.5, name: "map"}"#
        );
    }

    #[test]
    fn format_array_within_limit() {
        let v = Value::Array(vec![Value::I32(1), Value::I32(2)]);
        assert_eq!(format_compact(&v, &FormatLimits::default()), "[1, 2]");
    }

    #[test]
    fn format_array_truncated() {
        let v = Value::Array((0..12).map(Value::I32).collect());
        assert_eq!(
            format_compact(&v, &FormatLimits::default()),
            "[0, 1, 2, 3, 4, 5, 6, 7, …(+4)]"
        );
    }

    #[test]
    fn format_bytes() {
        assert_eq!(
            format_compact(&Value::Bytes(vec![]), &FormatLimits::default()),
            "bytes[0]"
        );
        assert_eq!(
            format_compact(&Value::Bytes(vec![0xde, 0xad]), &FormatLimits::default()),
            "bytes[2: de ad]"
        );
        let long: Vec<u8> = (0..20).collect();
        assert_eq!(
            format_compact(&Value::Bytes(long), &FormatLimits::default()),
            "bytes[20: 00 01 02 03 04 05 06 07 …]"
        );
    }

    #[test]
    fn format_long_string_truncated_at_char_boundary() {
        let limits = FormatLimits {
            string_preview: 3,
            ..FormatLimits::default()
        };
        assert_eq!(
            format_compact(&Value::String("こんにちは".to_owned()), &limits),
            r#""こんに…""#
        );
    }

    #[test]
    fn format_nested_struct_single_line() {
        let v = Value::Struct(vec![(
            "stamp".to_owned(),
            Value::Struct(vec![("sec".to_owned(), Value::I32(1))]),
        )]);
        assert_eq!(
            format_compact(&v, &FormatLimits::default()),
            "{stamp: {sec: 1}}"
        );
    }
}
