//! Dynamic ROS 1 message decode: no alignment padding, always little-endian, no encapsulation header (requirements §2.1-e).

use super::cdr::{DecodeError, DecodeErrorKind};
use super::msg_parser::{ArraySpec, FieldType, PrimitiveType, TypeRegistry};
use super::value::Value;

/// Decode one ROS 1 message body. `payload` is the bag record's data with nothing stripped.
pub fn decode_ros1_message(
    registry: &TypeRegistry,
    type_name: &str,
    payload: &[u8],
) -> Result<Value, DecodeError> {
    let mut cursor = Cursor {
        buf: payload,
        pos: 0,
    };
    decode_complex(registry, type_name, &mut cursor)
}

/// Read cursor. Deliberately not shared with `cdr.rs`: the absence of alignment is the whole difference, and abstracting over it invites breaking CDR.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

macro_rules! impl_read_num {
    ($fn_name:ident, $ty:ty) => {
        fn $fn_name(&mut self) -> Result<$ty, DecodeError> {
            const N: usize = std::mem::size_of::<$ty>();
            let bytes: [u8; N] = self.take(N)?.try_into().expect("take checked the length");
            Ok(<$ty>::from_le_bytes(bytes))
        }
    };
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.remaining() < n {
            return Err(DecodeError::at(
                DecodeErrorKind::Truncated {
                    needed: n,
                    remaining: self.remaining(),
                },
                self.pos,
            ));
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    impl_read_num!(read_u16, u16);
    impl_read_num!(read_i16, i16);
    impl_read_num!(read_u32, u32);
    impl_read_num!(read_i32, i32);
    impl_read_num!(read_u64, u64);
    impl_read_num!(read_i64, i64);
    impl_read_num!(read_f32, f32);
    impl_read_num!(read_f64, f64);
}

fn decode_complex(
    registry: &TypeRegistry,
    full_name: &str,
    cursor: &mut Cursor<'_>,
) -> Result<Value, DecodeError> {
    let def = registry.get(full_name).ok_or_else(|| {
        DecodeError::at(
            DecodeErrorKind::UnknownType(full_name.to_owned()),
            cursor.pos,
        )
    })?;
    let mut fields = Vec::with_capacity(def.fields.len());
    for field in &def.fields {
        let value = match field.array {
            ArraySpec::Scalar => {
                decode_scalar(registry, &field.ty, cursor).map_err(|e| e.prepend(&field.name))?
            }
            ArraySpec::Fixed(n) => decode_elements(registry, &field.ty, n, cursor, &field.name)?,
            ArraySpec::Sequence => {
                let count = cursor.read_u32().map_err(|e| e.prepend(&field.name))? as usize;
                // Every element takes at least one byte, so a count above the remaining size is corrupt; reject before allocating.
                if count > cursor.remaining() {
                    return Err(DecodeError::at(
                        DecodeErrorKind::LengthOverrun {
                            length: count,
                            remaining: cursor.remaining(),
                        },
                        cursor.pos,
                    )
                    .prepend(&field.name));
                }
                decode_elements(registry, &field.ty, count, cursor, &field.name)?
            }
        };
        fields.push((field.name.clone(), value));
    }
    Ok(Value::Struct(fields))
}

/// Arrays of 1-byte types are bulk-read into Bytes, same as the CDR path (Image/OccupancyGrid memory).
fn is_byte_like(ty: &FieldType) -> bool {
    matches!(
        ty,
        FieldType::Primitive(
            PrimitiveType::UInt8 | PrimitiveType::Byte | PrimitiveType::Char | PrimitiveType::Int8
        )
    )
}

fn decode_elements(
    registry: &TypeRegistry,
    ty: &FieldType,
    count: usize,
    cursor: &mut Cursor<'_>,
    field_name: &str,
) -> Result<Value, DecodeError> {
    if is_byte_like(ty) {
        let bytes = cursor.take(count).map_err(|e| e.prepend(field_name))?;
        return Ok(Value::Bytes(bytes.to_vec()));
    }
    let mut items = Vec::with_capacity(count);
    for i in 0..count {
        let item = decode_scalar(registry, ty, cursor)
            .map_err(|e| e.prepend(&format!("{field_name}[{i}]")))?;
        items.push(item);
    }
    Ok(Value::Array(items))
}

fn decode_scalar(
    registry: &TypeRegistry,
    ty: &FieldType,
    cursor: &mut Cursor<'_>,
) -> Result<Value, DecodeError> {
    match ty {
        FieldType::Primitive(prim) => decode_primitive(*prim, cursor),
        FieldType::Complex(full_name) => decode_complex(registry, full_name, cursor),
    }
}

fn decode_primitive(prim: PrimitiveType, cursor: &mut Cursor<'_>) -> Result<Value, DecodeError> {
    match prim {
        PrimitiveType::Bool => Ok(Value::Bool(cursor.read_u8()? != 0)),
        // ROS 1 `byte` is int8 and `char` is uint8, the reverse of ROS 2; both are read as one byte and shown unsigned (requirements d6).
        PrimitiveType::Byte | PrimitiveType::Char | PrimitiveType::UInt8 => {
            Ok(Value::U8(cursor.read_u8()?))
        }
        PrimitiveType::Int8 => Ok(Value::I8(cursor.read_u8()? as i8)),
        PrimitiveType::Int16 => Ok(Value::I16(cursor.read_i16()?)),
        PrimitiveType::UInt16 => Ok(Value::U16(cursor.read_u16()?)),
        PrimitiveType::Int32 => Ok(Value::I32(cursor.read_i32()?)),
        PrimitiveType::UInt32 => Ok(Value::U32(cursor.read_u32()?)),
        PrimitiveType::Int64 => Ok(Value::I64(cursor.read_i64()?)),
        PrimitiveType::UInt64 => Ok(Value::U64(cursor.read_u64()?)),
        PrimitiveType::Float32 => Ok(Value::F32(cursor.read_f32()?)),
        PrimitiveType::Float64 => Ok(Value::F64(cursor.read_f64()?)),
        PrimitiveType::String => decode_string(cursor),
    }
}

/// ROS 1 string = u32 length + UTF-8 bytes, with no trailing NUL (unlike CDR).
fn decode_string(cursor: &mut Cursor<'_>) -> Result<Value, DecodeError> {
    let len = cursor.read_u32()? as usize;
    if len > cursor.remaining() {
        return Err(DecodeError::at(
            DecodeErrorKind::LengthOverrun {
                length: len,
                remaining: cursor.remaining(),
            },
            cursor.pos,
        ));
    }
    let body_offset = cursor.pos;
    let bytes = cursor.take(len)?;
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(Value::String(s.to_owned())),
        Err(_) => Err(DecodeError::at(DecodeErrorKind::InvalidUtf8, body_offset)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bag::msgdef::{registry_for, rewrite_ros1_body};
    use crate::bag::reader::Connection;

    /// Registry from ROS 1 bodies, keyed by `pkg/Type` names as a bag would spell them.
    fn registry(defs: &[(&str, &str)]) -> TypeRegistry {
        let mut reg = TypeRegistry::new();
        for (name, body) in defs {
            let full = crate::bag::naming::ros1_type_to_ros2(name);
            reg.insert_msg(&full, &rewrite_ros1_body(body)).unwrap();
        }
        reg.insert_msg("builtin_interfaces/msg/Time", "int32 sec\nuint32 nanosec")
            .unwrap();
        reg.insert_msg(
            "builtin_interfaces/msg/Duration",
            "int32 sec\nuint32 nanosec",
        )
        .unwrap();
        reg.validate().unwrap();
        reg
    }

    fn s(pairs: &[(&str, Value)]) -> Value {
        Value::Struct(
            pairs
                .iter()
                .map(|(n, v)| ((*n).to_owned(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn r1_primitives_are_packed_with_no_padding() {
        let reg = registry(&[(
            "t/Prims",
            "bool b\nbyte y\nchar c\nint8 i8v\nuint8 u8v\nint16 i16v\nuint16 u16v\nint32 i32v\nuint32 u32v\nint64 i64v\nuint64 u64v\nfloat32 f32v\nfloat64 f64v",
        )]);
        let body: &[u8] = &[
            0x01, 0xAB, 0x41, 0xFE, 0xC8, // bool byte char int8 uint8
            0xFD, 0xFF, // int16 = -3, immediately after a 5-byte run (CDR would pad here)
            0xEF, 0xBE, // uint16
            0xFC, 0xFF, 0xFF, 0xFF, // int32 = -4
            0xEF, 0xBE, 0xAD, 0xDE, // uint32
            0xFB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // int64 = -5
            0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, // uint64
            0x00, 0x00, 0xC0, 0x3F, // float32 = 1.5
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xC0, // float64 = -2.5
        ];
        let v = decode_ros1_message(&reg, "t/msg/Prims", body).unwrap();
        assert_eq!(
            v,
            s(&[
                ("b", Value::Bool(true)),
                ("y", Value::U8(0xAB)),
                ("c", Value::U8(0x41)),
                ("i8v", Value::I8(-2)),
                ("u8v", Value::U8(200)),
                ("i16v", Value::I16(-3)),
                ("u16v", Value::U16(0xBEEF)),
                ("i32v", Value::I32(-4)),
                ("u32v", Value::U32(0xDEAD_BEEF)),
                ("i64v", Value::I64(-5)),
                ("u64v", Value::U64(0x0123_4567_89AB_CDEF)),
                ("f32v", Value::F32(1.5)),
                ("f64v", Value::F64(-2.5)),
            ])
        );
    }

    #[test]
    fn r2_strings_have_no_trailing_nul() {
        let reg = registry(&[("t/Str", "string a\nstring b\nint32 tail")]);
        let mut body = Vec::new();
        body.extend_from_slice(&3u32.to_le_bytes());
        body.extend_from_slice(b"map");
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&7i32.to_le_bytes());
        let v = decode_ros1_message(&reg, "t/msg/Str", &body).unwrap();
        assert_eq!(
            v,
            s(&[
                ("a", Value::String("map".to_owned())),
                ("b", Value::String(String::new())),
                // The int32 sits right after the strings with no alignment padding.
                ("tail", Value::I32(7)),
            ])
        );
    }

    #[test]
    fn r2_multibyte_utf8_and_invalid_utf8() {
        let reg = registry(&[("t/Str", "string a")]);
        let text = "ロボット";
        let mut body = (text.len() as u32).to_le_bytes().to_vec();
        body.extend_from_slice(text.as_bytes());
        assert_eq!(
            decode_ros1_message(&reg, "t/msg/Str", &body).unwrap(),
            s(&[("a", Value::String(text.to_owned()))])
        );
        let mut bad = 2u32.to_le_bytes().to_vec();
        bad.extend_from_slice(&[0xFF, 0xFE]);
        let e = decode_ros1_message(&reg, "t/msg/Str", &bad).unwrap_err();
        assert_eq!(e.kind, DecodeErrorKind::InvalidUtf8);
        assert_eq!(e.path, "a");
    }

    #[test]
    fn r3_time_and_duration_decode_as_sec_nanosec_structs() {
        let reg = registry(&[("t/Stamped", "time stamp\nduration timeout")]);
        let mut body = 1_700_000_000u32.to_le_bytes().to_vec();
        body.extend_from_slice(&250_000_000u32.to_le_bytes());
        body.extend_from_slice(&5u32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        let v = decode_ros1_message(&reg, "t/msg/Stamped", &body).unwrap();
        // The ROS 2 field names are what make the TF path and every renderer work unchanged.
        assert_eq!(
            v,
            s(&[
                (
                    "stamp",
                    s(&[
                        ("sec", Value::I32(1_700_000_000)),
                        ("nanosec", Value::U32(250_000_000)),
                    ])
                ),
                (
                    "timeout",
                    s(&[("sec", Value::I32(5)), ("nanosec", Value::U32(0))])
                ),
            ])
        );
    }

    #[test]
    fn r4_fixed_and_variable_arrays() {
        let reg = registry(&[("t/Arrays", "float64[3] fixed\nint32[] seq")]);
        let mut body = Vec::new();
        for x in [1.0f64, 2.0, 3.0] {
            body.extend_from_slice(&x.to_le_bytes());
        }
        body.extend_from_slice(&2u32.to_le_bytes());
        body.extend_from_slice(&(-1i32).to_le_bytes());
        body.extend_from_slice(&9i32.to_le_bytes());
        let v = decode_ros1_message(&reg, "t/msg/Arrays", &body).unwrap();
        assert_eq!(
            v,
            s(&[
                (
                    "fixed",
                    Value::Array(vec![Value::F64(1.0), Value::F64(2.0), Value::F64(3.0)])
                ),
                ("seq", Value::Array(vec![Value::I32(-1), Value::I32(9)])),
            ])
        );
    }

    #[test]
    fn r4_one_byte_arrays_collapse_into_bytes() {
        let reg = registry(&[("t/B", "uint8[] a\nint8[2] b\nchar[] c\nbyte[] d")]);
        let mut body = 3u32.to_le_bytes().to_vec();
        body.extend_from_slice(&[1, 2, 3]);
        body.extend_from_slice(&[0xFF, 0x01]);
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&[0x41]);
        body.extend_from_slice(&0u32.to_le_bytes());
        let v = decode_ros1_message(&reg, "t/msg/B", &body).unwrap();
        assert_eq!(
            v,
            s(&[
                ("a", Value::Bytes(vec![1, 2, 3])),
                ("b", Value::Bytes(vec![0xFF, 0x01])),
                ("c", Value::Bytes(vec![0x41])),
                ("d", Value::Bytes(vec![])),
            ])
        );
    }

    #[test]
    fn r5_nested_types_and_header_seq() {
        let reg = registry(&[
            ("t/Msg", "Header header\nPoint p"),
            ("std_msgs/Header", "uint32 seq\ntime stamp\nstring frame_id"),
            ("t/Point", "float64 x\nfloat64 y"),
        ]);
        let mut body = 42u32.to_le_bytes().to_vec();
        body.extend_from_slice(&3u32.to_le_bytes());
        body.extend_from_slice(&500u32.to_le_bytes());
        body.extend_from_slice(&3u32.to_le_bytes());
        body.extend_from_slice(b"map");
        body.extend_from_slice(&1.5f64.to_le_bytes());
        body.extend_from_slice(&(-0.5f64).to_le_bytes());
        let v = decode_ros1_message(&reg, "t/msg/Msg", &body).unwrap();
        let header = v.get("header").unwrap();
        // `seq` has no ROS 2 counterpart; it passes through as an extra field that renderers ignore (requirements d4).
        assert_eq!(header.get("seq"), Some(&Value::U32(42)));
        assert_eq!(
            header.get("frame_id"),
            Some(&Value::String("map".to_owned()))
        );
        assert_eq!(v.get("p").unwrap().get("y"), Some(&Value::F64(-0.5)));
    }

    #[test]
    fn r6_truncation_reports_the_failing_field_path() {
        let reg = registry(&[
            ("t/Msg", "Point[] pts"),
            ("t/Point", "float64 x\nfloat64 y"),
        ]);
        let mut body = 2u32.to_le_bytes().to_vec();
        body.extend_from_slice(&1.0f64.to_le_bytes());
        body.extend_from_slice(&2.0f64.to_le_bytes());
        body.extend_from_slice(&3.0f64.to_le_bytes());
        let e = decode_ros1_message(&reg, "t/msg/Msg", &body).unwrap_err();
        assert_eq!(e.path, "pts[1].y");
        assert!(matches!(e.kind, DecodeErrorKind::Truncated { .. }));
    }

    #[test]
    fn r7_oversized_length_prefixes_never_allocate() {
        let reg = registry(&[("t/Seq", "int32[] v")]);
        let body = 0xFFFF_FFFFu32.to_le_bytes();
        let e = decode_ros1_message(&reg, "t/msg/Seq", &body).unwrap_err();
        assert!(matches!(e.kind, DecodeErrorKind::LengthOverrun { .. }));
        assert_eq!(e.path, "v");

        let reg = registry(&[("t/Str", "string s")]);
        let e = decode_ros1_message(&reg, "t/msg/Str", &body).unwrap_err();
        assert!(matches!(e.kind, DecodeErrorKind::LengthOverrun { .. }));
    }

    #[test]
    fn r8_unknown_type_is_reported_by_name() {
        let reg = registry(&[("t/Msg", "int32 x")]);
        let e = decode_ros1_message(&reg, "t/msg/Absent", &[]).unwrap_err();
        assert_eq!(
            e.kind,
            DecodeErrorKind::UnknownType("t/msg/Absent".to_owned())
        );
    }

    /// A LaserScan sample built the way a bag stores one, decoded through the connection's own definition.
    #[test]
    fn r9_laser_scan_from_a_bag_style_definition() {
        let definition = [
            "Header header",
            "float32 angle_min",
            "float32 angle_max",
            "float32 angle_increment",
            "float32 time_increment",
            "float32 scan_time",
            "float32 range_min",
            "float32 range_max",
            "float32[] ranges",
            "float32[] intensities",
            "================================================================================",
            "MSG: std_msgs/Header",
            "uint32 seq",
            "time stamp",
            "string frame_id",
        ]
        .join("\n");
        let conn = Connection {
            id: 0,
            topic_raw: "scan".to_owned(),
            type_raw: "sensor_msgs/LaserScan".to_owned(),
            type_hash: "90c7ef2dc6895d81024acba2ac42f369".to_owned(),
            definition,
            message_encoding: "ros1".to_owned(),
            definition_encoding: "ros1msg".to_owned(),
        };
        let (root, reg) = registry_for(&conn).unwrap();
        assert_eq!(root, "sensor_msgs/msg/LaserScan");
        let mut body = 11u32.to_le_bytes().to_vec();
        body.extend_from_slice(&1_700_000_001u32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&5u32.to_le_bytes());
        body.extend_from_slice(b"laser");
        for x in [
            -std::f32::consts::PI,
            std::f32::consts::PI,
            0.01,
            0.0001,
            0.1,
            0.05,
            30.0,
        ] {
            body.extend_from_slice(&x.to_le_bytes());
        }
        body.extend_from_slice(&2u32.to_le_bytes());
        body.extend_from_slice(&1.25f32.to_le_bytes());
        body.extend_from_slice(&2.5f32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        let v = decode_ros1_message(&reg, &root, &body).unwrap();
        assert_eq!(
            v.get("header").unwrap().get("frame_id"),
            Some(&Value::String("laser".to_owned()))
        );
        assert_eq!(v.get("range_max"), Some(&Value::F32(30.0)));
        assert_eq!(
            v.get("ranges"),
            Some(&Value::Array(vec![Value::F32(1.25), Value::F32(2.5)]))
        );
        assert_eq!(v.get("intensities"), Some(&Value::Array(vec![])));
        // Dropping one byte must fail: that is what proves the decoder consumes exactly the record, the invariant the Python cross-check established on real bags.
        assert!(decode_ros1_message(&reg, &root, &body[..body.len() - 1]).is_err());
    }
}
