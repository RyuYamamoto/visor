//! Dynamic CDR decode (XCDR1, with 4-byte encapsulation header; spec: .plait/01-plan.md §3).

use std::fmt;

use super::msg_parser::{ArraySpec, FieldType, PrimitiveType, TypeRegistry};
use super::value::Value;

/// Kinds of decode failure.
#[derive(Debug, PartialEq, Eq)]
pub enum DecodeErrorKind {
    /// Fewer bytes remain than needed.
    Truncated { needed: usize, remaining: usize },
    /// Encapsulation ID other than CDR_BE(00 00) / CDR_LE(00 01).
    UnsupportedEncapsulation([u8; 2]),
    /// Type name not present in the registry.
    UnknownType(String),
    /// string field is not valid UTF-8.
    InvalidUtf8,
    /// Sequence length prefix exceeds remaining bytes (guards against OOM on corrupt input).
    LengthOverrun { length: usize, remaining: usize },
}

/// Decode error (offset is a byte position measured from just after the encapsulation header).
#[derive(Debug, PartialEq, Eq)]
pub struct DecodeError {
    pub kind: DecodeErrorKind,
    /// Path of the failing field (e.g. `transforms[0].header.stamp.sec`).
    pub path: String,
    pub offset: usize,
}

impl DecodeError {
    /// Shared with the ROS 1 decoder, which reports the same information from a different wire format.
    pub(crate) fn at(kind: DecodeErrorKind, offset: usize) -> Self {
        Self {
            kind,
            path: String::new(),
            offset,
        }
    }

    /// Prepend a segment to the field path while unwinding recursion (cost only on the error path).
    pub(crate) fn prepend(mut self, segment: &str) -> Self {
        self.path = if self.path.is_empty() {
            segment.to_owned()
        } else {
            format!("{segment}.{}", self.path)
        };
        self
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.path.is_empty() {
            write!(f, "at `{}` ", self.path)?;
        }
        write!(f, "(offset {}): ", self.offset)?;
        match &self.kind {
            DecodeErrorKind::Truncated { needed, remaining } => {
                write!(
                    f,
                    "truncated payload (needed {needed} bytes, {remaining} remaining)"
                )
            }
            DecodeErrorKind::UnsupportedEncapsulation(id) => {
                write!(
                    f,
                    "unsupported encapsulation id {:02x} {:02x}",
                    id[0], id[1]
                )
            }
            DecodeErrorKind::UnknownType(name) => write!(f, "unknown type `{name}`"),
            DecodeErrorKind::InvalidUtf8 => write!(f, "invalid UTF-8 in string"),
            DecodeErrorKind::LengthOverrun { length, remaining } => {
                write!(
                    f,
                    "sequence length {length} exceeds remaining {remaining} bytes"
                )
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Entry point. `payload` is the received bytes as-is, including the encapsulation header.
pub fn decode_message(
    registry: &TypeRegistry,
    type_name: &str,
    payload: &[u8],
) -> Result<Value, DecodeError> {
    if payload.len() < 4 {
        return Err(DecodeError::at(
            DecodeErrorKind::Truncated {
                needed: 4,
                remaining: payload.len(),
            },
            0,
        ));
    }
    let le = match [payload[0], payload[1]] {
        [0x00, 0x00] => false,
        [0x00, 0x01] => true,
        other => {
            return Err(DecodeError::at(
                DecodeErrorKind::UnsupportedEncapsulation(other),
                0,
            ));
        }
    };
    // Skip the 2-byte options field (payload[2..4]).
    let mut cursor = Cursor {
        buf: &payload[4..],
        pos: 0,
        le,
    };
    decode_complex(registry, type_name, &mut cursor)
}

/// Read cursor. `buf` starts just after the header, so `pos` is the distance from the alignment origin.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    le: bool,
}

macro_rules! impl_read_num {
    ($fn_name:ident, $ty:ty) => {
        fn $fn_name(&mut self) -> Result<$ty, DecodeError> {
            const N: usize = std::mem::size_of::<$ty>();
            self.align(N);
            let bytes: [u8; N] = self.take(N)?.try_into().expect("take checked the length");
            Ok(if self.le {
                <$ty>::from_le_bytes(bytes)
            } else {
                <$ty>::from_be_bytes(bytes)
            })
        }
    };
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// XCDR1 alignment: primitives align to their own size; padding bytes are not validated.
    fn align(&mut self, size: usize) {
        let rem = self.pos % size;
        if rem != 0 {
            self.pos += size - rem;
        }
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
                // Since every element is at least 1 byte, count can't exceed remaining; rejects corrupt lengths before allocating.
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

/// Arrays of 1-byte types are bulk-read into Bytes (int8 sign interpretation is the caller's concern).
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
        // fastcdr treats any non-zero as true, so values other than 0/1 are not errors.
        PrimitiveType::Bool => Ok(Value::Bool(cursor.read_u8()? != 0)),
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

/// string = u32 length (including NUL) + bytes. Strips the trailing NUL and strictly validates UTF-8.
fn decode_string(cursor: &mut Cursor<'_>) -> Result<Value, DecodeError> {
    let len = cursor.read_u32()? as usize;
    let body_offset = cursor.pos;
    let bytes = cursor.take(len)?;
    // Defensively accept length 0 as empty (spec requires at least the 1-byte NUL).
    let bytes = match bytes.split_last() {
        Some((0, rest)) => rest,
        _ => bytes,
    };
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(Value::String(s.to_owned())),
        Err(_) => Err(DecodeError::at(DecodeErrorKind::InvalidUtf8, body_offset)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Registry with the embedded types plus synthetic types for tests.
    fn test_registry() -> TypeRegistry {
        let mut reg = TypeRegistry::with_embedded().unwrap();
        for (name, text) in [
            (
                "test_msgs/msg/Prims",
                "bool b\nbyte y\nchar c\nint8 i8v\nuint8 u8v\nint16 i16v\nuint16 u16v\nint32 i32v\nuint32 u32v\nint64 i64v\nuint64 u64v\nfloat32 f32v\nfloat64 f64v",
            ),
            ("test_msgs/msg/AlignF64", "uint8 a\nfloat64 b"),
            ("test_msgs/msg/AlignI32", "uint8 a\nint32 b"),
            ("test_msgs/msg/AlignU16", "uint8 a\nuint16 b"),
            ("test_msgs/msg/FixedF64", "float64[3] m"),
            ("test_msgs/msg/SeqI32", "int32[] v"),
            ("test_msgs/msg/BytesSeq", "uint8[] d"),
            ("test_msgs/msg/BytesFixed", "uint8[4] d"),
            ("test_msgs/msg/ByteSeq", "byte[] d"),
            ("test_msgs/msg/CharSeq", "char[] d"),
            ("test_msgs/msg/Int8Seq", "int8[] d"),
            ("test_msgs/msg/Int8Fixed", "int8[3] d"),
            ("test_msgs/msg/PadFixed", "uint8 a\nfloat64[2] b"),
            ("test_msgs/msg/PadSeq", "uint8 a\nint32[] c"),
        ] {
            reg.insert_msg(name, text).unwrap();
        }
        reg
    }

    /// Prepend a CDR_LE header (00 01 00 00) to the body.
    fn le(body: &[u8]) -> Vec<u8> {
        [&[0x00, 0x01, 0x00, 0x00][..], body].concat()
    }

    /// Prepend a CDR_BE header (00 00 00 00) to the body.
    fn be(body: &[u8]) -> Vec<u8> {
        [&[0x00, 0x00, 0x00, 0x00][..], body].concat()
    }

    /// Shorthand for building a Value::Struct.
    fn s(pairs: &[(&str, Value)]) -> Value {
        Value::Struct(
            pairs
                .iter()
                .map(|(n, v)| ((*n).to_owned(), v.clone()))
                .collect(),
        )
    }

    fn expected_prims() -> Value {
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
    }

    #[test]
    fn c1_primitives_le() {
        let body: &[u8] = &[
            0x01, // bool = true
            0xAB, // byte = 0xAB
            0x41, // char = 'A'
            0xFE, // int8 = -2
            0xC8, // uint8 = 200
            0x00, // pad -> 2B boundary
            0xFD, 0xFF, // int16 = -3
            0xEF, 0xBE, // uint16 = 0xBEEF
            0x00, 0x00, // pad -> 4B boundary
            0xFC, 0xFF, 0xFF, 0xFF, // int32 = -4
            0xEF, 0xBE, 0xAD, 0xDE, // uint32 = 0xDEADBEEF
            0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0xFB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // int64 = -5
            0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, // uint64 = 0x0123456789ABCDEF
            0x00, 0x00, 0xC0, 0x3F, // float32 = 1.5
            0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xC0, // float64 = -2.5
        ];
        let v = decode_message(&test_registry(), "test_msgs/msg/Prims", &le(body)).unwrap();
        assert_eq!(v, expected_prims());
    }

    #[test]
    fn c1_primitives_be() {
        let body: &[u8] = &[
            0x01, // bool = true
            0xAB, // byte
            0x41, // char
            0xFE, // int8 = -2
            0xC8, // uint8 = 200
            0x00, // pad -> 2B boundary
            0xFF, 0xFD, // int16 = -3
            0xBE, 0xEF, // uint16
            0x00, 0x00, // pad -> 4B boundary
            0xFF, 0xFF, 0xFF, 0xFC, // int32 = -4
            0xDE, 0xAD, 0xBE, 0xEF, // uint32
            0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFB, // int64 = -5
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, // uint64
            0x3F, 0xC0, 0x00, 0x00, // float32 = 1.5
            0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0xC0, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // float64 = -2.5
        ];
        let v = decode_message(&test_registry(), "test_msgs/msg/Prims", &be(body)).unwrap();
        assert_eq!(v, expected_prims());
    }

    #[test]
    fn c2_string_variants() {
        let reg = test_registry();
        let ty = "std_msgs/msg/String";
        let cases: &[(&[u8], &str)] = &[
            // len=6 (incl NUL) + "hello\0"
            (
                &[0x06, 0x00, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o', 0x00],
                "hello",
            ),
            // Empty string: len=1 + NUL
            (&[0x01, 0x00, 0x00, 0x00, 0x00], ""),
            // Defensive acceptance: len=0 (no NUL)
            (&[0x00, 0x00, 0x00, 0x00], ""),
            // Non-ASCII UTF-8: "こんにちは" (15 bytes) + NUL = len 16
            (
                &[
                    0x10, 0x00, 0x00, 0x00, // len = 16
                    0xE3, 0x81, 0x93, 0xE3, 0x82, 0x93, 0xE3, 0x81, 0xAB, // ko n ni
                    0xE3, 0x81, 0xA1, 0xE3, 0x81, 0xAF, // chi ha
                    0x00, // NUL
                ],
                "こんにちは",
            ),
        ];
        for (body, expected) in cases {
            let v = decode_message(&reg, ty, &le(body)).unwrap();
            assert_eq!(v, s(&[("data", Value::String((*expected).to_owned()))]));
        }
        // BE: len=3 + "hi\0"
        let v = decode_message(&reg, ty, &be(&[0x00, 0x00, 0x00, 0x03, b'h', b'i', 0x00])).unwrap();
        assert_eq!(v, s(&[("data", Value::String("hi".to_owned()))]));
    }

    #[test]
    fn c2_string_invalid_utf8() {
        let e = decode_message(
            &test_registry(),
            "std_msgs/msg/String",
            &le(&[0x02, 0x00, 0x00, 0x00, 0xFF, 0x00]),
        )
        .unwrap_err();
        assert_eq!(e.kind, DecodeErrorKind::InvalidUtf8);
        assert_eq!(e.path, "data");
        assert_eq!(e.offset, 4);
    }

    #[test]
    fn c3_alignment_after_u8() {
        let reg = test_registry();
        // f64 after u8: pad 7 when origin is just after the header (a header-inclusive origin would give pad 3, catching the bug).
        let body_le: &[u8] = &[
            0x2A, // a = 42
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // b = 1.0
        ];
        let expected = s(&[("a", Value::U8(42)), ("b", Value::F64(1.0))]);
        assert_eq!(
            decode_message(&reg, "test_msgs/msg/AlignF64", &le(body_le)).unwrap(),
            expected
        );
        let body_be: &[u8] = &[
            0x2A, // a = 42
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // b = 1.0
        ];
        assert_eq!(
            decode_message(&reg, "test_msgs/msg/AlignF64", &be(body_be)).unwrap(),
            expected
        );

        // i32 after u8: pad 3
        let v = decode_message(
            &reg,
            "test_msgs/msg/AlignI32",
            &le(&[0x2A, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00]),
        )
        .unwrap();
        assert_eq!(v, s(&[("a", Value::U8(42)), ("b", Value::I32(7))]));

        // u16 after u8: pad 1
        let v = decode_message(
            &reg,
            "test_msgs/msg/AlignU16",
            &be(&[0x2A, 0x00, 0x12, 0x34]),
        )
        .unwrap();
        assert_eq!(v, s(&[("a", Value::U8(42)), ("b", Value::U16(0x1234))]));
    }

    #[test]
    fn c4_fixed_array_f64() {
        let reg = test_registry();
        // Three elements, no length prefix
        let body_le: &[u8] = &[
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // m[0] = 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xC0, // m[1] = -2.5
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x3F, // m[2] = 0.5
        ];
        let expected = s(&[(
            "m",
            Value::Array(vec![Value::F64(1.0), Value::F64(-2.5), Value::F64(0.5)]),
        )]);
        assert_eq!(
            decode_message(&reg, "test_msgs/msg/FixedF64", &le(body_le)).unwrap(),
            expected
        );
        let body_be: &[u8] = &[
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // m[0] = 1.0
            0xC0, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // m[1] = -2.5
            0x3F, 0xE0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // m[2] = 0.5
        ];
        assert_eq!(
            decode_message(&reg, "test_msgs/msg/FixedF64", &be(body_be)).unwrap(),
            expected
        );
    }

    #[test]
    fn c5_sequence_i32() {
        let reg = test_registry();
        let body_le: &[u8] = &[
            0x03, 0x00, 0x00, 0x00, // count = 3
            0x01, 0x00, 0x00, 0x00, // v[0] = 1
            0xFE, 0xFF, 0xFF, 0xFF, // v[1] = -2
            0x2A, 0x00, 0x00, 0x00, // v[2] = 42
        ];
        let expected = s(&[(
            "v",
            Value::Array(vec![Value::I32(1), Value::I32(-2), Value::I32(42)]),
        )]);
        assert_eq!(
            decode_message(&reg, "test_msgs/msg/SeqI32", &le(body_le)).unwrap(),
            expected
        );
        let body_be: &[u8] = &[
            0x00, 0x00, 0x00, 0x03, // count = 3
            0x00, 0x00, 0x00, 0x01, // v[0] = 1
            0xFF, 0xFF, 0xFF, 0xFE, // v[1] = -2
            0x00, 0x00, 0x00, 0x2A, // v[2] = 42
        ];
        assert_eq!(
            decode_message(&reg, "test_msgs/msg/SeqI32", &be(body_be)).unwrap(),
            expected
        );
        // Empty sequence
        let v =
            decode_message(&reg, "test_msgs/msg/SeqI32", &le(&[0x00, 0x00, 0x00, 0x00])).unwrap();
        assert_eq!(v, s(&[("v", Value::Array(vec![]))]));
    }

    #[test]
    fn c6_byte_like_arrays_become_bytes() {
        let reg = test_registry();
        let seq_body: &[u8] = &[0x03, 0x00, 0x00, 0x00, 0xAA, 0xBB, 0xCC];
        for ty in [
            "test_msgs/msg/BytesSeq",
            "test_msgs/msg/ByteSeq",
            "test_msgs/msg/CharSeq",
        ] {
            let v = decode_message(&reg, ty, &le(seq_body)).unwrap();
            assert_eq!(v, s(&[("d", Value::Bytes(vec![0xAA, 0xBB, 0xCC]))]), "{ty}");
        }
        // Byte payload is identical under BE
        let v = decode_message(
            &reg,
            "test_msgs/msg/BytesSeq",
            &be(&[0x00, 0x00, 0x00, 0x03, 0xAA, 0xBB, 0xCC]),
        )
        .unwrap();
        assert_eq!(v, s(&[("d", Value::Bytes(vec![0xAA, 0xBB, 0xCC]))]));
        // Fixed-length arrays have no length prefix
        let v = decode_message(
            &reg,
            "test_msgs/msg/BytesFixed",
            &le(&[0xDE, 0xAD, 0xBE, 0xEF]),
        )
        .unwrap();
        assert_eq!(v, s(&[("d", Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]))]));
        // Empty sequence
        let v = decode_message(
            &reg,
            "test_msgs/msg/BytesSeq",
            &le(&[0x00, 0x00, 0x00, 0x00]),
        )
        .unwrap();
        assert_eq!(v, s(&[("d", Value::Bytes(vec![]))]));
    }

    #[test]
    fn c6b_int8_arrays_become_bytes_with_raw_bit_representation() {
        let reg = test_registry();
        // -1 = 0xFF, -128 = 0x80 stored as raw bits (supports OccupancyGrid.data unknown = -1).
        let v = decode_message(
            &reg,
            "test_msgs/msg/Int8Seq",
            &le(&[0x03, 0x00, 0x00, 0x00, 0x00, 0x64, 0xFF]),
        )
        .unwrap();
        assert_eq!(v, s(&[("d", Value::Bytes(vec![0x00, 0x64, 0xFF]))]));
        let v = decode_message(&reg, "test_msgs/msg/Int8Fixed", &le(&[0x80, 0x01, 0x7F])).unwrap();
        assert_eq!(v, s(&[("d", Value::Bytes(vec![0x80, 0x01, 0x7F]))]));
        // Scalar int8 stays Value::I8 (same convention as the c1 Prims test)
        let v = decode_message(
            &reg,
            "test_msgs/msg/Int8Seq",
            &le(&[0x00, 0x00, 0x00, 0x00]),
        )
        .unwrap();
        assert_eq!(v, s(&[("d", Value::Bytes(vec![]))]));
    }

    /// LE body for TransformStamped (sec=7, nanosec=500, "map"->"base", t=(1,2,3), r=(0,0,0,1)).
    const TRANSFORM_STAMPED_LE: &[u8] = &[
        0x07, 0x00, 0x00, 0x00, // header.stamp.sec = 7
        0xF4, 0x01, 0x00, 0x00, // header.stamp.nanosec = 500
        0x04, 0x00, 0x00, 0x00, // frame_id len (incl NUL)
        b'm', b'a', b'p', 0x00, // "map\0"
        0x05, 0x00, 0x00, 0x00, // child_frame_id len
        b'b', b'a', b's', b'e', 0x00, // "base\0"
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary (25->32)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // translation.x = 1.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // translation.y = 2.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // translation.z = 3.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // rotation.x = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // rotation.y = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // rotation.z = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // rotation.w = 1.0
    ];

    fn expected_transform_stamped() -> Value {
        s(&[
            (
                "header",
                s(&[
                    (
                        "stamp",
                        s(&[("sec", Value::I32(7)), ("nanosec", Value::U32(500))]),
                    ),
                    ("frame_id", Value::String("map".to_owned())),
                ]),
            ),
            ("child_frame_id", Value::String("base".to_owned())),
            (
                "transform",
                s(&[
                    (
                        "translation",
                        s(&[
                            ("x", Value::F64(1.0)),
                            ("y", Value::F64(2.0)),
                            ("z", Value::F64(3.0)),
                        ]),
                    ),
                    (
                        "rotation",
                        s(&[
                            ("x", Value::F64(0.0)),
                            ("y", Value::F64(0.0)),
                            ("z", Value::F64(0.0)),
                            ("w", Value::F64(1.0)),
                        ]),
                    ),
                ]),
            ),
        ])
    }

    #[test]
    fn c7_nested_transform_stamped_le() {
        let v = decode_message(
            &test_registry(),
            "geometry_msgs/msg/TransformStamped",
            &le(TRANSFORM_STAMPED_LE),
        )
        .unwrap();
        assert_eq!(v, expected_transform_stamped());
    }

    #[test]
    fn c7_nested_transform_stamped_be() {
        let body: &[u8] = &[
            0x00, 0x00, 0x00, 0x07, // header.stamp.sec = 7
            0x00, 0x00, 0x01, 0xF4, // header.stamp.nanosec = 500
            0x00, 0x00, 0x00, 0x04, // frame_id len
            b'm', b'a', b'p', 0x00, // "map\0"
            0x00, 0x00, 0x00, 0x05, // child_frame_id len
            b'b', b'a', b's', b'e', 0x00, // "base\0"
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // translation.x = 1.0
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // translation.y = 2.0
            0x40, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // translation.z = 3.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // rotation.x = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // rotation.y = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // rotation.z = 0.0
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // rotation.w = 1.0
        ];
        let v = decode_message(
            &test_registry(),
            "geometry_msgs/msg/TransformStamped",
            &be(body),
        )
        .unwrap();
        assert_eq!(v, expected_transform_stamped());
    }

    /// LE body for TFMessage (2 transforms). Verifies 8B alignment continues across elements.
    const TF_MESSAGE_LE: &[u8] = &[
        0x02, 0x00, 0x00, 0x00, // transforms count = 2
        0x01, 0x00, 0x00, 0x00, // [0].header.stamp.sec = 1
        0x02, 0x00, 0x00, 0x00, // [0].header.stamp.nanosec = 2
        0x02, 0x00, 0x00, 0x00, // [0].frame_id len
        b'a', 0x00, // "a\0"
        0x00, 0x00, // pad -> 4B boundary (18->20)
        0x02, 0x00, 0x00, 0x00, // [0].child_frame_id len
        b'b', 0x00, // "b\0"
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary (26->32)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // [0].translation.x = 1.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].translation.y = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].translation.z = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].rotation.x = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].rotation.y = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].rotation.z = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // [0].rotation.w = 1.0
        0x03, 0x00, 0x00,
        0x00, // [1].header.stamp.sec = 3 (at 88, already 8B-aligned, no pad)
        0x04, 0x00, 0x00, 0x00, // [1].header.stamp.nanosec = 4
        0x02, 0x00, 0x00, 0x00, // [1].frame_id len
        b'c', 0x00, // "c\0"
        0x00, 0x00, // pad -> 4B boundary
        0x02, 0x00, 0x00, 0x00, // [1].child_frame_id len
        b'd', 0x00, // "d\0"
        0x00, 0x00, // pad -> 8B boundary (110->112)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x40, // [1].translation.x = 5.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].translation.y = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].translation.z = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].rotation.x = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].rotation.y = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].rotation.z = 0.0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // [1].rotation.w = 1.0
    ];

    fn tf_entry(sec: i32, nanosec: u32, frame: &str, child: &str, x: f64) -> Value {
        s(&[
            (
                "header",
                s(&[
                    (
                        "stamp",
                        s(&[("sec", Value::I32(sec)), ("nanosec", Value::U32(nanosec))]),
                    ),
                    ("frame_id", Value::String(frame.to_owned())),
                ]),
            ),
            ("child_frame_id", Value::String(child.to_owned())),
            (
                "transform",
                s(&[
                    (
                        "translation",
                        s(&[
                            ("x", Value::F64(x)),
                            ("y", Value::F64(0.0)),
                            ("z", Value::F64(0.0)),
                        ]),
                    ),
                    (
                        "rotation",
                        s(&[
                            ("x", Value::F64(0.0)),
                            ("y", Value::F64(0.0)),
                            ("z", Value::F64(0.0)),
                            ("w", Value::F64(1.0)),
                        ]),
                    ),
                ]),
            ),
        ])
    }

    #[test]
    fn c8_tf_message_sequence_of_nested_le() {
        let v = decode_message(
            &test_registry(),
            "tf2_msgs/msg/TFMessage",
            &le(TF_MESSAGE_LE),
        )
        .unwrap();
        let expected = s(&[(
            "transforms",
            Value::Array(vec![
                tf_entry(1, 2, "a", "b", 1.0),
                tf_entry(3, 4, "c", "d", 5.0),
            ]),
        )]);
        assert_eq!(v, expected);
    }

    #[test]
    fn c8_tf_message_sequence_of_nested_be() {
        let body: &[u8] = &[
            0x00, 0x00, 0x00, 0x02, // transforms count = 2
            0x00, 0x00, 0x00, 0x01, // [0].sec = 1
            0x00, 0x00, 0x00, 0x02, // [0].nanosec = 2
            0x00, 0x00, 0x00, 0x02, // [0].frame_id len
            b'a', 0x00, // "a\0"
            0x00, 0x00, // pad -> 4B boundary
            0x00, 0x00, 0x00, 0x02, // [0].child_frame_id len
            b'b', 0x00, // "b\0"
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].translation.x = 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].translation.y = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].translation.z = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].rotation.x = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].rotation.y = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].rotation.z = 0.0
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [0].rotation.w = 1.0
            0x00, 0x00, 0x00, 0x03, // [1].sec = 3
            0x00, 0x00, 0x00, 0x04, // [1].nanosec = 4
            0x00, 0x00, 0x00, 0x02, // [1].frame_id len
            b'c', 0x00, // "c\0"
            0x00, 0x00, // pad -> 4B boundary
            0x00, 0x00, 0x00, 0x02, // [1].child_frame_id len
            b'd', 0x00, // "d\0"
            0x00, 0x00, // pad -> 8B boundary
            0x40, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].translation.x = 5.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].translation.y = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].translation.z = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].rotation.x = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].rotation.y = 0.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].rotation.z = 0.0
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // [1].rotation.w = 1.0
        ];
        let v = decode_message(&test_registry(), "tf2_msgs/msg/TFMessage", &be(body)).unwrap();
        let expected = s(&[(
            "transforms",
            Value::Array(vec![
                tf_entry(1, 2, "a", "b", 1.0),
                tf_entry(3, 4, "c", "d", 5.0),
            ]),
        )]);
        assert_eq!(v, expected);
    }

    #[test]
    fn c9_padding_before_arrays() {
        let reg = test_registry();
        // Fixed-length f64 array after u8: pad 7 before the array
        let body: &[u8] = &[
            0x2A, // a = 42
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad -> 8B boundary
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // b[0] = 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // b[1] = 2.0
        ];
        let v = decode_message(&reg, "test_msgs/msg/PadFixed", &le(body)).unwrap();
        assert_eq!(
            v,
            s(&[
                ("a", Value::U8(42)),
                ("b", Value::Array(vec![Value::F64(1.0), Value::F64(2.0)])),
            ])
        );
        // Sequence after u8: pad 3 before the length prefix
        let body: &[u8] = &[
            0x2A, // a = 42
            0x00, 0x00, 0x00, // pad -> 4B boundary
            0x02, 0x00, 0x00, 0x00, // count = 2
            0x07, 0x00, 0x00, 0x00, // c[0] = 7
            0x08, 0x00, 0x00, 0x00, // c[1] = 8
        ];
        let v = decode_message(&reg, "test_msgs/msg/PadSeq", &le(body)).unwrap();
        assert_eq!(
            v,
            s(&[
                ("a", Value::U8(42)),
                ("c", Value::Array(vec![Value::I32(7), Value::I32(8)])),
            ])
        );
    }

    #[test]
    fn c10_encapsulation_errors() {
        let reg = test_registry();
        // PL_CDR_BE(00 02) is unsupported
        let e = decode_message(&reg, "std_msgs/msg/String", &[0x00, 0x02, 0x00, 0x00, 0x00])
            .unwrap_err();
        assert_eq!(
            e.kind,
            DecodeErrorKind::UnsupportedEncapsulation([0x00, 0x02])
        );
        // First byte other than 00
        let e = decode_message(&reg, "std_msgs/msg/String", &[0x01, 0x00, 0x00, 0x00]).unwrap_err();
        assert_eq!(
            e.kind,
            DecodeErrorKind::UnsupportedEncapsulation([0x01, 0x00])
        );
        // Header itself is under 4 bytes
        let e = decode_message(&reg, "std_msgs/msg/String", &[0x00, 0x01, 0x00]).unwrap_err();
        assert_eq!(
            e.kind,
            DecodeErrorKind::Truncated {
                needed: 4,
                remaining: 3
            }
        );
    }

    #[test]
    fn c11_truncation_carries_path_and_offset() {
        let reg = test_registry();
        // Cut off mid-way through the child_frame_id string body
        let e = decode_message(
            &reg,
            "geometry_msgs/msg/TransformStamped",
            &le(&TRANSFORM_STAMPED_LE[..22]),
        )
        .unwrap_err();
        assert_eq!(
            e.kind,
            DecodeErrorKind::Truncated {
                needed: 5,
                remaining: 2
            }
        );
        assert_eq!(e.path, "child_frame_id");
        assert_eq!(e.offset, 20);

        // Cut off mid translation.y (path should carry three nesting levels)
        let e = decode_message(
            &reg,
            "geometry_msgs/msg/TransformStamped",
            &le(&TRANSFORM_STAMPED_LE[..44]),
        )
        .unwrap_err();
        assert_eq!(e.path, "transform.translation.y");
        assert_eq!(e.offset, 40);

        // Cut off inside TFMessage's second element (path should carry the sequence index)
        let e =
            decode_message(&reg, "tf2_msgs/msg/TFMessage", &le(&TF_MESSAGE_LE[..90])).unwrap_err();
        assert_eq!(e.path, "transforms[1].header.stamp.sec");
        assert_eq!(e.offset, 88);
    }

    #[test]
    fn c12_sequence_length_overrun() {
        let reg = test_registry();
        let e = decode_message(&reg, "test_msgs/msg/SeqI32", &le(&[0xFF, 0xFF, 0xFF, 0xFF]))
            .unwrap_err();
        assert_eq!(
            e.kind,
            DecodeErrorKind::LengthOverrun {
                length: 0xFFFF_FFFF,
                remaining: 0
            }
        );
        assert_eq!(e.path, "v");
        // Same check applies to uint8-like types (bulk Bytes read)
        let e = decode_message(
            &reg,
            "test_msgs/msg/BytesSeq",
            &le(&[0xFF, 0xFF, 0xFF, 0xFF]),
        )
        .unwrap_err();
        assert!(matches!(e.kind, DecodeErrorKind::LengthOverrun { .. }));
    }

    #[test]
    fn c13_unknown_type() {
        let e = decode_message(&test_registry(), "test_msgs/msg/DoesNotExist", &le(&[0x00]))
            .unwrap_err();
        assert_eq!(
            e.kind,
            DecodeErrorKind::UnknownType("test_msgs/msg/DoesNotExist".to_owned())
        );
    }

    #[test]
    fn bounded_definitions_decode_the_same_bytes_as_unbounded_ones() {
        // XCDR1 puts no bound on the wire, so a `string<=N` / `T[<=N]` definition reads exactly what the unbounded one reads.
        let mut reg = test_registry();
        reg.insert_msg("test_msgs/msg/Bounded", "string<=8 s\nint32[<=4] v")
            .unwrap();
        reg.insert_msg("test_msgs/msg/Unbounded", "string s\nint32[] v")
            .unwrap();
        let body = le(&[
            0x03, 0x00, 0x00, 0x00, b'h', b'i', 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x05, 0x00,
            0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
        ]);
        let bounded = decode_message(&reg, "test_msgs/msg/Bounded", &body).unwrap();
        let unbounded = decode_message(&reg, "test_msgs/msg/Unbounded", &body).unwrap();
        assert_eq!(bounded, unbounded);
        assert_eq!(
            bounded,
            s(&[
                ("s", Value::String("hi".to_owned())),
                ("v", Value::Array(vec![Value::I32(5), Value::I32(6)])),
            ])
        );
    }

    #[test]
    fn trailing_bytes_are_ignored() {
        // CDR may append 4B-aligned trailing padding, so unconsumed bytes are not an error.
        let v = decode_message(
            &test_registry(),
            "std_msgs/msg/String",
            &le(&[0x03, 0x00, 0x00, 0x00, b'h', b'i', 0x00, 0x00]),
        )
        .unwrap();
        assert_eq!(v, s(&[("data", Value::String("hi".to_owned()))]));
    }
}
