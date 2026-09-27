pub mod cdr;
pub mod embedded;
pub mod msg_parser;
pub mod ros1;
pub mod value;

/// Both decoders report failures the same way, so the error type is addressed from here rather than from `cdr`.
pub use cdr::{DecodeError, DecodeErrorKind};
