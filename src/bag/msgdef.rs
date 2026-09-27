//! Turns a connection's concatenated message definition (`ros1msg` or `ros2msg`) into a per-connection TypeRegistry, falling back to the app's definitions when the bag's are missing or unreadable (requirements FR-3).

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use super::naming::ros1_type_to_ros2;
use super::reader::Connection;
use crate::decode::DecodeError;
use crate::decode::cdr::decode_message;
use crate::decode::msg_parser::{MsgParseError, TypeRegistry};
use crate::decode::ros1::decode_ros1_message;
use crate::decode::value::Value;

/// ROS 1 `time` and `duration` are primitives; a synthetic definition maps them onto ROS 2's field names (Q4 = A).
const TIME_BODY: &str = "int32 sec\nuint32 nanosec";
/// Token the rewrite emits; it stays in ROS 1 `pkg/Type` form so the existing parser normalizes it like any other reference.
const TIME_TOKEN: &str = "builtin_interfaces/Time";
/// Registry key the token resolves to.
const TIME_TYPE: &str = "builtin_interfaces/msg/Time";
/// Same treatment for `duration`, which shares the wire layout.
const DURATION_TOKEN: &str = "builtin_interfaces/Duration";
const DURATION_TYPE: &str = "builtin_interfaces/msg/Duration";

/// Why a connection's definition could not be turned into a registry.
#[derive(Debug)]
pub enum MsgDefError {
    /// A `====` separator was not followed by a `MSG: pkg/Type` line.
    MissingTypeHeader(usize),
    /// A type name in the definition is not `pkg/Type`.
    InvalidTypeName(String),
    /// The `.msg` body did not parse, or a reference stayed unresolved.
    Parse(MsgParseError),
}

impl fmt::Display for MsgDefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MsgDefError::MissingTypeHeader(line) => {
                write!(
                    f,
                    "line {line}: definition separator has no `MSG:` line after it"
                )
            }
            MsgDefError::InvalidTypeName(name) => {
                write!(f, "definition names type `{name}`, which is not `pkg/Type`")
            }
            MsgDefError::Parse(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MsgDefError {}

impl From<MsgParseError> for MsgDefError {
    fn from(e: MsgParseError) -> Self {
        MsgDefError::Parse(e)
    }
}

/// Split a concatenated definition into `(ROS 1 type name, body)`; the first entry is the root type.
pub fn split_definitions(
    root_type: &str,
    text: &str,
) -> Result<Vec<(String, String)>, MsgDefError> {
    let mut sections: Vec<(String, String)> = vec![(root_type.to_owned(), String::new())];
    let mut expecting_name = false;
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if is_separator(trimmed) {
            expecting_name = true;
            continue;
        }
        if expecting_name {
            // rosbag writes the separator and `MSG:` line as a pair, so anything else means the definition is not what we think.
            let name = trimmed
                .strip_prefix("MSG:")
                .ok_or(MsgDefError::MissingTypeHeader(idx + 1))?
                .trim();
            sections.push((name.to_owned(), String::new()));
            expecting_name = false;
            continue;
        }
        let body = &mut sections.last_mut().expect("root section exists").1;
        body.push_str(line);
        body.push('\n');
    }
    Ok(sections)
}

/// A line of `=` only (rosbag writes 80 of them) marks the boundary between definitions.
fn is_separator(trimmed: &str) -> bool {
    trimmed.len() >= 4 && trimmed.bytes().all(|b| b == b'=')
}

/// Rewrite a ROS 1 `.msg` body so the ROS 2 parser reads it: `time` / `duration` / bare `Header` get explicit types.
pub fn rewrite_ros1_body(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    for line in body.lines() {
        out.push_str(&rewrite_line(line));
        out.push('\n');
    }
    out
}

/// Rewrite one line's type token, leaving comments, constants and blank lines untouched.
fn rewrite_line(line: &str) -> String {
    let (code, comment) = match line.split_once('#') {
        Some((code, comment)) => (code, Some(comment)),
        None => (line, None),
    };
    // Constant lines (`uint8 FOO=1`) only ever use primitives, so there is no type token to map.
    if code.trim().is_empty() || code.contains('=') {
        return line.to_owned();
    }
    let mut parts = code.split_whitespace();
    let Some(token) = parts.next() else {
        return line.to_owned();
    };
    let rest: Vec<&str> = parts.collect();
    let mapped = map_type_token(token);
    let mut out = mapped;
    for part in rest {
        out.push(' ');
        out.push_str(part);
    }
    if let Some(comment) = comment {
        out.push_str(" #");
        out.push_str(comment);
    }
    out
}

/// Map a ROS 1 type token to its ROS 2 spelling, preserving any array suffix.
fn map_type_token(token: &str) -> String {
    let (base, suffix) = match token.find('[') {
        Some(at) => (&token[..at], &token[at..]),
        None => (token, ""),
    };
    let mapped = match base {
        "time" => TIME_TOKEN,
        "duration" => DURATION_TOKEN,
        // ROS 1 resolves a bare `Header` to std_msgs, not to the enclosing package (requirements d5).
        "Header" => "std_msgs/Header",
        other => other,
    };
    format!("{mapped}{suffix}")
}

/// Build the type registry for one connection: root type name plus every dependency it carries.
pub fn registry_for(conn: &Connection) -> Result<(String, TypeRegistry), MsgDefError> {
    let root = ros1_type_to_ros2(&conn.type_raw);
    let sections = split_definitions(&conn.type_raw, &conn.definition)?;
    let mut registry = TypeRegistry::new();
    for (index, (name, body)) in sections.iter().enumerate() {
        // The root keeps its alias-resolved name so the renderer registry and the TF path can find it.
        let full_name = if index == 0 {
            root.clone()
        } else {
            ros1_type_to_ros2(name)
        };
        if full_name.split('/').count() != 3 {
            return Err(MsgDefError::InvalidTypeName(name.clone()));
        }
        registry.insert_msg(&full_name, &rewrite_ros1_body(body))?;
    }
    for (name, body) in [(TIME_TYPE, TIME_BODY), (DURATION_TYPE, TIME_BODY)] {
        if registry.get(name).is_none() {
            registry.insert_msg(name, body)?;
        }
    }
    registry.validate()?;
    Ok((root, registry))
}

/// Build the registry for a rosbag2 `ros2msg` schema: the bag's definitions layered over a clone of `fallback`, so a bag definition always wins over a bundled one.
pub fn registry_for_ros2msg(
    conn: &Connection,
    fallback: Option<&TypeRegistry>,
) -> Result<(String, TypeRegistry), MsgDefError> {
    let root = ros1_type_to_ros2(&conn.type_raw);
    let sections = split_definitions(&conn.type_raw, &conn.definition)?;
    let mut registry = fallback.cloned().unwrap_or_default();
    for (index, (name, body)) in sections.iter().enumerate() {
        let full_name = if index == 0 {
            root.clone()
        } else {
            ros1_type_to_ros2(name)
        };
        if full_name.split('/').count() != 3 {
            return Err(MsgDefError::InvalidTypeName(name.clone()));
        }
        registry.insert_msg(&full_name, body)?;
    }
    registry.validate()?;
    Ok((root, registry))
}

/// How a connection's payload bytes are read: ROS 1 serialization, or CDR with the 4-byte encapsulation header (ROS 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decoder {
    Ros1,
    Cdr,
}

/// What one connection decodes as: its root type name, the registry describing it, and the wire format.
#[derive(Clone)]
pub struct ConnTypes {
    pub root_type: String,
    pub registry: Arc<TypeRegistry>,
    pub decoder: Decoder,
}

impl ConnTypes {
    /// Decode one payload with this connection's registry and wire format.
    pub fn decode(&self, payload: &[u8]) -> Result<Value, DecodeError> {
        match self.decoder {
            Decoder::Ros1 => decode_ros1_message(&self.registry, &self.root_type, payload),
            Decoder::Cdr => decode_message(&self.registry, &self.root_type, payload),
        }
    }
}

/// Per-connection registries, sharing one `Arc` across connections whose definitions are byte-identical.
#[derive(Default)]
pub struct RegistrySet {
    by_conn: HashMap<u32, ConnTypes>,
    /// Connections whose definition could not be interpreted, with the reason (that connection alone is dropped).
    unsupported: HashMap<u32, String>,
    shared: HashMap<String, Arc<TypeRegistry>>,
    /// Connections whose bag definition failed and were rescued by the fallback registry, with why; told to the user once.
    fallback_notices: Vec<(u32, String)>,
}

impl RegistrySet {
    /// Build registries for every connection; a failing definition costs that connection only, not the bag. `fallback` (the app's merged `.msg` registry) covers connections whose own definition is missing or unreadable.
    pub fn build(connections: &[Connection], fallback: Option<&Arc<TypeRegistry>>) -> Self {
        let mut set = Self::default();
        for conn in connections {
            let decoder = match conn.message_encoding.as_str() {
                "ros1" => Decoder::Ros1,
                "cdr" => Decoder::Cdr,
                other => {
                    set.unsupported
                        .insert(conn.id, format!("unsupported message encoding `{other}`"));
                    continue;
                }
            };
            // `/tf` alone can appear on six connections with the same definition, so cache by definition text.
            let key = format!(
                "{}\n{}\n{}",
                conn.type_raw, conn.definition_encoding, conn.definition
            );
            if let Some(registry) = set.shared.get(&key) {
                set.by_conn.insert(
                    conn.id,
                    ConnTypes {
                        root_type: ros1_type_to_ros2(&conn.type_raw),
                        registry: registry.clone(),
                        decoder,
                    },
                );
                continue;
            }
            let root_type = ros1_type_to_ros2(&conn.type_raw);
            let fallback_types = || {
                fallback
                    .filter(|registry| registry.get(&root_type).is_some())
                    .map(|registry| ConnTypes {
                        root_type: root_type.clone(),
                        registry: Arc::clone(registry),
                        decoder,
                    })
            };
            let own = match conn.definition_encoding.as_str() {
                "ros1msg" => Some(registry_for(conn)),
                "ros2msg" if !conn.definition.trim().is_empty() => {
                    Some(registry_for_ros2msg(conn, fallback.map(Arc::as_ref)))
                }
                _ => None,
            };
            match own {
                Some(Ok((root_type, registry))) => {
                    let registry = Arc::new(registry);
                    set.shared.insert(key, registry.clone());
                    set.by_conn.insert(
                        conn.id,
                        ConnTypes {
                            root_type,
                            registry,
                            decoder,
                        },
                    );
                }
                Some(Err(e)) => match fallback_types() {
                    Some(types) => {
                        set.fallback_notices.push((
                            conn.id,
                            format!("bag definition unreadable ({e}); using bundled definition"),
                        ));
                        set.by_conn.insert(conn.id, types);
                    }
                    None => {
                        set.unsupported.insert(conn.id, e.to_string());
                    }
                },
                None => match fallback_types() {
                    Some(types) => {
                        set.by_conn.insert(conn.id, types);
                    }
                    None => {
                        let why = match conn.definition_encoding.as_str() {
                            "" => "no definition in the bag".to_owned(),
                            "ros2msg" => "empty definition in the bag".to_owned(),
                            other => format!("definition encoding `{other}` is not parsed"),
                        };
                        set.unsupported.insert(
                            conn.id,
                            format!("{why} and no bundled definition for `{root_type}`"),
                        );
                    }
                },
            }
        }
        set
    }

    pub fn get(&self, conn: u32) -> Option<&ConnTypes> {
        self.by_conn.get(&conn)
    }

    /// Connections rescued by the fallback registry after their own definition failed, with the reason.
    pub fn fallback_notices(&self) -> &[(u32, String)] {
        &self.fallback_notices
    }

    /// Reasons individual connections were dropped, sorted by connection id for a stable notice.
    pub fn unsupported(&self) -> Vec<(u32, &str)> {
        let mut out: Vec<(u32, &str)> = self
            .unsupported
            .iter()
            .map(|(id, why)| (*id, why.as_str()))
            .collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    /// Number of distinct registries actually built (many connections share one).
    pub fn distinct_registries(&self) -> usize {
        self.shared.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::msg_parser::{ArraySpec, FieldType, PrimitiveType};

    const SEP: &str =
        "================================================================================";

    fn conn(type_raw: &str, definition: &str) -> Connection {
        Connection {
            id: 1,
            topic_raw: "/t".to_owned(),
            type_raw: type_raw.to_owned(),
            type_hash: "0".to_owned(),
            definition: definition.to_owned(),
            message_encoding: "ros1".to_owned(),
            definition_encoding: "ros1msg".to_owned(),
        }
    }

    /// A rosbag2 channel: CDR payloads and a `ros2msg` schema (empty `definition` models an old bag without one).
    fn ros2_conn(id: u32, type_raw: &str, definition: &str) -> Connection {
        Connection {
            id,
            topic_raw: "/t".to_owned(),
            type_raw: type_raw.to_owned(),
            type_hash: "RIHS01_0".to_owned(),
            definition: definition.to_owned(),
            message_encoding: "cdr".to_owned(),
            definition_encoding: "ros2msg".to_owned(),
        }
    }

    /// What rosbag2 (Iron and later) writes for `geometry_msgs/msg/PointStamped`: 3-segment `MSG:` names, ROS 2 field syntax.
    fn point_stamped_ros2msg() -> String {
        [
            "std_msgs/Header header",
            "Point point",
            SEP,
            "MSG: std_msgs/msg/Header",
            "builtin_interfaces/Time stamp",
            "string frame_id",
            SEP,
            "MSG: builtin_interfaces/msg/Time",
            "int32 sec",
            "uint32 nanosec",
            SEP,
            "MSG: geometry_msgs/msg/Point",
            "float64 x",
            "float64 y",
            "float64 z",
        ]
        .join("\n")
    }

    /// The real `nav_msgs/Odometry` definition as a bag records it, trimmed to the fields that matter here.
    fn odometry_definition() -> String {
        [
            "Header header",
            "string child_frame_id",
            "geometry_msgs/PoseWithCovariance pose",
            "geometry_msgs/TwistWithCovariance twist",
            SEP,
            "MSG: std_msgs/Header",
            "uint32 seq",
            "time stamp",
            "string frame_id",
            SEP,
            "MSG: geometry_msgs/PoseWithCovariance",
            "Pose pose",
            "float64[36] covariance",
            SEP,
            "MSG: geometry_msgs/Pose",
            "Point position",
            "Quaternion orientation",
            SEP,
            "MSG: geometry_msgs/Point",
            "float64 x",
            "float64 y",
            "float64 z",
            SEP,
            "MSG: geometry_msgs/Quaternion",
            "float64 x",
            "float64 y",
            "float64 z",
            "float64 w",
            SEP,
            "MSG: geometry_msgs/TwistWithCovariance",
            "Twist twist",
            "float64[36] covariance",
            SEP,
            "MSG: geometry_msgs/Twist",
            "Vector3  linear",
            "Vector3  angular",
            SEP,
            "MSG: geometry_msgs/Vector3",
            "float64 x",
            "float64 y",
            "float64 z",
        ]
        .join("\n")
    }

    #[test]
    fn t5_definitions_split_on_the_separator() {
        let sections = split_definitions("nav_msgs/Odometry", &odometry_definition()).unwrap();
        let names: Vec<&str> = sections.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "nav_msgs/Odometry",
                "std_msgs/Header",
                "geometry_msgs/PoseWithCovariance",
                "geometry_msgs/Pose",
                "geometry_msgs/Point",
                "geometry_msgs/Quaternion",
                "geometry_msgs/TwistWithCovariance",
                "geometry_msgs/Twist",
                "geometry_msgs/Vector3",
            ]
        );
        assert!(sections[0].1.contains("string child_frame_id"));
        assert!(!sections[0].1.contains("MSG:"));
        assert!(sections[1].1.contains("uint32 seq"));
    }

    #[test]
    fn t5_split_tolerates_blank_lines_and_a_trailing_newline() {
        let text = format!("int32 a\n\n{SEP}\nMSG: pkg/Dep\n\nint32 b\n");
        let sections = split_definitions("pkg/Root", &text).unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[1].0, "pkg/Dep");
        assert!(sections[1].1.contains("int32 b"));
    }

    #[test]
    fn t5_separator_without_a_msg_line_is_an_error() {
        let text = format!("int32 a\n{SEP}\nint32 b\n");
        assert!(matches!(
            split_definitions("pkg/Root", &text),
            Err(MsgDefError::MissingTypeHeader(3))
        ));
    }

    #[test]
    fn t6_type_tokens_are_rewritten_in_place() {
        let body = rewrite_ros1_body(
            "Header header\ntime stamp\nduration timeout\ntime[] stamps\nduration[2] pair\nstd_msgs/Header h2\nPose pose\nfloat64 x",
        );
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines[0], "std_msgs/Header header");
        assert_eq!(lines[1], "builtin_interfaces/Time stamp");
        assert_eq!(lines[2], "builtin_interfaces/Duration timeout");
        assert_eq!(lines[3], "builtin_interfaces/Time[] stamps");
        assert_eq!(lines[4], "builtin_interfaces/Duration[2] pair");
        assert_eq!(lines[5], "std_msgs/Header h2");
        assert_eq!(lines[6], "Pose pose");
        assert_eq!(lines[7], "float64 x");
    }

    #[test]
    fn t6_comments_constants_and_blanks_survive_rewriting() {
        let body = rewrite_ros1_body(
            "# a comment\n\nuint8 FOO=1\nstring NAME=time\ntime stamp # when\n  int32 x  ",
        );
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines[0], "# a comment");
        assert_eq!(lines[1], "");
        // Constants keep their exact text: rewriting `time` inside a string constant's value would corrupt it.
        assert_eq!(lines[2], "uint8 FOO=1");
        assert_eq!(lines[3], "string NAME=time");
        assert_eq!(lines[4], "builtin_interfaces/Time stamp # when");
        assert_eq!(lines[5], "int32 x");
    }

    #[test]
    fn t7_odometry_definition_becomes_a_valid_registry() {
        let (root, registry) =
            registry_for(&conn("nav_msgs/Odometry", &odometry_definition())).unwrap();
        assert_eq!(root, "nav_msgs/msg/Odometry");
        let header = registry.get("std_msgs/msg/Header").unwrap();
        assert_eq!(header.fields[0].name, "seq");
        // `time stamp` resolved through the injected synthetic definition.
        assert_eq!(
            header.fields[1].ty,
            FieldType::Complex(TIME_TYPE.to_owned())
        );
        let time = registry.get(TIME_TYPE).unwrap();
        assert_eq!(
            time.fields[0].ty,
            FieldType::Primitive(PrimitiveType::Int32)
        );
        assert_eq!(
            time.fields[1].ty,
            FieldType::Primitive(PrimitiveType::UInt32)
        );
        // Same-package relative names still resolve through msg_parser's existing rule.
        let pose_cov = registry
            .get("geometry_msgs/msg/PoseWithCovariance")
            .unwrap();
        assert_eq!(
            pose_cov.fields[0].ty,
            FieldType::Complex("geometry_msgs/msg/Pose".to_owned())
        );
        assert_eq!(pose_cov.fields[1].array, ArraySpec::Fixed(36));
    }

    #[test]
    fn t7_legacy_tf_message_registers_under_the_tf2_name() {
        let definition = [
            "geometry_msgs/TransformStamped[] transforms",
            SEP,
            "MSG: geometry_msgs/TransformStamped",
            "Header header",
            "string child_frame_id",
            "Transform transform",
            SEP,
            "MSG: std_msgs/Header",
            "uint32 seq",
            "time stamp",
            "string frame_id",
            SEP,
            "MSG: geometry_msgs/Transform",
            "Vector3 translation",
            "Quaternion rotation",
            SEP,
            "MSG: geometry_msgs/Vector3",
            "float64 x",
            "float64 y",
            "float64 z",
            SEP,
            "MSG: geometry_msgs/Quaternion",
            "float64 x",
            "float64 y",
            "float64 z",
            "float64 w",
        ]
        .join("\n");
        let (root, registry) = registry_for(&conn("tf/tfMessage", &definition)).unwrap();
        assert_eq!(root, "tf2_msgs/msg/TFMessage");
        assert!(registry.get("tf2_msgs/msg/TFMessage").is_some());
    }

    #[test]
    fn t8_a_broken_connection_is_dropped_while_the_others_survive() {
        let good = Connection {
            id: 1,
            ..conn("nav_msgs/Odometry", &odometry_definition())
        };
        let broken = Connection {
            id: 2,
            ..conn("pkg/Broken", "pkg/Missing dep")
        };
        let duplicate = Connection {
            id: 3,
            ..conn("nav_msgs/Odometry", &odometry_definition())
        };
        let set = RegistrySet::build(&[good, broken, duplicate], None);
        assert_eq!(set.get(1).unwrap().root_type, "nav_msgs/msg/Odometry");
        assert_eq!(set.get(1).unwrap().decoder, Decoder::Ros1);
        assert!(set.get(2).is_none());
        assert!(set.get(3).is_some());
        // Identical definitions share one registry rather than being parsed twice.
        assert_eq!(set.distinct_registries(), 1);
        assert!(Arc::ptr_eq(
            &set.get(1).unwrap().registry,
            &set.get(3).unwrap().registry
        ));
        let unsupported = set.unsupported();
        assert_eq!(unsupported.len(), 1);
        assert_eq!(unsupported[0].0, 2);
        assert!(unsupported[0].1.contains("Missing"), "{}", unsupported[0].1);
    }

    #[test]
    fn custom_types_register_even_though_nothing_renders_them() {
        let definition = [
            "Header header",
            "float64 max_velocity",
            "geometry_msgs/Point[] waypoints",
            SEP,
            "MSG: std_msgs/Header",
            "uint32 seq",
            "time stamp",
            "string frame_id",
            SEP,
            "MSG: geometry_msgs/Point",
            "float64 x",
            "float64 y",
            "float64 z",
        ]
        .join("\n");
        let (root, registry) =
            registry_for(&conn("prl_msgs/PathPlanningParameters", &definition)).unwrap();
        assert_eq!(root, "prl_msgs/msg/PathPlanningParameters");
        assert!(registry.get(&root).is_some());
    }

    #[test]
    fn ros2msg_definitions_split_on_three_segment_names_and_decode_with_cdr() {
        let set = RegistrySet::build(
            &[ros2_conn(
                1,
                "geometry_msgs/msg/PointStamped",
                &point_stamped_ros2msg(),
            )],
            None,
        );
        let types = set.get(1).unwrap();
        assert_eq!(types.root_type, "geometry_msgs/msg/PointStamped");
        assert_eq!(types.decoder, Decoder::Cdr);
        assert!(types.registry.get("builtin_interfaces/msg/Time").is_some());
        // No ROS 1 rewriting happened: the header's stamp is the type the bag named, not a synthesized one.
        let header = types.registry.get("std_msgs/msg/Header").unwrap();
        assert_eq!(
            header.fields[0].ty,
            FieldType::Complex("builtin_interfaces/msg/Time".to_owned())
        );
        // A CDR payload (LE header, sec, nanosec, frame_id "map", then x y z already 8-aligned) decodes through the connection.
        let mut payload = vec![0x00, 0x01, 0x00, 0x00];
        payload.extend_from_slice(&7i32.to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&4u32.to_le_bytes());
        payload.extend_from_slice(b"map\0");
        for v in [1.0f64, 2.0, 3.0] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let value = types.decode(&payload).unwrap();
        let Value::Struct(fields) = value else {
            panic!("expected a struct");
        };
        assert_eq!(fields[1].0, "point");
        assert!(set.fallback_notices().is_empty());
        assert!(set.unsupported().is_empty());
    }

    #[test]
    fn fallback_fills_missing_dependencies_but_the_bag_definition_wins() {
        let fallback = Arc::new(TypeRegistry::with_embedded().unwrap());
        // The bag names a Point with an extra field and omits every dependency: those come from the fallback.
        let definition = "std_msgs/Header header\nPoint point\n".to_owned()
            + SEP
            + "\nMSG: geometry_msgs/msg/Point\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w";
        let set = RegistrySet::build(
            &[ros2_conn(1, "geometry_msgs/msg/PointStamped", &definition)],
            Some(&fallback),
        );
        let types = set.get(1).unwrap();
        assert_eq!(
            types
                .registry
                .get("geometry_msgs/msg/Point")
                .unwrap()
                .fields
                .len(),
            4
        );
        assert!(types.registry.get("std_msgs/msg/Header").is_some());
        assert!(!Arc::ptr_eq(&types.registry, &fallback));
        assert!(set.fallback_notices().is_empty());
        // The fallback itself is untouched by the overlay.
        assert_eq!(
            fallback
                .get("geometry_msgs/msg/Point")
                .unwrap()
                .fields
                .len(),
            3
        );
    }

    #[test]
    fn empty_and_idl_definitions_use_the_fallback_root_silently() {
        let fallback = Arc::new(TypeRegistry::with_embedded().unwrap());
        let mut idl = ros2_conn(2, "sensor_msgs/msg/LaserScan", "module sensor_msgs {}");
        idl.definition_encoding = "ros2idl".to_owned();
        let mut none = ros2_conn(3, "nav_msgs/msg/Odometry", "");
        none.definition_encoding = String::new();
        let set = RegistrySet::build(
            &[ros2_conn(1, "std_msgs/msg/String", ""), idl, none],
            Some(&fallback),
        );
        for id in [1, 2, 3] {
            let types = set.get(id).unwrap();
            assert!(Arc::ptr_eq(&types.registry, &fallback), "conn {id}");
            assert_eq!(types.decoder, Decoder::Cdr);
        }
        assert!(set.fallback_notices().is_empty());
        // Without a fallback the same connections are dropped with a reason that names what was missing.
        let mut idl = ros2_conn(2, "sensor_msgs/msg/LaserScan", "module sensor_msgs {}");
        idl.definition_encoding = "ros2idl".to_owned();
        let set = RegistrySet::build(&[ros2_conn(1, "std_msgs/msg/String", ""), idl], None);
        let unsupported = set.unsupported();
        assert_eq!(unsupported.len(), 2);
        assert!(
            unsupported[0].1.contains("empty definition"),
            "{}",
            unsupported[0].1
        );
        assert!(unsupported[1].1.contains("ros2idl"), "{}", unsupported[1].1);
    }

    #[test]
    fn an_unreadable_ros2msg_falls_back_with_a_notice_or_is_dropped_with_the_reason() {
        let fallback = Arc::new(TypeRegistry::with_embedded().unwrap());
        // `wstring` is the one syntax the parser still refuses, so this definition is genuinely unreadable.
        let bad = ros2_conn(1, "std_msgs/msg/String", "wstring data");
        let set = RegistrySet::build(std::slice::from_ref(&bad), Some(&fallback));
        let types = set.get(1).unwrap();
        assert!(Arc::ptr_eq(&types.registry, &fallback));
        let notices = set.fallback_notices();
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].0, 1);
        assert!(notices[0].1.contains("wstring"), "{}", notices[0].1);
        assert!(notices[0].1.contains("bundled"), "{}", notices[0].1);
        let set = RegistrySet::build(std::slice::from_ref(&bad), None);
        assert!(set.get(1).is_none());
        assert!(set.unsupported()[0].1.contains("wstring"));
    }

    #[test]
    fn an_unknown_message_encoding_is_dropped_before_any_definition_is_read() {
        let mut odd = ros2_conn(1, "std_msgs/msg/String", "string data");
        odd.message_encoding = "protobuf".to_owned();
        let set = RegistrySet::build(&[odd], None);
        assert!(set.get(1).is_none());
        assert!(set.unsupported()[0].1.contains("protobuf"));
    }
}
