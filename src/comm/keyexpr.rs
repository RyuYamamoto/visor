//! Build/parse rmw_zenoh key expressions and mangle/demangle type names (spec: rmw_zenoh 0.2.4/0.2.9 liveliness_utils.cpp).

use std::fmt;

/// Reasons a key expression fails to parse.
#[derive(Debug, PartialEq, Eq)]
pub enum KeyExprError {
    /// Segment count does not match the spec.
    SegmentCount { key: String, actual: usize },
    /// domain_id is not a valid number.
    InvalidDomainId(String),
    /// liveliness token does not start with `@ros2_lv`.
    InvalidAdminSpace(String),
    /// Entity kind is not one of NN/MP/MS/SS/SC.
    UnknownEntityKind(String),
    /// Malformed type name.
    InvalidTypeName(String),
}

impl fmt::Display for KeyExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SegmentCount { key, actual } => {
                write!(
                    f,
                    "unexpected segment count {actual} in key expression `{key}`"
                )
            }
            Self::InvalidDomainId(s) => write!(f, "invalid domain_id `{s}`"),
            Self::InvalidAdminSpace(s) => {
                write!(f, "liveliness token does not start with @ros2_lv: `{s}`")
            }
            Self::UnknownEntityKind(s) => write!(f, "unknown entity kind `{s}`"),
            Self::InvalidTypeName(s) => write!(f, "invalid type name `{s}`"),
        }
    }
}

impl std::error::Error for KeyExprError {}

/// Replace every `/` with `%` (rmw_zenoh `mangle_name` equivalent).
pub fn mangle_name(s: &str) -> String {
    s.replace('/', "%")
}

/// Replace every `%` back with `/` (rmw_zenoh `demangle_name` equivalent).
pub fn demangle_name(s: &str) -> String {
    s.replace('%', "/")
}

/// DDS mangled type name to ROS form (e.g. `sensor_msgs::msg::dds_::LaserScan_` to `sensor_msgs/msg/LaserScan`).
pub fn dds_type_to_ros(dds: &str) -> Result<String, KeyExprError> {
    let err = || KeyExprError::InvalidTypeName(dds.to_owned());
    let parts: Vec<&str> = dds.split("::").collect();
    if parts.len() < 3 {
        return Err(err());
    }
    let name = parts[parts.len() - 1].strip_suffix('_').ok_or_else(err)?;
    if name.is_empty() || parts[parts.len() - 2] != "dds_" {
        return Err(err());
    }
    let namespace = &parts[..parts.len() - 2];
    if namespace.iter().any(|s| s.is_empty()) {
        return Err(err());
    }
    Ok(format!("{}/{}", namespace.join("/"), name))
}

/// ROS type name to DDS mangled type name (per rmw_zenoh `_create_type_name`; same rule for srv types).
pub fn ros_type_to_dds(ros: &str) -> Result<String, KeyExprError> {
    let err = || KeyExprError::InvalidTypeName(ros.to_owned());
    let parts: Vec<&str> = ros.split('/').collect();
    if parts.len() < 2 || parts.iter().any(|s| s.is_empty()) {
        return Err(err());
    }
    let (name, namespace) = parts.split_last().expect("len >= 2");
    Ok(format!("{}::dds_::{}_", namespace.join("::"), name))
}

/// Parsed topic-data key expression `<domain>/<topic>/<DDS type>/<type hash>`.
#[derive(Debug, PartialEq, Eq)]
pub struct TopicKeyExpr {
    pub domain_id: u32,
    /// ROS form with leading `/` restored (e.g. `/robot1/scan`).
    pub topic: String,
    /// Kept in DDS mangled form.
    pub type_name: String,
    /// `RIHS01_<hex>` string.
    pub type_hash: String,
}

/// Build a topic-data key expression `<domain>/<topic>/<DDS type>/<type hash>` (topic is ROS form with leading `/`).
pub fn build_topic_keyexpr(
    domain_id: u32,
    topic: &str,
    type_name: &str,
    type_hash: &str,
) -> String {
    let topic = topic.strip_prefix('/').unwrap_or(topic);
    format!("{domain_id}/{topic}/{type_name}/{type_hash}")
}

/// Parse a topic-data key expression (join the middle segments since inner `/` in a topic name spans several).
pub fn parse_topic_keyexpr(key: &str) -> Result<TopicKeyExpr, KeyExprError> {
    let parts: Vec<&str> = key.split('/').collect();
    if parts.len() < 4 {
        return Err(KeyExprError::SegmentCount {
            key: key.to_owned(),
            actual: parts.len(),
        });
    }
    let domain_id = parts[0]
        .parse::<u32>()
        .map_err(|_| KeyExprError::InvalidDomainId(parts[0].to_owned()))?;
    let type_hash = parts[parts.len() - 1].to_owned();
    let type_name = parts[parts.len() - 2].to_owned();
    let topic = format!("/{}", parts[1..parts.len() - 2].join("/"));
    Ok(TopicKeyExpr {
        domain_id,
        topic,
        type_name,
        type_hash,
    })
}

/// Subscription QoS compatible with any publisher (BEST_EFFORT+VOLATILE+defaults; the 6-part `:`-separated form per rmw_zenoh 0.2.9 qos_to_keyexpr).
pub const COMPATIBLE_SUBSCRIPTION_QOS: &str = "2::,:,:,:,,";

/// Node name we advertise in the subscriber liveliness token (shown in the rmw_zenoh graph).
pub const VIEWER_NODE_NAME: &str = "visor";

/// Build a subscriber (MS) liveliness token (per rmw_zenoh 0.2.9 liveliness_utils.cpp; declaring it clears the publisher's subscriber-count guard).
#[allow(clippy::too_many_arguments)]
pub fn build_subscription_token(
    domain_id: u32,
    zid: &str,
    nid: u64,
    eid: u64,
    node_name: &str,
    topic: &str,
    dds_type: &str,
    type_hash: &str,
    qos: &str,
) -> String {
    let enclave = mangle_name("/");
    let namespace = mangle_name("/");
    format!(
        "@ros2_lv/{domain_id}/{zid}/{nid}/{eid}/MS/{enclave}/{namespace}/{}/{}/{}/{}/{qos}",
        mangle_name(node_name),
        mangle_name(topic),
        mangle_name(dds_type),
        mangle_name(type_hash),
    )
}

/// Entity kind in a liveliness token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    /// NN
    Node,
    /// MP
    Publisher,
    /// MS
    Subscription,
    /// SS
    Service,
    /// SC
    Client,
}

impl EntityKind {
    fn from_entity_str(s: &str) -> Option<Self> {
        match s {
            "NN" => Some(Self::Node),
            "MP" => Some(Self::Publisher),
            "MS" => Some(Self::Subscription),
            "SS" => Some(Self::Service),
            "SC" => Some(Self::Client),
            _ => None,
        }
    }
}

/// Topic (or service) info carried in a liveliness token.
#[derive(Debug, PartialEq, Eq)]
pub struct TopicEntityInfo {
    /// Demangled ROS form (e.g. `/cmd_vel`).
    pub name: String,
    /// Demangled DDS form (e.g. `std_msgs::msg::dds_::String_`).
    pub type_name: String,
    /// `RIHS01_<hex>`
    pub type_hash: String,
    /// Raw QoS segment string (interpreted later).
    pub qos_raw: String,
}

/// Parsed liveliness token `@ros2_lv/...`.
#[derive(Debug, PartialEq, Eq)]
pub struct LivelinessToken {
    pub domain_id: u32,
    pub zid: String,
    pub nid: String,
    pub id: String,
    pub kind: EntityKind,
    /// Demangled.
    pub enclave: String,
    /// Demangled (e.g. `/`).
    pub namespace: String,
    pub node_name: String,
    /// `None` for `Node`.
    pub topic: Option<TopicEntityInfo>,
}

impl LivelinessToken {
    /// Full node name joining namespace and node_name (e.g. `/robot1/controller`).
    pub fn full_node_name(&self) -> String {
        if self.namespace.ends_with('/') {
            format!("{}{}", self.namespace, self.node_name)
        } else {
            format!("{}/{}", self.namespace, self.node_name)
        }
    }
}

/// Segment count of a node token (@ros2_lv through node_name).
const NODE_TOKEN_SEGMENTS: usize = 9;
/// Segment count of a topic/service token (through qos; QoS always contains delimiters so it is never empty).
const TOPIC_TOKEN_SEGMENTS: usize = 13;

/// Parse a liveliness token (names are mangled, so fixed-position parsing after `/` split works).
pub fn parse_liveliness_token(key: &str) -> Result<LivelinessToken, KeyExprError> {
    let parts: Vec<&str> = key.split('/').collect();
    if parts.len() < NODE_TOKEN_SEGMENTS {
        return Err(KeyExprError::SegmentCount {
            key: key.to_owned(),
            actual: parts.len(),
        });
    }
    if parts[0] != "@ros2_lv" {
        return Err(KeyExprError::InvalidAdminSpace(key.to_owned()));
    }
    let kind = EntityKind::from_entity_str(parts[5])
        .ok_or_else(|| KeyExprError::UnknownEntityKind(parts[5].to_owned()))?;
    let expected = if kind == EntityKind::Node {
        NODE_TOKEN_SEGMENTS
    } else {
        TOPIC_TOKEN_SEGMENTS
    };
    if parts.len() != expected {
        return Err(KeyExprError::SegmentCount {
            key: key.to_owned(),
            actual: parts.len(),
        });
    }
    let domain_id = parts[1]
        .parse::<u32>()
        .map_err(|_| KeyExprError::InvalidDomainId(parts[1].to_owned()))?;
    let topic = if kind == EntityKind::Node {
        None
    } else {
        Some(TopicEntityInfo {
            name: demangle_name(parts[9]),
            type_name: demangle_name(parts[10]),
            type_hash: demangle_name(parts[11]),
            qos_raw: parts[12].to_owned(),
        })
    };
    Ok(LivelinessToken {
        domain_id,
        zid: parts[2].to_owned(),
        nid: parts[3].to_owned(),
        id: parts[4].to_owned(),
        kind,
        enclave: demangle_name(parts[6]),
        namespace: demangle_name(parts[7]),
        node_name: demangle_name(parts[8]),
        topic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dds_to_ros_message_types() {
        assert_eq!(
            dds_type_to_ros("sensor_msgs::msg::dds_::LaserScan_").unwrap(),
            "sensor_msgs/msg/LaserScan"
        );
        assert_eq!(
            dds_type_to_ros("std_msgs::msg::dds_::String_").unwrap(),
            "std_msgs/msg/String"
        );
    }

    #[test]
    fn dds_to_ros_service_type() {
        assert_eq!(
            dds_type_to_ros("example_interfaces::srv::dds_::AddTwoInts_Request_").unwrap(),
            "example_interfaces/srv/AddTwoInts_Request"
        );
    }

    #[test]
    fn ros_to_dds_message_types() {
        assert_eq!(
            ros_type_to_dds("sensor_msgs/msg/LaserScan").unwrap(),
            "sensor_msgs::msg::dds_::LaserScan_"
        );
        assert_eq!(
            ros_type_to_dds("std_msgs/msg/String").unwrap(),
            "std_msgs::msg::dds_::String_"
        );
    }

    #[test]
    fn type_name_roundtrip() {
        for ros in [
            "sensor_msgs/msg/LaserScan",
            "tf2_msgs/msg/TFMessage",
            "nav_msgs/msg/Path",
        ] {
            assert_eq!(
                dds_type_to_ros(&ros_type_to_dds(ros).unwrap()).unwrap(),
                ros
            );
        }
    }

    #[test]
    fn dds_to_ros_rejects_invalid() {
        assert_eq!(
            dds_type_to_ros("std_msgs::msg::String_"),
            Err(KeyExprError::InvalidTypeName(
                "std_msgs::msg::String_".into()
            ))
        );
        assert!(dds_type_to_ros("std_msgs::msg::dds_::String").is_err());
        assert!(dds_type_to_ros("dds_::String_").is_err());
        assert!(dds_type_to_ros("String_").is_err());
    }

    #[test]
    fn ros_to_dds_rejects_invalid() {
        assert!(ros_type_to_dds("LaserScan").is_err());
        assert!(ros_type_to_dds("sensor_msgs//LaserScan").is_err());
        assert!(ros_type_to_dds("").is_err());
    }

    #[test]
    fn topic_keyexpr_root_topic() {
        let t = parse_topic_keyexpr("0/chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18").unwrap();
        assert_eq!(t.domain_id, 0);
        assert_eq!(t.topic, "/chatter");
        assert_eq!(t.type_name, "std_msgs::msg::dds_::String_");
        assert!(t.type_hash.starts_with("RIHS01_"));
    }

    #[test]
    fn topic_keyexpr_with_inner_slashes() {
        let t = parse_topic_keyexpr(
            "0/robot1/sensors/scan/sensor_msgs::msg::dds_::LaserScan_/RIHS01_xxx",
        )
        .unwrap();
        assert_eq!(t.topic, "/robot1/sensors/scan");
        assert_eq!(t.type_name, "sensor_msgs::msg::dds_::LaserScan_");
        assert_eq!(t.type_hash, "RIHS01_xxx");
    }

    #[test]
    fn topic_keyexpr_nonzero_domain() {
        let t = parse_topic_keyexpr("42/tf/tf2_msgs::msg::dds_::TFMessage_/RIHS01_yyy").unwrap();
        assert_eq!(t.domain_id, 42);
        assert_eq!(t.topic, "/tf");
    }

    #[test]
    fn build_topic_keyexpr_roundtrip() {
        let key = build_topic_keyexpr(
            0,
            "/robot1/sensors/scan",
            "sensor_msgs::msg::dds_::LaserScan_",
            "RIHS01_xxx",
        );
        assert_eq!(
            key,
            "0/robot1/sensors/scan/sensor_msgs::msg::dds_::LaserScan_/RIHS01_xxx"
        );
        let parsed = parse_topic_keyexpr(&key).unwrap();
        assert_eq!(parsed.topic, "/robot1/sensors/scan");
        assert_eq!(
            build_topic_keyexpr(42, "/tf", "tf2_msgs::msg::dds_::TFMessage_", "RIHS01_yyy"),
            "42/tf/tf2_msgs::msg::dds_::TFMessage_/RIHS01_yyy"
        );
    }

    #[test]
    fn topic_keyexpr_rejects_invalid() {
        assert_eq!(
            parse_topic_keyexpr("0/std_msgs::msg::dds_::String_/RIHS01_x"),
            Err(KeyExprError::SegmentCount {
                key: "0/std_msgs::msg::dds_::String_/RIHS01_x".into(),
                actual: 3,
            })
        );
        assert_eq!(
            parse_topic_keyexpr("abc/chatter/std_msgs::msg::dds_::String_/RIHS01_x"),
            Err(KeyExprError::InvalidDomainId("abc".into()))
        );
    }

    const ZID: &str = "f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6";

    #[test]
    fn liveliness_node_token() {
        let key = format!("@ros2_lv/0/{ZID}/0/0/NN/%/%/talker");
        let t = parse_liveliness_token(&key).unwrap();
        assert_eq!(t.domain_id, 0);
        assert_eq!(t.zid, ZID);
        assert_eq!(t.nid, "0");
        assert_eq!(t.id, "0");
        assert_eq!(t.kind, EntityKind::Node);
        assert_eq!(t.enclave, "/");
        assert_eq!(t.namespace, "/");
        assert_eq!(t.node_name, "talker");
        assert_eq!(t.full_node_name(), "/talker");
        assert_eq!(t.topic, None);
    }

    #[test]
    fn liveliness_publisher_token() {
        let key = format!(
            "@ros2_lv/0/{ZID}/0/10/MP/%/%/talker/%chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18/::,:,:,,"
        );
        let t = parse_liveliness_token(&key).unwrap();
        assert_eq!(t.kind, EntityKind::Publisher);
        let topic = t.topic.unwrap();
        assert_eq!(topic.name, "/chatter");
        assert_eq!(topic.type_name, "std_msgs::msg::dds_::String_");
        assert!(topic.type_hash.starts_with("RIHS01_"));
        assert_eq!(topic.qos_raw, "::,:,:,,");
    }

    #[test]
    fn liveliness_namespaced_tokens() {
        let key = format!("@ros2_lv/0/{ZID}/1/5/NN/%/%robot1/controller");
        let t = parse_liveliness_token(&key).unwrap();
        assert_eq!(t.namespace, "/robot1");
        assert_eq!(t.full_node_name(), "/robot1/controller");

        let key = format!(
            "@ros2_lv/0/{ZID}/1/6/MS/%/%robot1%sensors/lidar/%robot1%sensors%scan/sensor_msgs::msg::dds_::LaserScan_/RIHS01_xxx/2:1:2,10"
        );
        let t = parse_liveliness_token(&key).unwrap();
        assert_eq!(t.kind, EntityKind::Subscription);
        assert_eq!(t.namespace, "/robot1/sensors");
        assert_eq!(t.full_node_name(), "/robot1/sensors/lidar");
        let topic = t.topic.unwrap();
        assert_eq!(topic.name, "/robot1/sensors/scan");
        assert_eq!(topic.qos_raw, "2:1:2,10");
    }

    #[test]
    fn liveliness_service_and_client_kinds() {
        for (entity, kind) in [("SS", EntityKind::Service), ("SC", EntityKind::Client)] {
            let key = format!(
                "@ros2_lv/0/{ZID}/0/20/{entity}/%/%/adder/%add_two_ints/example_interfaces::srv::dds_::AddTwoInts_Request_/RIHS01_zzz/::,:,:,,"
            );
            let t = parse_liveliness_token(&key).unwrap();
            assert_eq!(t.kind, kind);
            assert_eq!(t.topic.unwrap().name, "/add_two_ints");
        }
    }

    #[test]
    fn liveliness_rejects_invalid() {
        assert_eq!(
            parse_liveliness_token(&format!("@wrong/0/{ZID}/0/0/NN/%/%/talker")),
            Err(KeyExprError::InvalidAdminSpace(format!(
                "@wrong/0/{ZID}/0/0/NN/%/%/talker"
            )))
        );
        assert_eq!(
            parse_liveliness_token(&format!("@ros2_lv/0/{ZID}/0/0/XX/%/%/talker")),
            Err(KeyExprError::UnknownEntityKind("XX".into()))
        );
        assert!(matches!(
            parse_liveliness_token("@ros2_lv/0/zid/0/0/NN/%/%"),
            Err(KeyExprError::SegmentCount { actual: 8, .. })
        ));
        assert!(matches!(
            parse_liveliness_token(&format!("@ros2_lv/0/{ZID}/0/0/NN/%/%/talker/extra")),
            Err(KeyExprError::SegmentCount { actual: 10, .. })
        ));
        assert!(matches!(
            parse_liveliness_token(&format!(
                "@ros2_lv/0/{ZID}/0/10/MP/%/%/talker/%chatter/std_msgs::msg::dds_::String_/RIHS01_x"
            )),
            Err(KeyExprError::SegmentCount { actual: 12, .. })
        ));
        assert_eq!(
            parse_liveliness_token(&format!("@ros2_lv/abc/{ZID}/0/0/NN/%/%/talker")),
            Err(KeyExprError::InvalidDomainId("abc".into()))
        );
    }

    #[test]
    fn subscription_token_roundtrips_through_parser() {
        let key = build_subscription_token(
            32,
            ZID,
            0,
            7,
            "visor",
            "/rldss_tracker/edge_map",
            "sensor_msgs::msg::dds_::PointCloud2_",
            "RIHS01_9198cabf",
            COMPATIBLE_SUBSCRIPTION_QOS,
        );
        let t = parse_liveliness_token(&key).unwrap();
        assert_eq!(t.domain_id, 32);
        assert_eq!(t.zid, ZID);
        assert_eq!(t.nid, "0");
        assert_eq!(t.id, "7");
        assert_eq!(t.kind, EntityKind::Subscription);
        assert_eq!(t.namespace, "/");
        assert_eq!(t.node_name, "visor");
        assert_eq!(t.full_node_name(), "/visor");
        let topic = t.topic.unwrap();
        assert_eq!(topic.name, "/rldss_tracker/edge_map");
        assert_eq!(topic.type_name, "sensor_msgs::msg::dds_::PointCloud2_");
        assert_eq!(topic.type_hash, "RIHS01_9198cabf");
        assert_eq!(topic.qos_raw, COMPATIBLE_SUBSCRIPTION_QOS);
    }

    #[test]
    fn compatible_subscription_qos_has_six_colon_parts() {
        assert_eq!(COMPATIBLE_SUBSCRIPTION_QOS.split(':').count(), 6);
        let parts: Vec<&str> = COMPATIBLE_SUBSCRIPTION_QOS.split(':').collect();
        assert_eq!(parts[0], "2");
        assert_eq!(parts[1], "");
        assert_eq!(parts[2].split(',').count(), 2);
        assert_eq!(parts[5].split(',').count(), 3);
    }

    #[test]
    fn mangle_demangle_roundtrip() {
        assert_eq!(mangle_name("/cmd_vel"), "%cmd_vel");
        assert_eq!(demangle_name("%cmd_vel"), "/cmd_vel");
        assert_eq!(
            demangle_name(&mangle_name("/robot1/sensors/scan")),
            "/robot1/sensors/scan"
        );
        assert_eq!(mangle_name("/"), "%");
        assert_eq!(demangle_name("%"), "/");
        assert_eq!(mangle_name(""), "");
        assert_eq!(demangle_name(""), "");
    }
}
