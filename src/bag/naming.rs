//! Topic/type name normalization between ROS 1 bags and the viewer's ROS 2 vocabulary (requirements §2.2 d2/d7, f2).

/// Legacy ROS 1 type names mapped to the modern name with the same wire structure; every entry needs a test (Q11 = B).
const TYPE_ALIASES: &[(&str, &str)] = &[
    // `tf/tfMessage` predates tf2 but is field-for-field identical, so aliasing the name is enough to reach the TF path.
    ("tf/tfMessage", "tf2_msgs/TFMessage"),
];

/// Normalize a bag topic name to absolute form (`odom` -> `/odom`); without it app.rs's exact `"/tf"` test never matches.
pub fn normalize_topic(raw: &str) -> String {
    let trimmed = raw.trim();
    let trimmed = trimmed.strip_suffix('/').unwrap_or(trimmed);
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.starts_with('/') {
        trimmed.to_owned()
    } else {
        format!("/{trimmed}")
    }
}

/// Resolve a ROS 1 type name (`pkg/Type`) to the viewer's `pkg/msg/Type` form, applying aliases first; already-normalized names pass through.
pub fn ros1_type_to_ros2(raw: &str) -> String {
    let raw = apply_alias(raw.trim());
    match raw.split('/').collect::<Vec<_>>().as_slice() {
        [pkg, ty] if !pkg.is_empty() && !ty.is_empty() => format!("{pkg}/msg/{ty}"),
        // Anything else (already 3-segment, or malformed) goes back untouched so the caller reports it as unknown.
        _ => raw.to_owned(),
    }
}

/// Replace a legacy ROS 1 type name with its modern equivalent (still in `pkg/Type` form).
fn apply_alias(raw: &str) -> &str {
    TYPE_ALIASES
        .iter()
        .find(|(from, _)| *from == raw)
        .map_or(raw, |(_, to)| *to)
}

/// Short type name for display (`sensor_msgs/msg/LaserScan` -> `LaserScan`).
pub fn short_type(ros_type: &str) -> &str {
    ros_type.rsplit('/').next().unwrap_or(ros_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_names_become_absolute() {
        assert_eq!(normalize_topic("/scan"), "/scan");
        // Relative names occur in real bags (requirements f2).
        assert_eq!(normalize_topic("odom"), "/odom");
        assert_eq!(normalize_topic("tf"), "/tf");
        assert_eq!(normalize_topic("robot/odom"), "/robot/odom");
        assert_eq!(normalize_topic("  /scan  "), "/scan");
        assert_eq!(normalize_topic("/scan/"), "/scan");
        assert_eq!(normalize_topic(""), "");
        assert_eq!(normalize_topic("/"), "");
    }

    #[test]
    fn type_names_gain_the_msg_segment() {
        assert_eq!(
            ros1_type_to_ros2("sensor_msgs/LaserScan"),
            "sensor_msgs/msg/LaserScan"
        );
        assert_eq!(
            ros1_type_to_ros2("prl_msgs/FollowTrajectoryGoal"),
            "prl_msgs/msg/FollowTrajectoryGoal"
        );
        // Idempotent, so re-normalizing an already-converted name is safe.
        assert_eq!(
            ros1_type_to_ros2("sensor_msgs/msg/LaserScan"),
            "sensor_msgs/msg/LaserScan"
        );
    }

    #[test]
    fn legacy_tf_type_is_aliased_onto_tf2() {
        assert_eq!(ros1_type_to_ros2("tf/tfMessage"), "tf2_msgs/msg/TFMessage");
        assert_eq!(
            ros1_type_to_ros2("tf2_msgs/TFMessage"),
            "tf2_msgs/msg/TFMessage"
        );
    }

    #[test]
    fn malformed_type_names_are_passed_through() {
        assert_eq!(ros1_type_to_ros2(""), "");
        assert_eq!(ros1_type_to_ros2("LaserScan"), "LaserScan");
        assert_eq!(ros1_type_to_ros2("a/b/c/d"), "a/b/c/d");
        assert_eq!(ros1_type_to_ros2("/LaserScan"), "/LaserScan");
    }

    #[test]
    fn short_type_takes_the_last_segment() {
        assert_eq!(short_type("sensor_msgs/msg/LaserScan"), "LaserScan");
        assert_eq!(short_type("LaserScan"), "LaserScan");
    }
}
