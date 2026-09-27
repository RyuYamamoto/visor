# About the bundled .msg definitions

The standard ROS 2 Jazzy message type definitions (`.msg`) are bundled as-is (unmodified)
as the type-information source for dynamic CDR decoding.

- Source: `/opt/ros/jazzy/share/<pkg>/msg/*.msg` (retrieved: 2026-07-21)
- License: Apache License 2.0 (see `LICENSE` in this directory)

| Package | Version | Upstream repo | Files |
|---|---|---|---|
| builtin_interfaces | 2.0.3 | ros2/rcl_interfaces | 2 |
| std_msgs | 5.3.6 | ros2/common_interfaces | 30 |
| geometry_msgs | 5.3.6 | ros2/common_interfaces | 32 |
| sensor_msgs | 5.3.6 | ros2/common_interfaces | 27 |
| nav_msgs | 5.3.6 | ros2/common_interfaces | 6 |
| tf2_msgs | 0.36.19 | ros2/geometry2 | 2 |
| map_msgs | 2.1.0 | ros-planning/navigation_msgs | 1 |
| visualization_msgs | 5.3.6 | ros2/common_interfaces | 4 |

104 files in total. `build.rs` scans this directory and embeds them into the binary
(adding a `.msg` picks it up automatically on the next build).

## Note: distribution differences

Because the definitions are pinned to Jazzy, if a ROS 2 node from a different distribution
publishes a type with a different field layout, decoding may "succeed but produce corrupted
values" (the RIHS01_... type hash in the keyexpr is not verified; resolution is by type name
only). If a decode result looks wrong, suspect a type-definition version mismatch.
