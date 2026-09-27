//! Renderer implementations (one display type = one file) and the builtin registration site.

pub mod laser_scan;
pub mod marker;
pub mod occupancy_grid;
pub mod odometry;
pub mod path;
pub mod point_cloud2;
pub mod urdf;

use crate::plugin::registry::{Registrar, RendererDescriptor};

/// Register the display types visor ships with, through exactly the same API a plugin uses (first match wins, so these resolve before plugins).
pub fn register_builtin(reg: &mut Registrar<'_>) {
    reg.renderer(RendererDescriptor::topic(
        "sensor_msgs/msg/LaserScan",
        "LaserScan",
        || Box::new(laser_scan::LaserScanRenderer::default()),
    ));
    reg.renderer(RendererDescriptor::topic(
        "sensor_msgs/msg/PointCloud2",
        "PointCloud2",
        || Box::new(point_cloud2::PointCloud2Renderer::default()),
    ));
    reg.renderer(RendererDescriptor::topic(
        "nav_msgs/msg/OccupancyGrid",
        "Map",
        || Box::new(occupancy_grid::OccupancyGridRenderer::default()),
    ));
    reg.renderer(RendererDescriptor::topic(
        "nav_msgs/msg/Path",
        "Path",
        || Box::new(path::PathRenderer::default()),
    ));
    reg.renderer(RendererDescriptor::topic(
        "nav_msgs/msg/Odometry",
        "Odometry",
        || Box::new(odometry::OdometryRenderer::default()),
    ));
    reg.renderer(RendererDescriptor::topic(
        "visualization_msgs/msg/Marker",
        "Marker",
        || Box::new(marker::MarkerRenderer::default()),
    ));
    reg.renderer(RendererDescriptor::topic(
        "visualization_msgs/msg/MarkerArray",
        "MarkerArray",
        || Box::new(marker::MarkerRenderer::default()),
    ));
    // Two entries, one label: the same display type reachable from a topic or from a file. Which `make` ran is how the renderer knows which of the two it is.
    reg.renderer(RendererDescriptor::topic_filtered(
        "std_msgs/msg/String",
        "RobotModel",
        || Box::new(urdf::UrdfRenderer::from_topic()),
        urdf::is_robot_description_topic,
    ));
    reg.renderer(RendererDescriptor::standalone("RobotModel", || {
        Box::new(urdf::UrdfRenderer::default())
    }));
}
