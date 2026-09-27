//! Visor plugin drawing the trail a TF frame leaves behind in the fixed frame (jsk_rviz_plugins' TFTrajectory in spirit).

mod renderer;
mod trajectory;

use visor::plugin::*;

pub use renderer::{TfTrajectoryRenderer, TrajectorySettings};
pub use trajectory::{Limits, Sample, Trajectory};

/// The plugin itself; it registers one standalone renderer and subscribes to nothing.
#[derive(Debug, Default)]
pub struct TfTrajectoryPlugin;

impl Plugin for TfTrajectoryPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo {
            id: "tf_trajectory",
            name: "TF Trajectory",
            version: env!("CARGO_PKG_VERSION"),
            api_version: PLUGIN_API_VERSION,
        }
    }

    fn register(&self, registrar: &mut Registrar<'_>) {
        registrar.renderer(
            RendererDescriptor::standalone("TFTrajectory", || {
                Box::new(TfTrajectoryRenderer::default())
            })
            .with_accent(theme::ODOM_DEFAULT),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_one_standalone_display_without_problems() {
        let mut registry = Registry::builtin();
        registry.add_plugin(&TfTrajectoryPlugin);
        let registry = registry.finish(&|_| None);
        assert!(
            registry.problems().is_empty(),
            "problems={:?}",
            registry.problems()
        );
        let entry = registry
            .find_standalone("tf_trajectory", "TFTrajectory")
            .expect("offered in the Displays other-sources menu");
        assert_eq!(entry.descriptor.accent, Some(theme::ODOM_DEFAULT));
        // Standalone entries are never matched against topics, so the Add dialog must not list it.
        assert!(
            registry
                .find_renderer("/tf", "tf2_msgs/msg/TFMessage")
                .is_none()
        );
    }
}
