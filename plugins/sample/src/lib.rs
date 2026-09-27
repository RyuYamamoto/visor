//! Sample visor plugin: a custom `.msg` type, a 3D renderer for it, and a dock panel sharing the renderer's state.

mod panel;
mod renderer;

use std::sync::{Arc, Mutex};

use visor::plugin::*;

pub use panel::FleetPanel;
pub use renderer::FleetRenderer;

/// ROS-form name of the custom type this plugin brings with it.
pub const FLEET_STATE_TYPE: &str = "sample_msgs/msg/FleetState";

/// One robot from the most recent FleetState message.
#[derive(Debug, Clone, PartialEq)]
pub struct RobotState {
    pub name: String,
    pub pose: nalgebra::Isometry3<f64>,
    pub battery: f32,
}

/// Latest fleet snapshot, shared between the renderer that decodes it and the panel that tabulates it.
#[derive(Debug, Default)]
pub struct FleetState {
    pub frame_id: String,
    pub stamp: TimeNs,
    pub robots: Vec<RobotState>,
}

/// Shared handle to the snapshot. A panel cannot subscribe on its own, so this is how it gets data.
pub type SharedFleet = Arc<Mutex<FleetState>>;

/// The plugin itself. One instance owns the state both extensions read.
pub struct SamplePlugin {
    shared: SharedFleet,
}

impl Default for SamplePlugin {
    fn default() -> Self {
        Self {
            shared: Arc::new(Mutex::new(FleetState::default())),
        }
    }
}

impl Plugin for SamplePlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo {
            id: "sample",
            name: "Visor Sample",
            version: env!("CARGO_PKG_VERSION"),
            api_version: PLUGIN_API_VERSION,
        }
    }

    fn register(&self, registrar: &mut Registrar<'_>) {
        registrar.msg(
            FLEET_STATE_TYPE,
            include_str!("../msgs/sample_msgs/msg/FleetState.msg"),
        );
        registrar.msg(
            "sample_msgs/msg/FleetRobot",
            include_str!("../msgs/sample_msgs/msg/FleetRobot.msg"),
        );
        let shared = Arc::clone(&self.shared);
        registrar.renderer(
            RendererDescriptor::topic(FLEET_STATE_TYPE, "FleetState", move || {
                Box::new(FleetRenderer::new(Arc::clone(&shared)))
            })
            .with_accent(theme::ACCENT_PURPLE),
        );
        let shared = Arc::clone(&self.shared);
        registrar.panel(PanelDescriptor::new("fleet", "Fleet", move || {
            Box::new(FleetPanel::new(Arc::clone(&shared)))
        }));
    }
}
