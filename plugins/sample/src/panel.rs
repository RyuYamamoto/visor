//! Fleet dock panel: tabulates the snapshot the renderer decoded (a panel cannot subscribe on its own).

use serde::{Deserialize, Serialize};
use visor::plugin::*;

use crate::SharedFleet;

/// Persisted panel state (stored under the config's `[[panels]]` entry for this panel's key).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FleetPanelSettings {
    /// Show the pose columns as well as name and battery.
    pub show_pose: bool,
}

pub struct FleetPanel {
    shared: SharedFleet,
    settings: FleetPanelSettings,
}

impl FleetPanel {
    pub fn new(shared: SharedFleet) -> Self {
        Self {
            shared,
            settings: FleetPanelSettings::default(),
        }
    }
}

impl PanelPlugin for FleetPanel {
    fn ui(&mut self, ui: &mut egui::Ui, ctx: &PanelContext<'_>) {
        let p = theme::ui::palette();
        ui.checkbox(&mut self.settings.show_pose, "Show pose");
        ui.separator();
        let Ok(state) = self.shared.lock() else {
            ui.colored_label(p.status_error, "shared state poisoned");
            return;
        };
        if state.robots.is_empty() {
            ui.colored_label(
                p.text_muted,
                "no fleet data — add a FleetState display first",
            );
            return;
        }
        ui.label(
            theme::machine_value(format!(
                "{} robot(s) in `{}`{}",
                state.robots.len(),
                state.frame_id,
                if ctx.is_playback { "  (replay)" } else { "" }
            ))
            .color(p.text_muted),
        );
        egui::ScrollArea::vertical().show(ui, |ui| {
            for robot in &state.robots {
                ui.horizontal(|ui| {
                    let color = if robot.battery < 20.0 {
                        p.status_warn
                    } else {
                        p.accent
                    };
                    ui.label(egui::RichText::new(&robot.name).color(color));
                    ui.label(
                        theme::machine_value(format!("{:.0}%", robot.battery)).color(p.text_muted),
                    );
                    if self.settings.show_pose {
                        let t = robot.pose.translation;
                        ui.label(
                            theme::machine_value(format!("{:.2}, {:.2}, {:.2}", t.x, t.y, t.z))
                                .color(p.instrument),
                        );
                    }
                });
            }
        });
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(&self.settings).ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(settings) = value.clone().try_into::<FleetPanelSettings>() {
            self.settings = settings;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FleetState;
    use std::sync::{Arc, Mutex};

    #[test]
    fn panel_settings_round_trip_through_toml() {
        let shared = Arc::new(Mutex::new(FleetState::default()));
        let mut panel = FleetPanel::new(Arc::clone(&shared));
        panel.settings.show_pose = true;
        let saved = panel.settings().expect("settings");
        let mut restored = FleetPanel::new(shared);
        restored.apply_settings(&saved);
        assert!(restored.settings.show_pose);
    }
}
