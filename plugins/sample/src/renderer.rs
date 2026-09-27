//! 3D renderer for sample_msgs/FleetState: one arrow plus an optional name/battery label per robot.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use visor::plugin::*;

use crate::{FleetState, RobotState, SharedFleet};

/// Persisted per-item settings (opaque toml under the config's display entry).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FleetSettings {
    pub show_labels: bool,
    /// Arrow shaft length [m].
    pub arrow_len: f32,
    /// Label height [m].
    pub label_height: f32,
}

impl Default for FleetSettings {
    fn default() -> Self {
        Self {
            show_labels: true,
            arrow_len: 0.6,
            label_height: 0.2,
        }
    }
}

/// Renderer state: the shared snapshot plus the baked batches and the generation that gates GPU upload.
pub struct FleetRenderer {
    shared: SharedFleet,
    settings: FleetSettings,
    /// Bumped only when the batches are rebuilt, so an unchanged frame skips the upload.
    generation: u64,
    /// Set when a new message or a settings change invalidated the bake.
    dirty: bool,
    mesh: Arc<Vec<Vertex>>,
    labels: Arc<Vec<Label>>,
    /// Fixed frame the current bake was made for; a change to it forces a rebake.
    baked_for: Option<String>,
    received: bool,
}

impl FleetRenderer {
    pub fn new(shared: SharedFleet) -> Self {
        Self {
            shared,
            settings: FleetSettings::default(),
            generation: 0,
            dirty: true,
            mesh: Arc::new(Vec::new()),
            labels: Arc::new(Vec::new()),
            baked_for: None,
            received: false,
        }
    }

    /// Bake arrows and labels into fixed-frame coordinates, bumping the generation because the contents changed.
    fn bake(&mut self, tf: &TfContext<'_>, state: &FleetState) -> Result<(), RenderStatus> {
        let Some(pose) = tf.resolve_at(&state.frame_id, state.stamp) else {
            return Err(RenderStatus::TfUnavailable {
                frame: state.frame_id.clone(),
            });
        };
        let mut mesh = Vec::with_capacity(state.robots.len() * arrow_mesh_vertex_count());
        let mut labels = Vec::with_capacity(state.robots.len());
        for robot in &state.robots {
            let world = pose * robot.pose;
            let iso = nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(
                    world.translation.x as f32,
                    world.translation.y as f32,
                    world.translation.z as f32,
                ),
                nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                    world.rotation.w as f32,
                    world.rotation.i as f32,
                    world.rotation.j as f32,
                    world.rotation.k as f32,
                )),
            );
            push_arrow_mesh(
                &mut mesh,
                &iso,
                self.settings.arrow_len,
                battery_color(robot.battery),
            );
            if self.settings.show_labels {
                let origin = iso * nalgebra::Point3::origin();
                labels.push(Label {
                    position: [origin.x, origin.y, origin.z + self.settings.arrow_len],
                    text: format!("{} {:.0}%", robot.name, robot.battery),
                    color: theme::to_linear_rgba(theme::LABEL_TEXT),
                    height_m: self.settings.label_height,
                });
            }
        }
        self.mesh = Arc::new(mesh);
        self.labels = Arc::new(labels);
        self.generation += 1;
        self.dirty = false;
        self.baked_for = Some(tf.fixed_frame.to_owned());
        Ok(())
    }
}

impl Renderer for FleetRenderer {
    fn on_message(&mut self, value: &Value) {
        let Ok((frame_id, stamp)) = extract_header(value) else {
            return;
        };
        let Some(Value::Array(entries)) = value.get("robots") else {
            return;
        };
        let robots: Vec<RobotState> = entries.iter().filter_map(parse_robot).collect();
        if let Ok(mut state) = self.shared.lock() {
            state.frame_id = frame_id;
            state.stamp = stamp;
            state.robots = robots;
        }
        self.received = true;
        self.dirty = true;
    }

    fn scene(&mut self, tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
        if !self.received {
            return Err(RenderStatus::NoData);
        }
        let snapshot = match self.shared.lock() {
            Ok(state) => FleetState {
                frame_id: state.frame_id.clone(),
                stamp: state.stamp,
                robots: state.robots.clone(),
            },
            Err(_) => return Err(RenderStatus::InvalidMessage("shared state poisoned".into())),
        };
        if snapshot.frame_id.is_empty() {
            return Err(RenderStatus::NoData);
        }
        if self.dirty || self.baked_for.as_deref() != Some(tf.fixed_frame) {
            self.bake(tf, &snapshot)?;
        }
        let mut batches = vec![SceneBatch::mesh(Arc::clone(&self.mesh), self.generation)];
        if !self.labels.is_empty() {
            batches.push(SceneBatch::labels(
                Arc::clone(&self.labels),
                self.generation,
            ));
        }
        Ok(batches)
    }

    fn reset(&mut self) {
        if let Ok(mut state) = self.shared.lock() {
            state.robots.clear();
        }
        self.received = false;
        self.dirty = true;
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let before = self.settings.clone();
        ui.checkbox(&mut self.settings.show_labels, "Show labels");
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Arrow (m)").color(theme::ui::palette().text_muted));
            ui.add(
                egui::DragValue::new(&mut self.settings.arrow_len)
                    .range(0.05..=5.0)
                    .speed(0.05),
            );
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Label (m)").color(theme::ui::palette().text_muted));
            ui.add(
                egui::DragValue::new(&mut self.settings.label_height)
                    .range(0.02..=2.0)
                    .speed(0.02),
            );
        });
        if self.settings != before {
            self.dirty = true;
        }
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(&self.settings).ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(settings) = value.clone().try_into::<FleetSettings>() {
            self.settings = settings;
            self.dirty = true;
        }
    }
}

/// Battery level as a colour: the theme's cyan when full, warning amber when low.
fn battery_color(battery: f32) -> [f32; 4] {
    let color = if battery < 20.0 {
        theme::ACCENT_AMBER
    } else {
        theme::ACCENT_CYAN
    };
    theme::to_linear_rgba(color)
}

/// One FleetRobot element; a malformed entry is skipped rather than failing the whole message.
fn parse_robot(value: &Value) -> Option<RobotState> {
    let Some(Value::String(name)) = value.get("name") else {
        return None;
    };
    let pose = extract_pose(value.get("pose")?).ok()?;
    let battery = match value.get("battery") {
        Some(Value::F32(v)) => *v,
        Some(Value::F64(v)) => *v as f32,
        _ => 0.0,
    };
    Some(RobotState {
        name: name.clone(),
        pose,
        battery,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn f64_struct(fields: &[(&str, f64)]) -> Value {
        Value::Struct(
            fields
                .iter()
                .map(|(n, v)| ((*n).to_owned(), Value::F64(*v)))
                .collect(),
        )
    }

    fn robot(name: &str, x: f64, battery: f32) -> Value {
        Value::Struct(vec![
            ("name".to_owned(), Value::String(name.to_owned())),
            (
                "pose".to_owned(),
                Value::Struct(vec![
                    (
                        "position".to_owned(),
                        f64_struct(&[("x", x), ("y", 0.0), ("z", 0.0)]),
                    ),
                    (
                        "orientation".to_owned(),
                        f64_struct(&[("x", 0.0), ("y", 0.0), ("z", 0.0), ("w", 1.0)]),
                    ),
                ]),
            ),
            ("battery".to_owned(), Value::F32(battery)),
        ])
    }

    fn fleet_message(robots: Vec<Value>) -> Value {
        Value::Struct(vec![
            (
                "header".to_owned(),
                Value::Struct(vec![
                    (
                        "stamp".to_owned(),
                        Value::Struct(vec![
                            ("sec".to_owned(), Value::I32(0)),
                            ("nanosec".to_owned(), Value::U32(0)),
                        ]),
                    ),
                    ("frame_id".to_owned(), Value::String("map".to_owned())),
                ]),
            ),
            ("robots".to_owned(), Value::Array(robots)),
        ])
    }

    fn shared() -> SharedFleet {
        Arc::new(Mutex::new(FleetState::default()))
    }

    #[test]
    fn a_message_lands_in_the_shared_state_the_panel_reads() {
        let shared = shared();
        let mut renderer = FleetRenderer::new(Arc::clone(&shared));
        renderer.on_message(&fleet_message(vec![
            robot("amr_1", 1.0, 80.0),
            robot("amr_2", 2.0, 10.0),
        ]));
        let state = shared.lock().unwrap();
        assert_eq!(state.frame_id, "map");
        assert_eq!(state.robots.len(), 2);
        assert_eq!(state.robots[0].name, "amr_1");
        assert_eq!(state.robots[1].battery, 10.0);
        assert_eq!(state.robots[1].pose.translation.x, 2.0);
    }

    #[test]
    fn malformed_entries_are_skipped_without_losing_the_rest() {
        let shared = shared();
        let mut renderer = FleetRenderer::new(Arc::clone(&shared));
        renderer.on_message(&fleet_message(vec![
            Value::Struct(vec![(
                "name".to_owned(),
                Value::String("no_pose".to_owned()),
            )]),
            robot("amr_1", 1.0, 50.0),
        ]));
        let state = shared.lock().unwrap();
        assert_eq!(state.robots.len(), 1);
        assert_eq!(state.robots[0].name, "amr_1");
    }

    #[test]
    fn no_message_reports_no_data_and_a_bake_bumps_the_generation_once() {
        let shared = shared();
        let mut renderer = FleetRenderer::new(Arc::clone(&shared));
        let buffer = TfBuffer::new();
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        assert!(matches!(renderer.scene(&tf), Err(RenderStatus::NoData)));
        renderer.on_message(&fleet_message(vec![robot("amr_1", 1.0, 80.0)]));
        // "map" resolves to itself, so no TF sample is needed to bake.
        let first = renderer.scene(&tf).expect("baked");
        assert_eq!(first[0].generation, 1);
        assert_eq!(first.len(), 2, "mesh plus labels");
        let second = renderer.scene(&tf).expect("baked");
        assert_eq!(second[0].generation, 1, "unchanged data must not re-bake");
    }

    #[test]
    fn settings_round_trip_and_force_a_rebake() {
        let shared = shared();
        let mut renderer = FleetRenderer::new(Arc::clone(&shared));
        renderer.settings = FleetSettings {
            show_labels: false,
            arrow_len: 1.5,
            label_height: 0.4,
        };
        let saved = renderer.settings().expect("settings");
        let mut restored = FleetRenderer::new(shared);
        restored.apply_settings(&saved);
        assert_eq!(restored.settings, renderer.settings);
        assert!(restored.dirty);
    }

    #[test]
    fn a_low_battery_robot_is_coloured_differently() {
        assert_ne!(battery_color(10.0), battery_color(90.0));
    }
}
