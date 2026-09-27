//! Odometry renderer (like RViz, lays out received poses as up-to-Keep 3D arrow meshes; keep/tolerance cap what is shown).

use std::collections::VecDeque;
use std::sync::Arc;

use egui::{Color32, RichText};
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};

use crate::decode::value::Value;
use crate::render::{
    RenderStatus, Renderer, SceneBatch, TfContext, Vertex, arrow_mesh_vertex_count, extract_header,
    extract_pose, push_arrow_mesh,
};
use crate::theme;

/// Default shaft length [m] (matches RViz Odometry's Shaft Length; head/radius scale with it).
const ARROW_LEN_DEFAULT: f32 = 1.0;
/// Default trail length (matches RViz Odometry's Keep default).
const KEEP_DEFAULT: usize = 100;
/// Default translation threshold [m] (matches RViz's Position Tolerance).
const POSITION_TOLERANCE_DEFAULT: f64 = 0.1;
/// Default rotation threshold [rad] (matches RViz's Angle Tolerance).
const ANGLE_TOLERANCE_DEFAULT: f64 = 0.1;

/// Latest pose extracted from an Odometry message (local coords relative to header.frame_id).
#[derive(Debug)]
struct OdomSample {
    frame_id: String,
    stamp: crate::tf::buffer::TimeNs,
    pose: Isometry3<f64>,
}

pub struct OdometryRenderer {
    latest: Option<OdomSample>,
    /// Accepted trail poses (held in source-frame coords; scene() bakes them all with one latest-stamp TF).
    trail: VecDeque<Isometry3<f64>>,
    parse_error: Option<String>,
    color: Color32,
    arrow_len: f32,
    keep: usize,
    position_tolerance: f64,
    angle_tolerance: f64,
    pose: Option<Isometry3<f32>>,
    pose_fixed: String,
    pose_dirty: bool,
    baked: Option<Arc<Vec<Vertex>>>,
    generation: u64,
    bake_dirty: bool,
}

impl Default for OdometryRenderer {
    fn default() -> Self {
        Self {
            latest: None,
            trail: VecDeque::new(),
            parse_error: None,
            color: theme::ODOM_DEFAULT,
            arrow_len: ARROW_LEN_DEFAULT,
            keep: KEEP_DEFAULT,
            position_tolerance: POSITION_TOLERANCE_DEFAULT,
            angle_tolerance: ANGLE_TOLERANCE_DEFAULT,
            pose: None,
            pose_fixed: String::new(),
            pose_dirty: false,
            baked: None,
            generation: 0,
            bake_dirty: false,
        }
    }
}

impl Renderer for OdometryRenderer {
    fn on_message(&mut self, value: &Value) {
        match extract_odom(value) {
            Ok(sample) => {
                // Prevent frame mixing: on a frame_id change, drop the old-frame trail.
                if self
                    .latest
                    .as_ref()
                    .is_some_and(|prev| prev.frame_id != sample.frame_id)
                {
                    self.trail.clear();
                }
                if should_append(
                    self.trail.back(),
                    &sample.pose,
                    self.position_tolerance,
                    self.angle_tolerance,
                ) {
                    self.trail.push_back(sample.pose);
                    while self.trail.len() > self.keep {
                        self.trail.pop_front();
                    }
                }
                self.latest = Some(sample);
                self.parse_error = None;
                self.pose_dirty = true;
                self.bake_dirty = true;
            }
            Err(e) => self.parse_error = Some(e),
        }
    }

    fn scene(&mut self, tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
        if let Some(error) = &self.parse_error {
            return Err(RenderStatus::InvalidMessage(error.clone()));
        }
        let Some(latest) = &self.latest else {
            return Err(RenderStatus::NoData);
        };
        if self.pose_dirty || self.pose_fixed != tf.fixed_frame {
            match tf.resolve_at(&latest.frame_id, latest.stamp) {
                Some(iso) => {
                    self.pose = Some(iso.cast::<f32>());
                    self.pose_fixed = tf.fixed_frame.to_owned();
                    self.pose_dirty = false;
                    self.bake_dirty = true;
                }
                None => {
                    if self.pose_fixed != tf.fixed_frame {
                        self.pose = None;
                        self.baked = None;
                        self.pose_fixed = tf.fixed_frame.to_owned();
                        self.pose_dirty = true;
                    }
                    // Keep emitting the old bake until TF catches up to the stamp (same convention as laser_scan).
                    if self.pose.is_none() {
                        return Err(RenderStatus::TfUnavailable {
                            frame: latest.frame_id.clone(),
                        });
                    }
                }
            }
        }
        let Some(pose) = &self.pose else {
            return Err(RenderStatus::TfUnavailable {
                frame: latest.frame_id.clone(),
            });
        };
        if !self.pose_dirty && self.bake_dirty {
            let color = theme::to_linear_rgba(self.color);
            let mut vertices = Vec::with_capacity(self.trail.len() * arrow_mesh_vertex_count());
            // Like RViz, lay out each accepted pose as a 3D arrow mesh (all baked with one latest-stamp TF).
            for iso in &self.trail {
                let world = *pose * iso.cast::<f32>();
                push_arrow_mesh(&mut vertices, &world, self.arrow_len, color);
            }
            self.baked = Some(Arc::new(vertices));
            self.generation += 1;
            self.bake_dirty = false;
        }
        match &self.baked {
            Some(vertices) => Ok(vec![SceneBatch::mesh(vertices.clone(), self.generation)]),
            None => Err(RenderStatus::TfUnavailable {
                frame: latest.frame_id.clone(),
            }),
        }
    }

    /// The trail is a history, so replaying from elsewhere must not draw a line joining the two positions.
    fn reset(&mut self) {
        self.trail.clear();
        self.latest = None;
        self.parse_error = None;
        self.pose = None;
        self.pose_dirty = true;
        self.bake_dirty = true;
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Color").color(p.text_muted));
            changed |= ui.color_edit_button_srgba(&mut self.color).changed();
            ui.label(RichText::new("Shaft (m)").color(p.text_muted));
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.arrow_len)
                        .range(0.05..=10.0)
                        .speed(0.05),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Keep").color(p.text_muted));
            if ui
                .add(egui::DragValue::new(&mut self.keep).range(1..=10_000))
                .changed()
            {
                while self.trail.len() > self.keep {
                    self.trail.pop_front();
                }
                changed = true;
            }
            if ui.button("Clear trail").clicked() {
                self.trail.clear();
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Tolerance (m / rad)").color(p.text_muted));
            ui.add(
                egui::DragValue::new(&mut self.position_tolerance)
                    .range(0.0..=10.0)
                    .speed(0.01),
            );
            ui.add(
                egui::DragValue::new(&mut self.angle_tolerance)
                    .range(0.0..=std::f64::consts::PI)
                    .speed(0.01),
            );
        });
        if changed {
            self.bake_dirty = true;
        }
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(OdometrySettings {
            color: self.color,
            arrow_len: self.arrow_len,
            keep: self.keep,
            position_tolerance: self.position_tolerance,
            angle_tolerance: self.angle_tolerance,
        })
        .ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(s) = value.clone().try_into::<OdometrySettings>() {
            self.color = s.color;
            self.arrow_len = s.arrow_len;
            self.keep = s.keep;
            self.position_tolerance = s.position_tolerance;
            self.angle_tolerance = s.angle_tolerance;
            while self.trail.len() > self.keep {
                self.trail.pop_front();
            }
            self.bake_dirty = true;
        }
    }
}

/// Persistence DTO for OdometryRenderer's user-editable settings (mirrors what settings_ui touches).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct OdometrySettings {
    #[serde(with = "crate::config::color_hex")]
    color: Color32,
    arrow_len: f32,
    keep: usize,
    position_tolerance: f64,
    angle_tolerance: f64,
}

impl Default for OdometrySettings {
    fn default() -> Self {
        Self {
            color: theme::ODOM_DEFAULT,
            arrow_len: ARROW_LEN_DEFAULT,
            keep: KEEP_DEFAULT,
            position_tolerance: POSITION_TOLERANCE_DEFAULT,
            angle_tolerance: ANGLE_TOLERANCE_DEFAULT,
        }
    }
}

/// Accept if translation >= pos_tol or rotation >= ang_tol from the last accepted pose (RViz Position/Angle Tolerance).
fn should_append(
    last: Option<&Isometry3<f64>>,
    new: &Isometry3<f64>,
    pos_tol: f64,
    ang_tol: f64,
) -> bool {
    let Some(last) = last else {
        return true;
    };
    let moved = (new.translation.vector - last.translation.vector).norm() >= pos_tol;
    let turned = last.rotation.angle_to(&new.rotation) >= ang_tol;
    moved || turned
}

/// Extract the latest pose from an Odometry Value (pose.pose is relative to header.frame_id; covariance/twist unused).
fn extract_odom(value: &Value) -> Result<OdomSample, String> {
    let (frame_id, stamp) = extract_header(value)?;
    let Some(pose_value) = value.get("pose").and_then(|p| p.get("pose")) else {
        return Err("missing field `pose.pose` (Pose)".to_owned());
    };
    Ok(OdomSample {
        frame_id,
        stamp,
        pose: extract_pose(pose_value)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::BatchData;
    use crate::tf::buffer::{TfBuffer, TfTransform, tf_update};

    #[test]
    fn settings_roundtrip_via_trait() {
        let r = OdometryRenderer {
            color: Color32::from_rgb(7, 7, 7),
            arrow_len: 2.0,
            keep: 42,
            position_tolerance: 0.5,
            angle_tolerance: 0.25,
            ..Default::default()
        };
        let value = r.settings().expect("odometry has settings");
        let mut restored = OdometryRenderer::default();
        restored.apply_settings(&value);
        assert_eq!(restored.color, r.color);
        assert_eq!(restored.arrow_len, r.arrow_len);
        assert_eq!(restored.keep, r.keep);
        assert_eq!(restored.position_tolerance, r.position_tolerance);
        assert_eq!(restored.angle_tolerance, r.angle_tolerance);
    }
    use nalgebra::{Translation3, UnitQuaternion, Vector3};

    fn iso(x: f64, y: f64, yaw: f64) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::new(x, y, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
        )
    }

    fn odom_value(frame_id: &str, x: f64, y: f64, yaw: f64) -> Value {
        let q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
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
                    ("frame_id".to_owned(), Value::String(frame_id.to_owned())),
                ]),
            ),
            (
                "child_frame_id".to_owned(),
                Value::String("base_link".to_owned()),
            ),
            (
                "pose".to_owned(),
                Value::Struct(vec![(
                    "pose".to_owned(),
                    Value::Struct(vec![
                        (
                            "position".to_owned(),
                            Value::Struct(vec![
                                ("x".to_owned(), Value::F64(x)),
                                ("y".to_owned(), Value::F64(y)),
                                ("z".to_owned(), Value::F64(0.0)),
                            ]),
                        ),
                        (
                            "orientation".to_owned(),
                            Value::Struct(vec![
                                ("x".to_owned(), Value::F64(q.i)),
                                ("y".to_owned(), Value::F64(q.j)),
                                ("z".to_owned(), Value::F64(q.k)),
                                ("w".to_owned(), Value::F64(q.w)),
                            ]),
                        ),
                    ]),
                )]),
            ),
        ])
    }

    fn tf_identity() -> TfBuffer {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(vec![TfTransform {
                parent: "map".to_owned(),
                child: "odom".to_owned(),
                stamp: 0,
                transform: Isometry3::identity(),
            }], true));
        buffer
    }

    #[test]
    fn reset_drops_the_trail_but_keeps_the_settings() {
        let mut r = OdometryRenderer {
            color: Color32::from_rgb(1, 2, 3),
            arrow_len: 3.0,
            keep: 500,
            ..Default::default()
        };
        for i in 0..5 {
            r.on_message(&odom_value("odom", f64::from(i), 0.0, 0.0));
        }
        assert_eq!(r.trail.len(), 5);
        r.reset();
        // Playback jumped, so the history is gone; what the user configured is not.
        assert!(r.trail.is_empty());
        assert!(r.latest.is_none());
        assert_eq!(r.color, Color32::from_rgb(1, 2, 3));
        assert_eq!(r.arrow_len, 3.0);
        assert_eq!(r.keep, 500);
        // The next message starts a fresh trail rather than joining the old one.
        r.on_message(&odom_value("odom", 100.0, 0.0, 0.0));
        assert_eq!(r.trail.len(), 1);
    }

    #[test]
    fn extracts_pose_from_pose_with_covariance() {
        let sample = extract_odom(&odom_value("odom", 1.0, 2.0, 0.0)).expect("valid odom");
        assert_eq!(sample.frame_id, "odom");
        assert_eq!(sample.pose.translation.vector.x, 1.0);
        assert_eq!(sample.pose.translation.vector.y, 2.0);
        assert!(extract_odom(&Value::Struct(vec![])).is_err());
    }

    #[test]
    fn should_append_honors_position_and_angle_tolerance() {
        let last = iso(0.0, 0.0, 0.0);
        // Always accept when the trail is empty.
        assert!(should_append(None, &last, 0.1, 0.1));
        // Both below tolerance -> drop.
        assert!(!should_append(Some(&last), &iso(0.05, 0.0, 0.05), 0.1, 0.1));
        // Position only exceeds (boundary value is accepted).
        assert!(should_append(Some(&last), &iso(0.1, 0.0, 0.0), 0.1, 0.1));
        // Rotation only exceeds.
        assert!(should_append(Some(&last), &iso(0.0, 0.0, 0.2), 0.1, 0.1));
    }

    #[test]
    fn trail_is_capped_at_keep() {
        let mut renderer = OdometryRenderer {
            keep: 3,
            ..Default::default()
        };
        for i in 0..10 {
            renderer.on_message(&odom_value("odom", i as f64, 0.0, 0.0));
        }
        assert_eq!(renderer.trail.len(), 3);
        // Oldest are dropped (7, 8, 9 remain).
        assert_eq!(renderer.trail[0].translation.vector.x, 7.0);
        assert_eq!(renderer.trail[2].translation.vector.x, 9.0);
    }

    #[test]
    fn frame_change_clears_trail() {
        let mut renderer = OdometryRenderer::default();
        renderer.on_message(&odom_value("odom", 0.0, 0.0, 0.0));
        renderer.on_message(&odom_value("odom", 1.0, 0.0, 0.0));
        assert_eq!(renderer.trail.len(), 2);
        renderer.on_message(&odom_value("odom2", 0.0, 0.0, 0.0));
        assert_eq!(renderer.trail.len(), 1);
    }

    #[test]
    fn arrow_mesh_is_triangle_list_spanning_origin_to_tip() {
        let color = [1.0, 1.0, 1.0, 1.0];
        let mut verts = Vec::new();
        push_arrow_mesh(&mut verts, &Isometry3::identity(), 1.0, color);
        // Circle segments x 5 faces x 3 vertices; TriangleList, so a multiple of 3.
        assert_eq!(verts.len(), arrow_mesh_vertex_count());
        assert_eq!(verts.len() % 3, 0);
        // Extends along +X: tip at shaft(1.0) + head(0.3) = 1.3, base at the origin.
        let max_x = verts.iter().map(|v| v.position[0]).fold(f32::MIN, f32::max);
        let min_x = verts.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min);
        assert!((max_x - 1.3).abs() < 1e-5, "tip at 1.3, got {max_x}");
        assert!(min_x.abs() < 1e-5, "base at 0, got {min_x}");
        // Radius is at most head_r = 0.1 (spread in +/-Y and +/-Z).
        let max_r = verts
            .iter()
            .map(|v| (v.position[1].powi(2) + v.position[2].powi(2)).sqrt())
            .fold(0.0f32, f32::max);
        assert!((max_r - 0.1).abs() < 1e-5, "head radius 0.1, got {max_r}");
        // Flat shading makes per-face brightness differ (shading is applied).
        let shades: Vec<f32> = verts.iter().step_by(3).map(|v| v.color[0]).collect();
        assert!(shades.iter().cloned().fold(f32::MAX, f32::min) < shades.iter().cloned().fold(f32::MIN, f32::max));
    }

    #[test]
    fn arrow_mesh_follows_pose_orientation() {
        // A +Z 90-degree rotation makes the +X arrow's tip point to fixed's +Y.
        let color = [1.0, 1.0, 1.0, 1.0];
        let mut verts = Vec::new();
        let pose = iso(2.0, 3.0, std::f64::consts::FRAC_PI_2).cast::<f32>();
        push_arrow_mesh(&mut verts, &pose, 1.0, color);
        let tip = verts
            .iter()
            .max_by(|a, b| a.position[1].total_cmp(&b.position[1]))
            .unwrap()
            .position;
        // Tip is near (2, 3+1.3, 0).
        assert!((tip[0] - 2.0).abs() < 1e-4);
        assert!((tip[1] - 4.3).abs() < 1e-4);
    }

    #[test]
    fn scene_bakes_one_arrow_mesh_per_trail_pose() {
        let mut renderer = OdometryRenderer::default();
        let buffer = tf_identity();
        renderer.on_message(&odom_value("odom", 0.0, 0.0, 0.0));
        renderer.on_message(&odom_value("odom", 1.0, 0.0, 0.0));
        renderer.on_message(&odom_value("odom", 2.0, 1.0, 0.0));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let batches = renderer.scene(&tf).expect("baked");
        let BatchData::Mesh(vertices) = &batches[0].data else {
            panic!("expected mesh batch");
        };
        // 3 accepted poses -> one arrow mesh each.
        assert_eq!(renderer.trail.len(), 3);
        assert_eq!(vertices.len(), 3 * arrow_mesh_vertex_count());
    }

    #[test]
    fn rebake_happens_only_on_new_message_or_settings() {
        let mut renderer = OdometryRenderer::default();
        let buffer = tf_identity();
        renderer.on_message(&odom_value("odom", 0.0, 0.0, 0.0));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let gen1 = renderer.scene(&tf).expect("baked")[0].generation;
        assert_eq!(renderer.scene(&tf).expect("cached")[0].generation, gen1);
        renderer.on_message(&odom_value("odom", 5.0, 0.0, 0.0));
        assert!(renderer.scene(&tf).expect("rebaked")[0].generation > gen1);
    }

    #[test]
    fn clearing_trail_removes_all_arrows() {
        let mut renderer = OdometryRenderer::default();
        let buffer = tf_identity();
        renderer.on_message(&odom_value("odom", 0.0, 0.0, 0.0));
        renderer.on_message(&odom_value("odom", 1.0, 0.0, 0.0));
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        renderer.scene(&tf).expect("baked");
        renderer.trail.clear();
        renderer.bake_dirty = true;
        let batches = renderer.scene(&tf).expect("rebaked");
        let BatchData::Mesh(vertices) = &batches[0].data else {
            panic!("expected mesh batch");
        };
        assert!(vertices.is_empty());
    }

    #[test]
    fn renderer_status_transitions() {
        let mut renderer = OdometryRenderer::default();
        let buffer = TfBuffer::new();
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        assert!(matches!(renderer.scene(&tf), Err(RenderStatus::NoData)));
        renderer.on_message(&Value::Struct(vec![]));
        assert!(matches!(
            renderer.scene(&tf),
            Err(RenderStatus::InvalidMessage(_))
        ));
        renderer.on_message(&odom_value("odom", 0.0, 0.0, 0.0));
        assert_eq!(
            renderer.scene(&tf).unwrap_err(),
            RenderStatus::TfUnavailable {
                frame: "odom".to_owned()
            }
        );
    }
}
