//! The TFTrajectory display: samples a TF frame's origin every frame and draws the trail it leaves in the fixed frame.

use std::sync::Arc;

use egui::{Color32, RichText};
use nalgebra::{Isometry3, Point3};
use serde::{Deserialize, Serialize};
use visor::plugin::{
    LINE_WIDTH_PX_DEFAULT, PointBatchBuilder, RenderStatus, Renderer, SceneBatch, SizeSpec,
    TfContext, TimeNs, Value, Vertex, color_hex, theme,
};

use crate::trajectory::{Limits, Sample, Trajectory};

/// How the polyline is drawn (mirrors RViz's Line Style; jsk_rviz_plugins' TFTrajectory uses the billboard form).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LineStyle {
    /// Camera-facing ribbons of Width meters, world-fixed (what jsk's TFTrajectory draws).
    #[default]
    Billboards,
    /// One-pixel wide lines, ignoring Width.
    Lines,
}

impl LineStyle {
    const ALL: [LineStyle; 2] = [LineStyle::Billboards, LineStyle::Lines];

    fn label(self) -> &'static str {
        match self {
            LineStyle::Billboards => "Billboards",
            LineStyle::Lines => "Lines",
        }
    }
}

/// Ribbon width [m] (world-fixed); matches jsk TFTrajectory's line_width default. Unused by the Lines style.
const WIDTH_DEFAULT: f32 = 0.01;
/// Lower bound on the ribbon width, so a width of 0 stays visible instead of collapsing.
const WIDTH_MIN: f32 = 0.001;
const HOLD_SEC_DEFAULT: f32 = 30.0;
const MIN_STEP_DEFAULT: f32 = 0.02;
/// Cap on samples; bounds the re-bake and the segment count (jsk splits at 1024 points per line for the same reason).
const MAX_SAMPLES_DEFAULT: usize = 1000;
/// Screen-fixed size of the current-position dot [px].
const CURRENT_DOT_PX: f32 = 8.0;
/// Opacity the age fade bottoms out at, so the oldest end stays faintly visible instead of vanishing early.
const FADE_MIN: f32 = 0.05;
/// scene() calls between refreshes of the frame-name list (~1 s at 30 fps); settings_ui gets no TF of its own.
const FRAME_LIST_INTERVAL: u32 = 30;

#[derive(Default)]
pub struct TfTrajectoryRenderer {
    settings: TrajectorySettings,
    trajectory: Trajectory,
    /// Fixed frame the samples are expressed in; a change to it clears them.
    sampled_in: Option<String>,
    /// Frame names cached from scene() for the settings combo.
    frames: Vec<String>,
    frames_countdown: u32,
    /// Baked batches, rebuilt when the samples, the settings, or the TF time behind the tail cut changed.
    baked: Vec<SceneBatch>,
    /// TF time the current bake was cut and faded against.
    baked_now: Option<TimeNs>,
    epoch: u64,
    dirty: bool,
}

impl Renderer for TfTrajectoryRenderer {
    /// Nothing to ingest: this display samples TF instead of subscribing to a topic.
    fn on_message(&mut self, _value: &Value) {}

    fn scene(&mut self, tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
        self.refresh_frame_list(tf);
        if self.settings.frame.is_empty() {
            return Err(RenderStatus::NoSource("no frame selected".to_owned()));
        }
        // Samples are fixed-frame coordinates, and transforms older than TF's window cannot be redone.
        if self.sampled_in.as_deref() != Some(tf.fixed_frame) {
            self.dirty |= self.trajectory.clear();
            self.sampled_in = Some(tf.fixed_frame.to_owned());
        }
        let Some((fixed_from_frame, stamp)) = tf.resolve_stamped(&self.settings.frame) else {
            return Err(RenderStatus::TfUnavailable {
                frame: self.settings.frame.clone(),
            });
        };
        let origin = fixed_from_frame.translation.vector;
        let position = Point3::new(origin.x as f32, origin.y as f32, origin.z as f32);
        self.dirty |= self
            .trajectory
            .observe(position, stamp, &self.settings.limits());
        // The tail cut and the fade both move with TF time, so a new time alone is reason to rebake.
        self.dirty |= self.trajectory.now() != self.baked_now && self.trajectory.len() > 1;
        if self.dirty {
            self.epoch += 1;
            self.baked_now = self.trajectory.now();
            self.baked = self.bake();
            self.dirty = false;
        }
        Ok(self.baked.clone())
    }

    fn reset(&mut self) {
        self.dirty |= self.trajectory.clear();
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        let mut changed = false;
        let mut frame_changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Frame").color(p.text_muted));
            let selected = if self.settings.frame.is_empty() {
                "select a frame"
            } else {
                self.settings.frame.as_str()
            };
            egui::ComboBox::from_id_salt("tf_trajectory_frame")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    if self.frames.is_empty() {
                        ui.label(RichText::new("waiting for TF").color(p.text_muted));
                    }
                    for name in &self.frames {
                        frame_changed |= ui
                            .selectable_value(&mut self.settings.frame, name.clone(), name.as_str())
                            .changed();
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Color").color(p.text_muted));
            changed |= ui
                .color_edit_button_srgba(&mut self.settings.color)
                .changed();
            ui.label(RichText::new("Alpha").color(p.text_muted));
            changed |= ui
                .add(egui::Slider::new(&mut self.settings.alpha, 0.0..=1.0))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Line style").color(p.text_muted));
            egui::ComboBox::from_id_salt("tf_trajectory_line_style")
                .selected_text(self.settings.line_style.label())
                .show_ui(ui, |ui| {
                    for style in LineStyle::ALL {
                        changed |= ui
                            .selectable_value(&mut self.settings.line_style, style, style.label())
                            .changed();
                    }
                });
            // Width is meaningless for one-pixel lines, so it is offered only for tubes (as RViz does).
            if self.settings.line_style == LineStyle::Billboards {
                ui.label(RichText::new("Width (m)").color(p.text_muted));
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.settings.width)
                            .range(0.002..=1.0)
                            .speed(0.005),
                    )
                    .changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Hold (s)").color(p.text_muted));
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.settings.hold_sec)
                        .range(1.0..=3600.0)
                        .speed(1.0),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Min step (m)").color(p.text_muted));
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.settings.min_step)
                        .range(0.0..=5.0)
                        .speed(0.01),
                )
                .changed();
            ui.label(RichText::new("Max samples").color(p.text_muted));
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.settings.max_samples)
                        .range(10..=20_000)
                        .speed(10.0),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            changed |= ui
                .checkbox(&mut self.settings.fade, "Fade with age")
                .changed();
            changed |= ui
                .checkbox(&mut self.settings.mark_current, "Mark current position")
                .changed();
        });
        ui.label(RichText::new(format!("samples: {}", self.trajectory.len())).color(p.text_muted));
        if frame_changed {
            self.trajectory.clear();
        }
        self.dirty |= changed || frame_changed;
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(self.settings.clone()).ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(settings) = value.clone().try_into::<TrajectorySettings>() {
            self.settings = settings;
            self.trajectory.clear();
            self.dirty = true;
        }
    }
}

impl TfTrajectoryRenderer {
    /// TfBuffer::frame_names allocates, so refresh the combo's choices about once a second rather than every frame.
    fn refresh_frame_list(&mut self, tf: &TfContext<'_>) {
        match self.frames_countdown.checked_sub(1) {
            Some(remaining) => self.frames_countdown = remaining,
            None => {
                self.frames = tf.buffer.frame_names();
                self.frames_countdown = FRAME_LIST_INTERVAL;
            }
        }
    }

    /// Turn the sample list into a line batch, plus the optional dot on the newest sample.
    fn bake(&self) -> Vec<SceneBatch> {
        let base = theme::to_linear_rgba(self.settings.color);
        let alpha = (base[3] * self.settings.alpha).clamp(0.0, 1.0);
        let samples = self.trajectory.samples();
        let cutoff = self.trajectory.cutoff(&self.settings.limits());
        let mut batches = Vec::new();
        let mut vertices = Vec::with_capacity(samples.len() + 1);
        for (index, sample) in samples.iter().enumerate() {
            // The oldest point is moved to where the window ends, so the tail retracts with the robot's own speed.
            let (position, stamp) = match (index, samples.get(1)) {
                (0, Some(next)) => trim_to_cutoff(sample, next, cutoff),
                _ => (sample.position, sample.stamp),
            };
            vertices.push(Vertex {
                position: [position.x, position.y, position.z],
                color: [base[0], base[1], base[2], alpha * self.fade_factor(stamp)],
            });
        }
        if vertices.len() > 1 {
            let points = Arc::new(vertices);
            batches.push(match self.settings.line_style {
                // A polyline needs mitered joins, or every corner shows a notch where two quads meet.
                LineStyle::Billboards => {
                    SceneBatch::ribbon(points, self.generation(0), self.line_size())
                }
                LineStyle::Lines => SceneBatch::lines_sized(
                    Arc::new(strip_to_line_list(&points)),
                    self.generation(0),
                    self.line_size(),
                ),
            });
        }
        if self.settings.mark_current
            && let Some(last) = samples.back()
        {
            let mut points = PointBatchBuilder::with_capacity(1);
            points.push(
                [last.position.x, last.position.y, last.position.z],
                quantize([base[0], base[1], base[2], alpha]),
            );
            batches.push(SceneBatch::points(
                points.build(),
                self.generation(1),
                &Isometry3::identity(),
                SizeSpec::Pixels(CURRENT_DOT_PX),
            ));
        }
        batches
    }

    /// Width spec handed to the line batch: a world-fixed ribbon, or one pixel.
    fn line_size(&self) -> SizeSpec {
        match self.settings.line_style {
            LineStyle::Billboards => SizeSpec::Meters(self.settings.width.max(WIDTH_MIN)),
            LineStyle::Lines => SizeSpec::Pixels(LINE_WIDTH_PX_DEFAULT),
        }
    }

    /// Opacity multiplier for one endpoint by its own age, so the gradient is smooth along the ribbon (1.0 unless the age fade is on).
    fn fade_factor(&self, stamp: Option<TimeNs>) -> f32 {
        if !self.settings.fade {
            return 1.0;
        }
        let (Some(now), Some(stamp)) = (self.trajectory.now(), stamp) else {
            return 1.0;
        };
        let age_sec = (now - stamp) as f32 / 1.0e9;
        (1.0 - age_sec / self.settings.hold_sec.max(WIDTH_MIN)).clamp(FADE_MIN, 1.0)
    }

    /// Batch generation; the slot index is folded in because the batch list shrinks and grows with the settings.
    fn generation(&self, slot: u64) -> u64 {
        (self.epoch << 32) | slot
    }
}

/// Expand a polyline into the segment pairs a LineList batch expects (the 1px style has no joins to worry about).
fn strip_to_line_list(points: &[Vertex]) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(points.len().saturating_sub(1) * 2);
    for pair in points.windows(2) {
        out.extend_from_slice(pair);
    }
    out
}

/// Start point of the oldest segment: the position at the cutoff time, interpolated between the two samples that straddle it.
fn trim_to_cutoff(
    from: &Sample,
    to: &Sample,
    cutoff: Option<TimeNs>,
) -> (Point3<f32>, Option<TimeNs>) {
    let (Some(cutoff), Some(a), Some(b)) = (cutoff, from.stamp, to.stamp) else {
        return (from.position, from.stamp);
    };
    if a >= cutoff || b <= a {
        return (from.position, from.stamp);
    }
    let t = ((cutoff - a) as f32 / (b - a) as f32).clamp(0.0, 1.0);
    (
        from.position + (to.position - from.position) * t,
        Some(cutoff),
    )
}

/// Quantizes a linear RGBA color to the packed 8-bit form the point pipeline reads.
fn quantize(rgba: [f32; 4]) -> [u8; 4] {
    rgba.map(|c| (c * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// Persistence DTO for the display's user-editable settings (the trail itself is never restored).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrajectorySettings {
    /// Target frame whose origin is sampled; empty means nothing is selected yet.
    pub frame: String,
    #[serde(with = "color_hex")]
    pub color: Color32,
    pub alpha: f32,
    pub line_style: LineStyle,
    /// Tube width [m]; kept across a style change so switching back restores it.
    pub width: f32,
    pub hold_sec: f32,
    pub min_step: f32,
    pub max_samples: usize,
    pub fade: bool,
    pub mark_current: bool,
}

impl Default for TrajectorySettings {
    fn default() -> Self {
        Self {
            frame: String::new(),
            color: theme::ODOM_DEFAULT,
            alpha: 1.0,
            line_style: LineStyle::Billboards,
            width: WIDTH_DEFAULT,
            hold_sec: HOLD_SEC_DEFAULT,
            min_step: MIN_STEP_DEFAULT,
            max_samples: MAX_SAMPLES_DEFAULT,
            fade: false,
            mark_current: false,
        }
    }
}

impl TrajectorySettings {
    fn limits(&self) -> Limits {
        Limits {
            min_step: self.min_step,
            hold_sec: self.hold_sec,
            max_samples: self.max_samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use visor::plugin::{BatchData, LIVE_EPOCH, TfBuffer, TfTransform, TfUpdate, TimeNs};

    const SEC: TimeNs = 1_000_000_000;

    /// Builds a buffer with map -> base_link at the given positions and times (the facade alone must suffice).
    fn tf_buffer(samples: &[(TimeNs, f64)]) -> TfBuffer {
        let mut buffer = TfBuffer::new();
        for &(stamp, x) in samples {
            push_tf(&mut buffer, stamp, x);
        }
        buffer
    }

    fn push_tf(buffer: &mut TfBuffer, stamp: TimeNs, x: f64) {
        buffer.insert(&TfUpdate {
            transforms: vec![TfTransform {
                parent: "map".to_owned(),
                child: "base_link".to_owned(),
                stamp,
                transform: Isometry3::translation(x, 0.0, 0.0),
            }],
            is_static: false,
            epoch: LIVE_EPOCH,
        });
    }

    /// One scene() per step, each advancing the TF time and the position (a sample only lands per scene() call).
    fn feed(renderer: &mut TfTrajectoryRenderer, steps: &[(TimeNs, f64)]) -> Vec<SceneBatch> {
        let mut buffer = TfBuffer::new();
        let mut batches = Vec::new();
        for &(stamp, x) in steps {
            push_tf(&mut buffer, stamp, x);
            batches = renderer
                .scene(&TfContext {
                    buffer: &buffer,
                    fixed_frame: "map",
                })
                .expect("ok");
        }
        batches
    }

    fn renderer_for(frame: &str) -> TfTrajectoryRenderer {
        let mut renderer = TfTrajectoryRenderer::default();
        renderer.settings.frame = frame.to_owned();
        renderer
    }

    fn ribbon_points(batches: &[SceneBatch]) -> &[Vertex] {
        let BatchData::Ribbon(points) = &batches[0].data else {
            panic!("expected a ribbon batch");
        };
        points
    }

    fn line_vertices(batches: &[SceneBatch]) -> &[Vertex] {
        let BatchData::Lines(vertices) = &batches[0].data else {
            panic!("expected a lines batch");
        };
        vertices
    }

    #[test]
    fn reports_no_source_until_a_frame_is_selected() {
        let mut renderer = TfTrajectoryRenderer::default();
        let buffer = tf_buffer(&[(SEC, 0.0)]);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        assert!(matches!(
            renderer.scene(&tf),
            Err(RenderStatus::NoSource(_))
        ));
        renderer.settings.frame = "base_link".to_owned();
        assert!(renderer.scene(&tf).is_ok());
    }

    #[test]
    fn reports_tf_unavailable_for_a_frame_that_is_not_in_tf() {
        let mut renderer = renderer_for("ghost");
        let buffer = tf_buffer(&[(SEC, 0.0)]);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        assert_eq!(
            renderer.scene(&tf).unwrap_err(),
            RenderStatus::TfUnavailable {
                frame: "ghost".to_owned()
            }
        );
    }

    #[test]
    fn a_single_sample_draws_nothing_without_erroring() {
        let mut renderer = renderer_for("base_link");
        let buffer = tf_buffer(&[(SEC, 0.0)]);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        assert_eq!(renderer.scene(&tf).expect("ok").len(), 0);
        assert_eq!(renderer.trajectory.len(), 1);
    }

    #[test]
    fn samples_become_one_ribbon_point_each() {
        let mut renderer = renderer_for("base_link");
        let batches = feed(&mut renderer, &[(SEC, 0.0), (2 * SEC, 1.0), (3 * SEC, 2.0)]);
        assert_eq!(renderer.trajectory.len(), 3);
        assert_eq!(batches.len(), 1);
        // A ribbon takes the polyline itself, so joins can be mitered; no duplicated endpoints.
        let points = ribbon_points(&batches);
        assert_eq!(points.len(), 3);
        assert_eq!(points[2].position, [2.0, 0.0, 0.0]);
    }

    #[test]
    fn the_lines_style_expands_the_polyline_into_segment_pairs() {
        let mut renderer = renderer_for("base_link");
        renderer.settings.line_style = LineStyle::Lines;
        let batches = feed(&mut renderer, &[(SEC, 0.0), (2 * SEC, 1.0), (3 * SEC, 2.0)]);
        let vertices = line_vertices(&batches);
        assert_eq!(vertices.len(), 2 * 2);
        assert_eq!(vertices[1].position, vertices[2].position);
    }

    #[test]
    fn the_tail_is_cut_at_the_window_edge_not_at_the_oldest_sample() {
        let mut renderer = renderer_for("base_link");
        renderer.settings.min_step = 0.0;
        renderer.settings.hold_sec = 2.0;
        // Cutoff is 1s behind the newest, which falls halfway along the first segment.
        let batches = feed(&mut renderer, &[(0, 0.0), (2 * SEC, 2.0), (3 * SEC, 3.0)]);
        let points = ribbon_points(&batches);
        assert_eq!(points.len(), 3);
        assert!(
            (points[0].position[0] - 1.0).abs() < 1e-5,
            "tail should sit at the cutoff, got {:?}",
            points[0].position
        );
        assert_eq!(points[1].position[0], 2.0);
    }

    #[test]
    fn line_style_selects_world_or_pixel_width() {
        let mut renderer = renderer_for("base_link");
        // jsk's TFTrajectory draws a world-width billboard line, so that is the default here too.
        assert_eq!(renderer.settings.line_style, LineStyle::Billboards);
        let batches = feed(&mut renderer, &[(SEC, 0.0), (2 * SEC, 1.0)]);
        assert_eq!(batches[0].size, SizeSpec::Meters(WIDTH_DEFAULT));
        assert!(matches!(batches[0].data, BatchData::Ribbon(_)));
        renderer.settings.line_style = LineStyle::Lines;
        renderer.dirty = true;
        let buffer = tf_buffer(&[(SEC, 0.0), (2 * SEC, 1.0)]);
        let batches = renderer
            .scene(&TfContext {
                buffer: &buffer,
                fixed_frame: "map",
            })
            .expect("ok");
        assert_eq!(batches[0].size, SizeSpec::Pixels(LINE_WIDTH_PX_DEFAULT));
        assert!(matches!(batches[0].data, BatchData::Lines(_)));
    }

    #[test]
    fn zero_width_clamps_instead_of_collapsing() {
        let mut renderer = renderer_for("base_link");
        renderer.settings.width = 0.0;
        let batches = feed(&mut renderer, &[(SEC, 0.0), (2 * SEC, 1.0)]);
        assert_eq!(batches[0].size, SizeSpec::Meters(WIDTH_MIN));
        assert_eq!(ribbon_points(&batches).len(), 2);
    }

    #[test]
    fn changing_the_fixed_frame_clears_the_trail() {
        let mut renderer = renderer_for("base_link");
        let mut buffer = tf_buffer(&[(SEC, 0.0), (2 * SEC, 1.0)]);
        buffer.insert(&TfUpdate {
            transforms: vec![TfTransform {
                parent: "odom".to_owned(),
                child: "map".to_owned(),
                stamp: 0,
                transform: Isometry3::translation(5.0, 0.0, 0.0),
            }],
            is_static: true,
            epoch: LIVE_EPOCH,
        });
        renderer
            .scene(&TfContext {
                buffer: &buffer,
                fixed_frame: "map",
            })
            .expect("ok");
        assert!(!renderer.trajectory.is_empty());
        renderer
            .scene(&TfContext {
                buffer: &buffer,
                fixed_frame: "odom",
            })
            .expect("ok");
        // Cleared, then re-seeded in the new fixed frame within the same call.
        assert_eq!(renderer.trajectory.len(), 1);
        let position = renderer.trajectory.samples()[0].position;
        assert!((position.x - 6.0).abs() < 1e-6, "position={position:?}");
    }

    #[test]
    fn changing_the_target_frame_clears_the_trail() {
        let mut renderer = renderer_for("base_link");
        let buffer = tf_buffer(&[(SEC, 0.0)]);
        renderer
            .scene(&TfContext {
                buffer: &buffer,
                fixed_frame: "map",
            })
            .expect("ok");
        assert_eq!(renderer.trajectory.len(), 1);
        let mut settings = renderer.settings.clone();
        settings.frame = "map".to_owned();
        renderer.apply_settings(&toml::Value::try_from(settings).expect("serializable"));
        assert!(renderer.trajectory.is_empty());
        assert_eq!(renderer.settings.frame, "map");
    }

    #[test]
    fn reset_drops_the_trail_but_keeps_the_settings() {
        let mut renderer = renderer_for("base_link");
        renderer.settings.width = 0.5;
        let buffer = tf_buffer(&[(SEC, 0.0)]);
        renderer
            .scene(&TfContext {
                buffer: &buffer,
                fixed_frame: "map",
            })
            .expect("ok");
        renderer.reset();
        assert!(renderer.trajectory.is_empty());
        assert_eq!(renderer.settings.width, 0.5);
        assert_eq!(renderer.settings.frame, "base_link");
    }

    #[test]
    fn an_unchanged_trail_keeps_its_generation() {
        let mut renderer = renderer_for("base_link");
        let first = feed(&mut renderer, &[(SEC, 0.0), (2 * SEC, 1.0)])[0].generation;
        // Re-scene with the same TF: nothing to append, so the bake and its generation stand.
        let buffer = tf_buffer(&[(SEC, 0.0), (2 * SEC, 1.0)]);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        assert_eq!(renderer.scene(&tf).expect("ok")[0].generation, first);
        renderer.settings.color = Color32::from_rgb(1, 2, 3);
        renderer.dirty = true;
        assert!(renderer.scene(&tf).expect("ok")[0].generation > first);
    }

    #[test]
    fn marking_the_current_position_adds_a_one_point_batch() {
        let mut renderer = renderer_for("base_link");
        renderer.settings.mark_current = true;
        let batches = feed(&mut renderer, &[(SEC, 0.0), (2 * SEC, 1.0)]);
        assert_eq!(batches.len(), 2);
        let BatchData::Points(points) = &batches[1].data else {
            panic!("expected a points batch");
        };
        assert_eq!(points.count, 1);
        assert_eq!(batches[1].size, SizeSpec::Pixels(CURRENT_DOT_PX));
        // The slot index keeps the two batches distinguishable as the list grows and shrinks.
        assert_ne!(batches[0].generation, batches[1].generation);
    }

    #[test]
    fn fade_makes_older_segments_more_transparent() {
        let mut renderer = renderer_for("base_link");
        renderer.settings.min_step = 0.0;
        renderer.settings.hold_sec = 10.0;
        let mut alphas = Vec::new();
        for fade in [false, true] {
            renderer.trajectory.clear();
            renderer.settings.fade = fade;
            renderer.dirty = true;
            let batches = feed(&mut renderer, &[(SEC, 0.0), (5 * SEC, 1.0), (9 * SEC, 2.0)]);
            let points = ribbon_points(&batches);
            alphas.push((points[0].color[3], points[2].color[3]));
        }
        let (off_old, off_new) = alphas[0];
        assert!((off_old - off_new).abs() < 1e-6, "fade off should be flat");
        let (on_old, on_new) = alphas[1];
        assert!(on_old < on_new, "old={on_old} new={on_new}");
    }

    #[test]
    fn settings_roundtrip_through_toml() {
        let mut renderer = renderer_for("base_link");
        renderer.settings = TrajectorySettings {
            frame: "laser".to_owned(),
            color: Color32::from_rgb(4, 5, 6),
            alpha: 0.25,
            line_style: LineStyle::Lines,
            width: 0.2,
            hold_sec: 12.0,
            min_step: 0.11,
            max_samples: 42,
            fade: true,
            mark_current: true,
        };
        let value = renderer.settings().expect("has settings");
        let mut restored = TfTrajectoryRenderer::default();
        restored.apply_settings(&value);
        assert_eq!(restored.settings, renderer.settings);
    }

    /// Reports the bake cost at the default cap; only the vertex count is asserted, so timing never fails CI.
    #[test]
    fn bake_cost_at_the_default_cap_is_reported() {
        let mut renderer = renderer_for("base_link");
        renderer.settings.min_step = 0.0;
        let limits = renderer.settings.limits();
        for i in 0..MAX_SAMPLES_DEFAULT {
            let x = i as f32 * 0.05;
            renderer.trajectory.observe(
                Point3::new(x, 0.0, 0.0),
                Some(i as TimeNs * 1_000_000),
                &limits,
            );
        }
        assert_eq!(renderer.trajectory.len(), MAX_SAMPLES_DEFAULT);
        let started = std::time::Instant::now();
        let batches = renderer.bake();
        let elapsed = started.elapsed();
        let bytes = size_of_val(ribbon_points(&batches));
        let vertices = ribbon_points(&batches).len();
        assert_eq!(vertices, MAX_SAMPLES_DEFAULT);
        println!(
            "bake {} samples -> {vertices} vertices ({} KiB) in {:.3} ms",
            MAX_SAMPLES_DEFAULT,
            bytes / 1024,
            elapsed.as_secs_f64() * 1.0e3
        );
    }

    #[test]
    fn missing_settings_fields_fall_back_to_defaults() {
        let mut renderer = TfTrajectoryRenderer::default();
        let partial: toml::Value = toml::from_str("frame = \"base_link\"").expect("valid toml");
        renderer.apply_settings(&partial);
        assert_eq!(renderer.settings.frame, "base_link");
        assert_eq!(renderer.settings.width, WIDTH_DEFAULT);
        assert_eq!(renderer.settings.color, theme::ODOM_DEFAULT);
    }
}
