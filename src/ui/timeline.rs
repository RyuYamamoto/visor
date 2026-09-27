//! Playback transport bar for bag mode: play/pause, scrub, speed, loop, step (FR-6). Pure UI; all state is borrowed.

use std::time::{Duration, Instant};

use crate::bag::{BagInfo, PlayState, PlaybackStatus, SPEEDS};
use crate::tf::buffer::TimeNs;
use crate::theme;

/// Height of the transport row, enough for buttons without crowding the 3D view.
const ROW_HEIGHT: f32 = 24.0;
/// Minimum gap between seeks while dragging; every seek decompresses chunks, so one per frame would saturate the player.
const SCRUB_INTERVAL: Duration = Duration::from_millis(100);
/// Breathing room between the scrub bar and the clock to its right.
const SLIDER_MARGIN: f32 = 12.0;
/// Floor for the scrub bar so a narrow window still leaves something draggable.
const MIN_SLIDER: f32 = 80.0;

/// Everything the bar reads and the two pieces of drag state it owns.
pub struct TimelineView<'a> {
    pub info: &'a BagInfo,
    pub status: &'a PlaybackStatus,
    /// Slider value while dragging, in seconds from the bag start.
    pub scrub: &'a mut Option<f64>,
    pub last_scrub_seek: &'a mut Option<Instant>,
}

/// What the user asked the player to do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimelineAction {
    Play,
    Pause,
    SeekTo(TimeNs),
    /// Jump back to the start.
    Restart,
    Step,
    StepBack,
    SetSpeed(f32),
    SetLoop(bool),
}

/// Draw the transport bar and return the one action the user triggered, if any.
pub fn show(ui: &mut egui::Ui, mut view: TimelineView<'_>) -> Option<TimelineAction> {
    let mut action = None;
    let total = view.info.duration_secs();
    let playing = view.status.state.is_playing();
    ui.horizontal(|ui| {
        ui.set_height(ROW_HEIGHT);
        if transport_button(ui, "⏮", "Back to start").clicked() {
            action = Some(TimelineAction::Restart);
        }
        // Message-level stepping is the only positioning that does not depend on how many pixels wide the bar is.
        if transport_button(ui, "«", "Step back one message (,)").clicked() {
            action = Some(TimelineAction::StepBack);
        }
        let (glyph, hint) = if playing {
            ("⏸", "Pause (Space)")
        } else {
            ("▶", "Play (Space)")
        };
        if transport_button(ui, glyph, hint).clicked() {
            action = Some(if playing {
                TimelineAction::Pause
            } else {
                TimelineAction::Play
            });
        }
        if transport_button(ui, "»", "Step forward one message (.)").clicked() {
            action = Some(TimelineAction::Step);
        }
        ui.separator();
        // Right-hand controls are laid out first so the slider takes exactly the space left over.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut looping = view.status.looping;
            if ui
                .checkbox(&mut looping, "Loop")
                .on_hover_text("Restart from the beginning at the end of the bag")
                .changed()
            {
                action = Some(TimelineAction::SetLoop(looping));
            }
            let speed = view.status.speed;
            egui::ComboBox::from_id_salt("bag_speed")
                .selected_text(theme::machine_value(format!("×{speed}")))
                .width(64.0)
                .show_ui(ui, |ui| {
                    for option in SPEEDS {
                        if ui
                            .selectable_label(
                                (option - speed).abs() < f32::EPSILON,
                                format!("×{option}"),
                            )
                            .clicked()
                        {
                            action = Some(TimelineAction::SetSpeed(option));
                        }
                    }
                });
            ui.label(clock_label(view.info, view.status, total));
            // A Slider does not grow on its own, so hand it whatever width the right-hand controls left over.
            ui.spacing_mut().slider_width = (ui.available_width() - SLIDER_MARGIN).max(MIN_SLIDER);
            if let Some(seek) = scrub_slider(ui, &mut view, total) {
                action = Some(seek);
            }
        });
    });
    action
}

/// The scrub bar. While dragging it shows the dragged position and seeks at most once per `SCRUB_INTERVAL`, plus once on release.
fn scrub_slider(
    ui: &mut egui::Ui,
    view: &mut TimelineView<'_>,
    total: f64,
) -> Option<TimelineAction> {
    let mut position = view
        .scrub
        .unwrap_or_else(|| view.info.offset_secs(view.status.playhead));
    let response = ui.add(
        egui::Slider::new(&mut position, 0.0..=total.max(f64::EPSILON))
            .show_value(false)
            .trailing_fill(true)
            // egui's default aiming snaps to the roundest value within ~1 point, which on a long recording
            // means whole seconds. Off, the bar maps linearly to the pointer (~total/width per pixel).
            .smart_aim(false),
    );
    let released = response.drag_stopped();
    if response.dragged() {
        // Hold the dragged value locally so the bar follows the pointer instead of snapping back to the playhead.
        *view.scrub = Some(position);
    } else if released || response.changed() {
        *view.scrub = None;
    } else {
        return None;
    }
    should_send_scrub(view.last_scrub_seek, Instant::now(), released || !response.dragged())
        .then(|| TimelineAction::SeekTo(view.info.time_at_offset(position)))
}

/// `132.45 / 228.10 s   1783300429.152` — bag-relative seconds for reading, the record's UNIX time for cross-referencing other logs (the form rosbag prints).
fn clock_label(info: &BagInfo, status: &PlaybackStatus, total: f64) -> egui::RichText {
    let offset = info.offset_secs(status.playhead);
    theme::machine_value(format!(
        "{offset:.2} / {total:.2} s   {}",
        unix_stamp(status.playhead)
    ))
    .color(theme::ui::palette().instrument)
}

/// Record time as UNIX `secs.mmm`, matching how rosbag prints stamps.
pub fn unix_stamp(time: TimeNs) -> String {
    format!(
        "{}.{:03}",
        time.div_euclid(1_000_000_000),
        time.rem_euclid(1_000_000_000) / 1_000_000
    )
}

/// Absolute record time as `HH:MM:SS.mmm` in UTC-independent day arithmetic (no chrono dependency).
pub fn wall_clock(time: TimeNs) -> String {
    let secs = time.div_euclid(1_000_000_000);
    let millis = time.rem_euclid(1_000_000_000) / 1_000_000;
    let day = secs.rem_euclid(86_400);
    format!(
        "{:02}:{:02}:{:02}.{millis:03}",
        day / 3600,
        (day % 3600) / 60,
        day % 60
    )
}

/// One fixed-width transport button; neutral text so the accent stays reserved for state, with hover carrying the affordance.
fn transport_button(ui: &mut egui::Ui, glyph: &str, hint: &str) -> egui::Response {
    let p = theme::ui::palette();
    let button =
        egui::Button::new(egui::RichText::new(glyph).color(p.text_primary)).fill(p.bg_widget);
    ui.add_sized(egui::vec2(28.0, ROW_HEIGHT - 2.0), button)
        .on_hover_text(hint)
}

/// Whether a scrub-driven seek should be sent now, throttling drags to one per interval.
pub fn should_send_scrub(last: &mut Option<Instant>, now: Instant, released: bool) -> bool {
    if released {
        *last = None;
        return true;
    }
    match last {
        Some(previous) if now.duration_since(*previous) < SCRUB_INTERVAL => false,
        _ => {
            *last = Some(now);
            true
        }
    }
}

/// State of playback rendered as a word, shared with the status bar's wording.
pub fn state_label(state: PlayState) -> &'static str {
    match state {
        PlayState::Stopped => "stopped",
        PlayState::Playing => "playing",
        PlayState::Paused => "paused",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_stamp_keeps_epoch_seconds_and_millis() {
        // The form rosbag prints, so a playhead can be pasted straight into another tool.
        assert_eq!(unix_stamp(1_783_300_429_152_000_000), "1783300429.152");
        assert_eq!(unix_stamp(0), "0.000");
        // Sub-millisecond digits are truncated, not rounded away into the next second.
        assert_eq!(unix_stamp(1_999_999_999), "1.999");
    }

    #[test]
    fn wall_clock_formats_hours_minutes_seconds_and_millis() {
        // 14:09:31.204 on an arbitrary day.
        let time = ((14 * 3600 + 9 * 60 + 31) as TimeNs) * 1_000_000_000 + 204_000_000;
        assert_eq!(wall_clock(time), "14:09:31.204");
        assert_eq!(wall_clock(0), "00:00:00.000");
        // A real bag's epoch timestamp still renders as a time of day.
        assert_eq!(wall_clock(1_767_000_000_000_000_000).len(), 12);
    }

    #[test]
    fn scrub_throttle_allows_one_seek_per_interval_and_always_one_on_release() {
        let mut last = None;
        let start = Instant::now();
        assert!(should_send_scrub(&mut last, start, false));
        // A second drag sample in the same window is suppressed.
        assert!(!should_send_scrub(&mut last, start + Duration::from_millis(10), false));
        assert!(should_send_scrub(&mut last, start + SCRUB_INTERVAL, false));
        // Release always lands, so the final position is never lost.
        assert!(should_send_scrub(&mut last, start + Duration::from_millis(1), true));
        assert_eq!(last, None);
    }

    #[test]
    fn state_labels_cover_every_transport_state() {
        assert_eq!(state_label(PlayState::Stopped), "stopped");
        assert_eq!(state_label(PlayState::Playing), "playing");
        assert_eq!(state_label(PlayState::Paused), "paused");
    }
}
