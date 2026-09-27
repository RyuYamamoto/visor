//! 2D display interface: a display item that draws into its own egui tab instead of the wgpu viewport.

use crate::decode::value::Value;
use crate::render::RenderStatus;

/// A topic-backed 2D view (object safe). Shares subscription, settings and lifecycle with 3D renderers; only the drawing path differs.
pub trait View2d {
    /// Ingest a decoded message (decoding already happened off the UI thread).
    fn on_message(&mut self, value: &Value);
    /// Draw the tab body; `topic` is the subscribed topic name, usable as a stable egui texture key.
    fn ui(&mut self, ui: &mut egui::Ui, topic: &str);
    /// Per-item settings panel, drawn in the expanded region of the Displays panel.
    fn settings_ui(&mut self, ui: &mut egui::Ui);
    /// Why nothing is drawable, in the shared Displays vocabulary (None = drawing normally).
    fn status(&self) -> Option<RenderStatus>;
    /// Current per-item settings as an opaque config value.
    fn settings(&self) -> Option<toml::Value> {
        None
    }
    /// Apply saved settings (unknown or missing entries fall back to each field's default).
    fn apply_settings(&mut self, _value: &toml::Value) {}
    /// Drop state accumulated over time because playback jumped; keep settings and GPU resources.
    fn reset(&mut self) {}
}
