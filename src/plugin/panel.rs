//! Dock-panel interface: a plugin-supplied tab that sits alongside Displays / Frames / 3D View.

use crate::comm::session::TopicRow;
use crate::tf::buffer::TfBuffer;

/// Read-only view of app state handed to a panel every frame. A panel cannot subscribe; share state with your own renderer instead.
pub struct PanelContext<'a> {
    pub tf: &'a TfBuffer,
    pub fixed_frame: Option<&'a str>,
    /// Latest discovery snapshot (live) or bag topic list.
    pub topics: &'a [TopicRow],
    /// True while an offline playback source is open rather than a live connection.
    pub is_playback: bool,
}

/// A dock panel (object safe). Must not request periodic repaints while idle; visor's drawing is event driven.
pub trait PanelPlugin {
    fn ui(&mut self, ui: &mut egui::Ui, ctx: &PanelContext<'_>);
    /// Panel state persisted under the config's `[[panels]]` entries (None = nothing to save).
    fn settings(&self) -> Option<toml::Value> {
        None
    }
    fn apply_settings(&mut self, _value: &toml::Value) {}
}
