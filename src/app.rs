//! egui application root; owns the egui_dock layout.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use egui_dock::{DockArea, DockState, NodeIndex, NodePath, Style, SurfaceIndex};
use nalgebra::Point3;

use crate::bag::{BagEvent, BagInfo, PlayState, PlaybackStatus};
use crate::comm::CommConfig;
use crate::comm::session::{ConnectionStatus, Notify, TopicRow};
use crate::config::{
    self, CURRENT_VERSION, CameraConfig, DisplayConfig, DisplayKind, FpsConfig, PanelConfig,
    TfConfig, ThemeConfig, TopDownConfig, ViewTypeConfig, ViewerConfig, ViewportConfig,
};
use crate::decode::msg_parser::TypeRegistry;
use crate::plugin::ids::PluginId;
use crate::plugin::panel::{PanelContext, PanelPlugin};
use crate::plugin::registry::{Problem, Registry, guarded};
use crate::plugin::view2d::View2d;
use crate::render::camera::ViewType;
use crate::render::viewport::{self, ViewportState};
use crate::render::{Companion, DisplayItemId, ItemScene, RenderStatus, Renderer, TfContext};
use crate::source::launch::Launch;
use crate::source::{Mode, Source};
use crate::tf::buffer::TfBuffer;
use crate::theme;
use crate::ui::display_list::{self, DisplayListAction, DisplayRow, ItemContentUi};
use crate::ui::frame_tree;
use crate::ui::timeline::{self, TimelineAction, TimelineView};
use crate::ui::topic_list::{self, GroupBy};

/// Minimum interval between comm-driven repaints (~30fps cap; input-driven repaints are unaffected and stay smooth).
const REPAINT_MIN_INTERVAL: Duration = Duration::from_millis(33);
/// Only type accepted for TF auto-subscription; /tf or /tf_static of other types are ignored.
const TF_MESSAGE_TYPE: &str = "tf2_msgs/msg/TFMessage";
/// The conventional TF topics, preferred whenever the graph has them.
const TF_TOPIC: &str = "/tf";
const TF_STATIC_TOPIC: &str = "/tf_static";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum Tab {
    View3d,
    Topics,
    Frames,
    Displays,
    /// One 2D display panel, keyed by topic name (stable across save/load, unlike a runtime item id).
    Image(String),
    /// A plugin-supplied panel, keyed by its qualified panel key (e.g. `sample::fleet`).
    Plugin(String),
}

/// TF auto-subscription tracking: the `(topic, type_hash)` subscribed per kind and the static topic's publisher count.
#[derive(Default)]
struct TfTracking {
    dyn_key: Option<(String, String)>,
    static_key: Option<(String, String)>,
    static_pub_count: usize,
}

impl TfTracking {
    /// Name of the dynamic TF topic currently subscribed, for the Frames panel.
    fn dyn_topic(&self) -> Option<&str> {
        self.dyn_key.as_ref().map(|(name, _)| name.as_str())
    }

    /// Name of the static TF topic currently subscribed, for the Frames panel.
    fn static_topic(&self) -> Option<&str> {
        self.static_key.as_ref().map(|(name, _)| name.as_str())
    }

    #[cfg(test)]
    fn subscribed(&self) -> bool {
        self.dyn_key.is_some() || self.static_key.is_some()
    }
}

/// TF subscribe/re-query command derived from a discovery snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TfAction {
    Subscribe { row: TopicRow, is_static: bool },
    RefreshStatic,
}

/// Pick the dynamic and static TF topics: a pinned name wins (and is not substituted when absent), else `/tf` / `/tf_static`, else the first other TFMessage topic by name (static when the name ends in `tf_static`). A recorder that logs TF under its own name (e.g. `/recorded/tf`) is what the last rule is for.
fn choose_tf_topics<'a>(
    topics: &'a [TopicRow],
    pins: &TfConfig,
) -> (Option<&'a TopicRow>, Option<&'a TopicRow>) {
    let mut tf_rows: Vec<&TopicRow> = topics
        .iter()
        .filter(|r| r.ros_type == TF_MESSAGE_TYPE)
        .collect();
    tf_rows.sort_by(|a, b| a.name.cmp(&b.name));
    let pick = |pin: &Option<String>, standard: &str, is_static: bool| -> Option<&'a TopicRow> {
        if let Some(name) = pin {
            return tf_rows.iter().copied().find(|r| &r.name == name);
        }
        tf_rows
            .iter()
            .copied()
            .find(|r| r.name == standard)
            .or_else(|| {
                tf_rows
                    .iter()
                    .copied()
                    .find(|r| r.name.ends_with("tf_static") == is_static)
            })
    };
    (
        pick(&pins.dynamic_topic, TF_TOPIC, false),
        pick(&pins.static_topic, TF_STATIC_TOPIC, true),
    )
}

/// Decides TF auto-subscription: subscribe when the chosen topic or its type_hash changed, re-query static on publisher increase.
fn plan_tf_actions(
    tracking: &mut TfTracking,
    topics: &[TopicRow],
    pins: &TfConfig,
) -> Vec<TfAction> {
    let mut actions = Vec::new();
    let (dynamic, static_) = choose_tf_topics(topics, pins);
    if let Some(row) = dynamic {
        let key = (row.name.clone(), row.type_hash.clone());
        if tracking.dyn_key.as_ref() != Some(&key) {
            tracking.dyn_key = Some(key);
            actions.push(TfAction::Subscribe {
                row: row.clone(),
                is_static: false,
            });
        }
    }
    if let Some(row) = static_ {
        let key = (row.name.clone(), row.type_hash.clone());
        if tracking.static_key.as_ref() != Some(&key) {
            tracking.static_key = Some(key);
            tracking.static_pub_count = row.publisher_count;
            actions.push(TfAction::Subscribe {
                row: row.clone(),
                is_static: true,
            });
        } else {
            // Late-publisher detection (like rmw_zenoh detect_late_publishers); re-query is always safe since inserts are idempotent.
            if row.publisher_count > tracking.static_pub_count {
                actions.push(TfAction::RefreshStatic);
            }
            tracking.static_pub_count = row.publisher_count;
        }
    }
    actions
}

/// What to do with a message carrying a given playback epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpochAction {
    /// First message of a newer generation: drop time-dependent state, then apply it.
    Reset,
    Accept,
    /// Produced before the latest jump, so applying it would repaint the state we just left.
    Drop,
}

/// Compares an incoming epoch against the one already applied, advancing `applied` when it grows.
fn epoch_action(applied: &mut u64, epoch: u64) -> EpochAction {
    if epoch > *applied {
        *applied = epoch;
        return EpochAction::Reset;
    }
    if epoch < *applied {
        return EpochAction::Drop;
    }
    EpochAction::Accept
}

/// Topic-subscription tracking of a display item; None on a standalone item, which subscribes to nothing.
struct DisplayTracking {
    /// Used to rebuild the subscribe key and decide offline; updated when type_hash changes.
    row: TopicRow,
    /// Whether the topic is present in the discovery snapshot.
    in_graph: bool,
    /// Publisher count from the last snapshot; an increase means late-publisher detection, triggering resubscribe -> history get.
    pub_count: usize,
}

/// Tracking for a renderer-requested companion subscription (e.g. OccupancyGrid's `_updates` diff topic).
struct CompanionState {
    /// Companion topic name (base topic + Companion::suffix).
    topic: String,
    /// Expected ROS-form type name, used for graph matching and the subscribe-key type.
    ros_type: &'static str,
}

/// One live subscription. Keyed by topic in `AppState::subscriptions`, so several items sharing a topic share one.
struct Subscription {
    /// type_hash it was subscribed with; a change means the key is stale and the topic must be resubscribed.
    type_hash: String,
    /// Receive counter; Hz display is future work, held only because the subscribe task produces it.
    _counter: Arc<AtomicU64>,
}

/// A display item's payload: a 3D scene renderer or a 2D view. Shares subscription/config/lifecycle with the other.
enum DisplayContent {
    /// 3D renderer producing SceneBatches for the wgpu viewport.
    Scene(Box<dyn Renderer>),
    /// 2D view rendered via egui's texture path, in its own dock tab.
    View2d(Box<dyn View2d>),
}

impl DisplayContent {
    /// Per-item settings as opaque toml (delegated to the renderer or 2D view).
    fn settings(&self) -> Option<toml::Value> {
        match self {
            DisplayContent::Scene(renderer) => renderer.settings(),
            DisplayContent::View2d(view) => view.settings(),
        }
    }

    /// Apply saved settings, containing a panic in plugin code so a bad restore degrades to defaults.
    fn apply_settings(&mut self, key: &str, value: &toml::Value) -> Result<(), String> {
        guarded(key, || match self {
            DisplayContent::Scene(renderer) => renderer.apply_settings(value),
            DisplayContent::View2d(view) => view.apply_settings(value),
        })
    }

    /// Companion subscription request (renderers only; 2D views never request one).
    fn companion(&self) -> Option<Companion> {
        match self {
            DisplayContent::Scene(renderer) => renderer.companion(),
            DisplayContent::View2d(_) => None,
        }
    }

    fn is_view2d(&self) -> bool {
        matches!(self, DisplayContent::View2d(_))
    }
}

impl ItemContentUi for DisplayContent {
    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        match self {
            DisplayContent::Scene(renderer) => renderer.settings_ui(ui),
            DisplayContent::View2d(view) => view.settings_ui(ui),
        }
    }
}

/// Which registration produced an item, so saving does not have to re-derive it from the title.
#[derive(Debug, Clone)]
struct ItemOrigin {
    plugin: PluginId,
    /// The display type's label (`RobotModel`, `LaserScan`, …); the config key for a standalone item.
    label: String,
}

/// One display item (3D scene renderer or 2D view).
struct DisplayItem {
    /// Stable id (GPU buffer key, removal, dock/panel identity); present on topic and standalone items alike.
    id: DisplayItemId,
    /// Where the display type came from; the save path reads its keys from here rather than from `title`.
    origin: ItemOrigin,
    /// Visibility toggle; OFF also unsubscribes so point-cloud-scale bandwidth isn't consumed in the background.
    visible: bool,
    /// ROS type this display type subscribes to; empty on a standalone item, which takes no topic at all.
    ros_type: String,
    /// Topic subscription tracking; None with a non-empty `ros_type` means a topic has not been assigned yet.
    tracking: Option<DisplayTracking>,
    /// Display name in "topic (short type)" form, or the renderer label for a standalone item.
    title: String,
    content: DisplayContent,
    /// Last scene() or decode failure; None means drawing normally.
    status: Option<RenderStatus>,
    /// Renderer-requested companion subscription, or None.
    companion: Option<CompanionState>,
}

impl DisplayItem {
    /// The subscribed topic name, or None when no topic is assigned (standalone, or added by display type).
    fn topic(&self) -> Option<&str> {
        self.tracking.as_ref().map(|t| t.row.name.as_str())
    }

    /// Whether this display type takes a topic at all (false for standalone items, which never show a Topic row).
    fn takes_topic(&self) -> bool {
        !self.ros_type.is_empty()
    }
}

/// The Add-display dialog's two entry points, as RViz names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AddDialogTab {
    /// Pick a display type, then assign its topic from the card (RViz's default tab).
    #[default]
    DisplayType,
    /// Pick a topic; its display type follows, with a picker when more than one answers for it.
    Topic,
}

/// Display-item subscription command derived from a discovery snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DisplayAction {
    Resubscribe { id: DisplayItemId },
}

/// One item's input to plan_display_actions, so planning stays testable without a Box<dyn Renderer>. tracking None = standalone.
struct DisplayPlan<'a> {
    id: DisplayItemId,
    visible: bool,
    tracking: Option<&'a mut DisplayTracking>,
}

/// Keeps the subscription when a topic disappears (avoids missed re-subscribe on reappearance) and resubscribes on type_hash change (same rule as TF).
fn plan_display_actions<'a>(
    items: impl IntoIterator<Item = DisplayPlan<'a>>,
    topics: &[TopicRow],
) -> Vec<DisplayAction> {
    let mut actions = Vec::new();
    for plan in items {
        // Standalone items subscribe to nothing, so a discovery snapshot never concerns them.
        let Some(tracking) = plan.tracking else {
            continue;
        };
        match topics.iter().find(|t| t.name == tracking.row.name) {
            Some(row) => {
                tracking.in_graph = true;
                let pub_increased = row.publisher_count > tracking.pub_count;
                tracking.pub_count = row.publisher_count;
                if row.type_hash != tracking.row.type_hash {
                    tracking.row = row.clone();
                    // Hidden items are unsubscribed, so don't resubscribe; they subscribe with the updated row when toggled ON.
                    if plan.visible {
                        actions.push(DisplayAction::Resubscribe { id: plan.id });
                    }
                } else if pub_increased && plan.visible {
                    // Late publisher (same motive as tf_static RefreshStatic): resubscribe to re-issue the history get.
                    actions.push(DisplayAction::Resubscribe { id: plan.id });
                }
            }
            None => tracking.in_graph = false,
        }
    }
    actions
}

/// Topics the current items need subscribed, with the graph row to subscribe them by. Visible in-graph items only, plus their companion topics; several items on one topic collapse to one entry.
fn wanted_subscriptions<'a>(
    items: impl IntoIterator<Item = &'a DisplayItem>,
    topics: &[TopicRow],
) -> HashMap<String, TopicRow> {
    let mut wanted = HashMap::new();
    for item in items {
        if !item.visible {
            continue;
        }
        if let Some(tracking) = &item.tracking
            && tracking.in_graph
        {
            wanted.insert(tracking.row.name.clone(), tracking.row.clone());
        }
        // The companion is subscribed from the graph row, since the renderer only names it by suffix and type.
        if let Some(companion) = &item.companion
            && let Some(row) = topics
                .iter()
                .find(|t| t.name == companion.topic && t.ros_type == companion.ros_type)
        {
            wanted.insert(row.name.clone(), row.clone());
        }
    }
    wanted
}

/// Bag playback state mirrored from the player, held only while a bag is open.
struct BagUi {
    info: BagInfo,
    status: PlaybackStatus,
    /// Slider position while the user drags it, in seconds from the bag start; None when not dragging.
    scrub: Option<f64>,
    /// When the last scrub-driven seek was sent, so dragging does not fire one per frame.
    last_scrub_seek: Option<std::time::Instant>,
}

/// What to apply a native file dialog's pick to; dialogs resolve asynchronously (see `AppState::spawn_dialog`).
enum DialogAction {
    /// Files for a registered file source: open and switch playback to them.
    SourceFiles,
    /// A file requested by a renderer's settings UI (display item id).
    PanelFile(DisplayItemId),
    /// A config save target; `and_close` quits once the save succeeds.
    SaveConfig { and_close: bool },
    /// A config to load.
    OpenConfig,
}

/// UI-side state for the selected (subscribed) topic.
struct AppState {
    source: Source,
    /// Everything visor can display, resolved once at startup from builtins plus plugins.
    registry: Arc<Registry>,
    /// Live-instantiated plugin panels, keyed by qualified panel key; created the first time a tab is drawn.
    panels: HashMap<String, Box<dyn PanelPlugin>>,
    /// Panel settings read from a config whose panel has not been instantiated yet.
    pending_panel_settings: HashMap<String, toml::Value>,
    /// Whether the Help > Plugins window is showing (UI-only, not persisted).
    plugins_dialog_open: bool,
    /// Registration and type-definition problems collected at startup, listed in the Plugins window.
    startup_problems: Vec<Problem>,
    /// Bag info and transport state; None in live mode and until the source reports itself open.
    bag: Option<BagUi>,
    /// Highest playback epoch already applied; anything older is stale and dropped (plan §5.4).
    applied_epoch: u64,
    /// Repaint request handed to whichever source is running; kept so a source switch can pass it on.
    notify: Notify,
    /// In-flight native file dialog: what to apply the pick to, and the channel it arrives on (see `spawn_dialog`).
    pending_dialog: Option<(DialogAction, std::sync::mpsc::Receiver<Vec<PathBuf>>)>,
    /// Endpoint being edited in the Source menu; applied to `live_config` on Connect.
    endpoint_edit: String,
    /// Domain id being edited in the Source menu; kept as text so partial input survives the frame.
    domain_edit: String,
    /// Connection live mode uses, resolved at startup even when launched with `--bag`.
    live_config: CommConfig,
    conn: ConnectionStatus,
    topics: Vec<TopicRow>,
    group_by: GroupBy,
    /// UI light/dark mode; the end of ui() reconciles it with theme::ui::mode() so menu and config share one path.
    theme: egui::Theme,
    tf_buffer: TfBuffer,
    tf_tracking: TfTracking,
    /// TF topics the user pinned in the Frames panel; the automatic rule applies to whichever kind is None.
    tf_topics: TfConfig,
    fixed_frame: Option<String>,
    /// While true, the fixed frame auto-follows the primary root; set false once the user picks one manually.
    fixed_frame_auto: bool,
    viewport: ViewportState,
    items: Vec<DisplayItem>,
    /// Live subscriptions keyed by topic, reconciled from the items rather than tracked per item, so several items can share one topic.
    subscriptions: HashMap<String, Subscription>,
    next_item_id: u64,
    /// State snapshot from the last save/load (excluding dock); the baseline for dirty detection.
    saved_config: Option<ViewerConfig>,
    /// Path last opened/saved; the Save overwrite target, None if unsaved.
    current_path: Option<PathBuf>,
    /// Config notice for the status bar (load failure, partial restore, and other user-visible messages).
    config_notice: Option<String>,
    /// Path of state.toml that records the last-used config path; None means don't remember.
    state_path: Option<PathBuf>,
    /// Default directory for file dialogs (XDG config dir); None if unresolvable.
    config_dir: Option<PathBuf>,
    /// Directory the last bag came from, used as the Open Bag dialog's starting point.
    bag_dir: Option<PathBuf>,
    /// Add-display dialog: open state (UI-only, not persisted).
    add_dialog_open: bool,
    /// Add-display dialog: which of the two entry points is showing.
    add_dialog_tab: AddDialogTab,
    /// Add-display dialog: search filter text.
    add_dialog_filter: String,
    /// Add-display dialog: currently highlighted topic name (By topic tab).
    add_dialog_selected: Option<String>,
    /// Add-display dialog, By topic tab: qualified key of the chosen display type; None = the highlighted topic's default.
    add_dialog_display_type: Option<String>,
    /// Add-display dialog, By display type tab: index into `renderer_entries()` of the chosen row.
    add_dialog_display_index: Option<usize>,
    /// Pending image-tab key renames (old_topic → new_topic) from topic switches; applied by ViewerApp to keep the tab in place.
    pending_image_rename: Vec<(String, String)>,
    /// Screen rect of the 3D View tab, captured each frame so the collapse arrow can sit on its left edge (RViz-style splitter placement).
    view_rect: egui::Rect,
    /// Set when an image item's visibility toggled this frame, so ViewerApp recomputes which image tabs are live (position preserved).
    image_visibility_dirty: bool,
}

impl AppState {
    /// Lets every 3D renderer take in its background results; true while any of them still has work outstanding. Hidden items are polled too, so a load that finishes off-screen is not left half-applied.
    fn poll_renderers(&mut self) -> bool {
        let mut busy = false;
        for item in self.items.iter_mut() {
            if let DisplayContent::Scene(renderer) = &mut item.content {
                busy |= renderer.poll();
            }
        }
        busy
    }

    /// Clears everything a display accumulated over time; called when playback jumps (FR-9).
    fn reset_renderers(&mut self) {
        for item in self.items.iter_mut() {
            match &mut item.content {
                DisplayContent::Scene(renderer) => renderer.reset(),
                DisplayContent::View2d(view) => view.reset(),
            }
        }
    }

    /// Applies the epoch rule to one incoming message; false means the message predates the latest jump and must be dropped.
    fn accept_epoch(&mut self, epoch: u64) -> bool {
        match epoch_action(&mut self.applied_epoch, epoch) {
            EpochAction::Reset => {
                // A backward jump would otherwise leave future TF samples in the buffer, freezing the robot at a pose it has not reached yet.
                self.tf_buffer = TfBuffer::new();
                self.reset_renderers();
                true
            }
            EpochAction::Accept => true,
            EpochAction::Drop => false,
        }
    }

    /// Drains the source channels; called at the top of every frame so it never stalls even with tabs closed.
    fn drain_channels(&mut self) {
        while let Ok(status) = self.source.conn_rx.try_recv() {
            self.conn = status;
        }
        self.drain_bag_events();
        let mut graph_changed = false;
        while let Ok(topics) = self.source.graph_rx.try_recv() {
            self.topics = topics;
            graph_changed = true;
        }
        if graph_changed {
            self.replan_graph();
        }
        while let Ok(message) = self.source.display_rx.try_recv() {
            // Messages produced before the latest jump would paint the pre-seek state over the new one.
            if !self.accept_epoch(message.epoch) {
                continue;
            }
            // Every item on the topic is fed, since one subscription now serves however many read it.
            for item in self.items.iter_mut() {
                if item
                    .tracking
                    .as_ref()
                    .is_some_and(|t| t.row.name == message.topic)
                {
                    match &message.result {
                        Ok(value) => match &mut item.content {
                            DisplayContent::Scene(renderer) => renderer.on_message(value),
                            // Images track their own status (decode/unsupported); mirror it into the Displays status.
                            DisplayContent::View2d(view) => {
                                view.on_message(value);
                                item.status = view.status();
                            }
                        },
                        // On decode failure, update only the status; don't hand it to the content.
                        Err(failure) => {
                            item.status =
                                Some(RenderStatus::InvalidMessage(failure.error.to_string()));
                        }
                    }
                    continue;
                }
                // Companion (diff-update) topics only feed the renderer; ignore decode failures to preserve the existing view.
                if item
                    .companion
                    .as_ref()
                    .is_some_and(|c| c.topic == message.topic)
                    && let (Ok(value), DisplayContent::Scene(renderer)) =
                        (&message.result, &mut item.content)
                {
                    renderer.on_companion(value);
                }
            }
        }
        while let Ok(update) = self.source.tf_rx.try_recv() {
            if !self.accept_epoch(update.epoch) {
                continue;
            }
            self.tf_buffer.insert(&update);
        }
        // Auto-follow the largest tree's root until manual selection, guarding against isolated small trees and root-change races.
        if self.fixed_frame_auto
            && let Some(root) = self.tf_buffer.primary_root()
            && self.fixed_frame.as_deref() != Some(root.as_str())
        {
            self.fixed_frame = Some(root);
        }
    }

    /// Takes in bag info, playback status and read failures; nothing here happens in live mode.
    fn drain_bag_events(&mut self) {
        let Some(bag_rx) = &self.source.event_rx else {
            return;
        };
        let events: Vec<BagEvent> = bag_rx.try_iter().collect();
        for event in events {
            match event {
                BagEvent::Opened(info) => {
                    let status = PlaybackStatus {
                        epoch: 0,
                        state: PlayState::Stopped,
                        playhead: info.start,
                        speed: 1.0,
                        looping: false,
                    };
                    self.bag = Some(BagUi {
                        info: *info,
                        status,
                        scrub: None,
                        last_scrub_seek: None,
                    });
                }
                BagEvent::Failed(reason) => {
                    self.bag = None;
                    self.note(reason);
                }
                BagEvent::Notice(notice) => self.note(notice),
                BagEvent::Status(status) => {
                    if let Some(bag) = &mut self.bag {
                        bag.status = status;
                    }
                }
            }
        }
    }

    /// Applies one timeline action to the player (no-op in live mode).
    fn apply_timeline_action(&mut self, action: TimelineAction) {
        let Some(handle) = self.source.playback() else {
            return;
        };
        match action {
            TimelineAction::Play => handle.play(),
            TimelineAction::Pause => handle.pause(),
            TimelineAction::SeekTo(time) => handle.seek(time),
            TimelineAction::Restart => {
                if let Some(bag) = &self.bag {
                    handle.seek(bag.info.start);
                }
            }
            TimelineAction::Step => handle.step(),
            TimelineAction::StepBack => handle.step_back(),
            TimelineAction::SetSpeed(speed) => handle.set_speed(speed),
            TimelineAction::SetLoop(enabled) => handle.set_loop(enabled),
        }
    }

    /// Toggles play/pause; bound to Space and to the timeline's own button.
    fn toggle_playback(&mut self) {
        let Some(bag) = &self.bag else {
            return;
        };
        let action = if bag.status.state.is_playing() {
            TimelineAction::Pause
        } else {
            TimelineAction::Play
        };
        self.apply_timeline_action(action);
    }

    /// Draws the timeline and forwards whatever the user did to the player.
    fn timeline_ui(&mut self, ui: &mut egui::Ui) {
        let Some(bag) = &mut self.bag else {
            return;
        };
        let action = timeline::show(
            ui,
            TimelineView {
                info: &bag.info,
                status: &bag.status,
                scrub: &mut bag.scrub,
                last_scrub_seek: &mut bag.last_scrub_seek,
            },
        );
        if let Some(action) = action {
            self.apply_timeline_action(action);
        }
    }

    /// Reapplies TF, display, and companion subscriptions together on graph update and right after config apply (idempotent).
    fn replan_graph(&mut self) {
        for action in plan_tf_actions(&mut self.tf_tracking, &self.topics, &self.tf_topics) {
            match action {
                TfAction::Subscribe { row, is_static } => {
                    // Say so when the rule fell back to a non-standard name, so a bag with an unexpected TF source is not silently trusted.
                    let (standard, pinned) = if is_static {
                        (TF_STATIC_TOPIC, self.tf_topics.static_topic.is_some())
                    } else {
                        (TF_TOPIC, self.tf_topics.dynamic_topic.is_some())
                    };
                    if row.name != standard && !pinned {
                        self.note(format!(
                            "TF from {} (no {standard} in the graph); change it in Frames if wrong",
                            row.name
                        ));
                    }
                    self.source.subscribe_tf(&row, is_static);
                }
                TfAction::RefreshStatic => self.source.refresh_tf_static(),
            }
        }
        let plans = self.items.iter_mut().map(|i| DisplayPlan {
            id: i.id,
            visible: i.visible,
            tracking: i.tracking.as_mut(),
        });
        let actions = plan_display_actions(plans, &self.topics);
        // A late publisher resubscribes on an unchanged key, which the diff below cannot see; name those topics explicitly.
        let mut forced = HashSet::new();
        for DisplayAction::Resubscribe { id } in actions {
            if let Some(topic) = self
                .items
                .iter()
                .find(|i| i.id == id)
                .and_then(|i| i.topic())
            {
                forced.insert(topic.to_owned());
            }
        }
        self.sync_subscriptions(&forced);
    }

    /// Brings the source's subscriptions in line with what the items need: subscribe what is missing or stale, drop what nothing wants. One subscription per topic however many items read it, so removing or hiding one item never cuts another's feed.
    fn sync_subscriptions(&mut self, forced: &HashSet<String>) {
        let wanted = wanted_subscriptions(&self.items, &self.topics);
        self.subscriptions
            .retain(|topic, _| match wanted.contains_key(topic) {
                true => true,
                false => {
                    self.source.unsubscribe_display(topic);
                    false
                }
            });
        for (topic, row) in wanted {
            let stale = self
                .subscriptions
                .get(&topic)
                .is_none_or(|s| s.type_hash != row.type_hash);
            if !stale && !forced.contains(&topic) {
                continue;
            }
            // The backend replaces a same-topic subscription in place, so a stale or forced one needs no unsubscribe first.
            let counter = self.source.subscribe_display(&row);
            self.subscriptions.insert(
                topic,
                Subscription {
                    type_hash: row.type_hash,
                    _counter: counter,
                },
            );
        }
    }

    /// Builds a DisplayItem from content and row (shared id/companion/title setup); does not subscribe.
    fn make_display_item(
        &mut self,
        row: TopicRow,
        content: DisplayContent,
        origin: ItemOrigin,
        in_graph: bool,
        visible: bool,
    ) -> DisplayItem {
        let id = DisplayItemId(self.next_item_id);
        self.next_item_id += 1;
        let companion = content.companion().map(|c| CompanionState {
            topic: format!("{}{}", row.name, c.suffix),
            ros_type: c.ros_type,
        });
        DisplayItem {
            id,
            title: format!("{} ({})", row.name, origin.label),
            origin,
            visible,
            ros_type: row.ros_type.clone(),
            tracking: Some(DisplayTracking {
                pub_count: row.publisher_count,
                row,
                in_graph,
            }),
            content,
            status: Some(RenderStatus::NoData),
            companion,
        }
    }

    /// Adds a display chosen by display type rather than by topic (RViz's By-display-type flow), by its index in `renderer_entries()`. A standalone one is ready as-is; a topic one starts with no topic, which the card's Topic dropdown assigns.
    fn add_display_type_item(&mut self, index: usize) {
        let registry = Arc::clone(&self.registry);
        let Some(entry) = registry.renderer_entries().get(index) else {
            return;
        };
        match entry.make() {
            Ok(renderer) => {
                let origin = ItemOrigin {
                    plugin: entry.plugin.clone(),
                    label: entry.descriptor.label.clone(),
                };
                let ros_type = entry.descriptor.ros_type.clone();
                self.push_unsubscribed_item(origin, ros_type, renderer, true);
            }
            Err(e) => self.note(e),
        }
    }

    /// Pushes an item that is not subscribing: a standalone display (`ros_type` empty) or one added by display type before a topic is assigned. status starts as None so the first scene() writes the renderer's own status (NoData would read as "waiting for data…", which no subscription will ever deliver).
    fn push_unsubscribed_item(
        &mut self,
        origin: ItemOrigin,
        ros_type: String,
        renderer: Box<dyn Renderer>,
        visible: bool,
    ) {
        let id = DisplayItemId(self.next_item_id);
        self.next_item_id += 1;
        self.items.push(DisplayItem {
            id,
            title: origin.label.clone(),
            origin,
            visible,
            ros_type,
            tracking: None,
            content: DisplayContent::Scene(renderer),
            status: None,
            companion: None,
        });
    }

    /// Restores a standalone item from config. Ok(Some(note)) = restored, but its settings-supplied source failed to load.
    fn restore_standalone_item(&mut self, dc: &DisplayConfig) -> Result<Option<String>, String> {
        let registry = Arc::clone(&self.registry);
        let restored = make_standalone_content(&registry, dc)?;
        self.push_unsubscribed_item(
            restored.origin,
            String::new(),
            restored.renderer,
            dc.visible,
        );
        Ok(restored.note)
    }

    /// Creates the matching content (3D renderer or 2D view), adds a display item, and starts subscribing. `key` is a qualified registry key; None takes the topic's default display type.
    fn add_display_item(&mut self, row: TopicRow, key: Option<&str>) {
        let Some((plugin, label)) = display_type_origin(&self.registry, &row, key) else {
            return;
        };
        let Some((content, origin)) = self.make_content(&plugin, &label, &row.name, &row.ros_type)
        else {
            return;
        };
        if self.image_topic_taken(&content, &row.name, None) {
            return;
        }
        let item = self.make_display_item(row, content, origin, true, true);
        self.items.push(item);
        self.sync_subscriptions(&HashSet::new());
    }

    /// Whether adding this content would put a second 2D view on a topic that already has one. Image tabs are keyed by topic in the dock (and in the saved layout), so two of them would collide; 3D displays have no such limit.
    fn image_topic_taken(
        &self,
        content: &DisplayContent,
        topic: &str,
        except: Option<DisplayItemId>,
    ) -> bool {
        content.is_view2d()
            && self
                .items
                .iter()
                .any(|i| Some(i.id) != except && i.content.is_view2d() && i.topic() == Some(topic))
    }

    /// Restores a display item that was saved by display type, before any topic was assigned to it.
    fn restore_unassigned_item(&mut self, dc: &DisplayConfig) -> Result<Option<String>, String> {
        let registry = Arc::clone(&self.registry);
        let Some(entry) = registry.find_display_type(&dc.plugin, &dc.label) else {
            return Err(missing_display_type(dc));
        };
        let mut renderer = entry.make()?;
        let mut note = None;
        if let Some(settings) = &dc.settings
            && let Err(e) = guarded(&entry.key, || renderer.apply_settings(settings))
        {
            note = Some(e);
        }
        let origin = ItemOrigin {
            plugin: entry.plugin.clone(),
            label: entry.descriptor.label.clone(),
        };
        let ros_type = entry.descriptor.ros_type.clone();
        self.push_unsubscribed_item(origin, ros_type, renderer, dc.visible);
        Ok(note)
    }

    /// Restores a display item from a config DisplayConfig: subscribe with the real row if in-graph, else restore offline.
    fn restore_display_item(&mut self, dc: &DisplayConfig) -> Result<Option<String>, String> {
        // A topic item with no topic was added by display type and never assigned one; it resolves by label alone.
        if dc.topic.is_empty() {
            return self.restore_unassigned_item(dc);
        }
        let Some((mut content, origin)) =
            self.make_content(&dc.plugin, &dc.label, &dc.topic, &dc.ros_type)
        else {
            return Err(missing_display_type(dc));
        };
        let mut note = None;
        if let Some(settings) = &dc.settings
            && let Err(e) = content.apply_settings(&origin.label, settings)
        {
            note = Some(e);
        }
        // In-graph gives the real row with type_hash/type_name_dds; otherwise a placeholder with no resolved subscribe key.
        let (row, in_graph) = match self.topics.iter().find(|t| t.name == dc.topic) {
            Some(row) => (row.clone(), true),
            None => (
                TopicRow {
                    name: dc.topic.clone(),
                    ros_type: dc.ros_type.clone(),
                    type_name_dds: String::new(),
                    type_hash: String::new(),
                    publisher_count: 0,
                    subscriber_count: 0,
                },
                false,
            ),
        };
        let item = self.make_display_item(row, content, origin, in_graph, dc.visible);
        self.items.push(item);
        // Subscribes only what is visible and in-graph; offline and hidden items are picked up later by replan_graph.
        self.sync_subscriptions(&HashSet::new());
        Ok(note)
    }

    /// Build display content for a topic: 2D views are checked first, then 3D renderers (`plugin` and `label` empty = the default display type).
    fn make_content(
        &self,
        plugin: &str,
        label: &str,
        topic: &str,
        ros_type: &str,
    ) -> Option<(DisplayContent, ItemOrigin)> {
        if let Some(entry) = self.registry.find_view2d_for(plugin, ros_type) {
            let origin = ItemOrigin {
                plugin: entry.plugin.clone(),
                label: entry.descriptor.label.clone(),
            };
            let view = entry.make().ok()?;
            return Some((DisplayContent::View2d(view), origin));
        }
        let entry = self
            .registry
            .find_renderer_as(plugin, label, topic, ros_type)?;
        let origin = ItemOrigin {
            plugin: entry.plugin.clone(),
            label: entry.descriptor.label.clone(),
        };
        let renderer = entry.make().ok()?;
        Some((DisplayContent::Scene(renderer), origin))
    }

    /// Card stripe color for a topic item, keyed by the resolved display type where there is one.
    fn topic_accent(&self, topic: &str, ros_type: &str) -> egui::Color32 {
        if let Some(entry) = self.registry.find_view2d(ros_type) {
            return entry.accent();
        }
        match self.registry.find_renderer(topic, ros_type) {
            Some(entry) => entry.accent(),
            None => theme::display_accent(ros_type),
        }
    }

    /// Removes an item; its topic is unsubscribed unless another item still reads it. GPU buffers are freed by the next frame's prepare retain.
    fn remove_display_item(&mut self, id: DisplayItemId) {
        if let Some(index) = self.items.iter().position(|i| i.id == id) {
            self.items.remove(index);
            self.sync_subscriptions(&HashSet::new());
        }
    }

    /// Removes the 2D item behind a closed image tab (image tabs are keyed by topic, so there is at most one).
    fn remove_image_item(&mut self, topic: &str) {
        if let Some(item) = self
            .items
            .iter()
            .find(|i| i.content.is_view2d() && i.topic() == Some(topic))
        {
            self.remove_display_item(item.id);
        }
    }

    /// Points an item at a compatible in-graph topic: subscribes, rebuilds content (state reset) while carrying settings over, and rebuilds the companion. Also the first assignment for an item added by display type, which has no old topic to leave.
    fn switch_display_topic(&mut self, id: DisplayItemId, new_topic: String) {
        let Some(index) = self.items.iter().position(|i| i.id == id) else {
            return;
        };
        // Standalone items take no topic at all.
        if !self.items[index].takes_topic() {
            return;
        }
        let old_topic = self.items[index].topic().map(str::to_owned);
        if old_topic.as_deref() == Some(new_topic.as_str()) {
            return;
        }
        let Some(new_row) = self.topics.iter().find(|t| t.name == new_topic).cloned() else {
            return;
        };
        let old_settings = self.items[index].content.settings();
        let plugin = self.items[index].origin.plugin.as_str().to_owned();
        let label = self.items[index].origin.label.clone();
        let Some((mut content, origin)) =
            self.make_content(&plugin, &label, &new_row.name, &new_row.ros_type)
        else {
            return;
        };
        if self.image_topic_taken(&content, &new_topic, Some(id)) {
            return;
        }
        if let Some(settings) = &old_settings
            && let Err(e) = content.apply_settings(&origin.label, settings)
        {
            self.note(e);
        }
        let companion = content.companion().map(|c| CompanionState {
            topic: format!("{}{}", new_row.name, c.suffix),
            ros_type: c.ros_type,
        });
        let item = &mut self.items[index];
        item.title = format!("{} ({})", new_row.name, origin.label);
        item.origin = origin;
        item.ros_type = new_row.ros_type.clone();
        item.tracking = Some(DisplayTracking {
            in_graph: true,
            pub_count: new_row.publisher_count,
            row: new_row,
        });
        let is_image = content.is_view2d();
        item.content = content;
        item.status = Some(RenderStatus::NoData);
        item.companion = companion;
        self.sync_subscriptions(&HashSet::new());
        // Rename the dock tab in place (rather than remove+recreate) so a moved image panel keeps its position.
        if is_image && let Some(old_topic) = old_topic {
            self.pending_image_rename.push((old_topic, new_topic));
        }
    }

    /// Emits errors and the like to both stderr and the status bar.
    fn note(&mut self, message: String) {
        eprintln!("visor: {message}");
        self.config_notice = Some(message);
    }

    /// On successful save, writes the last path to state.toml and updates current_path.
    fn remember_path(&mut self, path: PathBuf) {
        if let Some(state_path) = &self.state_path
            && let Err(e) = config::state::save(state_path, &path)
        {
            eprintln!("visor: failed to remember last config: {e}");
        }
        self.current_path = Some(path);
    }

    /// Serializes the current visualization state to the config DTO; dock is filled in by ViewerApp.
    fn to_config(&self) -> ViewerConfig {
        let cameras = &self.viewport.cameras;
        let orbit = &cameras.orbit;
        let topdown = &cameras.topdown;
        let fps = &cameras.fps;
        let mut hidden_frames: Vec<String> = self.viewport.hidden_frames.iter().cloned().collect();
        hidden_frames.sort();
        ViewerConfig {
            version: CURRENT_VERSION,
            fixed_frame: self.fixed_frame.clone(),
            fixed_frame_auto: self.fixed_frame_auto,
            tf: self.tf_topics.clone(),
            target_frame: self.viewport.target_frame.clone(),
            group_by: group_by_label(self.group_by).to_owned(),
            theme: theme_to_config(self.theme),
            camera: CameraConfig {
                view_type: view_type_to_config(cameras.view_type),
                target: [orbit.target.x, orbit.target.y, orbit.target.z],
                yaw: orbit.yaw,
                pitch: orbit.pitch,
                distance: orbit.distance,
                topdown: TopDownConfig {
                    center: topdown.center,
                    rotation: topdown.rotation,
                    half_height: topdown.half_height,
                },
                fps: FpsConfig {
                    eye: [fps.eye.x, fps.eye.y, fps.eye.z],
                    yaw: fps.yaw,
                    pitch: fps.pitch,
                },
            },
            viewport: ViewportConfig {
                show_names: self.viewport.show_names,
                show_links: self.viewport.show_links,
                tf_axis_len: self.viewport.tf_axis_len,
                tf_line_width: self.viewport.tf_line_width,
                hidden_frames,
            },
            displays: self
                .items
                .iter()
                .map(|item| match (&item.tracking, item.takes_topic()) {
                    (Some(tracking), _) => DisplayConfig {
                        kind: DisplayKind::Topic,
                        topic: tracking.row.name.clone(),
                        ros_type: tracking.row.ros_type.clone(),
                        label: self.non_default_label(item, &tracking.row),
                        plugin: item.origin.plugin.as_str().to_owned(),
                        visible: item.visible,
                        settings: item.content.settings(),
                    },
                    // Added by display type but never pointed at a topic: the label alone identifies it, and the empty topic is what marks it unassigned.
                    (None, true) => DisplayConfig {
                        kind: DisplayKind::Topic,
                        topic: String::new(),
                        ros_type: item.ros_type.clone(),
                        label: item.origin.label.clone(),
                        plugin: item.origin.plugin.as_str().to_owned(),
                        visible: item.visible,
                        settings: item.content.settings(),
                    },
                    // A standalone item is identified by its registry label; its source lives in the renderer's settings.
                    (None, false) => DisplayConfig {
                        kind: DisplayKind::Standalone,
                        topic: String::new(),
                        ros_type: String::new(),
                        label: item.origin.label.clone(),
                        plugin: item.origin.plugin.as_str().to_owned(),
                        visible: item.visible,
                        settings: item.content.settings(),
                    },
                })
                .collect(),
            panels: self.panel_configs(),
            dock: None,
        }
    }

    /// Display-type label to persist for a topic item: empty when it is the one the topic resolves to by default, so configs that never used the picker are unchanged.
    fn non_default_label(&self, item: &DisplayItem, row: &TopicRow) -> String {
        if item.content.is_view2d() {
            return String::new();
        }
        let default_key = self
            .registry
            .find_renderer(&row.name, &row.ros_type)
            .map(|e| e.key.as_str());
        let key = item.origin.plugin.qualify(&item.origin.label);
        match default_key == Some(key.as_str()) {
            true => String::new(),
            false => item.origin.label.clone(),
        }
    }

    /// Panel state to persist: whatever an instantiated panel reports, plus settings for panels not opened this session.
    fn panel_configs(&self) -> Vec<PanelConfig> {
        let mut configs: Vec<PanelConfig> = self
            .panels
            .iter()
            .filter_map(|(id, panel)| {
                Some(PanelConfig {
                    id: id.clone(),
                    settings: Some(panel.settings()?),
                })
            })
            .chain(
                self.pending_panel_settings
                    .iter()
                    .filter(|(id, _)| !self.panels.contains_key(*id))
                    .map(|(id, settings)| PanelConfig {
                        id: id.clone(),
                        settings: Some(settings.clone()),
                    }),
            )
            .collect();
        configs.sort_by(|a, b| a.id.cmp(&b.id));
        configs
    }

    /// Records the current state as "saved", updating the dirty-detection baseline.
    fn mark_saved(&mut self) {
        self.saved_config = Some(self.to_config());
    }

    /// Whether state differs from the saved snapshot; dock is excluded because its rects are noisy.
    fn is_dirty(&self) -> bool {
        self.saved_config.as_ref() != Some(&self.to_config())
    }

    /// Applies a config DTO to the visualization state; dock is restored by ViewerApp, and failures are collected into notices instead of crashing.
    fn apply_config(&mut self, config: &ViewerConfig) {
        let mut notes = Vec::new();
        if config.version > CURRENT_VERSION {
            notes.push(format!(
                "config version {} is newer than supported {} — applied best-effort",
                config.version, CURRENT_VERSION
            ));
        }
        // Unsubscribe all current display items before restoring.
        let ids: Vec<DisplayItemId> = self.items.iter().map(|i| i.id).collect();
        for id in ids {
            self.remove_display_item(id);
        }
        self.fixed_frame = config.fixed_frame.clone();
        self.fixed_frame_auto = config.fixed_frame_auto;
        if self.tf_topics != config.tf {
            self.tf_topics = config.tf.clone();
            // Same reset as the Frames panel; the replan at the end of this function resubscribes.
            self.reset_tf_subscriptions();
        }
        self.viewport.target_frame = config.target_frame.clone();
        self.group_by = group_by_from_label(&config.group_by);
        self.theme = theme_from_config(config.theme);
        let camera = &config.camera;
        let cameras = &mut self.viewport.cameras;
        cameras.view_type = view_type_from_config(camera.view_type);
        cameras.orbit.target = Point3::new(camera.target[0], camera.target[1], camera.target[2]);
        cameras.orbit.yaw = camera.yaw;
        cameras.orbit.pitch = camera.pitch;
        cameras.orbit.distance = camera.distance;
        cameras.topdown.center = camera.topdown.center;
        cameras.topdown.rotation = camera.topdown.rotation;
        cameras.topdown.half_height = camera.topdown.half_height;
        cameras.fps.eye = Point3::new(camera.fps.eye[0], camera.fps.eye[1], camera.fps.eye[2]);
        cameras.fps.yaw = camera.fps.yaw;
        cameras.fps.pitch = camera.fps.pitch;
        self.viewport.show_names = config.viewport.show_names;
        self.viewport.show_links = config.viewport.show_links;
        self.viewport.tf_axis_len = config.viewport.tf_axis_len;
        self.viewport.tf_line_width = config.viewport.tf_line_width;
        self.viewport.hidden_frames = config.viewport.hidden_frames.iter().cloned().collect();
        self.panels.clear();
        self.pending_panel_settings = config
            .panels
            .iter()
            .filter_map(|p| Some((p.id.clone(), p.settings.clone()?)))
            .collect();
        let mut missing = Vec::new();
        for dc in &config.displays {
            let restored = match dc.kind {
                DisplayKind::Topic => self.restore_display_item(dc),
                DisplayKind::Standalone => self.restore_standalone_item(dc),
            };
            match restored {
                Ok(note) => notes.extend(note),
                Err(e) => missing.push(e),
            }
        }
        // Apply subscription and type resolution once here for topics already in-graph at load time.
        self.replan_graph();
        if !missing.is_empty() {
            notes.push(format!(
                "{} display(s) not restored (unknown type or missing plugin): {}",
                missing.len(),
                missing.join(", ")
            ));
        }
        if !notes.is_empty() {
            for note in &notes {
                eprintln!("visor: {note}");
            }
            self.config_notice = Some(notes.join(" | "));
        }
    }

    /// Renders one plugin panel, instantiating it on first draw; a key with no registration shows a placeholder rather than crashing.
    fn plugin_panel_ui(&mut self, ui: &mut egui::Ui, key: &str) {
        let p = theme::ui::palette();
        if !self.panels.contains_key(key) {
            let registry = Arc::clone(&self.registry);
            let Some(entry) = registry.find_panel(key) else {
                ui.colored_label(p.text_muted, format!("plugin panel not available: {key}"));
                return;
            };
            let mut panel = match entry.make() {
                Ok(panel) => panel,
                Err(e) => {
                    ui.colored_label(p.status_error, &e);
                    return;
                }
            };
            if let Some(settings) = self.pending_panel_settings.get(key)
                && let Err(e) = guarded(key, || panel.apply_settings(settings))
            {
                eprintln!("visor: {e}");
            }
            self.panels.insert(key.to_owned(), panel);
        }
        // Take the panel out so it can be drawn while the rest of AppState is borrowed for its context.
        let Some(mut panel) = self.panels.remove(key) else {
            return;
        };
        let context = PanelContext {
            tf: &self.tf_buffer,
            fixed_frame: self.fixed_frame.as_deref(),
            topics: &self.topics,
            is_playback: self.source.is_files(),
        };
        panel.ui(ui, &context);
        self.panels.insert(key.to_owned(), panel);
    }

    /// Renders the Frames panel body (TF tree + display toggles); a returned frame sets the fixed frame manually.
    fn frames_ui(&mut self, ui: &mut egui::Ui) {
        let mut candidates: Vec<String> = self
            .topics
            .iter()
            .filter(|r| r.ros_type == TF_MESSAGE_TYPE)
            .map(|r| r.name.clone())
            .collect();
        candidates.sort();
        let response = frame_tree::show(
            ui,
            &self.tf_buffer,
            self.fixed_frame.as_deref(),
            &mut self.viewport.show_names,
            &mut self.viewport.show_links,
            &mut self.viewport.tf_axis_len,
            &mut self.viewport.tf_line_width,
            &mut self.viewport.hidden_frames,
            frame_tree::TfTopicPicker {
                candidates: &candidates,
                pins: &mut self.tf_topics,
                active_dynamic: self.tf_tracking.dyn_topic(),
                active_static: self.tf_tracking.static_topic(),
            },
        );
        if let Some(frame) = response.fixed_frame {
            self.fixed_frame = Some(frame);
            self.fixed_frame_auto = false;
        }
        if response.tf_topics_changed {
            self.reset_tf_subscriptions();
            self.replan_graph();
        }
    }

    /// Forget every TF source: unsubscribe both kinds in the backend (which also drops its static cache), discard updates already queued, and start the buffer over. The caller replans, which subscribes whatever the rule or the pins now choose; a pinned topic that is absent then stays unsubscribed instead of the old one lingering.
    fn reset_tf_subscriptions(&mut self) {
        self.source.unsubscribe_tf(false);
        self.source.unsubscribe_tf(true);
        while self.source.tf_rx.try_recv().is_ok() {}
        self.tf_buffer = TfBuffer::new();
        self.tf_tracking = TfTracking::default();
    }

    /// Renders the Displays panel body (item cards, Add, topic switch) and applies the resulting action.
    fn displays_ui(&mut self, ui: &mut egui::Ui) {
        let before: Vec<bool> = self.items.iter().map(|i| i.visible).collect();
        // Topics an image tab already holds; those are one per topic, so they are not offered to another 2D item.
        let taken: HashSet<&str> = self
            .items
            .iter()
            .filter(|i| i.content.is_view2d())
            .filter_map(|i| i.topic())
            .collect();
        // Precompute per-item topic options and accent so the row build only holds field borrows.
        let meta: Vec<(Vec<String>, egui::Color32)> = self
            .items
            .iter()
            .map(|i| {
                if !i.takes_topic() {
                    let accent = self
                        .registry
                        .find_standalone(i.origin.plugin.as_str(), &i.origin.label)
                        .map_or_else(
                            || theme::display_accent(&i.origin.label),
                            |entry| entry.accent(),
                        );
                    return (Vec::new(), accent);
                }
                let current = i.topic();
                let mut options = compatible_topics(
                    &self.registry,
                    &self.topics,
                    &i.origin,
                    &i.ros_type,
                    current,
                );
                if i.content.is_view2d() {
                    options.retain(|t| current == Some(t.as_str()) || !taken.contains(t.as_str()));
                }
                // Keyed by display type where there is one: /robot_description is a RobotModel, not a String.
                let accent = match current {
                    Some(topic) => self.topic_accent(topic, &i.ros_type),
                    None => self
                        .registry
                        .find_display_type(i.origin.plugin.as_str(), &i.origin.label)
                        .map_or_else(
                            || theme::display_accent(&i.origin.label),
                            |entry| entry.accent(),
                        ),
                };
                (options, accent)
            })
            .collect();
        let action = {
            let mut rows: Vec<DisplayRow<'_>> = self
                .items
                .iter_mut()
                .zip(&meta)
                .map(|(item, (compatible, accent))| DisplayRow {
                    id: item.id,
                    title: &item.title,
                    takes_topic: !item.ros_type.is_empty(),
                    topic: item.tracking.as_ref().map(|t| t.row.name.as_str()),
                    compatible_topics: compatible,
                    accent: *accent,
                    offline: item.tracking.as_ref().is_some_and(|t| !t.in_graph),
                    visible: &mut item.visible,
                    status: item.status.as_ref(),
                    content: &mut item.content,
                })
                .collect();
            display_list::show(ui, &mut rows)
        };
        // Reflect the visibility toggle into subscription (OFF = unsubscribe, unless another item still reads the topic); renderer state is kept so ON instantly restores the old view.
        let mut toggled = false;
        for (item, was_visible) in self.items.iter().zip(&before) {
            if item.visible == *was_visible {
                continue;
            }
            toggled = true;
            // Image visibility drives which image tabs are live; flag it so ViewerApp recomputes keeping positions.
            if item.content.is_view2d() {
                self.image_visibility_dirty = true;
            }
        }
        if toggled {
            self.sync_subscriptions(&HashSet::new());
        }
        match action {
            Some(DisplayListAction::Remove(id)) => self.remove_display_item(id),
            Some(DisplayListAction::OpenAddDialog) => self.add_dialog_open = true,
            Some(DisplayListAction::SwitchTopic { id, topic }) => {
                self.switch_display_topic(id, topic)
            }
            None => {}
        }
        self.handle_file_request();
    }

    /// Runs the native dialog for a file request raised in settings_ui and hands the choice back to that renderer.
    fn handle_file_request(&mut self) {
        // At most one per frame: the dialog is modal, and settings_ui only raises one on a button press.
        let found = self
            .items
            .iter_mut()
            .find_map(|item| match &mut item.content {
                DisplayContent::Scene(renderer) => renderer
                    .take_file_request()
                    .map(|request| (item.id, request)),
                DisplayContent::View2d(_) => None,
            });
        let Some((id, request)) = found else {
            return;
        };
        self.spawn_dialog(DialogAction::PanelFile(id), move || async move {
            let mut dialog =
                rfd::AsyncFileDialog::new().add_filter(request.filter_name, request.extensions);
            if let Some(dir) = &request.start_dir {
                dialog = dialog.set_directory(dir);
            }
            dialog
                .pick_file()
                .await
                .map(|file| vec![file.path().to_path_buf()])
                .unwrap_or_default()
        });
    }

    /// Runs a native file dialog off the UI thread and parks the pick for `poll_dialog`. Blocking the
    /// UI thread instead would freeze the panel on macOS, where the sheet is serviced by the main run
    /// loop. One dialog at a time; a request while one is open is dropped (the panel is modal anyway).
    fn spawn_dialog<Fut>(
        &mut self,
        action: DialogAction,
        build: impl FnOnce() -> Fut + Send + 'static,
    ) where
        Fut: std::future::Future<Output = Vec<PathBuf>>,
    {
        if self.pending_dialog.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let notify = self.notify.clone();
        let handle = self.source.handle();
        std::thread::spawn(move || {
            // block_on inside the comm runtime: the xdg-portal backend (zbus) needs its reactor.
            let _ = tx.send(handle.block_on(build()));
            notify();
        });
        self.pending_dialog = Some((action, rx));
    }

    /// Status bar in bag mode: file, length, message count and transport state instead of a connection.
    fn bag_status_bar(&self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        ui.horizontal(|ui| {
            mode_chip(ui, self.source.mode_tag(), true);
            let Some(bag) = &self.bag else {
                theme::status_led(ui, p.status_warn);
                ui.label(egui::RichText::new("opening bag…").color(p.text_muted));
                if let Some(notice) = &self.config_notice {
                    ui.separator();
                    ui.colored_label(p.status_error, notice);
                }
                return;
            };
            let info = &bag.info;
            theme::status_led(ui, p.accent);
            let names = theme::chip(ui, theme::machine_value(info.label()).color(p.instrument));
            // A merged set only shows the first name, so the rest live in the tooltip.
            if info.file_names.len() > 1 {
                names.on_hover_text(info.file_names.join("\n"));
            }
            // A sqlite3 bag has no chunks, so "none" would describe nothing; only chunked storage shows its compression.
            let compression = if info.chunk_count > 0 {
                format!(" / {}", info.compression)
            } else {
                String::new()
            };
            ui.label(
                theme::machine_value(format!(
                    "{:.1}s / {} msgs{compression}",
                    info.duration_secs(),
                    info.message_count,
                ))
                .color(p.text_muted)
                .small(),
            );
            let state = timeline::state_label(bag.status.state);
            ui.label(
                theme::machine_value(format!(
                    "{state} {:.2}s ×{}  {}",
                    info.offset_secs(bag.status.playhead),
                    bag.status.speed,
                    timeline::wall_clock(bag.status.playhead)
                ))
                .color(p.instrument),
            );
            if let Some(notice) = &self.config_notice {
                ui.separator();
                ui.colored_label(p.status_warn, notice);
            }
        });
    }

    fn status_bar(&self, ui: &mut egui::Ui) {
        if self.source.is_files() {
            self.bag_status_bar(ui);
            return;
        }
        let p = theme::ui::palette();
        let Mode::Live(config) = self.source.mode() else {
            return;
        };
        ui.horizontal(|ui| {
            mode_chip(ui, self.source.mode_tag(), false);
            // The LED + one word carry the connection state; the target lives in the chips beside them.
            match &self.conn {
                ConnectionStatus::Connecting => {
                    // Pulse only while connecting: motion marks the transient state, steady states hold still.
                    let phase = (ui.input(|i| i.time * std::f64::consts::TAU / 1.6).sin() * 0.5
                        + 0.5) as f32;
                    theme::status_led(ui, p.status_warn.gamma_multiply(0.35 + 0.65 * phase));
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(50));
                    ui.label("connecting…");
                }
                ConnectionStatus::Connected { .. } => {
                    theme::status_led(ui, p.accent);
                    ui.label("connected");
                }
                ConnectionStatus::Failed { .. } => {
                    theme::status_led(ui, p.status_error);
                    ui.label("connection failed");
                }
            }
            theme::chip(
                ui,
                theme::machine_value(config.endpoint.to_string()).color(p.instrument),
            );
            theme::chip(
                ui,
                theme::machine_value(format!("domain {}", config.domain_id)).color(p.instrument),
            );
            match &self.conn {
                ConnectionStatus::Connected { zid } => {
                    theme::chip(
                        ui,
                        theme::machine_value(format!("zid {zid}"))
                            .color(p.text_muted)
                            .small(),
                    );
                }
                ConnectionStatus::Failed { error, retry_in } => {
                    let text = if retry_in.is_zero() {
                        error.clone()
                    } else {
                        format!("{error} (retry in {}s)", retry_in.as_secs())
                    };
                    // zenoh errors carry file/line context and easily outrun the bar; elide and keep the full text on hover.
                    ui.add(
                        egui::Label::new(egui::RichText::new(&text).color(p.status_error))
                            .truncate(),
                    )
                    .on_hover_text(&text);
                }
                ConnectionStatus::Connecting => {}
            }
            if let Some(notice) = &self.config_notice {
                ui.separator();
                ui.colored_label(p.status_warn, notice);
            }
        });
    }
}

/// Which source the window is showing, as a chip at the left of the status bar (the modes are exclusive).
fn mode_chip(ui: &mut egui::Ui, text: &str, is_files: bool) {
    let p = theme::ui::palette();
    let color = if is_files {
        p.accent_secondary
    } else {
        p.accent
    };
    theme::chip(ui, theme::display_text(text).color(color).small().strong());
}

/// GroupBy to its stable, human-readable snake_case config label.
fn group_by_label(group_by: GroupBy) -> &'static str {
    match group_by {
        GroupBy::Flat => "flat",
        GroupBy::Namespace => "namespace",
        GroupBy::Type => "type",
    }
}

/// Config snake_case label to GroupBy; unknown values default to Namespace.
fn group_by_from_label(label: &str) -> GroupBy {
    match label {
        "flat" => GroupBy::Flat,
        "type" => GroupBy::Type,
        _ => GroupBy::Namespace,
    }
}

/// Runtime theme to its config DTO.
fn theme_to_config(theme: egui::Theme) -> ThemeConfig {
    match theme {
        egui::Theme::Dark => ThemeConfig::Dark,
        egui::Theme::Light => ThemeConfig::Light,
    }
}

/// Config DTO to the runtime theme.
fn theme_from_config(theme: ThemeConfig) -> egui::Theme {
    match theme {
        ThemeConfig::Dark => egui::Theme::Dark,
        ThemeConfig::Light => egui::Theme::Light,
    }
}

/// Runtime ViewType to its config DTO.
fn view_type_to_config(view_type: ViewType) -> ViewTypeConfig {
    match view_type {
        ViewType::Orbit => ViewTypeConfig::Orbit,
        ViewType::TopDownOrtho => ViewTypeConfig::TopDownOrtho,
        ViewType::Fps => ViewTypeConfig::Fps,
    }
}

/// Config DTO to runtime ViewType.
fn view_type_from_config(view_type: ViewTypeConfig) -> ViewType {
    match view_type {
        ViewTypeConfig::Orbit => ViewType::Orbit,
        ViewTypeConfig::TopDownOrtho => ViewType::TopDownOrtho,
        ViewTypeConfig::Fps => ViewType::Fps,
    }
}

/// One origin's block in the Plugins window: its name, version, and everything it registered.
fn plugin_section(
    ui: &mut egui::Ui,
    registry: &Registry,
    plugin: &PluginId,
    name: &str,
    version: &str,
) {
    let p = theme::ui::palette();
    let heading = if version.is_empty() {
        name.to_owned()
    } else {
        format!("{name}  v{version}")
    };
    // Section title, not state: the display face carries it and the accent stays reserved for live state.
    ui.label(theme::display_text(heading).color(p.text_primary).strong());
    let line = |ui: &mut egui::Ui, what: &str, items: Vec<String>| {
        if items.is_empty() {
            return;
        }
        ui.label(theme::machine_value(format!("{what}: {}", items.join(", "))).color(p.text_muted));
    };
    let displays: Vec<String> = registry
        .renderer_entries()
        .iter()
        .filter(|e| &e.plugin == plugin)
        .map(|e| {
            if e.descriptor.standalone {
                format!("{} (standalone)", e.descriptor.label)
            } else {
                e.descriptor.label.clone()
            }
        })
        .collect();
    line(ui, "3D displays", displays);
    line(
        ui,
        "2D displays",
        registry
            .view2d_entries()
            .iter()
            .filter(|e| &e.plugin == plugin)
            .map(|e| e.descriptor.label.clone())
            .collect(),
    );
    line(
        ui,
        "panels",
        registry
            .panel_entries()
            .iter()
            .filter(|e| &e.plugin == plugin)
            .map(|e| e.descriptor.title.clone())
            .collect(),
    );
    line(
        ui,
        "sources",
        registry
            .source_entries()
            .iter()
            .filter(|e| &e.plugin == plugin)
            .map(|e| e.descriptor.label.clone())
            .collect(),
    );
    let msgs = registry.msg_names(plugin);
    // The builtin set is 100+ definitions, so report its size rather than listing it.
    let msg_line = if plugin.is_builtin() {
        vec![format!("{} embedded definitions", msgs.len())]
    } else {
        msgs.iter().map(|m| (*m).to_owned()).collect()
    };
    line(ui, "message types", msg_line);
    ui.add_space(4.0);
}

/// Whether a topic can be added as a display (3D renderer or 2D view).
fn is_supported(registry: &Registry, topic: &str, ros_type: &str) -> bool {
    registry.find_view2d(ros_type).is_some() || registry.find_renderer(topic, ros_type).is_some()
}

/// Display types offered for a topic as (qualified key, label shown), in registration order with the default first. 2D views are keyed by type alone, so they always have exactly one and the Add dialog draws no picker for them.
fn display_type_choices(registry: &Registry, row: &TopicRow) -> Vec<(String, String)> {
    if let Some(entry) = registry.find_view2d(&row.ros_type) {
        return vec![(entry.key.clone(), entry.descriptor.label.clone())];
    }
    registry
        .find_renderers(&row.name, &row.ros_type)
        .map(|e| (e.key.clone(), e.descriptor.label.clone()))
        .collect()
}

/// One row of the By-display-type tab. Identified by its index in `renderer_entries()`, not by the qualified key: RobotModel deliberately registers the same label twice, once from a topic and once from a file.
struct DisplayTypeRow {
    index: usize,
    label: String,
    /// The ROS type it subscribes to; empty for a standalone display, which takes no topic.
    ros_type: String,
}

/// Every registered 3D display type, grouped by provider in registration order. `filter` is matched case-insensitively against the label and the ROS type.
fn display_types_by_provider(
    registry: &Registry,
    filter: &str,
) -> Vec<(String, Vec<DisplayTypeRow>)> {
    let mut groups: Vec<(String, Vec<DisplayTypeRow>)> = Vec::new();
    for (index, entry) in registry.renderer_entries().iter().enumerate() {
        let label = &entry.descriptor.label;
        let ros_type = &entry.descriptor.ros_type;
        if !filter.is_empty()
            && !label.to_lowercase().contains(filter)
            && !ros_type.to_lowercase().contains(filter)
        {
            continue;
        }
        let provider = entry.plugin.display_name().to_owned();
        let row = DisplayTypeRow {
            index,
            label: label.clone(),
            ros_type: ros_type.clone(),
        };
        match groups.iter_mut().find(|(name, _)| *name == provider) {
            Some((_, rows)) => rows.push(row),
            None => groups.push((provider, vec![row])),
        }
    }
    groups
}

/// Resolve the Add dialog's choice into the (plugin, label) pair make_content takes. None `key` = the topic's default, which is the empty pair. A key nothing answers for yields None, so the add is dropped rather than silently using another display type.
fn display_type_origin(
    registry: &Registry,
    row: &TopicRow,
    key: Option<&str>,
) -> Option<(String, String)> {
    let Some(key) = key else {
        return Some((String::new(), String::new()));
    };
    let entry = registry
        .find_renderers(&row.name, &row.ros_type)
        .find(|e| e.key == key)?;
    Some((
        entry.plugin.as_str().to_owned(),
        entry.descriptor.label.clone(),
    ))
}

/// A standalone item resolved from its config, ready to be pushed.
struct RestoredStandalone {
    origin: ItemOrigin,
    renderer: Box<dyn Renderer>,
    /// Settings or source failure worth reporting to the user (the item is restored either way).
    note: Option<String>,
}

/// Resolve a standalone item's config into a ready renderer (registry lookup + settings applied). Err = no such registration.
fn make_standalone_content(
    registry: &Registry,
    dc: &DisplayConfig,
) -> Result<RestoredStandalone, String> {
    let Some(entry) = registry.find_standalone(&dc.plugin, &dc.label) else {
        return Err(missing_display_label(dc));
    };
    let mut renderer = entry.make()?;
    let mut note = None;
    if let Some(settings) = &dc.settings
        && let Err(e) = guarded(&entry.key, || renderer.apply_settings(settings))
    {
        note = Some(e);
    }
    let note = note.or_else(|| {
        renderer
            .source_error()
            .map(|e| format!("{}: {e}", entry.key))
    });
    Ok(RestoredStandalone {
        origin: ItemOrigin {
            plugin: entry.plugin.clone(),
            label: entry.descriptor.label.clone(),
        },
        renderer,
        note,
    })
}

/// How a standalone item that could not be restored is reported (its provider is named when one was recorded).
fn missing_display_label(dc: &DisplayConfig) -> String {
    if dc.plugin.is_empty() {
        format!("{} (standalone)", dc.label)
    } else {
        format!("{} (standalone, plugin `{}`)", dc.label, dc.plugin)
    }
}

/// How a topic item that could not be restored is reported (its display type and provider are named when recorded).
fn missing_display_type(dc: &DisplayConfig) -> String {
    let mut what = dc.ros_type.clone();
    if !dc.label.is_empty() {
        what = format!("{what} as `{}`", dc.label);
    }
    if !dc.plugin.is_empty() {
        what = format!("{what}, plugin `{}`", dc.plugin);
    }
    format!("{} ({what})", dc.topic)
}

/// egui_dock style derived from egui's, with the active/focused tab carrying the UI accent.
fn dock_style(egui_style: &egui::Style) -> Style {
    let p = theme::ui::palette();
    let mut style = Style::from_egui(egui_style);
    style.tab_bar.bg_fill = p.bg_panel;
    style.tab_bar.hline_color = p.border;
    style.tab.inactive.text_color = p.text_muted;
    for active in [&mut style.tab.active, &mut style.tab.focused] {
        active.text_color = p.accent;
        active.bg_fill = p.bg_app;
    }
    style
}

/// Default dock: View3d on the right; Displays and Frames tabbed in the left pane (RViz-like, draggable/detachable/closeable).
fn default_dock() -> DockState<Tab> {
    build_dock(true, true)
}

/// Whether the dock currently contains the given tab.
fn has_tab(dock: &DockState<Tab>, target: &Tab) -> bool {
    dock.iter_all_tabs().any(|(_, tab)| tab == target)
}

/// Builds a clean main-surface dock: 3D View on the right, plus a left pane with whichever fixed panels are requested. Image tabs are re-added by sync_image_tabs. Rebuilding wholesale avoids the empty-node/rebalance bugs that surgical retain+split hit on restored layouts.
fn build_dock(show_displays: bool, show_frames: bool) -> DockState<Tab> {
    let mut left = Vec::new();
    if show_displays {
        left.push(Tab::Displays);
    }
    if show_frames {
        left.push(Tab::Frames);
    }
    let mut dock = DockState::new(vec![Tab::View3d]);
    if !left.is_empty() {
        dock.main_surface_mut()
            .split_left(NodeIndex::root(), 0.25, left);
    }
    dock
}

/// Adds a side panel back into the layout: into the sibling panel's leaf if present, else a fresh left split of the 3D View.
fn add_side_panel(dock: &mut DockState<Tab>, tab: Tab) {
    // A plugin panel joins whichever fixed panel is showing, so it lands in the familiar left pane.
    let sibling = match &tab {
        Tab::Displays => Tab::Frames,
        Tab::Frames => Tab::Displays,
        Tab::Plugin(_) if has_tab(dock, &Tab::Displays) => Tab::Displays,
        Tab::Plugin(_) => Tab::Frames,
        _ => return,
    };
    if let Some((node, _)) = dock.main_surface().find_tab(&sibling) {
        dock.set_focused_node_and_surface(NodePath {
            surface: SurfaceIndex::main(),
            node,
        });
        dock.push_to_focused_leaf(tab);
    } else if let Some((node, _)) = dock.main_surface().find_tab(&Tab::View3d) {
        dock.main_surface_mut().split_left(node, 0.25, vec![tab]);
    } else {
        dock.push_to_first_leaf(tab);
    }
}

/// Whether any non-View3d leaf sits on one side of the view center (i.e. there's a pane to collapse there).
fn side_has_content(dock: &DockState<Tab>, cx: f32, left: bool) -> bool {
    let surface = dock.main_surface();
    (0..surface.len()).any(|i| {
        let node = &surface[NodeIndex(i)];
        let Some(tabs) = node.tabs() else {
            return false;
        };
        if tabs.iter().any(|t| matches!(t, Tab::View3d)) {
            return false;
        }
        node.rect().is_some_and(|r| {
            if left {
                r.center().x < cx
            } else {
                r.center().x > cx
            }
        })
    })
}

/// Removes every non-View3d leaf on one side of the view (center x `cx`); used to fold a side while leaving the 3D view. remove_leaf rebalances, so no empty node lingers.
fn remove_side_leaves(dock: &mut DockState<Tab>, cx: f32, left: bool) {
    loop {
        let surface = dock.main_surface();
        let mut target = None;
        for i in 0..surface.len() {
            let node = &surface[NodeIndex(i)];
            let Some(tabs) = node.tabs() else { continue };
            if tabs.iter().any(|t| matches!(t, Tab::View3d)) {
                continue;
            }
            let Some(rect) = node.rect() else { continue };
            let on_side = if left {
                rect.center().x < cx
            } else {
                rect.center().x > cx
            };
            if on_side {
                target = Some(NodeIndex(i));
                break;
            }
        }
        match target {
            Some(idx) => dock.main_surface_mut().remove_leaf(idx),
            None => break,
        }
    }
}

/// Node of the leaf holding any existing image tab (images group into one leaf), or None.
fn image_leaf_node(dock: &DockState<Tab>) -> Option<NodeIndex> {
    let topic = dock.iter_all_tabs().find_map(|(_, tab)| match tab {
        Tab::Image(topic) => Some(topic.clone()),
        _ => None,
    })?;
    dock.main_surface()
        .find_tab(&Tab::Image(topic))
        .map(|(node, _)| node)
}

/// Adds an image tab: into the existing image leaf if any, else a new leaf right of the 3D View.
fn push_image_tab_into(dock: &mut DockState<Tab>, topic: String) {
    if let Some(node) = image_leaf_node(dock) {
        dock.set_focused_node_and_surface(NodePath {
            surface: SurfaceIndex::main(),
            node,
        });
        dock.push_to_focused_leaf(Tab::Image(topic));
    } else if let Some((node, _)) = dock.main_surface().find_tab(&Tab::View3d) {
        dock.main_surface_mut()
            .split_right(node, 0.65, vec![Tab::Image(topic)]);
    } else {
        dock.push_to_first_leaf(Tab::Image(topic));
    }
}

/// Renames an image tab's key in place (topic switch), keeping the tab where the user put it.
fn rename_image_tab(dock: &mut DockState<Tab>, old: &str, new: &str) {
    for (_, tab) in dock.iter_all_tabs_mut() {
        if let Tab::Image(topic) = tab
            && topic == old
        {
            *topic = new.to_owned();
        }
    }
}

/// Topics that can be added as a display (visualizable type); the caller greys out already-added ones.
fn addable_topics(topics: &[TopicRow], is_supported: impl Fn(&TopicRow) -> bool) -> Vec<TopicRow> {
    topics.iter().filter(|r| is_supported(r)).cloned().collect()
}

/// In-graph topics this item's own display type answers for (sorted, deduped), always including `current` so the dropdown shows it. `current` is None on an item added by display type, which has no topic yet.
fn compatible_topics(
    registry: &Registry,
    topics: &[TopicRow],
    origin: &ItemOrigin,
    ros_type: &str,
    current: Option<&str>,
) -> Vec<String> {
    // Same type is not enough when the type is shared: /robot_description and /chatter are both std_msgs/String but only one is a RobotModel, which the entry's own topic filter knows.
    let entry = registry.find_display_type(origin.plugin.as_str(), &origin.label);
    let mut names: Vec<String> = topics
        .iter()
        .filter(|r| match entry {
            Some(entry) => entry.matches(&r.name, &r.ros_type),
            // A 2D view, or a display type this build no longer has; fall back to the type-only rule.
            None => r.ros_type == ros_type,
        })
        .map(|r| r.name.clone())
        .collect();
    if let Some(current) = current
        && !names.iter().any(|n| n == current)
    {
        names.push(current.to_owned());
    }
    names.sort();
    names.dedup();
    names
}

pub struct ViewerApp {
    dock_state: DockState<Tab>,
    state: AppState,
    /// Whether the unsaved-changes confirmation modal is showing.
    close_dialog_open: bool,
    /// Skip the dirty check and let the next close request through (after choosing "quit without saving" etc. in the modal).
    force_close: bool,
    /// Left/right side panels collapsed to a splitter arrow. The live dock is derived from `saved_dock` minus the collapsed sides.
    left_collapsed: bool,
    right_collapsed: bool,
    /// Full layout captured when the first side collapses (the source of truth while anything is collapsed); None when nothing is collapsed.
    saved_dock: Option<DockState<Tab>>,
    /// View center x captured with `saved_dock`, used to classify leaves as left/right of the view.
    saved_view_cx: f32,
}

/// Which view edge a collapse arrow sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

/// Config operation deferred until after the UI to avoid a borrow conflict during menu drawing.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ConfigCommand {
    Open,
    Save,
    SaveAs,
    /// Pick files for a registered data source (carries its descriptor id) and switch to them.
    OpenFiles(String),
    /// Switch back to the zenoh connection.
    ConnectLive,
}

impl ViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        launch: Launch,
        registry: Arc<Registry>,
        types: Arc<TypeRegistry>,
        type_problems: Vec<Problem>,
    ) -> Self {
        let Launch {
            mode,
            config_path,
            live: live_config,
        } = launch;
        let ctx = cc.egui_ctx.clone();
        // Pace source-driven repaints to ~30fps: continuous data coalesces to one frame per interval, and the scheduled frame drains the trailing sample before going idle.
        let notify: Notify = Arc::new(move || ctx.request_repaint_after(REPAINT_MIN_INTERVAL));
        let source = Source::spawn(mode, notify.clone(), types, Arc::clone(&registry));
        let gpu_ready = match cc.wgpu_render_state.as_ref() {
            Some(render_state) => {
                viewport::init(render_state);
                let info = render_state.adapter.get_info();
                eprintln!("wgpu adapter: {} (backend: {:?})", info.name, info.backend);
                true
            }
            None => {
                eprintln!("wgpu render state unavailable; 3D view disabled");
                false
            }
        };
        let dock_state = default_dock();
        let endpoint_edit = live_config.endpoint.clone();
        let domain_edit = live_config.domain_id.to_string();
        let mut app = Self {
            dock_state,
            close_dialog_open: false,
            force_close: false,
            left_collapsed: false,
            right_collapsed: false,
            saved_dock: None,
            saved_view_cx: 0.0,
            state: AppState {
                subscriptions: HashMap::new(),
                source,
                registry,
                panels: HashMap::new(),
                pending_panel_settings: HashMap::new(),
                plugins_dialog_open: false,
                startup_problems: Vec::new(),
                bag: None,
                applied_epoch: 0,
                notify,
                pending_dialog: None,
                live_config,
                endpoint_edit,
                domain_edit,
                conn: ConnectionStatus::Connecting,
                topics: Vec::new(),
                group_by: GroupBy::default(),
                theme: theme::ui::mode(),
                tf_buffer: TfBuffer::new(),
                tf_tracking: TfTracking::default(),
                tf_topics: TfConfig::default(),
                fixed_frame: None,
                fixed_frame_auto: true,
                viewport: ViewportState::new(gpu_ready),
                items: Vec::new(),
                next_item_id: 0,
                saved_config: None,
                current_path: None,
                config_notice: None,
                state_path: config::paths::state_path(|k| std::env::var(k).ok()),
                config_dir: config::paths::config_dir(|k| std::env::var(k).ok()),
                bag_dir: None,
                add_dialog_open: false,
                add_dialog_tab: AddDialogTab::default(),
                add_dialog_filter: String::new(),
                add_dialog_selected: None,
                add_dialog_display_type: None,
                add_dialog_display_index: None,
                pending_image_rename: Vec::new(),
                view_rect: egui::Rect::ZERO,
                image_visibility_dirty: false,
            },
        };
        app.autoload(config_path);
        // After autoload, so a config notice does not overwrite the plugin summary (both are appended).
        app.report_startup_problems(type_problems);
        // If nothing was autoloaded (default start), baseline dirty detection on the initial state.
        if app.state.saved_config.is_none() {
            app.state.mark_saved();
        }
        app
    }

    /// Surfaces registration and type-definition problems: every line to stderr, a count to the status bar.
    fn report_startup_problems(&mut self, type_problems: Vec<Problem>) {
        let mut problems: Vec<Problem> = self.state.registry.problems().to_vec();
        problems.extend(type_problems);
        if problems.is_empty() {
            return;
        }
        for problem in &problems {
            eprintln!("visor: plugin {problem}");
        }
        let summary = format!("{} plugin problem(s) — see Help > Plugins", problems.len());
        self.state.config_notice = Some(match self.state.config_notice.take() {
            Some(existing) => format!("{existing} | {summary}"),
            None => summary,
        });
        self.state.startup_problems = problems;
    }

    /// Startup autoload: CLI arg, then the last path if it exists, else default start.
    fn autoload(&mut self, config_path: Option<PathBuf>) {
        let path = config_path.or_else(|| {
            self.state
                .state_path
                .as_deref()
                .and_then(config::state::load)
                .filter(|p| p.exists())
        });
        if let Some(path) = path {
            self.open_from(path);
        }
    }

    /// Builds the full config including dock; saves the logical layout (with hidden/collapsed tabs) so nothing is lost while folded.
    fn build_config(&self) -> ViewerConfig {
        let mut config = self.state.to_config();
        let dock = self.saved_dock.as_ref().unwrap_or(&self.dock_state);
        config.dock = toml::Value::try_from(dock).ok();
        config
    }

    /// Applies a config to the visualization state, including dock restore; failures notify instead of crashing.
    fn apply_config(&mut self, config: &ViewerConfig) {
        self.state.apply_config(config);
        // A restored layout replaces the dock, so drop any collapse/hidden state and recompute below.
        self.left_collapsed = false;
        self.right_collapsed = false;
        self.saved_dock = None;
        self.state.image_visibility_dirty = false;
        if let Some(dock) = &config.dock {
            match dock.clone().try_into::<DockState<Tab>>() {
                // A docked Topics tab means a pre-Task-11 config; that layout can't be cleanly retained, so rebuild the default dock. Otherwise restore as saved.
                Ok(dock_state) => {
                    let has_legacy_topics = dock_state
                        .iter_all_tabs()
                        .any(|(_, tab)| matches!(tab, Tab::Topics));
                    self.dock_state = if has_legacy_topics {
                        default_dock()
                    } else {
                        dock_state
                    };
                }
                Err(e) => self.state.note(format!("dock layout not restored: {e}")),
            }
        }
        if !has_tab(&self.dock_state, &Tab::View3d) {
            self.dock_state = default_dock();
        }
        // Exclude any config-hidden image tabs from the live dock (keeping them in saved_dock for restore).
        self.recompute_dock();
    }

    /// The layout to mutate: the saved full layout while a side is collapsed, else the live dock.
    fn logical_dock_mut(&mut self) -> &mut DockState<Tab> {
        self.saved_dock.as_mut().unwrap_or(&mut self.dock_state)
    }

    /// The layout to inspect: the saved full layout while anything is folded/hidden, else the live dock.
    fn logical_dock(&self) -> &DockState<Tab> {
        self.saved_dock.as_ref().unwrap_or(&self.dock_state)
    }

    /// Panels-menu toggle: removes the panel from the layout if present, else re-adds it (unfolding the left pane so it's visible).
    fn toggle_panel(&mut self, tab: Tab) {
        let present = has_tab(self.logical_dock(), &tab);
        {
            let dock = self.logical_dock_mut();
            if present {
                dock.main_surface_mut().retain_tabs(|t| *t != tab);
            } else {
                add_side_panel(dock, tab);
            }
        }
        if !present {
            self.left_collapsed = false;
        }
        self.recompute_dock();
    }

    /// Topics of image items currently hidden (visible=false); their tabs are excluded from the live dock but kept in the saved layout.
    fn hidden_image_topics(&self) -> HashSet<String> {
        self.state
            .items
            .iter()
            .filter(|i| i.content.is_view2d() && !i.visible)
            .filter_map(|i| i.topic().map(str::to_owned))
            .collect()
    }

    /// Derives the live dock from the saved full layout minus collapsed sides and hidden image tabs; restores the full layout when nothing is folded/hidden.
    fn recompute_dock(&mut self) {
        let hidden = self.hidden_image_topics();
        if !self.left_collapsed && !self.right_collapsed && hidden.is_empty() {
            if let Some(saved) = self.saved_dock.take() {
                self.dock_state = saved;
            }
            return;
        }
        if self.saved_dock.is_none() {
            self.saved_dock = Some(self.dock_state.clone());
            self.saved_view_cx = self.state.view_rect.center().x;
        }
        let mut dock = self.saved_dock.clone().unwrap();
        if self.left_collapsed {
            remove_side_leaves(&mut dock, self.saved_view_cx, true);
        }
        if self.right_collapsed {
            remove_side_leaves(&mut dock, self.saved_view_cx, false);
        }
        if !hidden.is_empty() {
            dock.main_surface_mut().retain_tabs(|tab| match tab {
                Tab::Image(topic) => !hidden.contains(topic),
                _ => true,
            });
        }
        self.dock_state = dock;
    }

    /// Image tab topics present in the given dock.
    fn image_tab_topics(dock: &DockState<Tab>) -> HashSet<String> {
        dock.iter_all_tabs()
            .filter_map(|(_, tab)| match tab {
                Tab::Image(topic) => Some(topic.clone()),
                _ => None,
            })
            .collect()
    }

    /// Reconciles image tabs with image items on the logical layout: removes tabs of deleted items, adds tabs for new visible ones; recomputes the live dock only when the tab set actually changed (so it never wipes drags every frame).
    fn sync_image_tabs(&mut self) {
        let all_topics: HashSet<String> = self
            .state
            .items
            .iter()
            .filter(|i| i.content.is_view2d())
            .filter_map(|i| i.topic().map(str::to_owned))
            .collect();
        let visible_topics: HashSet<String> = self
            .state
            .items
            .iter()
            .filter(|i| i.content.is_view2d() && i.visible)
            .filter_map(|i| i.topic().map(str::to_owned))
            .collect();
        let changed;
        {
            let dock = self.logical_dock_mut();
            let before = Self::image_tab_topics(dock);
            dock.main_surface_mut().retain_tabs(|tab| match tab {
                Tab::Image(topic) => all_topics.contains(topic),
                _ => true,
            });
            let existing = Self::image_tab_topics(dock);
            for topic in visible_topics.difference(&existing) {
                push_image_tab_into(dock, topic.clone());
            }
            changed = Self::image_tab_topics(dock) != before;
        }
        if changed && self.saved_dock.is_some() {
            self.recompute_dock();
        }
    }

    /// Applies pending image-tab key renames in place, preserving each tab's dock position across a topic switch.
    fn apply_image_renames(&mut self) {
        let renames = std::mem::take(&mut self.state.pending_image_rename);
        if renames.is_empty() {
            return;
        }
        let collapsed = self.saved_dock.is_some();
        {
            let dock = self.logical_dock_mut();
            for (old, new) in renames {
                rename_image_tab(dock, &old, &new);
            }
        }
        if collapsed {
            self.recompute_dock();
        }
    }

    /// Saves to the given path and remembers it as the last path; true if saved.
    fn save_to(&mut self, path: PathBuf) -> bool {
        match config::save(&self.build_config(), &path) {
            Ok(()) => {
                self.state.config_notice = None;
                self.state.remember_path(path);
                self.state.mark_saved();
                true
            }
            Err(e) => {
                self.state.note(format!("save failed: {e}"));
                false
            }
        }
    }

    /// Saves to the current file, or starts Save As if none; `and_close` quits once the save succeeds.
    fn save_active(&mut self, ctx: &egui::Context, and_close: bool) {
        match self.state.current_path.clone() {
            Some(path) => {
                if self.save_to(path) && and_close {
                    self.force_close = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            None => self.save_as(and_close),
        }
    }

    /// Loads and applies from the given path and remembers it as the last path.
    fn open_from(&mut self, path: PathBuf) {
        match config::load(&path) {
            Ok(config) => {
                self.state.config_notice = None;
                self.apply_config(&config);
                self.state.remember_path(path);
                self.state.mark_saved();
            }
            Err(e) => self
                .state
                .note(format!("could not load {}: {e}", path.display())),
        }
    }

    fn handle_config_command(&mut self, ctx: &egui::Context, command: ConfigCommand) {
        match command {
            ConfigCommand::Save => {
                self.save_active(ctx, false);
            }
            ConfigCommand::SaveAs => {
                self.save_as(false);
            }
            ConfigCommand::Open => self.open_dialog(),
            ConfigCommand::OpenFiles(source_id) => self.open_source_dialog(&source_id),
            ConfigCommand::ConnectLive => self.connect_live(),
        }
    }

    /// Prompts for files (or, for a source whose bags are directories, folders) and switches to them; several picks are merged in order.
    fn open_source_dialog(&mut self, source_id: &str) {
        let registry = Arc::clone(&self.state.registry);
        let Some(entry) = registry
            .source_entries()
            .iter()
            .find(|e| e.descriptor.id == source_id)
        else {
            return;
        };
        let label = entry.descriptor.label.clone();
        let extensions = entry.descriptor.extensions.clone();
        let pick_folders = entry.descriptor.pick_folders;
        let dir = self
            .state
            .bag_dir
            .clone()
            .or_else(|| self.state.config_dir.clone());
        self.state
            .spawn_dialog(DialogAction::SourceFiles, move || async move {
                let mut dialog = rfd::AsyncFileDialog::new().set_title(&label);
                if let Some(dir) = &dir {
                    dialog = dialog.set_directory(dir);
                }
                // A folder picker has nothing to filter; the source expands each directory to its files itself.
                let picked = if pick_folders {
                    dialog.pick_folders().await
                } else {
                    dialog.add_filter(&label, &extensions).pick_files().await
                };
                picked
                    .map(|files| files.iter().map(|f| f.path().to_path_buf()).collect())
                    .unwrap_or_default()
            });
    }

    /// Points the app at one or more files, remembering the directory for the next dialog.
    fn open_files(&mut self, paths: Vec<PathBuf>) {
        if paths.is_empty() {
            return;
        }
        if let Some(dir) = paths[0].parent()
            && let Some(state_path) = &self.state.state_path
        {
            self.state.bag_dir = Some(dir.to_path_buf());
            if let Err(e) = config::state::save_bag_dir(state_path, dir) {
                eprintln!("visor: failed to remember the bag directory: {e}");
            }
        }
        self.switch_source(Mode::Files(paths));
    }

    /// Returns to the zenoh connection resolved at startup (Source menu; the bag player stops on the way out).
    fn connect_live(&mut self) {
        let mode = Mode::Live(self.state.live_config.clone());
        self.switch_source(mode);
    }

    /// Restarts the source, drops everything time-dependent, and forces a resubscribe when the new graph lands.
    fn switch_source(&mut self, mode: Mode) {
        let notify = self.state.notify.clone();
        self.state.source.respawn(mode, notify);
        self.state.bag = None;
        // Live stamps LIVE_EPOCH (0) on everything, so a stale epoch from a bag would drop every live message.
        self.state.applied_epoch = 0;
        self.state.topics.clear();
        self.state.conn = ConnectionStatus::Connecting;
        self.state.tf_buffer = TfBuffer::new();
        self.state.tf_tracking = TfTracking::default();
        self.state.reset_renderers();
        self.state.config_notice = None;
        // Dropping what the old source was serving makes the new graph reissue every subscription.
        self.state.subscriptions.clear();
        for item in &mut self.state.items {
            if let Some(tracking) = &mut item.tracking {
                tracking.row.type_hash.clear();
                tracking.pub_count = 0;
                tracking.in_graph = false;
            }
            item.status = Some(RenderStatus::NoData);
        }
    }

    /// Prompts for a save location via the native dialog; the save (and optional quit) lands in `poll_dialog`.
    fn save_as(&mut self, and_close: bool) {
        let dir = self.state.config_dir.clone();
        self.state
            .spawn_dialog(DialogAction::SaveConfig { and_close }, move || async move {
                let mut dialog = rfd::AsyncFileDialog::new()
                    .add_filter("config (TOML)", &["toml"])
                    .set_file_name("viewer.toml");
                if let Some(dir) = &dir {
                    dialog = dialog.set_directory(dir);
                }
                dialog
                    .save_file()
                    .await
                    .map(|file| vec![file.path().to_path_buf()])
                    .unwrap_or_default()
            });
    }

    /// Prompts for a config via the native dialog; the open lands in `poll_dialog`.
    fn open_dialog(&mut self) {
        let dir = self.state.config_dir.clone();
        self.state
            .spawn_dialog(DialogAction::OpenConfig, move || async move {
                let mut dialog = rfd::AsyncFileDialog::new().add_filter("config (TOML)", &["toml"]);
                if let Some(dir) = &dir {
                    dialog = dialog.set_directory(dir);
                }
                dialog
                    .pick_file()
                    .await
                    .map(|file| vec![file.path().to_path_buf()])
                    .unwrap_or_default()
            });
    }

    /// Applies the pick of a finished native dialog (`spawn_dialog`); cancel arrives as an empty pick.
    fn poll_dialog(&mut self, ctx: &egui::Context) {
        let Some((_, rx)) = &self.state.pending_dialog else {
            return;
        };
        let paths = match rx.try_recv() {
            Ok(paths) => paths,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.state.pending_dialog = None;
                return;
            }
        };
        let (action, _) = self.state.pending_dialog.take().expect("checked above");
        match action {
            DialogAction::SourceFiles => self.open_files(paths),
            DialogAction::PanelFile(id) => {
                if let Some(path) = paths.first()
                    && let Some(item) = self.state.items.iter_mut().find(|i| i.id == id)
                    && let DisplayContent::Scene(renderer) = &mut item.content
                {
                    renderer.on_file_picked(path);
                }
            }
            DialogAction::SaveConfig { and_close } => {
                if let Some(path) = paths.first()
                    && self.save_to(path.clone())
                    && and_close
                {
                    self.force_close = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            DialogAction::OpenConfig => {
                if let Some(path) = paths.first() {
                    self.open_from(path.clone());
                }
            }
        }
    }

    /// Draws an RViz-style collapse arrow on one edge of the 3D view (the splitter), vertically centered; returns true when clicked.
    fn collapse_arrow(&self, ui: &egui::Ui, side: Side) -> bool {
        let rect = self.state.view_rect;
        if rect.width() < 1.0 {
            return false;
        }
        let p = theme::ui::palette();
        // Hide while dragging a tab so the arrow doesn't cover egui_dock's edge drop zones (blocking re-docking).
        if ui.ctx().dragged_id().is_some() {
            return false;
        }
        let size = egui::vec2(14.0, 46.0);
        let collapsed = match side {
            Side::Left => self.left_collapsed,
            Side::Right => self.right_collapsed,
        };
        // Arrow points toward where the pane would go: outward to collapse, inward (toward the view) to reveal.
        let (arrow, hint, x) = match side {
            Side::Left if collapsed => ("›", "Show panels", rect.left()),
            Side::Left => ("‹", "Hide panels", rect.left()),
            Side::Right if collapsed => ("‹", "Show panels", rect.right() - size.x),
            Side::Right => ("›", "Hide panels", rect.right() - size.x),
        };
        let pos = egui::pos2(x, rect.center().y - size.y / 2.0);
        let id = match side {
            Side::Left => "collapse_arrow_left",
            Side::Right => "collapse_arrow_right",
        };
        let mut clicked = false;
        egui::Area::new(egui::Id::new(id))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ui.ctx(), |ui| {
                let button = egui::Button::new(egui::RichText::new(arrow).color(p.text_primary))
                    .fill(p.overlay_bg)
                    .corner_radius(4);
                if ui.add_sized(size, button).on_hover_text(hint).clicked() {
                    clicked = true;
                }
            });
        clicked
    }

    /// Draws the Add-display modal: RViz's two entry points, By display type (pick a display, assign its topic later) and By topic (pick a topic, its display type follows).
    fn show_add_dialog(&mut self, ctx: &egui::Context) {
        if !self.state.add_dialog_open {
            return;
        }
        let p = theme::ui::palette();
        let mut do_add = false;
        let mut close = false;
        let registry = Arc::clone(&self.state.registry);
        let response = egui::Modal::new(egui::Id::new("add_display_modal")).show(ctx, |ui| {
            ui.set_width(460.0);
            ui.heading("Add display");
            ui.horizontal(|ui| {
                for (tab, text) in [
                    (AddDialogTab::DisplayType, "By display type"),
                    (AddDialogTab::Topic, "By topic"),
                ] {
                    if ui
                        .selectable_label(self.state.add_dialog_tab == tab, text)
                        .clicked()
                    {
                        self.state.add_dialog_tab = tab;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Filter").color(p.text_muted));
                ui.text_edit_singleline(&mut self.state.add_dialog_filter);
            });
            ui.add_space(4.0);
            let can_add = egui::Frame::default()
                .fill(p.bg_app)
                .corner_radius(4)
                .inner_margin(egui::Margin::same(4))
                .show(ui, |ui| {
                    ui.set_height(320.0);
                    match self.state.add_dialog_tab {
                        AddDialogTab::DisplayType => self.add_dialog_by_display_type(ui, &registry),
                        AddDialogTab::Topic => self.add_dialog_by_topic(ui, &registry),
                    }
                })
                .inner;
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(can_add, egui::Button::new("Add")).clicked() {
                    do_add = true;
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
        if response.should_close() {
            close = true;
        }
        if do_add {
            match self.state.add_dialog_tab {
                AddDialogTab::DisplayType => {
                    if let Some(index) = self.state.add_dialog_display_index {
                        self.state.add_display_type_item(index);
                    }
                }
                AddDialogTab::Topic => {
                    if let Some(name) = self.state.add_dialog_selected.clone()
                        && let Some(row) =
                            self.state.topics.iter().find(|t| t.name == name).cloned()
                    {
                        let key = self.state.add_dialog_display_type.clone();
                        self.state.add_display_item(row, key.as_deref());
                    }
                }
            }
            self.close_add_dialog();
        } else if close {
            self.close_add_dialog();
        }
    }

    /// By-display-type tab: every registered 3D display grouped by who provides it. Returns whether something is selected.
    fn add_dialog_by_display_type(&mut self, ui: &mut egui::Ui, registry: &Registry) -> bool {
        let p = theme::ui::palette();
        let filter = self.state.add_dialog_filter.to_lowercase();
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                let mut shown = 0;
                for (provider, rows) in display_types_by_provider(registry, &filter) {
                    ui.label(egui::RichText::new(provider).color(p.text_muted).small());
                    for row in rows {
                        shown += 1;
                        let selected = self.state.add_dialog_display_index == Some(row.index);
                        let hit = ui
                            .indent(row.index, |ui| {
                                ui.horizontal(|ui| {
                                    let hit = ui.selectable_label(selected, &row.label);
                                    // Standalone displays take no topic, which is what lets them be added here with nothing else to pick.
                                    let detail = match row.ros_type.is_empty() {
                                        true => "(no topic)",
                                        false => row.ros_type.as_str(),
                                    };
                                    ui.label(
                                        egui::RichText::new(detail).color(p.text_muted).small(),
                                    );
                                    hit
                                })
                                .inner
                            })
                            .inner;
                        if hit.clicked() {
                            self.state.add_dialog_display_index = Some(row.index);
                        }
                    }
                }
                if shown == 0 {
                    ui.colored_label(p.text_muted, "no display type matches the filter");
                }
            });
        self.state.add_dialog_display_index.is_some()
    }

    /// By-topic tab: the filterable topic list, plus a display-type picker when the highlighted topic has more than one. Returns whether something is selected.
    fn add_dialog_by_topic(&mut self, ui: &mut egui::Ui, registry: &Registry) -> bool {
        let p = theme::ui::palette();
        let addable = addable_topics(&self.state.topics, |r| {
            is_supported(registry, &r.name, &r.ros_type)
        });
        // Only topics that genuinely cannot take another item are inert; a topic already carrying a 3D display can take another.
        let added: HashSet<String> = self
            .state
            .items
            .iter()
            .filter(|i| i.content.is_view2d())
            .filter_map(|i| i.topic().map(str::to_owned))
            .collect();
        let before = self.state.add_dialog_selected.clone();
        topic_list::show_selector(
            ui,
            &addable,
            &mut self.state.group_by,
            &self.state.add_dialog_filter,
            &added,
            &mut self.state.add_dialog_selected,
        );
        // A newly highlighted topic starts on its default display type.
        if self.state.add_dialog_selected != before {
            self.state.add_dialog_display_type = None;
        }
        let choices = self
            .state
            .add_dialog_selected
            .as_ref()
            .and_then(|name| self.state.topics.iter().find(|t| &t.name == name))
            .map(|row| display_type_choices(registry, row))
            .unwrap_or_default();
        // Only worth a row when the topic really has a choice; almost every type resolves to exactly one.
        if choices.len() > 1 {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Display type").color(p.text_muted));
                let selected = self
                    .state
                    .add_dialog_display_type
                    .clone()
                    .unwrap_or_else(|| choices[0].0.clone());
                let shown = choices
                    .iter()
                    .find(|(key, _)| *key == selected)
                    .map_or("", |(_, label)| label.as_str());
                egui::ComboBox::from_id_salt("add_display_type")
                    .selected_text(shown)
                    .show_ui(ui, |ui| {
                        for (key, label) in &choices {
                            if ui.selectable_label(&selected == key, label).clicked() {
                                self.state.add_dialog_display_type = Some(key.clone());
                            }
                        }
                    });
            });
        }
        self.state.add_dialog_selected.is_some()
    }

    /// Lists loaded plugins with what each registered, plus every startup problem (FR-8.3).
    fn show_plugins_dialog(&mut self, ctx: &egui::Context) {
        if !self.state.plugins_dialog_open {
            return;
        }
        let p = theme::ui::palette();
        let registry = Arc::clone(&self.state.registry);
        let mut open = true;
        egui::Window::new("Plugins")
            .open(&mut open)
            .default_width(520.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(420.0)
                    .show(ui, |ui| {
                        plugin_section(ui, &registry, &PluginId::Builtin, "visor (builtin)", "");
                        for info in registry.plugins() {
                            let id = PluginId::Plugin(info.id.to_owned());
                            plugin_section(ui, &registry, &id, info.name, info.version);
                        }
                        let env_msgs = registry.msg_names(&PluginId::Env);
                        if !env_msgs.is_empty() {
                            plugin_section(ui, &registry, &PluginId::Env, "VISOR_MSG_PATHS", "");
                        }
                        if registry.plugins().is_empty() {
                            ui.colored_label(p.text_muted, "no plugins registered");
                        }
                        if self.state.startup_problems.is_empty() {
                            return;
                        }
                        ui.separator();
                        ui.label(
                            egui::RichText::new("Problems")
                                .color(p.status_warn)
                                .strong(),
                        );
                        for problem in &self.state.startup_problems {
                            ui.colored_label(p.status_warn, problem.to_string());
                        }
                    });
            });
        self.state.plugins_dialog_open = open;
    }

    /// Resets and hides the Add-display dialog.
    fn close_add_dialog(&mut self) {
        self.state.add_dialog_open = false;
        self.state.add_dialog_selected = None;
        self.state.add_dialog_display_type = None;
        self.state.add_dialog_display_index = None;
        self.state.add_dialog_filter.clear();
    }

    /// Intercepts the window close (X) and shows the confirmation modal if there are unsaved changes.
    fn handle_close(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.viewport().close_requested())
            && !self.force_close
            && self.state.is_dirty()
        {
            // Cancel this frame's close and open the modal.
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.close_dialog_open = true;
        }
        if !self.close_dialog_open {
            return;
        }
        egui::Modal::new(egui::Id::new("unsaved_changes_modal")).show(ctx, |ui| {
            ui.set_width(320.0);
            ui.heading("Unsaved changes");
            ui.label("Save the current config before quitting?");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save and quit").clicked() {
                    self.close_dialog_open = false;
                    // Quit only once the save succeeds; a cancelled Save As leaves the app open.
                    self.save_active(ctx, true);
                }
                if ui.button("Quit without saving").clicked() {
                    self.close_dialog_open = false;
                    self.force_close = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                if ui.button("Cancel").clicked() {
                    self.close_dialog_open = false;
                }
            });
        });
    }
}

struct ViewerTabViewer<'a> {
    state: &'a mut AppState,
}

impl egui_dock::TabViewer for ViewerTabViewer<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        let text = match tab {
            Tab::View3d => "3D View".to_owned(),
            Tab::Topics => "Topics".to_owned(),
            Tab::Frames => "Frames".to_owned(),
            Tab::Displays => "Displays".to_owned(),
            // Show just the leaf topic name so multiple camera tabs stay readable.
            Tab::Image(topic) => topic.rsplit('/').next().unwrap_or(topic).to_owned(),
            Tab::Plugin(key) => match self.state.registry.find_panel(key) {
                Some(entry) => entry.descriptor.title.clone(),
                None => key.clone(),
            },
        };
        // Tabs carry the display face (egui_dock lays titles out with the shared Button style, so the family is set per title).
        theme::display_text(text).size(13.0).into()
    }

    /// Only image tabs carry a ✕ (which deletes that display). Displays/Frames/View3d are hidden via the side collapse arrows instead.
    fn is_closeable(&self, tab: &Self::Tab) -> bool {
        matches!(tab, Tab::Image(_))
    }

    /// Closing an image tab removes its display item (Displays panel and dock stay in sync).
    fn on_close(&mut self, tab: &mut Self::Tab) -> egui_dock::widgets::tab_viewer::OnCloseResponse {
        if let Tab::Image(topic) = tab {
            self.state.remove_image_item(topic);
        }
        egui_dock::widgets::tab_viewer::OnCloseResponse::Close
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::View3d => {
                // Remember the view's screen rect so ViewerApp can place the collapse arrow on its left edge.
                self.state.view_rect = ui.max_rect();
                // Extraction (scene calls) happens in app.rs, which knows render, keeping the policy of no ui -> render dependency.
                let mut item_scenes = Vec::new();
                let live_ids: Vec<DisplayItemId> = self.state.items.iter().map(|i| i.id).collect();
                if let Some(fixed) = self.state.fixed_frame.as_deref() {
                    let tf = TfContext {
                        buffer: &self.state.tf_buffer,
                        fixed_frame: fixed,
                    };
                    for item in &mut self.state.items {
                        if !item.visible {
                            continue;
                        }
                        // Only 3D renderers contribute to the wgpu viewport; image items render in their own tab.
                        let DisplayContent::Scene(renderer) = &mut item.content else {
                            continue;
                        };
                        match renderer.scene(&tf) {
                            Ok(batches) => {
                                item.status = None;
                                item_scenes.push(ItemScene {
                                    id: item.id,
                                    batches,
                                });
                            }
                            Err(status) => item.status = Some(status),
                        }
                    }
                }
                viewport::show(
                    ui,
                    &mut self.state.viewport,
                    &self.state.tf_buffer,
                    self.state.fixed_frame.as_deref(),
                    item_scenes,
                    live_ids,
                );
            }
            // Legacy tab kept only for old-config deserialization; rebuilt away on load, so unreachable in practice.
            Tab::Topics => {}
            Tab::Frames => self.state.frames_ui(ui),
            Tab::Displays => self.state.displays_ui(ui),
            Tab::Image(topic) => {
                // Find the 2D view for this topic; a stale tab with no item shows a harmless notice.
                let view = self.state.items.iter_mut().find_map(|item| {
                    match (item.topic() == Some(topic.as_str()), &mut item.content) {
                        (true, DisplayContent::View2d(view)) => Some(view),
                        _ => None,
                    }
                });
                match view {
                    Some(view) => view.ui(ui, topic),
                    None => {
                        ui.colored_label(
                            theme::ui::palette().text_muted,
                            "no 2D display for this topic",
                        );
                    }
                }
            }
            Tab::Plugin(key) => self.state.plugin_panel_ui(ui, key),
        }
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.state.drain_channels();
        self.poll_dialog(ui.ctx());
        // A finished background load has to reach the screen without user input, but only while one is running.
        if self.state.poll_renderers() {
            ui.ctx().request_repaint_after(REPAINT_MIN_INTERVAL);
        }
        let mut command = None;
        let mut panel_toggle = None;
        egui::Panel::top("menu_bar").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open…").clicked() {
                        command = Some(ConfigCommand::Open);
                    }
                    if ui.button("Save").clicked() {
                        command = Some(ConfigCommand::Save);
                    }
                    if ui.button("Save As…").clicked() {
                        command = Some(ConfigCommand::SaveAs);
                    }
                });
                // Source menu: the modes are exclusive, so radios both show which one is live and switch to the other.
                ui.menu_button("Source", |ui| {
                    let is_files = self.state.source.is_files();
                    let live = format!("Live — {}", self.state.live_config.endpoint);
                    if ui
                        .radio(!is_files, live)
                        .on_hover_text("Connect to the zenoh router (stops file playback)")
                        .clicked()
                    {
                        command = Some(ConfigCommand::ConnectLive);
                    }
                    for entry in self.state.registry.source_entries() {
                        let selected = is_files
                            && self.state.source.mode_tag() == entry.descriptor.id.to_uppercase();
                        if ui
                            .radio(selected, format!("{}…", entry.descriptor.label))
                            .clicked()
                        {
                            command = Some(ConfigCommand::OpenFiles(entry.descriptor.id.clone()));
                        }
                    }
                    // Editable connection: Connect applies the fields to live_config and reconnects.
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Endpoint");
                        ui.text_edit_singleline(&mut self.state.endpoint_edit);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Domain id");
                        ui.text_edit_singleline(&mut self.state.domain_edit);
                    });
                    let domain = self.state.domain_edit.trim().parse::<u32>();
                    let endpoint = self.state.endpoint_edit.trim();
                    let ready = domain.is_ok() && !endpoint.is_empty();
                    if ui
                        .add_enabled(ready, egui::Button::new("Connect"))
                        .on_hover_text("Connect to this endpoint (stops file playback)")
                        .clicked()
                    {
                        self.state.live_config = CommConfig {
                            endpoint: endpoint.to_owned(),
                            domain_id: domain.expect("gated by ready"),
                        };
                        command = Some(ConfigCommand::ConnectLive);
                        ui.close();
                    }
                });
                // Panels menu (RViz-style): explicit show/hide per panel, and the way back if one is ever removed.
                ui.menu_button("Panels", |ui| {
                    for (label, tab) in [
                        ("Displays".to_owned(), Tab::Displays),
                        ("Frames".to_owned(), Tab::Frames),
                    ]
                    .into_iter()
                    .chain(
                        self.state
                            .registry
                            .panel_entries()
                            .iter()
                            .map(|e| (e.descriptor.title.clone(), Tab::Plugin(e.key.clone()))),
                    ) {
                        let mut shown = has_tab(self.logical_dock(), &tab);
                        if ui.checkbox(&mut shown, label).clicked() {
                            panel_toggle = Some(tab);
                        }
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.menu_button("Theme", |ui| {
                        for (theme, label) in
                            [(egui::Theme::Dark, "Dark"), (egui::Theme::Light, "Light")]
                        {
                            if ui.radio(self.state.theme == theme, label).clicked() {
                                self.state.theme = theme;
                                ui.close();
                            }
                        }
                    });
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Plugins…").clicked() {
                        self.state.plugins_dialog_open = true;
                        ui.close();
                    }
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        theme::display_text("VISOR")
                            .color(theme::ui::palette().accent)
                            .strong(),
                    );
                });
            });
        });
        egui::Panel::bottom("status_bar").show(ui, |ui| {
            self.state.status_bar(ui);
        });
        // Added after the status bar so it sits above it; only bag mode has a transport to show (Q12 = A).
        if self.state.bag.is_some() {
            egui::Panel::bottom("timeline").show(ui, |ui| {
                self.state.timeline_ui(ui);
            });
        }
        DockArea::new(&mut self.dock_state)
            .style(dock_style(ui.style().as_ref()))
            // Only image tabs get a ✕; drop the close-all and the (non-reclaiming) leaf collapse buttons.
            .show_leaf_collapse_buttons(false)
            .show_leaf_close_all_buttons(false)
            .show_inside(
                ui,
                &mut ViewerTabViewer {
                    state: &mut self.state,
                },
            );
        // Show a side's collapse arrow only when it has a pane (or is already collapsed), so an empty edge stays free for docking.
        let cx = self.state.view_rect.center().x;
        let show_left = self.left_collapsed || side_has_content(&self.dock_state, cx, true);
        let show_right = self.right_collapsed || side_has_content(&self.dock_state, cx, false);
        let mut recompute = false;
        if show_left && self.collapse_arrow(ui, Side::Left) {
            self.left_collapsed = !self.left_collapsed;
            recompute = true;
        }
        if show_right && self.collapse_arrow(ui, Side::Right) {
            self.right_collapsed = !self.right_collapsed;
            recompute = true;
        }
        if recompute {
            self.recompute_dock();
        }
        if let Some(command) = command {
            self.handle_config_command(ui.ctx(), command);
        }
        if let Some(tab) = panel_toggle {
            self.toggle_panel(tab);
        }
        // Rename switched image tabs in place before syncing so a moved panel isn't torn down and recreated.
        self.apply_image_renames();
        // Keep image tabs in sync with image items (adds/removes from this frame's actions and config apply).
        self.sync_image_tabs();
        // A visibility toggle changes which image tabs are live; recompute keeps hidden tabs' positions in saved_dock.
        if std::mem::take(&mut self.state.image_visibility_dirty) {
            self.recompute_dock();
        }
        self.show_add_dialog(&ui.ctx().clone());
        self.show_plugins_dialog(&ui.ctx().clone());
        self.handle_close(&ui.ctx().clone());
        // Transport keys, suppressed while a text field (the Add dialog's filter) has focus.
        // `,` / `.` step one message, which positions far more finely than the bar's pixel resolution allows.
        if self.state.bag.is_some() && !ui.ctx().egui_wants_keyboard_input() {
            let (space, back, forward) = ui.ctx().input(|i| {
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::Comma),
                    i.key_pressed(egui::Key::Period),
                )
            });
            if space {
                self.state.toggle_playback();
            }
            if back {
                self.state.apply_timeline_action(TimelineAction::StepBack);
            }
            if forward {
                self.state.apply_timeline_action(TimelineAction::Step);
            }
        }
        // Reconciled at the end of the frame: this Ui's style was snapshotted before it ran, so switching mid-frame would draw the new palette against the old widget fills.
        if self.state.theme != theme::ui::mode() {
            theme::ui::apply(ui.ctx(), self.state.theme);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::session::LIVE_EPOCH;

    /// Registry with only the builtins, which is the state every test here reasons about.
    fn test_registry() -> Arc<Registry> {
        Registry::builtin().finish(&|_| None)
    }

    #[test]
    fn t18_epoch_resets_once_per_generation_and_drops_stale_messages() {
        let mut applied = 0;
        // The live path never moves the epoch, so it never resets anything.
        assert_eq!(epoch_action(&mut applied, LIVE_EPOCH), EpochAction::Accept);
        assert_eq!(epoch_action(&mut applied, LIVE_EPOCH), EpochAction::Accept);
        assert_eq!(applied, 0);
        // A bag's first generation resets once, then the rest of that generation is accepted.
        assert_eq!(epoch_action(&mut applied, 1), EpochAction::Reset);
        assert_eq!(epoch_action(&mut applied, 1), EpochAction::Accept);
        assert_eq!(epoch_action(&mut applied, 1), EpochAction::Accept);
        // A seek bumps it: one reset, then messages still in flight from before are discarded.
        assert_eq!(epoch_action(&mut applied, 2), EpochAction::Reset);
        assert_eq!(epoch_action(&mut applied, 1), EpochAction::Drop);
        assert_eq!(epoch_action(&mut applied, 2), EpochAction::Accept);
        // Skipping generations (a burst of seeks coalesced) still resets exactly once.
        assert_eq!(epoch_action(&mut applied, 7), EpochAction::Reset);
        assert_eq!(epoch_action(&mut applied, 7), EpochAction::Accept);
        assert_eq!(applied, 7);
    }

    #[test]
    fn dock_state_roundtrips_through_toml() {
        let mut dock = DockState::new(vec![Tab::View3d]);
        let surface = dock.main_surface_mut();
        let [_view3d, topics] = surface.split_left(NodeIndex::root(), 0.25, vec![Tab::Topics]);
        surface.split_below(topics, 0.5, vec![Tab::Displays, Tab::Frames]);
        // A dynamic image tab must survive the toml round-trip (topic-name-keyed).
        dock.push_to_first_leaf(Tab::Image("/camera/image_raw".to_owned()));
        let value = toml::Value::try_from(&dock).expect("dock -> toml::Value");
        let back: DockState<Tab> = value.try_into().expect("toml::Value -> dock");
        assert_eq!(
            back.main_surface().num_tabs(),
            dock.main_surface().num_tabs()
        );
        assert!(
            back.iter_all_tabs()
                .any(|(_, tab)| matches!(tab, Tab::Image(topic) if topic == "/camera/image_raw"))
        );
    }

    #[test]
    fn a_plugin_tab_survives_the_toml_round_trip_even_without_that_plugin() {
        let mut dock = default_dock();
        add_side_panel(&mut dock, Tab::Plugin("sample::fleet".to_owned()));
        assert!(has_tab(&dock, &Tab::Plugin("sample::fleet".to_owned())));
        let value = toml::Value::try_from(&dock).expect("dock -> toml::Value");
        // A build without the plugin must still deserialize the layout; the tab then draws a placeholder.
        let back: DockState<Tab> = value.try_into().expect("toml::Value -> dock");
        assert!(has_tab(&back, &Tab::Plugin("sample::fleet".to_owned())));
        assert!(test_registry().find_panel("sample::fleet").is_none());
    }

    #[test]
    fn a_display_from_a_missing_plugin_is_reported_rather_than_resolved_to_a_builtin() {
        let registry = test_registry();
        // Same label as the builtin standalone entry, but attributed to a plugin this build does not have.
        let config = DisplayConfig {
            plugin: "sample".to_owned(),
            ..standalone_config(None)
        };
        let Err(error) = make_standalone_content(&registry, &config) else {
            panic!("a builtin must not answer for a plugin-provided item");
        };
        assert!(error.contains("RobotModel"), "error={error}");
        assert!(error.contains("sample"), "error={error}");
        // The same holds for topic items, whose message names the recorded provider.
        let topic_config = DisplayConfig {
            topic: "/scan".to_owned(),
            ros_type: "sensor_msgs/msg/LaserScan".to_owned(),
            plugin: "sample".to_owned(),
            ..Default::default()
        };
        assert!(
            registry
                .find_renderer_as("sample", "", "/scan", "sensor_msgs/msg/LaserScan")
                .is_none()
        );
        let message = missing_display_type(&topic_config);
        assert!(message.contains("plugin `sample`"), "message={message}");
        // A recorded display type is named too, so a removed alternative is distinguishable from a removed plugin.
        let labelled = DisplayConfig {
            label: "AltDisplay".to_owned(),
            ..topic_config
        };
        let message = missing_display_type(&labelled);
        assert!(
            message.contains("as `AltDisplay`"),
            "message={message}"
        );
        assert!(message.contains("plugin `sample`"), "message={message}");
    }

    #[test]
    fn display_type_choices_lists_the_default_first() {
        let registry = test_registry();
        // Shipping builtins offer exactly one type per topic, so the Add dialog draws no picker.
        assert_eq!(
            display_type_choices(
                &registry,
                &row("/map", "nav_msgs/msg/OccupancyGrid", "h", 1)
            ),
            vec![("Map".to_owned(), "Map".to_owned())]
        );
        assert_eq!(
            display_type_choices(&registry, &row("/image", "sensor_msgs/msg/Image", "h", 1)),
            vec![("Image".to_owned(), "Image".to_owned())]
        );
        assert!(
            display_type_choices(&registry, &row("/chatter", "std_msgs/msg/String", "h", 1))
                .is_empty()
        );
    }

    #[test]
    fn display_types_are_grouped_by_provider_and_filtered_by_label_or_type() {
        let registry = test_registry();
        let groups = display_types_by_provider(&registry, "");
        // A builtin-only build has exactly one group, and standalone entries are in it too.
        let names: Vec<&str> = groups.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec![PluginId::Builtin.display_name()]);
        let rows = &groups[0].1;
        let map = rows
            .iter()
            .find(|r| r.label == "Map")
            .expect("Map is offered");
        assert_eq!(map.ros_type, "nav_msgs/msg/OccupancyGrid");
        assert_eq!(registry.renderer_entries()[map.index].key, "Map");
        // RobotModel is registered twice on purpose (from a topic, from a file), so the rows carry distinct indices.
        let robot: Vec<&DisplayTypeRow> = rows.iter().filter(|r| r.label == "RobotModel").collect();
        assert_eq!(robot.len(), 2);
        assert_eq!(robot[0].ros_type, "std_msgs/msg/String");
        // A standalone display carries no ROS type, which is how the tab marks it as taking no topic.
        assert_eq!(robot[1].ros_type, "");
        assert_ne!(robot[0].index, robot[1].index);
        // The filter matches the label or the ROS type.
        let by_type = display_types_by_provider(&registry, "occupancy");
        let labels: Vec<&str> = by_type[0].1.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, vec!["Map"]);
        let by_label = display_types_by_provider(&registry, "laserscan");
        assert_eq!(by_label[0].1[0].label, "LaserScan");
        assert!(display_types_by_provider(&registry, "nothing").is_empty());
    }

    #[test]
    fn a_picked_display_type_resolves_to_its_own_provider_not_to_a_builtin() {
        use crate::decode::value::Value;
        use crate::plugin::registry::{Registrar, RendererDescriptor};
        use crate::plugin::{PLUGIN_API_VERSION, Plugin, PluginInfo};
        use crate::render::SceneBatch;

        struct NullRenderer;
        impl Renderer for NullRenderer {
            fn on_message(&mut self, _value: &Value) {}
            fn scene(&mut self, _tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
                Err(RenderStatus::NoData)
            }
            fn settings_ui(&mut self, _ui: &mut egui::Ui) {}
        }
        struct MapPlugin;
        impl Plugin for MapPlugin {
            fn info(&self) -> PluginInfo {
                PluginInfo {
                    id: "sample",
                    name: "Test",
                    version: "0.0.0",
                    api_version: PLUGIN_API_VERSION,
                }
            }
            fn register(&self, reg: &mut Registrar<'_>) {
                reg.renderer(RendererDescriptor::topic(
                    "nav_msgs/msg/OccupancyGrid",
                    "MyMap",
                    || Box::new(NullRenderer),
                ));
            }
        }
        let mut registry = Registry::builtin();
        registry.add_plugin(&MapPlugin);
        let registry = registry.finish(&|_| None);
        let map = row("/local_map", "nav_msgs/msg/OccupancyGrid", "h", 1);
        let keys: Vec<String> = display_type_choices(&registry, &map)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(keys, vec!["Map".to_owned(), "sample::MyMap".to_owned()]);
        // The picker hands back a qualified key, so a plugin entry must not be looked up as a builtin.
        assert_eq!(
            display_type_origin(&registry, &map, Some("sample::MyMap")),
            Some(("sample".to_owned(), "MyMap".to_owned()))
        );
        assert_eq!(
            display_type_origin(&registry, &map, Some("Map")),
            Some((String::new(), "Map".to_owned()))
        );
        // No pick = the topic's default, which make_content resolves on its own.
        assert_eq!(
            display_type_origin(&registry, &map, None),
            Some((String::new(), String::new()))
        );
        // A key nothing answers for drops the add instead of falling back to another display type.
        assert_eq!(display_type_origin(&registry, &map, Some("Gone")), None);
    }

    #[test]
    fn default_dock_contains_fixed_panels() {
        let dock = default_dock();
        assert!(has_tab(&dock, &Tab::View3d));
        assert!(has_tab(&dock, &Tab::Displays));
        assert!(has_tab(&dock, &Tab::Frames));
        assert!(!has_tab(&dock, &Tab::Topics));
    }

    #[test]
    fn build_dock_reflects_requested_panels() {
        let both = build_dock(true, true);
        assert!(has_tab(&both, &Tab::View3d));
        assert!(has_tab(&both, &Tab::Displays));
        assert!(has_tab(&both, &Tab::Frames));

        let none = build_dock(false, false);
        assert!(has_tab(&none, &Tab::View3d));
        assert!(!has_tab(&none, &Tab::Displays));
        assert!(!has_tab(&none, &Tab::Frames));

        let displays_only = build_dock(true, false);
        assert!(has_tab(&displays_only, &Tab::Displays));
        assert!(!has_tab(&displays_only, &Tab::Frames));
        assert!(has_tab(&displays_only, &Tab::View3d));
    }

    #[test]
    fn addable_topics_keeps_only_visualizable_types() {
        let topics = vec![
            row("/scan", "sensor_msgs/msg/LaserScan", "h1", 1),
            row("/chatter", "std_msgs/msg/String", "h2", 1),
        ];
        let registry = test_registry();
        let addable = addable_topics(&topics, |r| is_supported(&registry, &r.name, &r.ros_type));
        let names: Vec<&str> = addable.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["/scan"]);
    }

    fn builtin_origin(label: &str) -> ItemOrigin {
        ItemOrigin {
            plugin: PluginId::Builtin,
            label: label.to_owned(),
        }
    }

    #[test]
    fn compatible_topics_lists_same_type_and_includes_current() {
        let registry = test_registry();
        let origin = builtin_origin("LaserScan");
        let topics = vec![
            row("/scan", "sensor_msgs/msg/LaserScan", "h1", 1),
            row("/scan2", "sensor_msgs/msg/LaserScan", "h2", 1),
            row("/points", "sensor_msgs/msg/PointCloud2", "h3", 1),
        ];
        let ros_type = "sensor_msgs/msg/LaserScan";
        let compat = compatible_topics(&registry, &topics, &origin, ros_type, Some("/scan"));
        assert_eq!(compat, vec!["/scan".to_owned(), "/scan2".to_owned()]);
        // A current topic missing from the graph is still offered.
        let compat = compatible_topics(&registry, &topics, &origin, ros_type, Some("/gone"));
        assert!(compat.contains(&"/gone".to_owned()));
        assert!(compat.contains(&"/scan".to_owned()));
        assert!(!compat.iter().any(|n| n == "/points"));
        // Added by display type: every matching topic is offered, and nothing is pinned.
        let compat = compatible_topics(&registry, &topics, &origin, ros_type, None);
        assert_eq!(compat, vec!["/scan".to_owned(), "/scan2".to_owned()]);
    }

    #[test]
    fn compatible_topics_respects_the_topic_name_filter() {
        let registry = test_registry();
        let topics = vec![
            row("/robot_description", "std_msgs/msg/String", "h1", 1),
            row("/robot1/robot_description", "std_msgs/msg/String", "h2", 1),
            row("/chatter", "std_msgs/msg/String", "h3", 1),
        ];
        // Same type, different display type: switching a RobotModel to /chatter would draw nothing.
        let compat = compatible_topics(
            &registry,
            &topics,
            &builtin_origin("RobotModel"),
            "std_msgs/msg/String",
            Some("/robot_description"),
        );
        assert_eq!(
            compat,
            vec![
                "/robot1/robot_description".to_owned(),
                "/robot_description".to_owned()
            ]
        );
        // Types outside the 3D registry (images) keep the type-only rule.
        let images = vec![
            row("/cam_a/image_raw", "sensor_msgs/msg/Image", "h4", 1),
            row("/cam_b/image_raw", "sensor_msgs/msg/Image", "h5", 1),
        ];
        let compat = compatible_topics(
            &registry,
            &images,
            &builtin_origin("Image"),
            "sensor_msgs/msg/Image",
            Some("/cam_a/image_raw"),
        );
        assert_eq!(compat.len(), 2);
    }

    #[test]
    fn image_tabs_go_to_a_separate_leaf_and_group_together() {
        let mut dock = default_dock();
        push_image_tab_into(&mut dock, "/cam_a".to_owned());
        let (v3d, _) = dock.main_surface().find_tab(&Tab::View3d).unwrap();
        let (img_a, _) = dock
            .main_surface()
            .find_tab(&Tab::Image("/cam_a".to_owned()))
            .unwrap();
        // First image must not land in the 3D View leaf.
        assert_ne!(v3d, img_a);
        // A second image groups into the same leaf as the first.
        push_image_tab_into(&mut dock, "/cam_b".to_owned());
        let (img_b, _) = dock
            .main_surface()
            .find_tab(&Tab::Image("/cam_b".to_owned()))
            .unwrap();
        assert_eq!(img_a, img_b);
    }

    #[test]
    fn rename_image_tab_keeps_position_and_swaps_key() {
        let mut dock = default_dock();
        push_image_tab_into(&mut dock, "/cam_a".to_owned());
        let (before, _) = dock
            .main_surface()
            .find_tab(&Tab::Image("/cam_a".to_owned()))
            .unwrap();
        rename_image_tab(&mut dock, "/cam_a", "/cam_c");
        assert!(!has_tab(&dock, &Tab::Image("/cam_a".to_owned())));
        let (after, _) = dock
            .main_surface()
            .find_tab(&Tab::Image("/cam_c".to_owned()))
            .unwrap();
        // Same leaf as before: the moved panel is not torn down.
        assert_eq!(before, after);
    }

    #[test]
    fn group_by_label_roundtrips() {
        for g in [GroupBy::Flat, GroupBy::Namespace, GroupBy::Type] {
            assert_eq!(group_by_from_label(group_by_label(g)), g);
        }
        assert_eq!(group_by_from_label("unknown"), GroupBy::Namespace);
    }

    #[test]
    fn restored_offline_item_resubscribes_when_topic_appears() {
        // Offline item restored from config (type_hash unresolved, in_graph false, pub_count 0).
        let mut item = DisplayTracking {
            in_graph: false,
            pub_count: 0,
            ..tracking("/scan", "")
        };
        assert!(plan_one(0, true, &mut item, &[]).is_empty());
        assert!(!item.in_graph);
        // Saved topic appears with a real hash -> resubscribe + row update + online (AC-6).
        let topics = vec![row("/scan", "sensor_msgs/msg/LaserScan", "h1", 1)];
        assert_eq!(
            plan_one(0, true, &mut item, &topics),
            vec![DisplayAction::Resubscribe {
                id: DisplayItemId(0)
            }]
        );
        assert!(item.in_graph);
        assert_eq!(item.row.type_hash, "h1");
    }

    fn row(name: &str, ros_type: &str, hash: &str, publisher_count: usize) -> TopicRow {
        TopicRow {
            name: name.to_owned(),
            ros_type: ros_type.to_owned(),
            type_name_dds: "tf2_msgs::msg::dds_::TFMessage_".to_owned(),
            type_hash: hash.to_owned(),
            publisher_count,
            subscriber_count: 0,
        }
    }

    /// The automatic rule (nothing pinned), which every pre-existing TF test exercises.
    fn plan_auto(tracking: &mut TfTracking, topics: &[TopicRow]) -> Vec<TfAction> {
        plan_tf_actions(tracking, topics, &TfConfig::default())
    }

    #[test]
    fn subscribes_tf_topics_once_when_they_appear() {
        let mut tracking = TfTracking::default();
        let topics = vec![
            row("/tf", TF_MESSAGE_TYPE, "h1", 1),
            row("/tf_static", TF_MESSAGE_TYPE, "h2", 1),
        ];
        let actions = plan_auto(&mut tracking, &topics);
        assert_eq!(
            actions,
            vec![
                TfAction::Subscribe {
                    row: topics[0].clone(),
                    is_static: false,
                },
                TfAction::Subscribe {
                    row: topics[1].clone(),
                    is_static: true,
                },
            ]
        );
        assert!(tracking.subscribed());
        // Re-evaluating the same snapshot does nothing.
        assert!(plan_auto(&mut tracking, &topics).is_empty());
    }

    #[test]
    fn ignores_tf_topics_with_wrong_type() {
        let mut tracking = TfTracking::default();
        let topics = vec![row("/tf", "std_msgs/msg/String", "h1", 1)];
        assert!(plan_auto(&mut tracking, &topics).is_empty());
        assert!(!tracking.subscribed());
    }

    #[test]
    fn resubscribes_when_type_hash_changes() {
        let mut tracking = TfTracking::default();
        plan_auto(&mut tracking, &[row("/tf", TF_MESSAGE_TYPE, "h1", 1)]);
        let changed = row("/tf", TF_MESSAGE_TYPE, "h2", 1);
        assert_eq!(
            plan_auto(&mut tracking, std::slice::from_ref(&changed)),
            vec![TfAction::Subscribe {
                row: changed,
                is_static: false,
            }]
        );
    }

    #[test]
    fn keeps_subscription_when_topic_disappears_from_graph() {
        let mut tracking = TfTracking::default();
        plan_auto(&mut tracking, &[row("/tf", TF_MESSAGE_TYPE, "h1", 1)]);
        assert!(plan_auto(&mut tracking, &[]).is_empty());
        assert!(tracking.subscribed());
    }

    fn tracking(name: &str, hash: &str) -> DisplayTracking {
        DisplayTracking {
            row: row(name, "sensor_msgs/msg/LaserScan", hash, 1),
            in_graph: true,
            pub_count: 1,
        }
    }

    /// Plan a single topic item (the common shape in these tests).
    fn plan_one(
        id: u64,
        visible: bool,
        tracking: &mut DisplayTracking,
        topics: &[TopicRow],
    ) -> Vec<DisplayAction> {
        let plan = DisplayPlan {
            id: DisplayItemId(id),
            visible,
            tracking: Some(tracking),
        };
        plan_display_actions([plan], topics)
    }

    /// Standalone DisplayConfig for the registered RobotModel renderer.
    fn standalone_config(settings: Option<&str>) -> DisplayConfig {
        DisplayConfig {
            kind: DisplayKind::Standalone,
            label: "RobotModel".to_owned(),
            visible: true,
            settings: settings.map(|s| toml::from_str(s).expect("valid settings")),
            ..Default::default()
        }
    }

    #[test]
    fn standalone_config_restores_the_registered_renderer_with_its_settings() {
        let registry = test_registry();
        let restored = make_standalone_content(
            &registry,
            &standalone_config(Some("path = \"\"\nalpha = 0.5")),
        )
        .expect("RobotModel is registered");
        assert_eq!(restored.origin.label, "RobotModel");
        assert!(restored.origin.plugin.is_builtin());
        assert_eq!(restored.note, None);
        // The settings really reached the renderer (they come back out unchanged).
        let settings = restored.renderer.settings().expect("has settings");
        assert_eq!(
            settings.get("alpha").and_then(toml::Value::as_float),
            Some(0.5)
        );
        // Missing settings is fine (a freshly added item is saved before any source is chosen).
        assert!(make_standalone_content(&registry, &standalone_config(None)).is_ok());
    }

    #[test]
    fn standalone_config_with_unloadable_source_restores_with_a_note() {
        let config = standalone_config(Some("path = \"/nonexistent/robot.urdf\""));
        let restored = make_standalone_content(&test_registry(), &config).expect("still restored");
        let note = restored.note.expect("source failure is reported");
        assert!(note.contains("RobotModel"), "note={note}");
        assert!(note.contains("/nonexistent/robot.urdf"), "note={note}");
    }

    #[test]
    fn standalone_config_with_unknown_label_is_reported_as_missing() {
        let config = DisplayConfig {
            label: "FromTheFuture".to_owned(),
            ..standalone_config(None)
        };
        let Err(error) = make_standalone_content(&test_registry(), &config) else {
            panic!("an unknown label must not resolve to a renderer");
        };
        assert!(error.contains("FromTheFuture"), "error={error}");
    }

    /// A topic item carrying no renderer state, which is all wanted_subscriptions reads.
    fn subscriber(id: u64, topic: &str, hash: &str, visible: bool, in_graph: bool) -> DisplayItem {
        struct NullRenderer;
        impl Renderer for NullRenderer {
            fn on_message(&mut self, _value: &crate::decode::value::Value) {}
            fn scene(
                &mut self,
                _tf: &TfContext<'_>,
            ) -> Result<Vec<crate::render::SceneBatch>, RenderStatus> {
                Err(RenderStatus::NoData)
            }
            fn settings_ui(&mut self, _ui: &mut egui::Ui) {}
        }
        let row = row(topic, "nav_msgs/msg/OccupancyGrid", hash, 1);
        DisplayItem {
            id: DisplayItemId(id),
            origin: builtin_origin("Map"),
            visible,
            ros_type: row.ros_type.clone(),
            tracking: Some(DisplayTracking {
                pub_count: row.publisher_count,
                row,
                in_graph,
            }),
            title: topic.to_owned(),
            content: DisplayContent::Scene(Box::new(NullRenderer)),
            status: None,
            companion: None,
        }
    }

    #[test]
    fn several_items_on_one_topic_want_a_single_subscription() {
        let topics = vec![row("/local_map", "nav_msgs/msg/OccupancyGrid", "h1", 1)];
        // Two display types on the same topic collapse to one entry, so removing one never cuts the other's feed.
        let items = vec![
            subscriber(0, "/local_map", "h1", true, true),
            subscriber(1, "/local_map", "h1", true, true),
        ];
        let wanted = wanted_subscriptions(&items, &topics);
        assert_eq!(wanted.len(), 1);
        assert_eq!(wanted["/local_map"].type_hash, "h1");
        // The topic stays wanted while any one of them is visible, and is dropped once none is.
        let mut items = items;
        items[0].visible = false;
        assert_eq!(wanted_subscriptions(&items, &topics).len(), 1);
        items[1].visible = false;
        assert!(wanted_subscriptions(&items, &topics).is_empty());
    }

    #[test]
    fn hidden_offline_and_standalone_items_want_nothing() {
        let topics = vec![row("/map", "nav_msgs/msg/OccupancyGrid", "h1", 1)];
        assert!(
            wanted_subscriptions(&[subscriber(0, "/map", "h1", false, true)], &topics).is_empty()
        );
        // Offline keeps the item but not the subscription; replan_graph resubscribes when the topic returns.
        assert!(
            wanted_subscriptions(&[subscriber(0, "/map", "h1", true, false)], &topics).is_empty()
        );
        let mut standalone = subscriber(0, "/map", "h1", true, true);
        standalone.tracking = None;
        standalone.ros_type = String::new();
        assert!(wanted_subscriptions(&[standalone], &topics).is_empty());
    }

    #[test]
    fn a_companion_is_wanted_only_while_the_graph_offers_its_type() {
        let mut item = subscriber(0, "/map", "h1", true, true);
        item.companion = Some(CompanionState {
            topic: "/map_updates".to_owned(),
            ros_type: "map_msgs/msg/OccupancyGridUpdate",
        });
        let base = row("/map", "nav_msgs/msg/OccupancyGrid", "h1", 1);
        // Absent from the graph: only the base topic is wanted.
        let wanted = wanted_subscriptions([&item], std::slice::from_ref(&base));
        assert_eq!(wanted.keys().collect::<Vec<_>>(), vec!["/map"]);
        // Present but under another type: still not it (the name alone does not identify a companion).
        let wrong = row("/map_updates", "std_msgs/msg/String", "h2", 1);
        let wanted = wanted_subscriptions([&item], &[base.clone(), wrong]);
        assert_eq!(wanted.keys().collect::<Vec<_>>(), vec!["/map"]);
        let right = row("/map_updates", "map_msgs/msg/OccupancyGridUpdate", "h3", 1);
        let wanted = wanted_subscriptions([&item], &[base, right]);
        let mut names: Vec<&String> = wanted.keys().collect();
        names.sort();
        assert_eq!(names, vec!["/map", "/map_updates"]);
        // A hidden item drops its companion along with its own topic.
        item.visible = false;
        assert!(wanted_subscriptions([&item], &[]).is_empty());
    }

    #[test]
    fn standalone_items_produce_no_actions_on_any_snapshot() {
        let plan = DisplayPlan {
            id: DisplayItemId(0),
            visible: true,
            tracking: None,
        };
        let topics = vec![row("/scan", "sensor_msgs/msg/LaserScan", "h1", 1)];
        assert!(plan_display_actions([plan], &topics).is_empty());
        // A snapshot with nothing in it is equally uneventful (no offline flag to flip either).
        let plan = DisplayPlan {
            id: DisplayItemId(0),
            visible: true,
            tracking: None,
        };
        assert!(plan_display_actions([plan], &[]).is_empty());
    }

    #[test]
    fn display_items_keep_subscription_and_go_offline_when_topic_disappears() {
        let mut item = tracking("/scan", "h1");
        assert!(plan_one(0, true, &mut item, &[]).is_empty());
        assert!(!item.in_graph);
        // Reappears with the same hash -> back online while keeping the subscription (no resubscribe).
        let topics = vec![row("/scan", "sensor_msgs/msg/LaserScan", "h1", 1)];
        assert!(plan_one(0, true, &mut item, &topics).is_empty());
        assert!(item.in_graph);
    }

    #[test]
    fn display_items_resubscribe_on_type_hash_change() {
        let mut items = [tracking("/scan", "h1"), tracking("/scan2", "h1")];
        let topics = vec![
            row("/scan", "sensor_msgs/msg/LaserScan", "h2", 1),
            row("/scan2", "sensor_msgs/msg/LaserScan", "h1", 1),
        ];
        let plans: Vec<DisplayPlan<'_>> = items
            .iter_mut()
            .enumerate()
            .map(|(i, tracking)| DisplayPlan {
                id: DisplayItemId(i as u64),
                visible: true,
                tracking: Some(tracking),
            })
            .collect();
        assert_eq!(
            plan_display_actions(plans, &topics),
            vec![DisplayAction::Resubscribe {
                id: DisplayItemId(0)
            }]
        );
        // row is updated to the new hash, used to build the next resubscribe key.
        assert_eq!(items[0].row.type_hash, "h2");
        assert_eq!(items[1].row.type_hash, "h1");
    }

    #[test]
    fn hidden_display_items_update_row_but_do_not_resubscribe() {
        let mut item = tracking("/points", "h1");
        let topics = vec![row("/points", "sensor_msgs/msg/PointCloud2", "h2", 1)];
        // Hidden = unsubscribed, so no Resubscribe; row is updated for the subscribe key used when toggled ON.
        assert!(plan_one(0, false, &mut item, &topics).is_empty());
        assert_eq!(item.row.type_hash, "h2");
    }

    #[test]
    fn display_items_unchanged_snapshot_produces_no_actions() {
        let mut item = tracking("/scan", "h1");
        let topics = vec![row("/scan", "sensor_msgs/msg/LaserScan", "h1", 1)];
        assert!(plan_one(0, true, &mut item, &topics).is_empty());
        assert!(item.in_graph);
    }

    #[test]
    fn display_items_resubscribe_on_publisher_increase() {
        let mut item = tracking("/map", "h1");
        // A decrease does nothing (pub_count still tracks it).
        let topics = vec![row("/map", "sensor_msgs/msg/LaserScan", "h1", 0)];
        assert!(plan_one(0, true, &mut item, &topics).is_empty());
        assert_eq!(item.pub_count, 0);
        // Increase = late publisher -> resubscribe (re-runs the history get right after declaration).
        let topics = vec![row("/map", "sensor_msgs/msg/LaserScan", "h1", 1)];
        assert_eq!(
            plan_one(0, true, &mut item, &topics),
            vec![DisplayAction::Resubscribe {
                id: DisplayItemId(0)
            }]
        );
        // Same count is a no-op.
        assert!(plan_one(0, true, &mut item, &topics).is_empty());
    }

    #[test]
    fn hidden_display_items_track_publisher_count_without_resubscribe() {
        let mut item = DisplayTracking {
            pub_count: 0,
            ..tracking("/map", "h1")
        };
        let topics = vec![row("/map", "sensor_msgs/msg/LaserScan", "h1", 2)];
        assert!(plan_one(0, false, &mut item, &topics).is_empty());
        assert_eq!(item.pub_count, 2);
    }

    #[test]
    fn static_publisher_increase_triggers_history_refresh() {
        let mut tracking = TfTracking::default();
        plan_auto(
            &mut tracking,
            &[row("/tf_static", TF_MESSAGE_TYPE, "h1", 1)],
        );
        // A decrease does nothing.
        assert!(
            plan_auto(
                &mut tracking,
                &[row("/tf_static", TF_MESSAGE_TYPE, "h1", 0)]
            )
            .is_empty()
        );
        // Recovery (increase) re-queries to fetch the late publisher's cache.
        assert_eq!(
            plan_auto(
                &mut tracking,
                &[row("/tf_static", TF_MESSAGE_TYPE, "h1", 1)]
            ),
            vec![TfAction::RefreshStatic]
        );
        assert_eq!(
            plan_auto(
                &mut tracking,
                &[row("/tf_static", TF_MESSAGE_TYPE, "h1", 3)]
            ),
            vec![TfAction::RefreshStatic]
        );
    }

    #[test]
    fn without_tf_any_tfmessage_topic_is_used_and_named() {
        // A recording may carry TF only under a namespaced topic such as /recorded/tf, with no /tf_static at all.
        let mut tracking = TfTracking::default();
        let topics = vec![
            row("/recorded/tf", TF_MESSAGE_TYPE, "bag1", 1),
            row("/odom", "nav_msgs/msg/Odometry", "bag1", 1),
        ];
        assert_eq!(
            plan_auto(&mut tracking, &topics),
            vec![TfAction::Subscribe {
                row: topics[0].clone(),
                is_static: false,
            }]
        );
        assert_eq!(tracking.dyn_topic(), Some("/recorded/tf"));
        assert_eq!(tracking.static_topic(), None);
        // A namespaced pair sorts out by suffix: `robot/tf_static` is the static one, `robot/tf` the dynamic one.
        let mut tracking = TfTracking::default();
        let topics = vec![
            row("/robot/tf_static", TF_MESSAGE_TYPE, "h1", 1),
            row("/robot/tf", TF_MESSAGE_TYPE, "h1", 1),
        ];
        let actions = plan_auto(&mut tracking, &topics);
        assert_eq!(actions.len(), 2);
        assert!(
            matches!(&actions[0], TfAction::Subscribe { row, is_static: false } if row.name == "/robot/tf")
        );
        assert!(
            matches!(&actions[1], TfAction::Subscribe { row, is_static: true } if row.name == "/robot/tf_static")
        );
    }

    #[test]
    fn the_standard_tf_topics_win_over_other_tfmessage_topics() {
        let mut tracking = TfTracking::default();
        let topics = vec![
            row("/recorded/tf", TF_MESSAGE_TYPE, "h1", 1),
            row("/tf", TF_MESSAGE_TYPE, "h1", 1),
        ];
        let actions = plan_auto(&mut tracking, &topics);
        assert_eq!(actions.len(), 1);
        assert_eq!(tracking.dyn_topic(), Some("/tf"));
    }

    #[test]
    fn a_pinned_topic_wins_and_is_not_substituted_when_absent() {
        let pins = TfConfig {
            dynamic_topic: Some("/recorded/tf".to_owned()),
            static_topic: None,
        };
        let mut tracking = TfTracking::default();
        let topics = vec![
            row("/tf", TF_MESSAGE_TYPE, "h1", 1),
            row("/recorded/tf", TF_MESSAGE_TYPE, "h1", 1),
            row("/tf_static", TF_MESSAGE_TYPE, "h1", 1),
        ];
        let actions = plan_tf_actions(&mut tracking, &topics, &pins);
        assert_eq!(tracking.dyn_topic(), Some("/recorded/tf"));
        // The static kind is not pinned, so it still follows the rule.
        assert_eq!(tracking.static_topic(), Some("/tf_static"));
        assert_eq!(actions.len(), 2);
        // Pinning a topic that is not in the graph subscribes nothing rather than quietly taking /tf.
        let mut tracking = TfTracking::default();
        let pins = TfConfig {
            dynamic_topic: Some("/elsewhere/tf".to_owned()),
            static_topic: None,
        };
        plan_tf_actions(&mut tracking, &topics, &pins);
        assert_eq!(tracking.dyn_topic(), None);
        assert_eq!(tracking.static_topic(), Some("/tf_static"));
    }

    #[test]
    fn changing_the_topic_resubscribes_even_when_the_hash_is_the_same() {
        // Every row of one bag shares the same type_hash, so the topic name has to be part of the key.
        let mut tracking = TfTracking::default();
        plan_auto(
            &mut tracking,
            &[row("/recorded/tf", TF_MESSAGE_TYPE, "bag1", 1)],
        );
        let pins = TfConfig {
            dynamic_topic: Some("/other/tf".to_owned()),
            static_topic: None,
        };
        let topics = [
            row("/recorded/tf", TF_MESSAGE_TYPE, "bag1", 1),
            row("/other/tf", TF_MESSAGE_TYPE, "bag1", 1),
        ];
        assert_eq!(
            plan_tf_actions(&mut tracking, &topics, &pins),
            vec![TfAction::Subscribe {
                row: topics[1].clone(),
                is_static: false,
            }]
        );
    }
}
