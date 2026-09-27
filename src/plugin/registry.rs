//! Runtime registry of every extension point, built once at startup from builtins plus plugins.

use std::borrow::Cow;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;

use crate::comm::session::Notify;
use crate::decode::msg_parser::TypeRegistry;
use crate::render::Renderer;
use crate::source::{SourceBackend, SourceChannels};

use super::ids::{PluginId, is_valid_plugin_id, plugin_id_from_config};
use super::panel::PanelPlugin;
use super::types::{self, MsgEntry};
use super::view2d::View2d;
use super::{PLUGIN_API_VERSION, Plugin, PluginInfo};

/// Factory for a 3D renderer, called once per added display item.
pub type MakeRenderer = Arc<dyn Fn() -> Box<dyn Renderer> + Send + Sync>;
/// Extra topic-name condition for types the ROS type alone cannot decide.
pub type TopicFilter = Arc<dyn Fn(&str) -> bool + Send + Sync>;
/// Factory for a 2D view, called once per added display item.
pub type MakeView2d = Arc<dyn Fn() -> Box<dyn View2d> + Send + Sync>;
/// Factory for a dock panel, called the first time that panel is opened.
pub type MakePanel = Arc<dyn Fn() -> Box<dyn PanelPlugin> + Send + Sync>;
/// Factory for a file-backed data source: `(paths, generation, channels, notify, app type registry)`; the registry is the fallback for files whose own definitions are missing or unreadable.
pub type MakeSource = Arc<
    dyn Fn(
            &[PathBuf],
            u64,
            SourceChannels,
            Notify,
            Arc<TypeRegistry>,
        ) -> Result<Box<dyn SourceBackend>, String>
        + Send
        + Sync,
>;

/// One 3D display type. `topic` / `topic_filtered` / `standalone` are the three shapes a renderer can take.
pub struct RendererDescriptor {
    /// Short type name shown in the UI, and the stable key that selects a standalone renderer in a config.
    pub label: String,
    /// Supported schema name in ROS form; empty for standalone entries.
    pub ros_type: String,
    /// Needs no topic subscription (the Add dialog shows it as taking no topic); never matched against topics.
    pub standalone: bool,
    pub topic_filter: Option<TopicFilter>,
    pub make: MakeRenderer,
    /// Left stripe color of the Displays card; None falls back to `theme::display_accent`.
    pub accent: Option<egui::Color32>,
}

impl RendererDescriptor {
    /// Topic-subscribing renderer keyed by type name alone (the common case).
    pub fn topic(
        ros_type: impl Into<String>,
        label: impl Into<String>,
        make: impl Fn() -> Box<dyn Renderer> + Send + Sync + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            ros_type: ros_type.into(),
            standalone: false,
            topic_filter: None,
            make: Arc::new(make),
            accent: None,
        }
    }

    /// Topic-subscribing renderer that also requires the topic name to satisfy `filter`.
    pub fn topic_filtered(
        ros_type: impl Into<String>,
        label: impl Into<String>,
        make: impl Fn() -> Box<dyn Renderer> + Send + Sync + 'static,
        filter: impl Fn(&str) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            topic_filter: Some(Arc::new(filter)),
            ..Self::topic(ros_type, label, make)
        }
    }

    /// Non-topic renderer added by label, sourced from something other than a subscription (files, TF, parameters).
    pub fn standalone(
        label: impl Into<String>,
        make: impl Fn() -> Box<dyn Renderer> + Send + Sync + 'static,
    ) -> Self {
        Self {
            standalone: true,
            ..Self::topic(String::new(), label, make)
        }
    }

    pub fn with_accent(mut self, color: egui::Color32) -> Self {
        self.accent = Some(color);
        self
    }

    /// Whether this entry answers for the given topic (standalone entries never do).
    fn matches(&self, topic: &str, ros_type: &str) -> bool {
        !self.standalone
            && self.ros_type == ros_type
            && self.topic_filter.as_ref().is_none_or(|f| f(topic))
    }
}

/// One 2D display type, keyed by ROS type name (2D views have no standalone or filtered shape).
pub struct View2dDescriptor {
    pub label: String,
    pub ros_type: String,
    pub make: MakeView2d,
    pub accent: Option<egui::Color32>,
}

impl View2dDescriptor {
    pub fn new(
        ros_type: impl Into<String>,
        label: impl Into<String>,
        make: impl Fn() -> Box<dyn View2d> + Send + Sync + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            ros_type: ros_type.into(),
            make: Arc::new(make),
            accent: None,
        }
    }

    pub fn with_accent(mut self, color: egui::Color32) -> Self {
        self.accent = Some(color);
        self
    }
}

/// One dock panel offered in the Panels menu.
pub struct PanelDescriptor {
    /// Stable id, namespaced by the plugin; part of the persisted dock layout.
    pub id: String,
    /// Tab title.
    pub title: String,
    pub make: MakePanel,
}

impl PanelDescriptor {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        make: impl Fn() -> Box<dyn PanelPlugin> + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            make: Arc::new(make),
        }
    }
}

/// One file-backed data source (the Live connection is not a descriptor; it needs no files).
pub struct SourceDescriptor {
    pub id: String,
    /// Shown in the Source menu, the file dialog filter, and the status-bar mode chip.
    pub label: String,
    /// Accepted extensions without the dot.
    pub extensions: Vec<String>,
    /// The Source menu opens a folder picker instead of a file picker (a rosbag2 bag is a directory); several folders merge like several files. Extensions still route paths given on the command line.
    pub pick_folders: bool,
    pub make: MakeSource,
}

impl SourceDescriptor {
    /// Make the Source menu pick directories for this source; `Source::start` expands each one to its files before choosing the descriptor.
    pub fn folders(mut self) -> Self {
        self.pick_folders = true;
        self
    }

    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        extensions: &[&str],
        make: impl Fn(
            &[PathBuf],
            u64,
            SourceChannels,
            Notify,
            Arc<TypeRegistry>,
        ) -> Result<Box<dyn SourceBackend>, String>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            extensions: extensions.iter().map(|e| (*e).to_owned()).collect(),
            pick_folders: false,
            make: Arc::new(make),
        }
    }
}

/// A registered 3D display type together with who registered it.
pub struct RendererEntry {
    pub plugin: PluginId,
    /// Qualified key (`label` for builtin, `plugin::label` otherwise); used for display and duplicate detection.
    pub key: String,
    pub descriptor: RendererDescriptor,
}

impl RendererEntry {
    /// Construct the renderer, containing a panic in the factory so one bad plugin cannot take the app down.
    pub fn make(&self) -> Result<Box<dyn Renderer>, String> {
        guard(&self.key, || (self.descriptor.make)())
    }

    /// Whether this display type answers for the given topic (standalone entries never do).
    pub fn matches(&self, topic: &str, ros_type: &str) -> bool {
        self.descriptor.matches(topic, ros_type)
    }

    /// Card stripe color: the descriptor's own accent, else the theme's label-keyed fallback.
    pub fn accent(&self) -> egui::Color32 {
        self.descriptor
            .accent
            .unwrap_or_else(|| crate::theme::display_accent(&self.descriptor.label))
    }
}

/// A registered 2D display type together with who registered it.
pub struct View2dEntry {
    pub plugin: PluginId,
    pub key: String,
    pub descriptor: View2dDescriptor,
}

impl View2dEntry {
    pub fn make(&self) -> Result<Box<dyn View2d>, String> {
        guard(&self.key, || (self.descriptor.make)())
    }

    pub fn accent(&self) -> egui::Color32 {
        self.descriptor
            .accent
            .unwrap_or_else(|| crate::theme::display_accent(&self.descriptor.label))
    }
}

/// A registered dock panel together with who registered it.
pub struct PanelEntry {
    pub plugin: PluginId,
    /// Qualified panel key; this is what the persisted dock layout stores.
    pub key: String,
    pub descriptor: PanelDescriptor,
}

impl PanelEntry {
    pub fn make(&self) -> Result<Box<dyn PanelPlugin>, String> {
        guard(&self.key, || (self.descriptor.make)())
    }
}

/// A registered data source together with who registered it.
pub struct SourceEntry {
    pub plugin: PluginId,
    pub key: String,
    pub descriptor: SourceDescriptor,
}

/// Run a plugin-supplied factory with unwinding contained; the message names the registration that failed.
fn guard<T>(key: &str, f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(AssertUnwindSafe(f))
        .map_err(|_| format!("`{key}` panicked while being constructed"))
}

/// Run a plugin-supplied call with unwinding contained (config restore and other one-shot entry points).
pub fn guarded(what: &str, f: impl FnOnce()) -> Result<(), String> {
    guard(what, f)
}

/// What kind of registration problem was found at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemKind {
    /// The plugin was built against a different `PLUGIN_API_VERSION`.
    ApiVersion,
    /// The plugin id is not a usable namespace.
    InvalidId,
    /// Two plugins claim the same id.
    DuplicatePlugin,
    /// The same `(plugin, label)` standalone display was registered twice.
    DuplicateStandalone,
    /// The same panel key was registered twice.
    DuplicatePanel,
    /// A 2D display type was already registered for that ROS type.
    DuplicateView2d,
    /// A data source id was registered twice.
    DuplicateSource,
    /// A `.msg` definition was already supplied by a higher-priority origin.
    DuplicateMsg,
    /// A `.msg` definition was dropped because a referenced type is missing.
    UnresolvedMsg,
    /// A `.msg` definition failed to parse.
    MsgParse,
    /// A `VISOR_MSG_PATHS` root could not be read.
    MsgPath,
    /// A plugin factory or settings call panicked.
    Panic,
}

impl ProblemKind {
    /// Short tag used in log lines and the Plugins dialog.
    pub fn tag(self) -> &'static str {
        match self {
            ProblemKind::ApiVersion => "api version",
            ProblemKind::InvalidId => "invalid id",
            ProblemKind::DuplicatePlugin => "duplicate plugin",
            ProblemKind::DuplicateStandalone => "duplicate standalone",
            ProblemKind::DuplicatePanel => "duplicate panel",
            ProblemKind::DuplicateView2d => "duplicate 2D view",
            ProblemKind::DuplicateSource => "duplicate source",
            ProblemKind::DuplicateMsg => "duplicate msg",
            ProblemKind::UnresolvedMsg => "unresolved msg",
            ProblemKind::MsgParse => "msg parse",
            ProblemKind::MsgPath => "msg path",
            ProblemKind::Panic => "panic",
        }
    }
}

/// One startup problem, collected rather than raised so a broken plugin degrades instead of aborting.
#[derive(Debug, Clone)]
pub struct Problem {
    pub plugin: PluginId,
    pub kind: ProblemKind,
    pub message: String,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {}",
            self.plugin.display_name(),
            self.kind.tag(),
            self.message
        )
    }
}

/// Everything the app can be extended with, resolved once at startup and then read only.
#[derive(Default)]
pub struct Registry {
    renderers: Vec<RendererEntry>,
    views2d: Vec<View2dEntry>,
    panels: Vec<PanelEntry>,
    sources: Vec<SourceEntry>,
    msgs: Vec<MsgEntry>,
    plugins: Vec<PluginInfo>,
    problems: Vec<Problem>,
}

impl Registry {
    /// Registry holding only what visor ships with; plugins are added on top with `add_plugin`.
    pub fn builtin() -> Self {
        let mut registry = Self::default();
        for (full_name, text) in crate::decode::embedded::EMBEDDED_MSGS {
            registry.msgs.push(MsgEntry {
                origin: PluginId::Builtin,
                full_name: (*full_name).to_owned(),
                text: Cow::Borrowed(text),
            });
        }
        let mut registrar = Registrar {
            registry: &mut registry,
            plugin: PluginId::Builtin,
        };
        crate::render::renderers::register_builtin(&mut registrar);
        crate::image::register_builtin(&mut registrar);
        crate::source::register_builtin(&mut registrar);
        registry
    }

    /// Registry with nothing at all; only tests and tooling that supply their own entries want this.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Let one plugin register its extensions. Version and id problems reject the whole plugin rather than half-registering it.
    pub fn add_plugin(&mut self, plugin: &dyn Plugin) {
        let info = plugin.info();
        let id = PluginId::Plugin(info.id.to_owned());
        if info.api_version != PLUGIN_API_VERSION {
            self.problems.push(Problem {
                plugin: id,
                kind: ProblemKind::ApiVersion,
                message: format!(
                    "`{}` was built against plugin API {} but this visor provides {PLUGIN_API_VERSION}",
                    info.id, info.api_version
                ),
            });
            return;
        }
        if !is_valid_plugin_id(info.id) {
            self.problems.push(Problem {
                plugin: id,
                kind: ProblemKind::InvalidId,
                message: format!("`{}` is not a valid plugin id ([a-z0-9_]+)", info.id),
            });
            return;
        }
        if self.plugins.iter().any(|p| p.id == info.id) {
            self.problems.push(Problem {
                plugin: id,
                kind: ProblemKind::DuplicatePlugin,
                message: format!("plugin id `{}` is already registered", info.id),
            });
            return;
        }
        self.plugins.push(info);
        let mut registrar = Registrar {
            registry: self,
            plugin: id,
        };
        plugin.register(&mut registrar);
    }

    /// Close the registry: detect collisions, pull in `VISOR_MSG_PATHS`, and freeze it behind an Arc.
    pub fn finish(mut self, env: &dyn Fn(&str) -> Option<String>) -> Arc<Self> {
        self.resolve_renderer_conflicts();
        self.resolve_view2d_conflicts();
        self.resolve_panel_conflicts();
        self.resolve_source_conflicts();
        let mut env_msgs = Vec::new();
        types::scan_env_paths(env, &mut env_msgs, &mut self.problems);
        self.msgs.extend(env_msgs);
        Arc::new(self)
    }

    /// Topic entries for the same ROS type coexist as alternatives the user picks between; only standalone keys must be unique.
    fn resolve_renderer_conflicts(&mut self) {
        let mut kept: Vec<RendererEntry> = Vec::new();
        for entry in std::mem::take(&mut self.renderers) {
            if entry.descriptor.standalone
                && kept
                    .iter()
                    .any(|k| k.descriptor.standalone && k.key == entry.key)
            {
                self.problems.push(Problem {
                    plugin: entry.plugin.clone(),
                    kind: ProblemKind::DuplicateStandalone,
                    message: format!(
                        "standalone display `{}` is already registered; the later one is ignored",
                        entry.key
                    ),
                });
                continue;
            }
            kept.push(entry);
        }
        self.renderers = kept;
    }

    fn resolve_view2d_conflicts(&mut self) {
        let mut kept: Vec<View2dEntry> = Vec::new();
        for entry in std::mem::take(&mut self.views2d) {
            if let Some(first) = kept
                .iter()
                .find(|k| k.descriptor.ros_type == entry.descriptor.ros_type)
            {
                self.problems.push(Problem {
                    plugin: entry.plugin.clone(),
                    kind: ProblemKind::DuplicateView2d,
                    message: format!(
                        "`{}` is already shown as 2D by `{}`; the later one is ignored",
                        entry.descriptor.ros_type, first.key
                    ),
                });
                continue;
            }
            kept.push(entry);
        }
        self.views2d = kept;
    }

    fn resolve_panel_conflicts(&mut self) {
        let mut kept: Vec<PanelEntry> = Vec::new();
        for entry in std::mem::take(&mut self.panels) {
            if kept.iter().any(|k| k.key == entry.key) {
                self.problems.push(Problem {
                    plugin: entry.plugin.clone(),
                    kind: ProblemKind::DuplicatePanel,
                    message: format!("panel `{}` is already registered", entry.key),
                });
                continue;
            }
            kept.push(entry);
        }
        self.panels = kept;
    }

    fn resolve_source_conflicts(&mut self) {
        let mut kept: Vec<SourceEntry> = Vec::new();
        for entry in std::mem::take(&mut self.sources) {
            if kept.iter().any(|k| k.key == entry.key) {
                self.problems.push(Problem {
                    plugin: entry.plugin.clone(),
                    kind: ProblemKind::DuplicateSource,
                    message: format!("data source `{}` is already registered", entry.key),
                });
                continue;
            }
            kept.push(entry);
        }
        self.sources = kept;
    }

    /// Merge every `.msg` origin into one type registry (builtin first, then plugins, then the environment).
    pub fn build_type_registry(&self) -> (Arc<TypeRegistry>, Vec<Problem>) {
        let (registry, problems) = types::build(&self.msgs);
        (Arc::new(registry), problems)
    }

    /// Every 3D display type that answers for a topic, in registration order (builtins first); what the Add dialog offers.
    pub fn find_renderers(
        &self,
        topic: &str,
        ros_type: &str,
    ) -> impl Iterator<Item = &RendererEntry> {
        self.renderers
            .iter()
            .filter(move |e| e.descriptor.matches(topic, ros_type))
    }

    /// The default 3D display type for a topic, first match wins (builtins are registered before plugins).
    pub fn find_renderer(&self, topic: &str, ros_type: &str) -> Option<&RendererEntry> {
        self.find_renderers(topic, ros_type).next()
    }

    /// Config restore: honour the recorded provider and label, so an item never silently comes back as a different display type.
    pub fn find_renderer_as(
        &self,
        plugin: &str,
        label: &str,
        topic: &str,
        ros_type: &str,
    ) -> Option<&RendererEntry> {
        if plugin.is_empty() && label.is_empty() {
            return self.find_renderer(topic, ros_type);
        }
        let want = plugin_id_from_config(plugin);
        self.find_renderers(topic, ros_type)
            .find(|e| e.plugin == want && (label.is_empty() || e.descriptor.label == label))
    }

    /// A 3D display type by provider and label alone, with no topic in hand; how the By-display-type picker and a topic-less item resolve theirs (`plugin` empty = builtin).
    pub fn find_display_type(&self, plugin: &str, label: &str) -> Option<&RendererEntry> {
        let want = plugin_id_from_config(plugin);
        self.renderers
            .iter()
            .find(|e| e.plugin == want && e.descriptor.label == label)
    }

    /// The 2D display type for a topic (checked before renderers, as image types always were).
    pub fn find_view2d(&self, ros_type: &str) -> Option<&View2dEntry> {
        self.views2d
            .iter()
            .find(|e| e.descriptor.ros_type == ros_type)
    }

    pub fn find_view2d_for(&self, plugin: &str, ros_type: &str) -> Option<&View2dEntry> {
        if plugin.is_empty() {
            return self.find_view2d(ros_type);
        }
        let want = plugin_id_from_config(plugin);
        self.views2d
            .iter()
            .find(|e| e.plugin == want && e.descriptor.ros_type == ros_type)
    }

    /// Standalone (non-topic) entries; in the Add dialog they sit among the rest, marked as taking no topic.
    pub fn standalone_entries(&self) -> impl Iterator<Item = &RendererEntry> {
        self.renderers.iter().filter(|e| e.descriptor.standalone)
    }

    /// Resolve a standalone item from its config keys (`plugin` empty = builtin).
    pub fn find_standalone(&self, plugin: &str, label: &str) -> Option<&RendererEntry> {
        let want = plugin_id_from_config(plugin);
        self.standalone_entries()
            .find(|e| e.plugin == want && e.descriptor.label == label)
    }

    pub fn renderer_entries(&self) -> &[RendererEntry] {
        &self.renderers
    }

    pub fn view2d_entries(&self) -> &[View2dEntry] {
        &self.views2d
    }

    pub fn panel_entries(&self) -> &[PanelEntry] {
        &self.panels
    }

    pub fn find_panel(&self, key: &str) -> Option<&PanelEntry> {
        self.panels.iter().find(|e| e.key == key)
    }

    pub fn source_entries(&self) -> &[SourceEntry] {
        &self.sources
    }

    /// The data source that claims a file's extension (case-insensitive), or None if nothing handles it.
    pub fn find_source_for_path(&self, path: &std::path::Path) -> Option<&SourceEntry> {
        let extension = path.extension()?.to_str()?.to_ascii_lowercase();
        self.sources.iter().find(|e| {
            e.descriptor
                .extensions
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(&extension))
        })
    }

    /// `.msg` type names contributed by one origin, for the Plugins dialog.
    pub fn msg_names(&self, plugin: &PluginId) -> Vec<&str> {
        self.msgs
            .iter()
            .filter(|m| &m.origin == plugin)
            .map(|m| m.full_name.as_str())
            .collect()
    }

    pub fn plugins(&self) -> &[PluginInfo] {
        &self.plugins
    }

    pub fn problems(&self) -> &[Problem] {
        &self.problems
    }
}

/// Namespaced view of the registry handed to one plugin; registrations are qualified automatically.
pub struct Registrar<'a> {
    registry: &'a mut Registry,
    plugin: PluginId,
}

impl Registrar<'_> {
    /// Which plugin is registering (builtins register through this same path).
    pub fn plugin_id(&self) -> &PluginId {
        &self.plugin
    }

    pub fn renderer(&mut self, descriptor: RendererDescriptor) {
        let key = self.plugin.qualify(&descriptor.label);
        self.registry.renderers.push(RendererEntry {
            plugin: self.plugin.clone(),
            key,
            descriptor,
        });
    }

    pub fn view2d(&mut self, descriptor: View2dDescriptor) {
        let key = self.plugin.qualify(&descriptor.label);
        self.registry.views2d.push(View2dEntry {
            plugin: self.plugin.clone(),
            key,
            descriptor,
        });
    }

    pub fn panel(&mut self, descriptor: PanelDescriptor) {
        let key = self.plugin.qualify(&descriptor.id);
        self.registry.panels.push(PanelEntry {
            plugin: self.plugin.clone(),
            key,
            descriptor,
        });
    }

    pub fn source(&mut self, descriptor: SourceDescriptor) {
        let key = self.plugin.qualify(&descriptor.id);
        self.registry.sources.push(SourceEntry {
            plugin: self.plugin.clone(),
            key,
            descriptor,
        });
    }

    /// Supply a `.msg` definition so the live decoder understands a type that is not in `assets/msgs`.
    pub fn msg(&mut self, full_name: impl Into<String>, text: impl Into<Cow<'static, str>>) {
        self.registry.msgs.push(MsgEntry {
            origin: self.plugin.clone(),
            full_name: full_name.into(),
            text: text.into(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::value::Value;
    use crate::render::{RenderStatus, SceneBatch, TfContext};

    struct NullRenderer;

    impl Renderer for NullRenderer {
        fn on_message(&mut self, _value: &Value) {}
        fn scene(&mut self, _tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
            Err(RenderStatus::NoData)
        }
        fn settings_ui(&mut self, _ui: &mut egui::Ui) {}
    }

    fn null() -> Box<dyn Renderer> {
        Box::new(NullRenderer)
    }

    struct TestPlugin {
        info: PluginInfo,
        register: fn(&mut Registrar<'_>),
    }

    impl Plugin for TestPlugin {
        fn info(&self) -> PluginInfo {
            self.info
        }
        fn register(&self, reg: &mut Registrar<'_>) {
            (self.register)(reg);
        }
    }

    fn plugin(id: &'static str, register: fn(&mut Registrar<'_>)) -> TestPlugin {
        TestPlugin {
            info: PluginInfo {
                id,
                name: "Test",
                version: "0.0.0",
                api_version: PLUGIN_API_VERSION,
            },
            register,
        }
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn finish(registry: Registry) -> Arc<Registry> {
        registry.finish(&no_env)
    }

    #[test]
    fn builtin_registry_resolves_every_shipped_display_type() {
        let registry = finish(Registry::builtin());
        for (topic, ros_type, label) in [
            ("/scan", "sensor_msgs/msg/LaserScan", "LaserScan"),
            ("/points", "sensor_msgs/msg/PointCloud2", "PointCloud2"),
            ("/map", "nav_msgs/msg/OccupancyGrid", "Map"),
            ("/plan", "nav_msgs/msg/Path", "Path"),
            ("/odom", "nav_msgs/msg/Odometry", "Odometry"),
            ("/marker", "visualization_msgs/msg/Marker", "Marker"),
            (
                "/markers",
                "visualization_msgs/msg/MarkerArray",
                "MarkerArray",
            ),
            ("/robot_description", "std_msgs/msg/String", "RobotModel"),
        ] {
            let found = registry
                .find_renderer(topic, ros_type)
                .unwrap_or_else(|| panic!("{ros_type} registered"));
            assert_eq!(found.descriptor.label, label);
            assert_eq!(found.key, label);
            assert!(found.plugin.is_builtin());
            assert!(found.make().is_ok());
        }
        assert!(registry.problems().is_empty(), "{:?}", registry.problems());
    }

    #[test]
    fn builtin_topic_filter_and_standalone_rules_are_unchanged() {
        let registry = finish(Registry::builtin());
        for topic in ["/robot_description", "/ns/robot_description"] {
            assert_eq!(
                registry
                    .find_renderer(topic, "std_msgs/msg/String")
                    .expect("registered")
                    .descriptor
                    .label,
                "RobotModel"
            );
        }
        for topic in [
            "/chatter",
            "/my_robot_description",
            "/robot_description_raw",
        ] {
            assert!(
                registry
                    .find_renderer(topic, "std_msgs/msg/String")
                    .is_none(),
                "{topic}"
            );
        }
        let listed: Vec<&str> = registry
            .standalone_entries()
            .map(|e| e.descriptor.label.as_str())
            .collect();
        assert_eq!(listed, vec!["RobotModel"]);
        assert!(registry.find_renderer("/robot_description", "").is_none());
        assert!(registry.find_standalone("", "RobotModel").is_some());
        assert!(registry.find_standalone("sample", "RobotModel").is_none());
    }

    #[test]
    fn builtin_2d_views_cover_both_image_types() {
        let registry = finish(Registry::builtin());
        for (ros_type, label) in [
            ("sensor_msgs/msg/Image", "Image"),
            ("sensor_msgs/msg/CompressedImage", "CompressedImage"),
        ] {
            let found = registry.find_view2d(ros_type).expect("registered");
            assert_eq!(found.descriptor.label, label);
            assert!(found.make().is_ok());
        }
        assert!(registry.find_view2d("sensor_msgs/msg/LaserScan").is_none());
    }

    #[test]
    fn a_plugin_entry_for_a_builtin_type_becomes_a_selectable_alternative() {
        let mut registry = Registry::builtin();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.renderer(RendererDescriptor::topic(
                "sensor_msgs/msg/LaserScan",
                "MyScan",
                null,
            ));
        }));
        let registry = finish(registry);
        // The builtin stays the default, and the plugin one is offered next to it rather than reported as a problem.
        let keys: Vec<&str> = registry
            .find_renderers("/scan", "sensor_msgs/msg/LaserScan")
            .map(|e| e.key.as_str())
            .collect();
        assert_eq!(keys, vec!["LaserScan", "sample::MyScan"]);
        assert_eq!(
            registry
                .find_renderer("/scan", "sensor_msgs/msg/LaserScan")
                .expect("registered")
                .key,
            "LaserScan"
        );
        assert!(registry.problems().is_empty(), "{:?}", registry.problems());
    }

    #[test]
    fn find_renderer_as_pins_the_recorded_provider_and_label() {
        let mut registry = Registry::builtin();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.renderer(RendererDescriptor::topic(
                "sensor_msgs/msg/LaserScan",
                "MyScan",
                null,
            ));
        }));
        let registry = finish(registry);
        let as_key = |plugin: &str, label: &str| {
            registry
                .find_renderer_as(plugin, label, "/scan", "sensor_msgs/msg/LaserScan")
                .map(|e| e.key.clone())
        };
        // Both keys empty = pre-picker configs, which keep resolving to the default.
        assert_eq!(as_key("", ""), Some("LaserScan".to_owned()));
        assert_eq!(
            as_key("sample", "MyScan"),
            Some("sample::MyScan".to_owned())
        );
        assert_eq!(as_key("sample", ""), Some("sample::MyScan".to_owned()));
        assert_eq!(as_key("", "LaserScan"), Some("LaserScan".to_owned()));
        // A label that no registration answers for is a restore failure, not a silent fallback.
        assert_eq!(as_key("sample", "Gone"), None);
        assert_eq!(as_key("", "MyScan"), None);
    }

    #[test]
    fn a_plugin_can_claim_a_type_visor_does_not_handle() {
        let mut registry = Registry::builtin();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.renderer(RendererDescriptor::topic(
                "sample_msgs/msg/FleetState",
                "FleetState",
                null,
            ));
        }));
        let registry = finish(registry);
        let found = registry
            .find_renderer("/fleet", "sample_msgs/msg/FleetState")
            .expect("registered");
        assert_eq!(found.key, "sample::FleetState");
        assert_eq!(found.plugin, PluginId::Plugin("sample".to_owned()));
        assert!(registry.problems().is_empty(), "{:?}", registry.problems());
    }

    #[test]
    fn a_plugin_filtered_entry_coexists_with_the_builtin_one() {
        let mut registry = Registry::builtin();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.renderer(RendererDescriptor::topic_filtered(
                "std_msgs/msg/String",
                "Banner",
                null,
                |topic| topic.ends_with("/banner"),
            ));
        }));
        let registry = finish(registry);
        assert_eq!(
            registry
                .find_renderer("/robot_description", "std_msgs/msg/String")
                .expect("registered")
                .key,
            "RobotModel"
        );
        assert_eq!(
            registry
                .find_renderer("/ui/banner", "std_msgs/msg/String")
                .expect("registered")
                .key,
            "sample::Banner"
        );
        assert!(registry.problems().is_empty(), "{:?}", registry.problems());
    }

    #[test]
    fn the_same_label_from_two_plugins_stays_distinguishable() {
        let mut registry = Registry::empty();
        registry.add_plugin(&plugin("one", |reg| {
            reg.renderer(RendererDescriptor::standalone("Fleet", null));
        }));
        registry.add_plugin(&plugin("two", |reg| {
            reg.renderer(RendererDescriptor::standalone("Fleet", null));
        }));
        let registry = finish(registry);
        assert_eq!(registry.standalone_entries().count(), 2);
        assert_eq!(
            registry.find_standalone("one", "Fleet").expect("one").key,
            "one::Fleet"
        );
        assert_eq!(
            registry.find_standalone("two", "Fleet").expect("two").key,
            "two::Fleet"
        );
        assert!(registry.problems().is_empty(), "{:?}", registry.problems());
    }

    #[test]
    fn duplicate_standalone_within_one_plugin_keeps_the_first_and_warns() {
        let mut registry = Registry::empty();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.renderer(RendererDescriptor::standalone("Fleet", null));
            reg.renderer(RendererDescriptor::standalone("Fleet", null));
        }));
        let registry = finish(registry);
        assert_eq!(registry.standalone_entries().count(), 1);
        assert_eq!(registry.problems().len(), 1);
        assert_eq!(
            registry.problems()[0].kind,
            ProblemKind::DuplicateStandalone
        );
    }

    #[test]
    fn an_api_version_mismatch_rejects_the_whole_plugin() {
        let mut registry = Registry::empty();
        registry.add_plugin(&TestPlugin {
            info: PluginInfo {
                id: "sample",
                name: "Test",
                version: "0.0.0",
                api_version: PLUGIN_API_VERSION + 1,
            },
            register: |reg| reg.renderer(RendererDescriptor::standalone("Fleet", null)),
        });
        let registry = finish(registry);
        assert_eq!(registry.standalone_entries().count(), 0);
        assert!(registry.plugins().is_empty());
        assert_eq!(registry.problems().len(), 1);
        assert_eq!(registry.problems()[0].kind, ProblemKind::ApiVersion);
    }

    #[test]
    fn an_invalid_or_repeated_id_rejects_the_whole_plugin() {
        let mut registry = Registry::empty();
        registry.add_plugin(&plugin("Sample", |reg| {
            reg.renderer(RendererDescriptor::standalone("A", null));
        }));
        registry.add_plugin(&plugin("ok", |reg| {
            reg.renderer(RendererDescriptor::standalone("B", null));
        }));
        registry.add_plugin(&plugin("ok", |reg| {
            reg.renderer(RendererDescriptor::standalone("C", null));
        }));
        let registry = finish(registry);
        let labels: Vec<&str> = registry
            .standalone_entries()
            .map(|e| e.descriptor.label.as_str())
            .collect();
        assert_eq!(labels, vec!["B"]);
        assert_eq!(registry.plugins().len(), 1);
        let kinds: Vec<ProblemKind> = registry.problems().iter().map(|p| p.kind).collect();
        assert_eq!(
            kinds,
            vec![ProblemKind::InvalidId, ProblemKind::DuplicatePlugin]
        );
    }

    #[test]
    fn a_panicking_factory_is_reported_instead_of_unwinding_into_the_app() {
        let mut registry = Registry::empty();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.renderer(RendererDescriptor::standalone("Boom", || {
                panic!("factory exploded")
            }));
        }));
        let registry = finish(registry);
        let entry = registry.find_standalone("sample", "Boom").expect("kept");
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let error = entry.make().err();
        std::panic::set_hook(previous);
        assert!(error.is_some_and(|e| e.contains("sample::Boom")));
    }

    #[test]
    fn panels_are_namespaced_and_duplicates_are_dropped() {
        let mut registry = Registry::empty();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.panel(PanelDescriptor::new("fleet", "Fleet", || {
                unreachable!("not constructed in this test")
            }));
            reg.panel(PanelDescriptor::new("fleet", "Fleet again", || {
                unreachable!("not constructed in this test")
            }));
        }));
        let registry = finish(registry);
        assert_eq!(registry.panel_entries().len(), 1);
        assert_eq!(registry.panel_entries()[0].key, "sample::fleet");
        assert!(registry.find_panel("sample::fleet").is_some());
        assert!(registry.find_panel("fleet").is_none());
        assert_eq!(registry.problems()[0].kind, ProblemKind::DuplicatePanel);
    }

    #[test]
    fn builtin_sources_claim_the_bag_and_rosbag2_extensions() {
        let registry = finish(Registry::builtin());
        let entry = registry
            .find_source_for_path(std::path::Path::new("/tmp/run_0.BAG"))
            .expect("bag source registered");
        assert_eq!(entry.descriptor.id, "bag");
        // ROS 1 bags are files, so the menu picks files.
        assert!(!entry.descriptor.pick_folders);
        for name in [
            "/tmp/log_0.mcap",
            "/tmp/log_0.db3",
            "/tmp/log/metadata.yaml",
        ] {
            let entry = registry
                .find_source_for_path(std::path::Path::new(name))
                .unwrap_or_else(|| panic!("rosbag2 source claims {name}"));
            assert_eq!(entry.descriptor.id, "rosbag2");
            // A rosbag2 bag is a directory, so the menu picks folders (several at once, like several .bag files).
            assert!(entry.descriptor.pick_folders);
        }
        assert!(
            registry
                .find_source_for_path(std::path::Path::new("/tmp/log.csv"))
                .is_none()
        );
    }

    #[test]
    fn plugin_msg_definitions_reach_the_merged_type_registry() {
        let mut registry = Registry::builtin();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.msg(
                "sample_msgs/msg/FleetState",
                "std_msgs/Header header\nint32 robots\n",
            );
        }));
        let registry = finish(registry);
        let (types, problems) = registry.build_type_registry();
        assert!(problems.is_empty(), "{problems:?}");
        let def = types.get("sample_msgs/msg/FleetState").expect("registered");
        assert_eq!(def.fields[1].name, "robots");
        assert!(types.get("std_msgs/msg/Header").is_some());
        assert_eq!(
            registry.msg_names(&PluginId::Plugin("sample".to_owned())),
            vec!["sample_msgs/msg/FleetState"]
        );
    }

    #[test]
    fn a_plugin_cannot_override_a_builtin_msg_definition() {
        let mut registry = Registry::builtin();
        registry.add_plugin(&plugin("sample", |reg| {
            reg.msg("std_msgs/msg/Header", "int32 bogus\n");
        }));
        let registry = finish(registry);
        let (types, problems) = registry.build_type_registry();
        assert_eq!(
            types
                .get("std_msgs/msg/Header")
                .expect("builtin kept")
                .fields[0]
                .name,
            "stamp"
        );
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].kind, ProblemKind::DuplicateMsg);
    }

    #[test]
    fn environment_definitions_come_last_and_only_fill_gaps() {
        let root = std::env::temp_dir().join(format!("visor_registry_env_{}", std::process::id()));
        let msg_dir = root.join("std_msgs").join("msg");
        std::fs::create_dir_all(&msg_dir).expect("temp dir");
        std::fs::write(msg_dir.join("Header.msg"), "int32 bogus\n").expect("write");
        let extra = root.join("extra_msgs").join("msg");
        std::fs::create_dir_all(&extra).expect("temp dir");
        std::fs::write(extra.join("Ping.msg"), "int32 seq\n").expect("write");
        let value = root.display().to_string();
        let env = |key: &str| (key == types::MSG_PATHS_ENV).then(|| value.clone());
        let registry = Registry::builtin().finish(&env);
        let (typedefs, problems) = registry.build_type_registry();
        assert!(typedefs.get("extra_msgs/msg/Ping").is_some());
        assert_eq!(
            typedefs.get("std_msgs/msg/Header").expect("kept").fields[0].name,
            "stamp"
        );
        assert!(
            problems
                .iter()
                .any(|p| p.kind == ProblemKind::DuplicateMsg && p.plugin == PluginId::Env)
        );
        std::fs::remove_dir_all(&root).ok();
    }
}
