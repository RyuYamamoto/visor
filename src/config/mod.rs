//! Config schema (DTOs) persisting the viewer's visualization state, TOML I/O, and CLI `--config` splitting.

pub mod color_hex;
pub mod paths;
pub mod state;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Config schema version (used for backward-compat checks; bump on breaking changes).
pub const CURRENT_VERSION: u32 = 1;

/// Root DTO for persisted viewer state (conversion to/from runtime structs lives in app.rs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewerConfig {
    pub version: u32,
    pub fixed_frame: Option<String>,
    pub fixed_frame_auto: bool,
    /// Target Frame the camera follows (translation-only); None = world-fixed.
    pub target_frame: Option<String>,
    pub group_by: String,
    /// UI light/dark mode; absent from configs written before the theme switch existed and when it is the default Dark.
    #[serde(skip_serializing_if = "ThemeConfig::is_dark")]
    pub theme: ThemeConfig,
    pub camera: CameraConfig,
    pub viewport: ViewportConfig,
    /// Pinned TF topics; absent while the automatic rule (`/tf` / `/tf_static`, else any TFMessage topic) is in force.
    #[serde(skip_serializing_if = "TfConfig::is_auto")]
    pub tf: TfConfig,
    pub displays: Vec<DisplayConfig>,
    /// Plugin panel state; absent from configs written before panels existed and when no panel has settings.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub panels: Vec<PanelConfig>,
    /// Embeds egui_dock's DockState as-is (representation delegated to egui_dock's serde).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dock: Option<toml::Value>,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            version: CURRENT_VERSION,
            fixed_frame: None,
            fixed_frame_auto: true,
            target_frame: None,
            group_by: "namespace".to_owned(),
            theme: ThemeConfig::default(),
            camera: CameraConfig::default(),
            viewport: ViewportConfig::default(),
            tf: TfConfig::default(),
            displays: Vec::new(),
            panels: Vec::new(),
            dock: None,
        }
    }
}

/// Which topics feed the TF buffer. None per kind means the automatic rule; a name pins that topic even when the graph also has `/tf`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TfConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_topic: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_topic: Option<String>,
}

impl TfConfig {
    /// Whether both kinds follow the automatic rule, so the section is left out of a config that never pinned anything.
    pub fn is_auto(&self) -> bool {
        self.dynamic_topic.is_none() && self.static_topic.is_none()
    }
}

/// UI light/dark mode (mirrors egui::Theme; kept separate so config does not depend on egui).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThemeConfig {
    #[default]
    Dark,
    Light,
}

impl ThemeConfig {
    /// Whether this is the default, so a Dark config stays byte-identical to one written before the theme switch existed.
    pub fn is_dark(&self) -> bool {
        matches!(self, Self::Dark)
    }
}

/// Active view type (mirrors render::camera::ViewType; kept separate so config does not depend on render).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ViewTypeConfig {
    #[default]
    Orbit,
    TopDownOrtho,
    Fps,
}

/// Persistence DTO for the whole camera state: active view type plus all three views' poses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub view_type: ViewTypeConfig,
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub topdown: TopDownConfig,
    pub fps: FpsConfig,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            view_type: ViewTypeConfig::default(),
            target: [0.0, 0.0, 0.0],
            yaw: -45.0_f32.to_radians(),
            pitch: 35.0_f32.to_radians(),
            distance: 10.0,
            topdown: TopDownConfig::default(),
            fps: FpsConfig::default(),
        }
    }
}

/// Persistence DTO for TopDownCamera.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TopDownConfig {
    pub center: [f32; 2],
    pub rotation: f32,
    pub half_height: f32,
}

impl Default for TopDownConfig {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            rotation: 0.0,
            half_height: 10.0,
        }
    }
}

/// Persistence DTO for FpsCamera.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FpsConfig {
    pub eye: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for FpsConfig {
    fn default() -> Self {
        Self {
            eye: [-5.0, -5.0, 3.0],
            yaw: 45.0_f32.to_radians(),
            pitch: -20.0_f32.to_radians(),
        }
    }
}

/// Persistence DTO for ViewportState (gpu_ready and camera are managed separately).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewportConfig {
    pub show_names: bool,
    pub show_links: bool,
    pub tf_axis_len: f32,
    pub tf_line_width: f32,
    /// Hidden frames (the HashSet is sorted for stable output).
    pub hidden_frames: Vec<String>,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            show_names: true,
            show_links: true,
            tf_axis_len: 0.3,
            tf_line_width: 0.02,
            hidden_frames: Vec::new(),
        }
    }
}

/// Display item kind (absent in older configs, so the default keeps them readable as topic items).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DisplayKind {
    #[default]
    Topic,
    Standalone,
}

impl DisplayKind {
    /// Used to skip serializing the default, so topic entries produce the same TOML as before.
    fn is_topic(&self) -> bool {
        matches!(self, DisplayKind::Topic)
    }
}

/// Persistence DTO for one display item (Vec order = insertion order).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplayConfig {
    /// Item kind; not written when topic (the common case).
    #[serde(skip_serializing_if = "DisplayKind::is_topic")]
    pub kind: DisplayKind,
    /// Topic name, required to rebuild the subscription key. Empty for a standalone item.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub topic: String,
    /// ROS-form type name, required for renderer selection and the subscription key type. Empty for a standalone item.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub ros_type: String,
    /// Registry label of the display type: always set for a standalone item, and for a topic item only when it is not the type that topic resolves to by default (empty = default).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub label: String,
    /// Id of the plugin that provides this display type; empty means visor itself, which is what every pre-plugin config says.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub plugin: String,
    pub visible: bool,
    /// Renderer-specific settings (held opaquely; the config schema does not know its contents, a standalone item's source included).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<toml::Value>,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            kind: DisplayKind::Topic,
            topic: String::new(),
            ros_type: String::new(),
            label: String::new(),
            plugin: String::new(),
            visible: true,
            settings: None,
        }
    }
}

/// Persistence DTO for one plugin panel's state (opaque settings, keyed by qualified panel id).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PanelConfig {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<toml::Value>,
}

/// Writes config to TOML (creates parent directories automatically).
pub fn save(config: &ViewerConfig, path: &Path) -> Result<(), String> {
    let text =
        toml::to_string_pretty(config).map_err(|e| format!("failed to serialize config: {e}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

/// Reads config from TOML (missing/corrupt/schema-mismatch returns an error string).
pub fn load(path: &Path) -> Result<ViewerConfig, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

/// Extracts `--config <path>` from the args and returns the rest for `CommConfig::resolve`.
pub fn split_config_arg<I>(args: I) -> Result<(Option<PathBuf>, Vec<String>), String>
where
    I: IntoIterator<Item = String>,
{
    let mut config = None;
    let mut rest = Vec::new();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        if arg == "--config" {
            let value = it.next().ok_or("--config requires a value")?;
            config = Some(PathBuf::from(value));
        } else {
            rest.push(arg);
        }
    }
    Ok((config, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ViewerConfig {
        ViewerConfig {
            version: CURRENT_VERSION,
            fixed_frame: Some("map".to_owned()),
            fixed_frame_auto: false,
            target_frame: Some("base_link".to_owned()),
            group_by: "namespace".to_owned(),
            theme: ThemeConfig::Light,
            tf: TfConfig::default(),
            camera: CameraConfig {
                view_type: ViewTypeConfig::TopDownOrtho,
                target: [1.0, 2.0, 3.0],
                yaw: -0.785,
                pitch: 0.611,
                distance: 12.5,
                topdown: TopDownConfig {
                    center: [4.0, -5.0],
                    rotation: 0.25,
                    half_height: 20.0,
                },
                fps: FpsConfig {
                    eye: [-1.0, -2.0, 1.5],
                    yaw: 0.1,
                    pitch: -0.2,
                },
            },
            viewport: ViewportConfig {
                show_names: true,
                show_links: false,
                tf_axis_len: 0.5,
                tf_line_width: 0.03,
                hidden_frames: vec!["base_link".to_owned(), "odom".to_owned()],
            },
            displays: vec![
                DisplayConfig {
                    topic: "/scan".to_owned(),
                    ros_type: "sensor_msgs/msg/LaserScan".to_owned(),
                    visible: true,
                    settings: Some(toml::from_str("color_mode = \"intensity\"").unwrap()),
                    ..Default::default()
                },
                DisplayConfig {
                    kind: DisplayKind::Standalone,
                    label: "RobotModel".to_owned(),
                    visible: true,
                    settings: Some(toml::from_str("path = \"/tmp/robot.urdf\"").unwrap()),
                    ..Default::default()
                },
            ],
            panels: Vec::new(),
            dock: None,
        }
    }

    #[test]
    fn config_roundtrips_through_toml() {
        let config = sample();
        let text = toml::to_string_pretty(&config).unwrap();
        let back: ViewerConfig = toml::from_str(&text).unwrap();
        assert_eq!(back, config);
    }

    #[test]
    fn serialized_toml_is_human_readable() {
        let text = toml::to_string_pretty(&sample()).unwrap();
        println!("{text}");
        assert!(text.contains("version = 1"));
        assert!(text.contains("[camera]"));
        assert!(text.contains("[viewport]"));
        assert!(text.contains("[[displays]]"));
        assert!(text.contains("color_mode = \"intensity\""));
    }

    #[test]
    fn connection_settings_are_never_serialized() {
        let text = toml::to_string_pretty(&sample()).unwrap();
        assert!(!text.contains("endpoint"));
        assert!(!text.contains("domain"));
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let back: ViewerConfig = toml::from_str("version = 1\n").unwrap();
        assert_eq!(back, ViewerConfig::default());
    }

    #[test]
    fn pinned_tf_topics_roundtrip_and_the_automatic_rule_writes_nothing() {
        // The default (automatic) config stays byte-identical to one written before TF topics were configurable.
        let text = toml::to_string_pretty(&sample()).unwrap();
        assert!(!text.contains("[tf]"), "{text}");
        let mut config = sample();
        config.tf.dynamic_topic = Some("/recorded/tf".to_owned());
        let text = toml::to_string_pretty(&config).unwrap();
        assert!(text.contains("[tf]"), "{text}");
        assert!(text.contains("dynamic_topic = \"/recorded/tf\""), "{text}");
        assert!(!text.contains("static_topic"), "{text}");
        let back: ViewerConfig = toml::from_str(&text).unwrap();
        assert_eq!(back, config);
        assert!(!back.tf.is_auto());
    }

    #[test]
    fn old_camera_config_without_view_fields_falls_back() {
        let text = "\
version = 1
[camera]
target = [1.0, 2.0, 3.0]
yaw = -0.785
pitch = 0.611
distance = 12.5
";
        let back: ViewerConfig = toml::from_str(text).unwrap();
        assert_eq!(back.camera.view_type, ViewTypeConfig::Orbit);
        assert_eq!(back.camera.target, [1.0, 2.0, 3.0]);
        assert_eq!(back.camera.topdown, TopDownConfig::default());
        assert_eq!(back.camera.fps, FpsConfig::default());
        assert_eq!(back.target_frame, None);
    }

    #[test]
    fn a_light_theme_roundtrips_and_dark_stays_absent_from_the_output() {
        let text = toml::to_string_pretty(&sample()).unwrap();
        assert!(text.contains("theme = \"light\""));
        let back: ViewerConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.theme, ThemeConfig::Light);

        let dark = ViewerConfig::default();
        let text = toml::to_string_pretty(&dark).unwrap();
        assert!(!text.contains("theme"));
    }

    #[test]
    fn a_config_without_a_theme_reads_as_dark() {
        let back: ViewerConfig = toml::from_str("version = 1\n").unwrap();
        assert_eq!(back.theme, ThemeConfig::Dark);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let text = "version = 1\nfuture_field = 42\n";
        let back: ViewerConfig = toml::from_str(text).unwrap();
        assert_eq!(back.version, 1);
    }

    #[test]
    fn standalone_display_config_roundtrips() {
        let text = toml::to_string_pretty(&sample()).unwrap();
        assert!(text.contains("kind = \"standalone\""));
        assert!(text.contains("label = \"RobotModel\""));
        assert!(text.contains("path = \"/tmp/robot.urdf\""));
        let back: ViewerConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.displays, sample().displays);
    }

    #[test]
    fn topic_display_config_does_not_emit_kind_or_label() {
        let config = ViewerConfig {
            displays: vec![DisplayConfig {
                topic: "/scan".to_owned(),
                ros_type: "sensor_msgs/msg/LaserScan".to_owned(),
                visible: true,
                ..Default::default()
            }],
            ..Default::default()
        };
        let text = toml::to_string_pretty(&config).unwrap();
        assert!(!text.contains("kind"), "{text}");
        assert!(!text.contains("label"), "{text}");
        // A builtin display type records no provider, so pre-plugin configs are byte-identical.
        assert!(!text.contains("plugin"), "{text}");
        assert!(!text.contains("panels"), "{text}");
    }

    #[test]
    fn plugin_provided_displays_record_their_provider() {
        let config = ViewerConfig {
            displays: vec![DisplayConfig {
                kind: DisplayKind::Standalone,
                label: "Fleet".to_owned(),
                plugin: "sample".to_owned(),
                visible: true,
                ..Default::default()
            }],
            panels: vec![PanelConfig {
                id: "sample::fleet".to_owned(),
                settings: Some(toml::from_str("show_ids = true").unwrap()),
            }],
            ..Default::default()
        };
        let text = toml::to_string_pretty(&config).unwrap();
        assert!(text.contains("plugin = \"sample\""), "{text}");
        assert!(text.contains("[[panels]]"), "{text}");
        let back: ViewerConfig = toml::from_str(&text).unwrap();
        assert_eq!(back, config);
    }

    #[test]
    fn old_config_without_plugin_or_panels_reads_as_builtin() {
        let text = "\
version = 1
[[displays]]
kind = \"standalone\"
label = \"RobotModel\"
visible = true
";
        let back: ViewerConfig = toml::from_str(text).unwrap();
        assert!(back.displays[0].plugin.is_empty());
        assert_eq!(back.displays[0].label, "RobotModel");
        assert!(back.panels.is_empty());
    }

    #[test]
    fn old_config_without_kind_reads_as_topic() {
        let text = "\
version = 1
[[displays]]
topic = \"/scan\"
ros_type = \"sensor_msgs/msg/LaserScan\"
visible = true
";
        let back: ViewerConfig = toml::from_str(text).unwrap();
        assert_eq!(back.displays[0].kind, DisplayKind::Topic);
        assert_eq!(back.displays[0].topic, "/scan");
        assert!(back.displays[0].label.is_empty());
    }

    #[test]
    fn bundled_sample_config_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/sample.toml");
        let config = load(&path).expect("config/sample.toml loads");
        assert_eq!(config.version, CURRENT_VERSION);
        assert!(!config.displays.is_empty());
        assert!(config.dock.is_some());
    }

    #[test]
    fn split_config_arg_extracts_and_passes_rest() {
        let args = ["--config", "/tmp/a.toml", "--domain-id", "3"]
            .into_iter()
            .map(str::to_owned);
        let (config, rest) = split_config_arg(args).unwrap();
        assert_eq!(config, Some(PathBuf::from("/tmp/a.toml")));
        assert_eq!(rest, vec!["--domain-id".to_owned(), "3".to_owned()]);
    }

    #[test]
    fn split_config_arg_is_transparent_without_flag() {
        let args = ["--endpoint", "tcp/x:7447"].into_iter().map(str::to_owned);
        let (config, rest) = split_config_arg(args).unwrap();
        assert_eq!(config, None);
        assert_eq!(rest, vec!["--endpoint".to_owned(), "tcp/x:7447".to_owned()]);
    }

    #[test]
    fn split_config_arg_errors_on_missing_value() {
        let args = ["--config"].into_iter().map(str::to_owned);
        assert!(split_config_arg(args).is_err());
    }
}
