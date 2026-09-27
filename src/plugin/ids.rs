//! Plugin identity and the qualified keys that keep one plugin's registrations from colliding with another's.

/// Where a registration came from. Registrations are namespaced by this, so two plugins may reuse the same label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum PluginId {
    /// Built into visor itself. Its labels stay unqualified, which is what keeps existing configs resolving.
    #[default]
    Builtin,
    Plugin(String),
    /// Supplied through the environment (`VISOR_MSG_PATHS`); only .msg definitions ever use this.
    Env,
}

impl PluginId {
    /// Config-facing id: empty for builtin, so a config written without plugins looks exactly as it did before.
    pub fn as_str(&self) -> &str {
        match self {
            PluginId::Builtin => "",
            PluginId::Plugin(id) => id,
            PluginId::Env => ENV_ID,
        }
    }

    pub fn is_builtin(&self) -> bool {
        matches!(self, PluginId::Builtin)
    }

    /// Runtime key that namespaces a label: `label` for builtin, `id::label` for a plugin.
    pub fn qualify(&self, label: &str) -> String {
        if self.is_builtin() {
            label.to_owned()
        } else {
            format!("{}{SEPARATOR}{label}", self.as_str())
        }
    }

    /// Human-readable origin for problem messages and the Plugins dialog.
    pub fn display_name(&self) -> &str {
        match self {
            PluginId::Builtin => "builtin",
            PluginId::Plugin(id) => id,
            PluginId::Env => "$VISOR_MSG_PATHS",
        }
    }
}

/// Separator between a plugin id and a label inside a qualified key.
pub const SEPARATOR: &str = "::";

/// Reserved id used by environment-supplied definitions; a plugin may not claim it.
const ENV_ID: &str = "env";

/// Whether an id can namespace registrations: non-empty `[a-z0-9_]+` and not the reserved `env` (empty is builtin's own id).
pub fn is_valid_plugin_id(id: &str) -> bool {
    !id.is_empty()
        && id != ENV_ID
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Resolve a config's `plugin` field to a PluginId (empty = builtin).
pub fn plugin_id_from_config(plugin: &str) -> PluginId {
    if plugin.is_empty() {
        PluginId::Builtin
    } else {
        PluginId::Plugin(plugin.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_labels_stay_unqualified() {
        assert_eq!(PluginId::Builtin.qualify("RobotModel"), "RobotModel");
        assert_eq!(PluginId::Builtin.as_str(), "");
        assert!(PluginId::Builtin.is_builtin());
    }

    #[test]
    fn plugin_labels_are_namespaced_by_id() {
        let id = PluginId::Plugin("sample".to_owned());
        assert_eq!(id.qualify("FleetState"), "sample::FleetState");
        assert_eq!(id.as_str(), "sample");
        assert!(!id.is_builtin());
    }

    #[test]
    fn id_validation_rejects_empty_reserved_and_odd_characters() {
        assert!(is_valid_plugin_id("sample"));
        assert!(is_valid_plugin_id("fleet_view2"));
        assert!(!is_valid_plugin_id(""));
        assert!(!is_valid_plugin_id("env"));
        assert!(!is_valid_plugin_id("Sample"));
        assert!(!is_valid_plugin_id("my-plugin"));
        assert!(!is_valid_plugin_id("a::b"));
    }

    #[test]
    fn config_plugin_field_maps_empty_to_builtin() {
        assert_eq!(plugin_id_from_config(""), PluginId::Builtin);
        assert_eq!(
            plugin_id_from_config("sample"),
            PluginId::Plugin("sample".to_owned())
        );
    }
}
