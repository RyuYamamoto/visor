//! Remembers the most recently used config path and bag directory (small file state.toml, separate from the config itself).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Contents of state.toml; every field is optional so an older file still loads.
#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LastConfig {
    pub last_config: Option<PathBuf>,
    /// Directory of the last bag opened, used only as the file dialog's starting point (Q8 = C).
    pub last_bag_dir: Option<PathBuf>,
}

/// Reads the whole state file (missing/corrupt yields defaults).
pub fn load_all(path: &Path) -> LastConfig {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str::<LastConfig>(&text).ok())
        .unwrap_or_default()
}

/// Reads the last config path from state.toml (missing/corrupt returns None).
pub fn load(path: &Path) -> Option<PathBuf> {
    load_all(path).last_config
}

/// Records the last config path, keeping the other remembered values.
pub fn save(path: &Path, config_path: &Path) -> Result<(), String> {
    let mut state = load_all(path);
    state.last_config = Some(config_path.to_path_buf());
    write(path, &state)
}

/// Records the directory a bag was opened from, keeping the other remembered values.
pub fn save_bag_dir(path: &Path, dir: &Path) -> Result<(), String> {
    let mut state = load_all(path);
    state.last_bag_dir = Some(dir.to_path_buf());
    write(path, &state)
}

/// Writes state.toml, creating parent directories automatically.
fn write(path: &Path, state: &LastConfig) -> Result<(), String> {
    let text = toml::to_string(state).map_err(|e| format!("failed to serialize state: {e}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_last_config_path() {
        let dir = std::env::temp_dir().join(format!("visor_state_test_{}", std::process::id()));
        let state = dir.join("state.toml");
        let target = PathBuf::from("/home/u/.config/visor/my.toml");
        save(&state, &target).unwrap();
        assert_eq!(load(&state), Some(target));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bag_dir_and_config_path_do_not_overwrite_each_other() {
        let dir = std::env::temp_dir().join(format!("visor_state_bag_{}", std::process::id()));
        let state = dir.join("state.toml");
        let config = PathBuf::from("/home/u/.config/visor/my.toml");
        let bags = PathBuf::from("/home/u/bags");
        save(&state, &config).unwrap();
        save_bag_dir(&state, &bags).unwrap();
        let all = load_all(&state);
        assert_eq!(all.last_config, Some(config.clone()));
        assert_eq!(all.last_bag_dir, Some(bags));
        // Writing the config path again must not drop the bag directory.
        save(&state, &config).unwrap();
        assert!(load_all(&state).last_bag_dir.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_is_none() {
        assert_eq!(load(Path::new("/nonexistent/visor/state.toml")), None);
        assert_eq!(
            load_all(Path::new("/nonexistent/visor/state.toml")),
            LastConfig::default()
        );
    }
}
