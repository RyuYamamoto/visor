//! Path resolution for the config dir and state file (XDG convention, `APPDATA` on Windows; testable via env injection).

use std::path::PathBuf;

/// App-specific directory name directly under the config root.
pub const APP_DIR: &str = "visor";
/// Small file recording the "last path" (managed separately from the config itself).
pub const STATE_FILE: &str = "state.toml";

/// The user's home directory: `HOME` first (Unix, Git Bash), then `USERPROFILE` (Windows cmd / PowerShell); empty values count as unset.
pub fn home_dir(env: impl Fn(&str) -> Option<String>) -> Option<String> {
    env("HOME")
        .filter(|s| !s.is_empty())
        .or_else(|| env("USERPROFILE").filter(|s| !s.is_empty()))
}

/// Returns `<XDG_CONFIG_HOME, else APPDATA, else $HOME/.config>/visor` (None if none is set = feature disabled). `APPDATA` beats `HOME` so Git Bash and PowerShell agree on one location.
pub fn config_dir(env: impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    if let Some(xdg) = env("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(xdg).join(APP_DIR));
    }
    if let Some(appdata) = env("APPDATA").filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(appdata).join(APP_DIR));
    }
    let home = home_dir(env)?;
    Some(PathBuf::from(home).join(".config").join(APP_DIR))
}

/// Full path to state.toml recording the last path.
pub fn state_path(env: impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    config_dir(env).map(|dir| dir.join(STATE_FILE))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| (*v).to_owned())
        }
    }

    #[test]
    fn prefers_xdg_config_home() {
        let dir = config_dir(env_of(&[
            ("XDG_CONFIG_HOME", "/tmp/xdg"),
            ("HOME", "/home/u"),
        ]));
        assert_eq!(dir, Some(PathBuf::from("/tmp/xdg/visor")));
    }

    #[test]
    fn falls_back_to_home_dotconfig() {
        let dir = config_dir(env_of(&[("HOME", "/home/u")]));
        assert_eq!(dir, Some(PathBuf::from("/home/u/.config/visor")));
    }

    #[test]
    fn empty_xdg_is_ignored() {
        let dir = config_dir(env_of(&[("XDG_CONFIG_HOME", ""), ("HOME", "/home/u")]));
        assert_eq!(dir, Some(PathBuf::from("/home/u/.config/visor")));
    }

    #[test]
    fn none_when_no_env() {
        assert_eq!(config_dir(env_of(&[])), None);
        assert_eq!(state_path(env_of(&[])), None);
    }

    #[test]
    fn state_path_is_under_config_dir() {
        let path = state_path(env_of(&[("HOME", "/home/u")]));
        assert_eq!(
            path,
            Some(PathBuf::from("/home/u/.config/visor/state.toml"))
        );
    }

    #[test]
    fn appdata_used_when_xdg_unset() {
        let appdata = r"C:\Users\u\AppData\Roaming";
        let dir = config_dir(env_of(&[
            ("APPDATA", appdata),
            ("USERPROFILE", r"C:\Users\u"),
        ]));
        assert_eq!(dir, Some(PathBuf::from(appdata).join("visor")));
    }

    #[test]
    fn xdg_beats_appdata() {
        let dir = config_dir(env_of(&[
            ("XDG_CONFIG_HOME", "/tmp/xdg"),
            ("APPDATA", r"C:\Users\u\AppData\Roaming"),
        ]));
        assert_eq!(dir, Some(PathBuf::from("/tmp/xdg/visor")));
    }

    #[test]
    fn appdata_beats_home() {
        let appdata = r"C:\Users\u\AppData\Roaming";
        let dir = config_dir(env_of(&[("HOME", "/home/u"), ("APPDATA", appdata)]));
        assert_eq!(dir, Some(PathBuf::from(appdata).join("visor")));
    }

    #[test]
    fn home_dir_falls_back_to_userprofile() {
        assert_eq!(
            home_dir(env_of(&[
                ("HOME", "/home/u"),
                ("USERPROFILE", r"C:\Users\u")
            ])),
            Some("/home/u".to_owned())
        );
        assert_eq!(
            home_dir(env_of(&[("USERPROFILE", r"C:\Users\u")])),
            Some(r"C:\Users\u".to_owned())
        );
        assert_eq!(
            home_dir(env_of(&[("HOME", ""), ("USERPROFILE", r"C:\Users\u")])),
            Some(r"C:\Users\u".to_owned())
        );
        assert_eq!(home_dir(env_of(&[])), None);
        let dir = config_dir(env_of(&[("USERPROFILE", r"C:\Users\u")]));
        assert_eq!(
            dir,
            Some(PathBuf::from(r"C:\Users\u").join(".config").join("visor"))
        );
    }
}
