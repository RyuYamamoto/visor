//! Command-line resolution of which source to start (requirements FR-1, G14).

use std::path::PathBuf;

use super::Mode;
use crate::comm::CommConfig;
use crate::config;

/// One-line usage, shared by the GUI's argument errors.
pub const USAGE: &str = "usage: visor [--config <path>] [--bag <path>…] [--endpoint <ep>] [--domain-id <n>]\n\
     \n\
     <path> is a ROS 1 .bag, or a ROS 2 bag: its directory, its metadata.yaml, or the .mcap / .db3 files themselves\n\
     several bags are merged by record time, so `visor --bag run_*.bag` replays a split recording as one timeline";

/// What to start with: a source, the config file to autoload, and the connection live mode uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub mode: Mode,
    pub config_path: Option<PathBuf>,
    /// Connection for live mode, resolved even when starting from a bag so the Source menu can switch to it.
    pub live: CommConfig,
}

/// Resolve arguments into a launch plan; `--bag` with an explicit connection argument is rejected as a mode conflict.
pub fn resolve<I, F>(args: I, env: F) -> Result<Launch, String>
where
    I: IntoIterator<Item = String>,
    F: Fn(&str) -> Option<String>,
{
    let (config_path, rest) = config::split_config_arg(args)?;
    let (bags, rest) = split_bag_arg(rest)?;
    let bags = expand_directories(bags)?;
    if bags.is_empty() {
        let live = CommConfig::resolve(rest, env)?;
        return Ok(Launch {
            mode: Mode::Live(live.clone()),
            config_path,
            live,
        });
    }
    // Only explicit arguments conflict: ROS_STATIC_PEERS and friends are permanently set by robot-connection shells (plan §2.4).
    if let Some(conflict) = rest
        .iter()
        .find(|arg| *arg == "--endpoint" || *arg == "--domain-id")
    {
        return Err(format!("--bag cannot be combined with {conflict}"));
    }
    if let Some(unknown) = rest.first() {
        return Err(format!("unknown argument `{unknown}`"));
    }
    // Connection arguments are rejected above, so this is env-or-default: what "switch to live" will use.
    let live = CommConfig::resolve(Vec::new(), env)?;
    Ok(Launch {
        mode: Mode::Files(bags),
        config_path,
        live,
    })
}

/// Replace a rosbag2 directory by its `metadata.yaml`, or by its `.mcap` / `.db3` files in name order when the yaml is missing; files pass through untouched. Done here, before a source descriptor is chosen by extension.
pub fn expand_directories(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        if !path.is_dir() {
            out.push(path);
            continue;
        }
        let metadata = path.join("metadata.yaml");
        if metadata.is_file() {
            out.push(metadata);
            continue;
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                    e.eq_ignore_ascii_case("mcap") || e.eq_ignore_ascii_case("db3")
                })
            })
            .collect();
        if files.is_empty() {
            return Err(format!(
                "{}: no metadata.yaml and no .mcap / .db3 files in the directory",
                path.display()
            ));
        }
        files.sort();
        out.extend(files);
    }
    Ok(out)
}

/// Extract `--bag <path>…`, returning the paths and the remaining arguments.
/// Every following non-flag argument is a path, so a shell glob (`--bag run_*.bag`) works; `--bag` may also repeat.
fn split_bag_arg(args: Vec<String>) -> Result<(Vec<PathBuf>, Vec<String>), String> {
    let mut bags: Vec<PathBuf> = Vec::new();
    let mut rest = Vec::new();
    let args: Vec<String> = args.into_iter().collect();
    let mut at = 0;
    while at < args.len() {
        if args[at] != "--bag" {
            rest.push(args[at].clone());
            at += 1;
            continue;
        }
        at += 1;
        let before = bags.len();
        while at < args.len() && !args[at].starts_with("--") {
            bags.push(PathBuf::from(&args[at]));
            at += 1;
        }
        // A flag with nothing after it is a typo, not "play every bag".
        if bags.len() == before {
            return Err("--bag requires a value".to_owned());
        }
    }
    Ok((bags, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn t17_no_arguments_means_live_with_defaults() {
        let launch = resolve(args(&[]), no_env).unwrap();
        assert!(matches!(launch.mode, Mode::Live(_)));
        assert_eq!(launch.config_path, None);
    }

    #[test]
    fn t17_bag_alone_selects_bag_mode() {
        let launch = resolve(args(&["--bag", "/tmp/a.bag"]), no_env).unwrap();
        assert_eq!(launch.mode, Mode::Files(vec![PathBuf::from("/tmp/a.bag")]));
    }

    #[test]
    fn t17_bag_takes_several_paths_from_a_glob_or_repetition() {
        // `visor --bag run_*.bag` after shell expansion.
        let launch = resolve(args(&["--bag", "a_0.bag", "a_1.bag", "a_2.bag"]), no_env).unwrap();
        assert_eq!(
            launch.mode,
            Mode::Files(vec![
                PathBuf::from("a_0.bag"),
                PathBuf::from("a_1.bag"),
                PathBuf::from("a_2.bag"),
            ])
        );
        // Repeating the flag works too, and a later flag ends the path list.
        let launch = resolve(
            args(&[
                "--bag",
                "a.bag",
                "--bag",
                "b.bag",
                "--config",
                "/tmp/v.toml",
            ]),
            no_env,
        )
        .unwrap();
        assert_eq!(
            launch.mode,
            Mode::Files(vec![PathBuf::from("a.bag"), PathBuf::from("b.bag")])
        );
        assert_eq!(launch.config_path, Some(PathBuf::from("/tmp/v.toml")));
    }

    #[test]
    fn t17_bag_and_an_explicit_connection_argument_conflict() {
        for extra in [
            args(&["--endpoint", "tcp/x:7447"]),
            args(&["--domain-id", "3"]),
        ] {
            let mut all = args(&["--bag", "/tmp/a.bag"]);
            all.extend(extra);
            let error = resolve(all, no_env).unwrap_err();
            assert!(error.contains("--bag cannot be combined"), "{error}");
        }
    }

    #[test]
    fn t17_bag_ignores_connection_environment_variables() {
        // A shell that exports ROS_STATIC_PEERS for robot access must not make --bag unusable.
        let env = |key: &str| match key {
            "ROS_STATIC_PEERS" => Some("tcp/robot:7447".to_owned()),
            "ROS_DOMAIN_ID" => Some("32".to_owned()),
            _ => None,
        };
        let launch = resolve(args(&["--bag", "/tmp/a.bag"]), env).unwrap();
        assert_eq!(launch.mode, Mode::Files(vec![PathBuf::from("/tmp/a.bag")]));
        // The environment still decides where "switch to live" would connect.
        assert_eq!(launch.live.endpoint, "tcp/robot:7447");
        assert_eq!(launch.live.domain_id, 32);
    }

    #[test]
    fn t17_live_config_is_available_in_both_modes() {
        let launch = resolve(args(&["--endpoint", "tcp/sim:7447"]), no_env).unwrap();
        assert_eq!(launch.mode, Mode::Live(launch.live.clone()));
        // Starting from a bag with nothing set still yields the default connection to switch to.
        let launch = resolve(args(&["--bag", "/tmp/a.bag"]), no_env).unwrap();
        assert_eq!(launch.live.endpoint, crate::comm::DEFAULT_ENDPOINT);
    }

    #[test]
    fn t17_bag_combines_with_config() {
        let launch = resolve(
            args(&["--config", "/tmp/v.toml", "--bag", "/tmp/a.bag"]),
            no_env,
        )
        .unwrap();
        assert_eq!(launch.mode, Mode::Files(vec![PathBuf::from("/tmp/a.bag")]));
        assert_eq!(launch.config_path, Some(PathBuf::from("/tmp/v.toml")));
    }

    #[test]
    fn t17_missing_values_and_unknown_arguments_are_rejected() {
        assert!(resolve(args(&["--bag"]), no_env).is_err());
        // A dangling repetition is also a typo.
        assert!(resolve(args(&["--bag", "a.bag", "--bag"]), no_env).is_err());
        assert!(resolve(args(&["--bag", "/tmp/a.bag", "--bogus"]), no_env).is_err());
        assert!(resolve(args(&["--bogus"]), no_env).is_err());
    }

    #[test]
    fn a_bag_directory_expands_to_its_metadata_or_its_storage_files() {
        let root = std::env::temp_dir().join(format!("visor_launch_test_{}", std::process::id()));
        let with_yaml = root.join("with_yaml");
        let files_only = root.join("files_only");
        let empty = root.join("empty");
        for dir in [&with_yaml, &files_only, &empty] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(
            with_yaml.join("metadata.yaml"),
            "rosbag2_bagfile_information:\n",
        )
        .unwrap();
        std::fs::write(with_yaml.join("with_yaml_0.mcap"), b"").unwrap();
        std::fs::write(files_only.join("run_1.mcap"), b"").unwrap();
        std::fs::write(files_only.join("run_0.mcap"), b"").unwrap();
        std::fs::write(files_only.join("notes.txt"), b"").unwrap();
        // metadata.yaml wins when present, so the rosbag2 source expands it (and checks its compression settings).
        let launch = resolve(args(&["--bag", with_yaml.to_str().unwrap()]), no_env).unwrap();
        assert_eq!(
            launch.mode,
            Mode::Files(vec![with_yaml.join("metadata.yaml")])
        );
        // Without it, the storage files are listed in name order and unrelated files are ignored.
        let launch = resolve(args(&["--bag", files_only.to_str().unwrap()]), no_env).unwrap();
        assert_eq!(
            launch.mode,
            Mode::Files(vec![
                files_only.join("run_0.mcap"),
                files_only.join("run_1.mcap")
            ])
        );
        // An empty directory is an error naming the directory, not an empty source.
        let error = resolve(args(&["--bag", empty.to_str().unwrap()]), no_env).unwrap_err();
        assert!(error.contains("no metadata.yaml"), "{error}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn t17_live_mode_still_takes_endpoint_and_domain() {
        let launch = resolve(
            args(&["--endpoint", "tcp/sim:7447", "--domain-id", "3"]),
            no_env,
        )
        .unwrap();
        let Mode::Live(config) = launch.mode else {
            panic!("expected live mode");
        };
        assert_eq!(config.endpoint, "tcp/sim:7447");
        assert_eq!(config.domain_id, 3);
    }
}
