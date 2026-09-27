//! rosbag2 bag directories: `metadata.yaml` → storage files (`relative_file_paths`), and the storage-format dispatch for `.mcap` / `.db3`.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use super::reader::BagError;
use super::storage::Storage;
use super::{mcap, sqlite3};

/// `OpenStorage` for rosbag2 files: the extension picks the storage plugin, as `storage_id` does in rosbag2.
pub fn open_storage(
    path: &Path,
    cache: usize,
    cancel: &AtomicBool,
) -> Result<Box<dyn Storage>, BagError> {
    match storage_kind(path) {
        Some(StorageKind::Mcap) => mcap::open_storage(path, cache, cancel),
        Some(StorageKind::Sqlite3) => sqlite3::open_storage(path, cache, cancel),
        None => Err(BagError::Unsupported(format!(
            "`{}` is not a rosbag2 storage file (.mcap or .db3)",
            path.display()
        ))),
    }
}

/// Replace every `metadata.yaml` in `paths` by the storage files it lists; `.mcap` / `.db3` pass through. Mixed storage is refused.
pub fn expand_paths(paths: &[PathBuf]) -> Result<Vec<PathBuf>, BagError> {
    let mut files = Vec::new();
    for path in paths {
        if is_yaml(path) {
            files.extend(files_of_metadata(path)?);
        } else {
            files.push(path.clone());
        }
    }
    let mut kinds: Vec<Option<StorageKind>> = files.iter().map(|p| storage_kind(p)).collect();
    kinds.dedup();
    if kinds.len() > 1 {
        return Err(BagError::Malformed(
            "mixed storage: a set has to be all .mcap or all .db3".to_owned(),
        ));
    }
    Ok(files)
}

/// The storage files a `metadata.yaml` names, resolved against its directory; rosbag2's own file / message compression is refused.
fn files_of_metadata(path: &Path) -> Result<Vec<PathBuf>, BagError> {
    let text = std::fs::read_to_string(path)?;
    if let Some((mode, format)) = compression_of(&text) {
        return Err(BagError::Unsupported(format!(
            "rosbag2 {mode} compression ({format}); record without `--compression-mode`, or decompress the bag first"
        )));
    }
    let names = relative_file_paths_of(&text);
    if names.is_empty() {
        return Err(BagError::Malformed(format!(
            "`{}` has no `relative_file_paths`; not a rosbag2 metadata.yaml",
            path.display()
        )));
    }
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    Ok(names.iter().map(|name| dir.join(name)).collect())
}

/// The `- name` items under `relative_file_paths:` (a line-based read: the only key this viewer needs from the file).
pub fn relative_file_paths_of(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = text.lines().map(str::trim);
    if !lines.any(|line| line == "relative_file_paths:") {
        return out;
    }
    for line in lines {
        match line.strip_prefix('-') {
            Some(item) => out.push(unquote(item.trim()).to_owned()),
            None if line.is_empty() => continue,
            None => break,
        }
    }
    out
}

/// `(compression_mode, compression_format)` when the bag uses rosbag2's file / message compression; None for `""` / `NONE`.
pub fn compression_of(text: &str) -> Option<(String, String)> {
    let value = |key: &str| {
        text.lines()
            .map(str::trim)
            .find_map(|line| line.strip_prefix(key))
            .map(|rest| unquote(rest.trim()).to_owned())
            .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("none"))
    };
    let mode = value("compression_mode:")?;
    Some((mode, value("compression_format:").unwrap_or_default()))
}

fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
        .unwrap_or(value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageKind {
    Mcap,
    Sqlite3,
}

fn storage_kind(path: &Path) -> Option<StorageKind> {
    match extension(path).as_str() {
        "mcap" => Some(StorageKind::Mcap),
        "db3" => Some(StorageKind::Sqlite3),
        _ => None,
    }
}

fn is_yaml(path: &Path) -> bool {
    matches!(extension(path).as_str(), "yaml" | "yml")
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed copy of what rosbag2 Jazzy writes for a single-file mcap recording.
    const METADATA: &str = "rosbag2_bagfile_information:\n  version: 9\n  storage_identifier: mcap\n  duration:\n    nanoseconds: 4000000000\n  starting_time:\n    nanoseconds_since_epoch: 1700000000000000000\n  message_count: 12\n  topics_with_message_count:\n    - topic_metadata:\n        name: /chatter\n        type: std_msgs/msg/String\n        serialization_format: cdr\n        offered_qos_profiles: \"- history: 3\\n  depth: 0\\n\"\n        type_description_hash: RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18\n      message_count: 12\n  compression_format: \"\"\n  compression_mode: \"\"\n  relative_file_paths:\n    - run_0.mcap\n    - 'run_1.mcap'\n  files:\n    - path: run_0.mcap\n      starting_time:\n        nanoseconds_since_epoch: 1700000000000000000\n      duration:\n        nanoseconds: 4000000000\n      message_count: 12\n  custom_data: ~\n  ros_distro: jazzy\n";

    #[test]
    fn relative_file_paths_come_out_in_order_and_unquoted() {
        assert_eq!(
            relative_file_paths_of(METADATA),
            vec!["run_0.mcap".to_owned(), "run_1.mcap".to_owned()]
        );
        // The topic list's own `- ` items are not mistaken for files, and a file without the key yields nothing.
        assert!(relative_file_paths_of("rosbag2_bagfile_information:\n  version: 9\n").is_empty());
    }

    #[test]
    fn compression_is_detected_only_when_set() {
        assert_eq!(compression_of(METADATA), None);
        let compressed = METADATA
            .replace("compression_format: \"\"", "compression_format: zstd")
            .replace("compression_mode: \"\"", "compression_mode: FILE");
        assert_eq!(
            compression_of(&compressed),
            Some(("FILE".to_owned(), "zstd".to_owned()))
        );
        assert_eq!(
            compression_of(&METADATA.replace("compression_mode: \"\"", "compression_mode: NONE")),
            None
        );
    }

    #[test]
    fn metadata_expands_to_its_storage_files_next_to_it() {
        let dir = std::env::temp_dir().join(format!("visor_rosbag2_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let metadata = dir.join("metadata.yaml");
        std::fs::write(&metadata, METADATA).unwrap();
        let files = expand_paths(&[metadata.clone(), PathBuf::from("/x/extra.mcap")]).unwrap();
        assert_eq!(
            files,
            vec![
                dir.join("run_0.mcap"),
                dir.join("run_1.mcap"),
                PathBuf::from("/x/extra.mcap"),
            ]
        );
        // File compression is refused up front, with the mode named.
        let compressed = METADATA.replace("compression_mode: \"\"", "compression_mode: MESSAGE");
        std::fs::write(&metadata, compressed).unwrap();
        assert!(matches!(
            expand_paths(std::slice::from_ref(&metadata)),
            Err(BagError::Unsupported(_))
        ));
        // A yaml that is not rosbag2's is malformed rather than silently empty.
        std::fs::write(&metadata, "hello: world\n").unwrap();
        assert!(matches!(
            expand_paths(std::slice::from_ref(&metadata)),
            Err(BagError::Malformed(_))
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mixed_storage_is_refused_and_plain_files_pass_through() {
        assert!(matches!(
            expand_paths(&[PathBuf::from("a_0.mcap"), PathBuf::from("a_1.DB3")]),
            Err(BagError::Malformed(_))
        ));
        let files = expand_paths(&[PathBuf::from("a_0.mcap"), PathBuf::from("a_1.MCAP")]).unwrap();
        assert_eq!(files.len(), 2);
        // A file of neither storage kind is reported when opened, not when the set is assembled.
        assert!(matches!(
            open_storage(Path::new("a.csv"), 1, &AtomicBool::new(false)),
            Err(BagError::Unsupported(_))
        ));
    }
}
