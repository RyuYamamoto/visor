//! Merging .msg definitions from every origin into one TypeRegistry, plus the `VISOR_MSG_PATHS` scan.

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::decode::msg_parser::TypeRegistry;

use super::ids::PluginId;
use super::registry::{Problem, ProblemKind};

/// Environment variable holding extra `.msg` roots (a PATH-style list: `:` on Unix, `;` on Windows), each laid out as `<root>/<pkg>/msg/*.msg`.
pub const MSG_PATHS_ENV: &str = "VISOR_MSG_PATHS";

/// One `.msg` definition awaiting registration, tagged with where it came from.
#[derive(Debug, Clone)]
pub struct MsgEntry {
    pub origin: PluginId,
    /// Fully qualified `pkg/msg/Type`.
    pub full_name: String,
    pub text: Cow<'static, str>,
}

/// Merge definitions in priority order (first wins) and drop whatever cannot be resolved, reporting both as problems.
pub fn build(entries: &[MsgEntry]) -> (TypeRegistry, Vec<Problem>) {
    let mut registry = TypeRegistry::new();
    let mut problems = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for entry in entries {
        if !seen.insert(entry.full_name.as_str()) {
            problems.push(Problem {
                plugin: entry.origin.clone(),
                kind: ProblemKind::DuplicateMsg,
                message: format!(
                    "`{}` is already defined by an earlier origin; this definition is ignored",
                    entry.full_name
                ),
            });
            continue;
        }
        if let Err(e) = registry.insert_msg(&entry.full_name, &entry.text) {
            problems.push(Problem {
                plugin: entry.origin.clone(),
                kind: ProblemKind::MsgParse,
                message: e.to_string(),
            });
        }
    }
    problems.extend(drop_unresolved(&mut registry, entries));
    (registry, problems)
}

/// Remove every definition with an unresolvable reference, repeating until the set is closed, so one bad type cannot take the rest down.
fn drop_unresolved(registry: &mut TypeRegistry, entries: &[MsgEntry]) -> Vec<Problem> {
    let mut problems = Vec::new();
    loop {
        let unresolved = registry.unresolved();
        if unresolved.is_empty() {
            return problems;
        }
        for (type_name, field, referenced) in unresolved {
            if !registry.remove(&type_name) {
                continue;
            }
            let origin = entries
                .iter()
                .find(|e| e.full_name == type_name)
                .map_or(PluginId::Builtin, |e| e.origin.clone());
            problems.push(Problem {
                plugin: origin,
                kind: ProblemKind::UnresolvedMsg,
                message: format!(
                    "`{type_name}` dropped: field `{field}` references unknown type `{referenced}`"
                ),
            });
        }
    }
}

/// Scan the roots named by `VISOR_MSG_PATHS`, appending `<root>/<pkg>/msg/*.msg`; unreadable roots become one problem each.
pub fn scan_env_paths(
    env: &dyn Fn(&str) -> Option<String>,
    out: &mut Vec<MsgEntry>,
    problems: &mut Vec<Problem>,
) {
    let Some(value) = env(MSG_PATHS_ENV) else {
        return;
    };
    for root in std::env::split_paths(&value).filter(|root| !root.as_os_str().is_empty()) {
        match scan_root(&root) {
            Ok(found) => out.extend(found),
            Err(e) => problems.push(Problem {
                plugin: PluginId::Env,
                kind: ProblemKind::MsgPath,
                message: format!("{}: {e}", root.display()),
            }),
        }
    }
}

/// Read one root laid out like `assets/msgs`, returning its definitions sorted for deterministic priority.
fn scan_root(root: &Path) -> Result<Vec<MsgEntry>, String> {
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    let packages = std::fs::read_dir(root).map_err(|e| e.to_string())?;
    for package in packages {
        let package = package.map_err(|e| e.to_string())?.path();
        let Some(pkg) = package.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let msg_dir = package.join("msg");
        let Ok(files) = std::fs::read_dir(&msg_dir) else {
            continue;
        };
        for file in files {
            let path = file.map_err(|e| e.to_string())?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("msg") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|n| n.to_str()) else {
                continue;
            };
            found.push((format!("{pkg}/msg/{stem}"), path));
        }
    }
    found.sort();
    let mut entries = Vec::with_capacity(found.len());
    for (full_name, path) in found {
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        entries.push(MsgEntry {
            origin: PluginId::Env,
            full_name,
            text: Cow::Owned(text),
        });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(origin: PluginId, full_name: &str, text: &str) -> MsgEntry {
        MsgEntry {
            origin,
            full_name: full_name.to_owned(),
            text: Cow::Owned(text.to_owned()),
        }
    }

    fn plugin(id: &str) -> PluginId {
        PluginId::Plugin(id.to_owned())
    }

    #[test]
    fn first_origin_wins_and_the_later_duplicate_is_reported() {
        let entries = vec![
            entry(PluginId::Builtin, "std_msgs/msg/Header", "int32 seq"),
            entry(plugin("sample"), "std_msgs/msg/Header", "float64 other"),
        ];
        let (registry, problems) = build(&entries);
        let header = registry.get("std_msgs/msg/Header").expect("kept");
        assert_eq!(header.fields[0].name, "seq");
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].kind, ProblemKind::DuplicateMsg);
        assert_eq!(problems[0].plugin, plugin("sample"));
    }

    #[test]
    fn a_parse_failure_is_reported_without_losing_the_other_definitions() {
        let entries = vec![
            entry(plugin("sample"), "sample_msgs/msg/Good", "int32 x"),
            entry(plugin("sample"), "sample_msgs/msg/Bad", "wstring s"),
        ];
        let (registry, problems) = build(&entries);
        assert!(registry.get("sample_msgs/msg/Good").is_some());
        assert!(registry.get("sample_msgs/msg/Bad").is_none());
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].kind, ProblemKind::MsgParse);
    }

    #[test]
    fn unresolved_references_are_dropped_transitively_and_leave_unrelated_types_alone() {
        let entries = vec![
            entry(plugin("sample"), "t/msg/A", "t/B dep"),
            entry(plugin("sample"), "t/msg/B", "t/C dep"),
            entry(plugin("sample"), "t/msg/D", "int32 x"),
        ];
        let (registry, problems) = build(&entries);
        assert!(registry.get("t/msg/D").is_some());
        assert!(registry.get("t/msg/A").is_none());
        assert!(registry.get("t/msg/B").is_none());
        let dropped: Vec<&str> = problems
            .iter()
            .filter(|p| p.kind == ProblemKind::UnresolvedMsg)
            .map(|p| p.message.as_str())
            .collect();
        assert_eq!(dropped.len(), 2, "{dropped:?}");
        assert!(dropped.iter().any(|m| m.contains("t/msg/B")));
        assert!(dropped.iter().any(|m| m.contains("t/msg/A")));
    }

    #[test]
    fn env_scan_reads_a_package_layout_and_reports_bad_roots() {
        let root = std::env::temp_dir().join(format!("visor_msg_scan_{}", std::process::id()));
        let msg_dir = root.join("sample_msgs").join("msg");
        std::fs::create_dir_all(&msg_dir).expect("temp msg dir");
        std::fs::write(msg_dir.join("Ping.msg"), "int32 seq\n").expect("write msg");
        std::fs::write(msg_dir.join("notes.txt"), "ignored").expect("write txt");
        let value = std::env::join_paths([root.clone(), root.join("missing")])
            .unwrap()
            .into_string()
            .unwrap();
        let env = |key: &str| (key == MSG_PATHS_ENV).then(|| value.clone());
        let mut entries = Vec::new();
        let mut problems = Vec::new();
        scan_env_paths(&env, &mut entries, &mut problems);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].full_name, "sample_msgs/msg/Ping");
        assert_eq!(entries[0].origin, PluginId::Env);
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].kind, ProblemKind::MsgPath);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn no_env_variable_adds_nothing() {
        let mut entries = Vec::new();
        let mut problems = Vec::new();
        scan_env_paths(&|_| None, &mut entries, &mut problems);
        assert!(entries.is_empty() && problems.is_empty());
    }
}
