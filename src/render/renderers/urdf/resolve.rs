//! URDF resource URI -> absolute path, and `$(find pkg)` -> package directory. Pure functions with env lookup and file existence injected, so every search branch is unit-testable without a ROS 2 installation.

use std::path::{Path, PathBuf};

/// Mesh search roots the user supplies (a PATH-style list: `:` on Unix, `;` on Windows).
pub const MESH_ROOTS_ENV: &str = "VISOR_MESH_ROOTS";
/// Alias accepted when [`MESH_ROOTS_ENV`] is unset (the name used by this project's task description).
pub const MESH_ROOTS_ENV_ALIAS: &str = "ROS2_VIEWER_MESH_ROOTS";

/// Search roots in priority order: `package://` tries them top to bottom and takes the first hit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeshRoots {
    /// User roots: the settings_ui list first, then the ones from the environment.
    pub user: Vec<PathBuf>,
    /// AMENT_PREFIX_PATH prefixes, searched as `<prefix>/share/<pkg>/<rel>`.
    pub ament: Vec<PathBuf>,
    /// ROS_PACKAGE_PATH entries, searched as `<path>/<pkg>/<rel>`.
    pub ros_package_path: Vec<PathBuf>,
}

impl MeshRoots {
    /// Build from the settings_ui list plus the environment (`env` is injected so tests need no real variables).
    pub fn from_env(ui_roots: &[String], env: impl Fn(&str) -> Option<String>) -> Self {
        let home = crate::config::paths::home_dir(&env);
        let mut user: Vec<PathBuf> = ui_roots
            .iter()
            .map(|root| root.trim())
            .filter(|root| !root.is_empty())
            .map(|root| expand_tilde(root, home.as_deref()))
            .collect();
        user.extend(env_user_roots(&env));
        Self {
            user,
            ament: split_paths(env("AMENT_PREFIX_PATH")),
            ros_package_path: split_paths(env("ROS_PACKAGE_PATH")),
        }
    }
}

/// User roots coming from the environment (settings_ui lists them read-only so an env root never looks like it vanished).
pub fn env_user_roots(env: impl Fn(&str) -> Option<String>) -> Vec<PathBuf> {
    let home = crate::config::paths::home_dir(&env);
    let raw = env(MESH_ROOTS_ENV)
        .filter(|v| !v.is_empty())
        .or_else(|| env(MESH_ROOTS_ENV_ALIAS));
    split_paths(raw)
        .into_iter()
        .map(|entry| expand_tilde(&entry.to_string_lossy(), home.as_deref()))
        .collect()
}

/// Why a URI could not be resolved, with every path that was tried (so the user can see which root is missing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveError {
    pub reason: String,
    pub tried: Vec<PathBuf>,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)?;
        if !self.tried.is_empty() {
            let tried: Vec<String> = self.tried.iter().map(|p| p.display().to_string()).collect();
            write!(f, " — tried: {}", tried.join(", "))?;
        }
        Ok(())
    }
}

impl ResolveError {
    fn plain(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            tried: Vec::new(),
        }
    }
}

/// Resolve a URDF `<mesh filename>` to an absolute path; `exists` is the file-existence probe. Only `package://` probes the filesystem, the other forms are handed back as-is so a missing file surfaces as the OS error at read time.
pub fn resolve(
    uri: &str,
    urdf_dir: Option<&Path>,
    roots: &MeshRoots,
    exists: impl Fn(&Path) -> bool,
) -> Result<PathBuf, ResolveError> {
    let uri = uri.trim();
    if uri.is_empty() {
        return Err(ResolveError::plain("empty mesh filename"));
    }
    if let Some(rest) = uri.strip_prefix("package://") {
        return resolve_package(rest, roots, urdf_dir, exists);
    }
    if let Some(rest) = uri.strip_prefix("file://") {
        // Only the local form `file:///abs/path` is supported; `file://host/path` would need a network mount.
        return match rest.strip_prefix('/') {
            Some(_) => Ok(PathBuf::from(rest)),
            None => Err(ResolveError::plain(format!(
                "unsupported file URI with a host component: {uri}"
            ))),
        };
    }
    if let Some(scheme) = unsupported_scheme(uri) {
        return Err(ResolveError::plain(format!(
            "unsupported URI scheme `{scheme}://` ({uri})"
        )));
    }
    let path = Path::new(uri);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    match urdf_dir {
        Some(dir) => Ok(dir.join(path)),
        None => Err(ResolveError::plain(format!(
            "relative mesh path with no URDF directory to resolve against: {uri}"
        ))),
    }
}

/// The `<pkg>/<rel>` part of a `package://` URI -> the first candidate that exists.
fn resolve_package(
    rest: &str,
    roots: &MeshRoots,
    urdf_dir: Option<&Path>,
    exists: impl Fn(&Path) -> bool,
) -> Result<PathBuf, ResolveError> {
    let (package, relative) = rest.split_once('/').unwrap_or((rest, ""));
    check_package(package)
        .map_err(|_| ResolveError::plain(format!("invalid package name in package://{rest}")))?;
    if relative.is_empty() {
        return Err(ResolveError::plain(format!(
            "package://{rest} has no file path after the package name"
        )));
    }
    // `..` would let a URDF reach outside the package, and outside every configured root.
    if Path::new(relative)
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(ResolveError::plain(format!(
            "`..` is not allowed in package://{rest}"
        )));
    }
    let tried: Vec<PathBuf> = package_dirs(package, roots, urdf_dir)
        .into_iter()
        .map(|dir| dir.join(relative))
        .collect();
    if let Some(hit) = tried.iter().find(|candidate| exists(candidate)) {
        return Ok(hit.clone());
    }
    let reason = if tried.is_empty() {
        format!(
            "package `{package}` not found: no mesh roots configured (add one below or set {MESH_ROOTS_ENV})"
        )
    } else {
        format!("package `{package}` does not provide `{relative}`")
    };
    Err(ResolveError { reason, tried })
}

/// Directory of a ROS package for xacro's `$(find pkg)`: the first [`package_dirs`] candidate that is a directory (`is_dir` is the probe); the error lists every candidate.
pub fn find_package(
    package: &str,
    roots: &MeshRoots,
    urdf_dir: Option<&Path>,
    is_dir: impl Fn(&Path) -> bool,
) -> Result<PathBuf, ResolveError> {
    check_package(package)?;
    let tried = package_dirs(package, roots, urdf_dir);
    if let Some(hit) = tried.iter().find(|candidate| is_dir(candidate)) {
        return Ok(hit.clone());
    }
    let reason = if tried.is_empty() {
        format!(
            "package `{package}` not found: no mesh roots configured (add one or set {MESH_ROOTS_ENV})"
        )
    } else {
        format!("package `{package}` not found")
    };
    Err(ResolveError { reason, tried })
}

/// Candidate directories of `package` in priority order (user roots as `<root>/<pkg>` / `<root>/share/<pkg>` / the root itself when its basename is the package, then `AMENT_PREFIX_PATH`, `ROS_PACKAGE_PATH`, and every ancestor of `urdf_dir` with the same three layouts). `package://` meshes and `$(find)` share this list so an include and its meshes agree on where a package is; the ancestors cover the URDF's own package and the sibling packages of its workspace without any root being configured.
pub fn package_dirs(package: &str, roots: &MeshRoots, urdf_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let push = |dirs: &mut Vec<PathBuf>, dir: PathBuf| {
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    };
    let layouts = |dirs: &mut Vec<PathBuf>, root: &Path| {
        push(dirs, root.join(package));
        push(dirs, root.join("share").join(package));
        if root.file_name().is_some_and(|name| name == package) {
            push(dirs, root.to_path_buf());
        }
    };
    for root in &roots.user {
        layouts(&mut dirs, root);
    }
    for prefix in &roots.ament {
        push(&mut dirs, prefix.join("share").join(package));
    }
    for path in &roots.ros_package_path {
        push(&mut dirs, path.join(package));
    }
    if let Some(dir) = urdf_dir {
        for ancestor in dir.ancestors() {
            layouts(&mut dirs, ancestor);
        }
    }
    dirs
}

/// Reject package names that are empty, `.` / `..`, or contain a path separator (any of them would escape the search roots).
fn check_package(package: &str) -> Result<(), ResolveError> {
    let mut components = Path::new(package).components();
    let single_normal_component = matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(_)), None)
    );
    if !single_normal_component || package.contains(['/', '\\']) {
        return Err(ResolveError::plain(format!(
            "invalid package name `{package}`"
        )));
    }
    Ok(())
}

/// URI scheme of a non-path string (`model://`, `http://`, …); None for plain paths.
fn unsupported_scheme(uri: &str) -> Option<&str> {
    let (scheme, _) = uri.split_once("://")?;
    (!scheme.is_empty()
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')))
    .then_some(scheme)
}

/// PATH-style env value -> paths through `std::env::split_paths` (`:` on Unix, `;` on Windows, no cfg needed); entries are trimmed and blank ones dropped.
fn split_paths(value: Option<String>) -> Vec<PathBuf> {
    value
        .into_iter()
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .filter_map(|entry| {
            let trimmed = entry.to_string_lossy().trim().to_owned();
            (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
        })
        .collect()
}

/// Expand a leading `~/` (roots are typed by hand in settings_ui, where `~` is the natural thing to write).
fn expand_tilde(root: &str, home: Option<&str>) -> PathBuf {
    match (root.strip_prefix("~/"), home) {
        (Some(rest), Some(home)) => Path::new(home).join(rest),
        _ => PathBuf::from(root),
    }
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

    /// Existence probe backed by a fixed list of "files on disk".
    fn fs_of<'a>(files: &'a [&'a str]) -> impl Fn(&Path) -> bool + 'a {
        move |path| files.iter().any(|f| Path::new(f) == path)
    }

    /// Build a PATH-style env value the way the OS expects it (`:` or `;`), so tests never hard-code a separator.
    fn joined(parts: &[&str]) -> String {
        std::env::join_paths(parts).unwrap().into_string().unwrap()
    }

    fn roots(user: &[&str], ament: &[&str], ros_package_path: &[&str]) -> MeshRoots {
        let to_paths = |v: &[&str]| v.iter().map(PathBuf::from).collect();
        MeshRoots {
            user: to_paths(user),
            ament: to_paths(ament),
            ros_package_path: to_paths(ros_package_path),
        }
    }

    #[test]
    fn user_roots_win_then_ament_then_ros_package_path() {
        let roots = roots(&["/user"], &["/opt/ros"], &["/src"]);
        let uri = "package://pkg/meshes/a.stl";
        let all = fs_of(&[
            "/user/pkg/meshes/a.stl",
            "/opt/ros/share/pkg/meshes/a.stl",
            "/src/pkg/meshes/a.stl",
        ]);
        assert_eq!(
            resolve(uri, None, &roots, all).unwrap(),
            PathBuf::from("/user/pkg/meshes/a.stl")
        );
        let ament_only = fs_of(&["/opt/ros/share/pkg/meshes/a.stl", "/src/pkg/meshes/a.stl"]);
        assert_eq!(
            resolve(uri, None, &roots, ament_only).unwrap(),
            PathBuf::from("/opt/ros/share/pkg/meshes/a.stl")
        );
        let ros_only = fs_of(&["/src/pkg/meshes/a.stl"]);
        assert_eq!(
            resolve(uri, None, &roots, ros_only).unwrap(),
            PathBuf::from("/src/pkg/meshes/a.stl")
        );
    }

    #[test]
    fn user_roots_accept_three_layouts() {
        let uri = "package://pkg/meshes/a.stl";
        let source = roots(&["/ws"], &[], &[]);
        assert_eq!(
            resolve(uri, None, &source, fs_of(&["/ws/pkg/meshes/a.stl"])).unwrap(),
            PathBuf::from("/ws/pkg/meshes/a.stl")
        );
        assert_eq!(
            resolve(uri, None, &source, fs_of(&["/ws/share/pkg/meshes/a.stl"])).unwrap(),
            PathBuf::from("/ws/share/pkg/meshes/a.stl")
        );
        let direct = roots(&["/ws/pkg"], &[], &[]);
        assert_eq!(
            resolve(uri, None, &direct, fs_of(&["/ws/pkg/meshes/a.stl"])).unwrap(),
            PathBuf::from("/ws/pkg/meshes/a.stl")
        );
        // A root whose basename differs never gets the bare layout, so nothing outside the package is picked up.
        let other = roots(&["/ws/other"], &[], &[]);
        assert!(resolve(uri, None, &other, fs_of(&["/ws/other/meshes/a.stl"])).is_err());
    }

    #[test]
    fn file_absolute_and_relative_forms_need_no_probe() {
        let none = roots(&[], &[], &[]);
        let never = |_: &Path| false;
        assert_eq!(
            resolve("file:///abs/a.stl", None, &none, never).unwrap(),
            PathBuf::from("/abs/a.stl")
        );
        // An absolute path of the host OS (drive-qualified on Windows) comes back untouched without any probe.
        let abs = std::env::current_dir().unwrap().join("a.stl");
        assert_eq!(
            resolve(abs.to_str().unwrap(), None, &none, never).unwrap(),
            abs
        );
        assert_eq!(
            resolve("meshes/a.stl", Some(Path::new("/robot/urdf")), &none, never).unwrap(),
            PathBuf::from("/robot/urdf/meshes/a.stl")
        );
        assert!(resolve("file://host/a.stl", None, &none, never).is_err());
        let error = resolve("meshes/a.stl", None, &none, never).unwrap_err();
        assert!(error.reason.contains("URDF directory"), "{error}");
    }

    #[test]
    fn unresolved_package_lists_every_candidate() {
        let roots = roots(&["/user", "/ws/pkg"], &["/opt/ros"], &["/src"]);
        let error =
            resolve("package://pkg/meshes/a.stl", None, &roots, |_: &Path| false).unwrap_err();
        assert_eq!(
            error.tried,
            vec![
                PathBuf::from("/user/pkg/meshes/a.stl"),
                PathBuf::from("/user/share/pkg/meshes/a.stl"),
                PathBuf::from("/ws/pkg/pkg/meshes/a.stl"),
                PathBuf::from("/ws/pkg/share/pkg/meshes/a.stl"),
                PathBuf::from("/ws/pkg/meshes/a.stl"),
                PathBuf::from("/opt/ros/share/pkg/meshes/a.stl"),
                PathBuf::from("/src/pkg/meshes/a.stl"),
            ]
        );
        let text = error.to_string();
        assert!(text.contains("pkg"), "{text}");
        // Same join steps as resolve_package (the relative part is one segment), so the separators match on every OS.
        let ament_candidate = Path::new("/opt/ros").join("share").join("pkg").join("meshes/a.stl");
        assert!(
            text.contains(&ament_candidate.display().to_string()),
            "{text}"
        );
        let empty = resolve(
            "package://pkg/meshes/a.stl",
            None,
            &MeshRoots::default(),
            |_: &Path| false,
        )
        .unwrap_err();
        assert!(empty.tried.is_empty());
        assert!(empty.reason.contains("no mesh roots"), "{empty}");
    }

    fn paths(list: &[&str]) -> Vec<PathBuf> {
        list.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn the_urdf_directory_ancestors_are_searched_after_every_root() {
        let urdf_dir = Path::new("/ws/src/robot_description/urdf");
        let none = roots(&[], &[], &[]);
        // The URDF's own package is found through the ancestor whose basename matches.
        let own = resolve(
            "package://robot_description/meshes/a.stl",
            Some(urdf_dir),
            &none,
            fs_of(&["/ws/src/robot_description/meshes/a.stl"]),
        )
        .unwrap();
        assert_eq!(own, PathBuf::from("/ws/src/robot_description/meshes/a.stl"));
        // A sibling package of the same workspace is found through a higher ancestor.
        let sibling = resolve(
            "package://other/meshes/b.stl",
            Some(urdf_dir),
            &none,
            fs_of(&["/ws/src/other/meshes/b.stl"]),
        )
        .unwrap();
        assert_eq!(sibling, PathBuf::from("/ws/src/other/meshes/b.stl"));
        // A configured root wins over the ancestors when both have the file.
        let user = roots(&["/user"], &[], &[]);
        let both = fs_of(&["/user/other/meshes/b.stl", "/ws/src/other/meshes/b.stl"]);
        assert_eq!(
            resolve("package://other/meshes/b.stl", Some(urdf_dir), &user, both).unwrap(),
            PathBuf::from("/user/other/meshes/b.stl")
        );
        assert_eq!(
            package_dirs("other", &none, Some(Path::new("/a/b"))),
            paths(&[
                "/a/b/other",
                "/a/b/share/other",
                "/a/other",
                "/a/share/other",
                "/other",
                "/share/other"
            ])
        );
        // Duplicates are dropped when a root is also an ancestor.
        let dup = roots(&["/a"], &[], &[]);
        assert_eq!(
            package_dirs("other", &dup, Some(Path::new("/a"))),
            paths(&["/a/other", "/a/share/other", "/other", "/share/other"])
        );
        // Without a URDF directory the list is exactly the configured roots, as before.
        assert_eq!(
            package_dirs(
                "pkg",
                &roots(&["/user", "/ws/pkg"], &["/opt/ros"], &["/src"]),
                None
            ),
            paths(&[
                "/user/pkg",
                "/user/share/pkg",
                "/ws/pkg/pkg",
                "/ws/pkg/share/pkg",
                "/ws/pkg",
                "/opt/ros/share/pkg",
                "/src/pkg"
            ])
        );
    }

    #[test]
    fn find_package_returns_the_first_existing_directory_and_lists_the_rest() {
        let roots = roots(&["/user"], &["/opt/ros"], &[]);
        let dirs = |list: &'static [&'static str]| {
            move |path: &Path| list.iter().any(|d| Path::new(d) == path)
        };
        assert_eq!(
            find_package("pkg", &roots, None, dirs(&["/opt/ros/share/pkg"])).unwrap(),
            PathBuf::from("/opt/ros/share/pkg")
        );
        assert_eq!(
            find_package(
                "pkg",
                &roots,
                None,
                dirs(&["/user/pkg", "/opt/ros/share/pkg"])
            )
            .unwrap(),
            PathBuf::from("/user/pkg")
        );
        let error = find_package("pkg", &roots, None, |_: &Path| false).unwrap_err();
        assert_eq!(
            error.tried,
            paths(&["/user/pkg", "/user/share/pkg", "/opt/ros/share/pkg"])
        );
        // Built with the same joins as package_dirs so the separators match on every OS.
        let ament_dir = Path::new("/opt/ros").join("share").join("pkg");
        assert!(
            error.to_string().contains(&ament_dir.display().to_string()),
            "{error}"
        );
        let empty = find_package("pkg", &MeshRoots::default(), None, |_: &Path| true).unwrap_err();
        assert!(empty.tried.is_empty());
        assert!(empty.reason.contains("no mesh roots"), "{empty}");
        // The ancestors alone are enough when the xacro lives inside its workspace.
        let found = find_package(
            "pkg",
            &MeshRoots::default(),
            Some(Path::new("/ws/src/robot/urdf")),
            dirs(&["/ws/src/pkg"]),
        )
        .unwrap();
        assert_eq!(found, PathBuf::from("/ws/src/pkg"));
        for bad in ["", ".", "..", "a/b", "/abs", "a/"] {
            let error = find_package(bad, &roots, None, |_: &Path| true).unwrap_err();
            assert!(error.tried.is_empty(), "{bad} -> {error}");
        }
    }

    #[test]
    fn malformed_package_uris_are_rejected() {
        let roots = roots(&["/user"], &[], &[]);
        for uri in [
            "package://",
            "package:///meshes/a.stl",
            "package://pkg",
            "package://pkg/",
            "package://../etc/passwd",
            "package://./a.stl",
            "package://pkg/../../etc/passwd",
            "package://pkg/meshes/../../../etc/passwd",
            "",
            "   ",
        ] {
            let error = resolve(uri, None, &roots, |_: &Path| true).unwrap_err();
            assert!(error.tried.is_empty(), "{uri} -> {error}");
        }
        let error = resolve("package://pkg/meshes/../a.stl", None, &roots, |_: &Path| {
            true
        })
        .unwrap_err();
        assert!(error.reason.contains(".."), "{error}");
    }

    #[test]
    fn other_schemes_report_that_they_are_unsupported() {
        let roots = roots(&["/user"], &[], &[]);
        for uri in [
            "model://turtlebot/meshes/a.dae",
            "http://example.com/a.stl",
            "https://example.com/a.stl",
        ] {
            let error = resolve(uri, None, &roots, |_: &Path| true).unwrap_err();
            assert!(error.reason.contains("unsupported"), "{uri} -> {error}");
        }
    }

    #[test]
    fn env_roots_are_appended_after_the_ui_list() {
        let mesh_roots = joined(&["/env/a", "/env/b"]);
        let ament = joined(&["/opt/ros/humble", "/opt/overlay"]);
        let pairs = [
            ("VISOR_MESH_ROOTS", mesh_roots.as_str()),
            ("AMENT_PREFIX_PATH", ament.as_str()),
            ("ROS_PACKAGE_PATH", "/src"),
            ("HOME", "/home/u"),
        ];
        let roots = MeshRoots::from_env(&["/ui".to_owned(), "  ".to_owned()], env_of(&pairs));
        assert_eq!(
            roots.user,
            vec![
                PathBuf::from("/ui"),
                PathBuf::from("/env/a"),
                PathBuf::from("/env/b")
            ]
        );
        assert_eq!(
            roots.ament,
            vec![
                PathBuf::from("/opt/ros/humble"),
                PathBuf::from("/opt/overlay")
            ]
        );
        assert_eq!(roots.ros_package_path, vec![PathBuf::from("/src")]);
    }

    #[test]
    fn env_root_variable_alias_and_tilde_expansion() {
        let alias = env_of(&[("ROS2_VIEWER_MESH_ROOTS", "/alias"), ("HOME", "/home/u")]);
        assert_eq!(env_user_roots(alias), vec![PathBuf::from("/alias")]);
        let both = env_of(&[
            ("VISOR_MESH_ROOTS", "/primary"),
            ("ROS2_VIEWER_MESH_ROOTS", "/alias"),
        ]);
        assert_eq!(env_user_roots(both), vec![PathBuf::from("/primary")]);
        let empty_primary = env_of(&[
            ("VISOR_MESH_ROOTS", ""),
            ("ROS2_VIEWER_MESH_ROOTS", "/alias"),
        ]);
        assert_eq!(env_user_roots(empty_primary), vec![PathBuf::from("/alias")]);
        let with_blank = joined(&["~/ws", "", " /abs "]);
        let tilde = [
            ("VISOR_MESH_ROOTS", with_blank.as_str()),
            ("HOME", "/home/u"),
        ];
        assert_eq!(
            env_user_roots(env_of(&tilde)),
            vec![PathBuf::from("/home/u/ws"), PathBuf::from("/abs")]
        );
        let homeless = env_of(&[("VISOR_MESH_ROOTS", "~/ws")]);
        assert_eq!(env_user_roots(homeless), vec![PathBuf::from("~/ws")]);
        // Windows shells set USERPROFILE instead of HOME; `~/` must expand from that too.
        let userprofile = env_of(&[("VISOR_MESH_ROOTS", "~/ws"), ("USERPROFILE", "/home/u")]);
        assert_eq!(
            env_user_roots(userprofile),
            vec![PathBuf::from("/home/u/ws")]
        );
        let ui = MeshRoots::from_env(&["~/ui".to_owned()], env_of(&[("HOME", "/home/u")]));
        assert_eq!(ui.user, vec![PathBuf::from("/home/u/ui")]);
        assert_eq!(env_user_roots(env_of(&[])), Vec::<PathBuf>::new());
    }
}
