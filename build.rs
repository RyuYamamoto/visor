//! Scans assets/msgs/<pkg>/msg/*.msg and generates the compile-time `EMBEDDED_MSGS` table (key `pkg/msg/Type`, value the file body) into OUT_DIR; std only, table generation only (no parsing).

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Regenerate when .msg files are added, removed, or changed
    println!("cargo::rerun-if-changed=assets/msgs");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let msgs_root = Path::new(&manifest_dir).join("assets").join("msgs");

    let mut entries: Vec<(String, String)> = Vec::new();
    for pkg_entry in fs::read_dir(&msgs_root).expect("assets/msgs not found") {
        let pkg_entry = pkg_entry.expect("read_dir entry");
        if !pkg_entry.file_type().expect("file_type").is_dir() {
            continue;
        }
        let pkg = pkg_entry
            .file_name()
            .into_string()
            .expect("package dir name is not UTF-8");
        let msg_dir = pkg_entry.path().join("msg");
        if !msg_dir.is_dir() {
            continue;
        }
        for msg_entry in fs::read_dir(&msg_dir).expect("read msg dir") {
            let path = msg_entry.expect("read_dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("msg") {
                continue;
            }
            let type_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("msg file name is not UTF-8");
            entries.push((
                format!("{pkg}/msg/{type_name}"),
                path.to_str().expect("msg path is not UTF-8").to_owned(),
            ));
        }
    }
    // Make the generated code deterministic (read_dir order is environment-dependent)
    entries.sort();

    let mut out = String::new();
    out.push_str("/// (`pkg/msg/Type`, .msg body) table generated from assets/msgs by build.rs\n");
    out.push_str("pub static EMBEDDED_MSGS: &[(&str, &str)] = &[\n");
    for (full_name, path) in &entries {
        out.push_str(&format!("    ({full_name:?}, include_str!({path:?})),\n"));
    }
    out.push_str("];\n");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    fs::write(Path::new(&out_dir).join("embedded_msgs.rs"), out).expect("write embedded_msgs.rs");
}
