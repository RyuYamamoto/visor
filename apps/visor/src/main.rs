//! eframe startup and plugin registration; no other logic here.

// Release builds on Windows are GUI-subsystem executables (no console window); debug keeps the console so `cargo run` still shows logs.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use visor::plugin::registry::Registry;
use visor::run;
use visor_plugin_tf_trajectory::TfTrajectoryPlugin;

fn main() -> eframe::Result {
    let launch = run::resolve_launch();
    let mut registry = Registry::builtin();
    registry.add_plugin(&TfTrajectoryPlugin);
    run::run(launch, registry.finish(&run::env_lookup))
}
