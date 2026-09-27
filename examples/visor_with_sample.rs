//! What a binary that links a plugin looks like: build the registry, add the plugin, run.

use visor::plugin::registry::Registry;
use visor::run;
use visor_plugin_sample::SamplePlugin;

fn main() -> eframe::Result {
    let launch = run::resolve_launch();
    let mut registry = Registry::builtin();
    registry.add_plugin(&SamplePlugin::default());
    run::run(launch, registry.finish(&run::env_lookup))
}
