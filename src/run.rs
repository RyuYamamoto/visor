//! eframe startup, shared by visor's own binary and by any binary that links extra plugins.

use std::sync::Arc;

use crate::app::ViewerApp;
use crate::plugin::registry::Registry;
use crate::source::launch::Launch;
use crate::theme;

/// Open the viewer window. Build the registry first, add plugins to it, then hand it over here.
pub fn run(launch: Launch, registry: Arc<Registry>) -> eframe::Result {
    let (types, type_problems) = registry.build_type_registry();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Visor")
            // Match the .desktop file basename so GNOME associates the window with the app.
            .with_app_id("dev.visor.viewer")
            // Icon of the running window itself (needed separately from the launcher's .desktop/hicolor).
            .with_icon(
                eframe::icon_data::from_png_bytes(include_bytes!("../assets/icons/icon_256.png"))
                    .expect("embedded window icon PNG should decode"),
            )
            .with_inner_size([1280.0, 800.0]),
        // For 3D viewport depth testing (egui-wgpu attaches Depth32Float to the shared render pass).
        depth_buffer: 32,
        // Frame latency 1 (vs default 2) shortens how long X11 live resize shows stale buffer content in the newly exposed area.
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            surface: eframe::egui_wgpu::SurfaceConfig::LOW_LATENCY,
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "Visor",
        options,
        Box::new(move |cc| {
            theme::install_fonts(&cc.egui_ctx);
            theme::ui::apply(&cc.egui_ctx, egui::Theme::Dark);
            Ok(Box::new(ViewerApp::new(
                cc,
                launch.clone(),
                Arc::clone(&registry),
                Arc::clone(&types),
                type_problems.clone(),
            )))
        }),
    )
}

/// Resolve command-line arguments, printing usage and exiting 2 on error (shared by every visor binary).
pub fn resolve_launch() -> Launch {
    match crate::source::launch::resolve(std::env::args().skip(1), |key| std::env::var(key).ok()) {
        Ok(launch) => launch,
        Err(e) => {
            eprintln!("visor: {e}");
            eprintln!("{}", crate::source::launch::USAGE);
            std::process::exit(2);
        }
    }
}

/// Read one environment variable, the form `Registry::finish` takes so tests can inject their own.
pub fn env_lookup(key: &str) -> Option<String> {
    std::env::var(key).ok()
}
