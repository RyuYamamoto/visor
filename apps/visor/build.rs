//! Embeds the exe icon and version info into visor.exe when targeting Windows; a no-op on every other target.

const ICON: &str = "../../assets/icons/visor.ico";

fn main() {
    println!("cargo::rerun-if-changed={ICON}");
    // Decided at build time from the target, not with #[cfg]: build scripts run on the host, which may differ when cross-compiling.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    winresource::WindowsResource::new()
        .set_icon(ICON)
        .compile()
        .expect("embed the Windows icon resource (needs rc.exe from the Windows SDK)");
}
