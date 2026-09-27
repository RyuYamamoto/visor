//! Scene colors (fixed in both modes) at the top level, mode-dependent UI colors in [`ui`]; keep all color literals in this file only (CLAUDE.md rule).

use egui::{Color32, CornerRadius, Stroke};

/// 3D viewport clear color (fixed: the scene keeps the dark control-room look in light mode too). Deepened so the cyan accents glow harder and panels step off the backdrop.
pub const VIEWPORT_BG: Color32 = Color32::from_rgb(0x06, 0x08, 0x0C);
/// Text of 3D frame-name labels (drawn over scene content, so fixed like the rest of the scene).
pub const LABEL_TEXT: Color32 = Color32::from_rgb(0xD5, 0xE3, 0xEE);
/// Leader line from a 3D frame-name label to its frame origin.
pub const LABEL_LEADER: Color32 = Color32::from_rgb(0x75, 0x87, 0x9B);
/// Primary accent (scene highlight; the UI counterpart is `ui.accent`).
pub const ACCENT_CYAN: Color32 = Color32::from_rgb(0x00, 0xE5, 0xFF);
/// Secondary accent (TF/marker identity colors; the UI counterpart is `ui.accent_secondary`).
pub const ACCENT_PURPLE: Color32 = Color32::from_rgb(0xB3, 0x88, 0xFF);

/// 3D axis-triad X axis (ROS/RViz red; a separate constant since its role differs from STATUS_ERROR).
pub const AXIS_X: Color32 = Color32::from_rgb(0xE5, 0x3E, 0x4E);
/// 3D axis-triad Y axis (green).
pub const AXIS_Y: Color32 = Color32::from_rgb(0x3E, 0xD5, 0x6B);
/// 3D axis-triad Z axis (blue).
pub const AXIS_Z: Color32 = Color32::from_rgb(0x3E, 0x7A, 0xF5);
/// 3D TF parent-child link line: neutral slate, since the link only shows structure while the axis triad carries the meaning.
pub const TF_LINK: Color32 = Color32::from_rgb(0x5E, 0x6E, 0x82);
/// Default Flat-mode color for point renderers (LaserScan / PointCloud2); white to match RViz.
pub const POINT_FLAT_DEFAULT: Color32 = Color32::WHITE;
/// Intensity colormap gradient stops (low→high: purple→cyan→white, vivid on a dark background).
pub const POINT_COLORMAP: [Color32; 4] = [
    Color32::from_rgb(0x2A, 0x12, 0x52),
    Color32::from_rgb(0xB3, 0x88, 0xFF),
    Color32::from_rgb(0x00, 0xE5, 0xFF),
    Color32::from_rgb(0xFF, 0xFF, 0xFF),
];
/// OccupancyGrid free (value 0) cell color (alpha MAP_FREE_ALPHA; faint cyan so reachable area floats over the background).
pub const MAP_FREE: Color32 = Color32::from_rgb(0x7A, 0xF0, 0xFF);
/// Free-cell opacity (just enough for grid lines to show through).
pub const MAP_FREE_ALPHA: f32 = 0.12;
/// OccupancyGrid occupied (value 100) cell color (bright cyan so walls appear to glow).
pub const MAP_OCCUPIED: Color32 = ACCENT_CYAN;
/// Occupied-cell opacity.
pub const MAP_OCCUPIED_ALPHA: f32 = 1.0;
/// OccupancyGrid unknown (value -1) cell color (very faint purple, distinguishable from free).
pub const MAP_UNKNOWN: Color32 = ACCENT_PURPLE;
/// Unknown-cell opacity.
pub const MAP_UNKNOWN_ALPHA: f32 = 0.06;

/// RViz map palette (matches rviz_default_plugins palette_builder.cpp; 0=white → 100=black).
pub fn rviz_map_palette(index: u8) -> [u8; 4] {
    let i = index as u32;
    match i {
        0..=100 => {
            let v = (255 - (255 * i) / 100) as u8;
            [v, v, v, 255]
        }
        101..=127 => [0, 255, 0, 255],
        128..=254 => [255, ((255 * (i - 128)) / 126) as u8, 0, 255],
        _ => [0x70, 0x89, 0x86, 255],
    }
}

/// RViz costmap palette (0=transparent, 1–98=blue→red, 99=cyan, 100=magenta).
pub fn rviz_costmap_palette(index: u8) -> [u8; 4] {
    let i = index as u32;
    match i {
        0 => [0, 0, 0, 0],
        1..=98 => {
            let v = ((255 * i) / 100) as u8;
            [v, 0, 255 - v, 255]
        }
        99 => [0, 255, 255, 255],
        100 => [255, 0, 255, 255],
        101..=127 => [0, 255, 0, 255],
        128..=254 => [255, ((255 * (i - 128)) / 126) as u8, 0, 255],
        _ => [0x70, 0x89, 0x86, 255],
    }
}

/// RViz raw palette (linear gray using the value directly as brightness).
pub fn rviz_raw_palette(index: u8) -> [u8; 4] {
    [index, index, index, 255]
}

/// World-fixed directional light for mesh shading (not normalized; both the CPU bake and mesh.wgsl normalize it).
pub const MESH_LIGHT_DIR: [f32; 3] = [0.4, 0.5, 1.0];
/// Ambient term of the mesh lambert shading (floor brightness on faces facing away from the light).
pub const MESH_AMBIENT: f32 = 0.35;
/// Diffuse term of the mesh lambert shading.
pub const MESH_DIFFUSE: f32 = 0.65;
/// Mesh light color (white for now; kept as a constant so the look can be pushed toward cyan later).
pub const MESH_LIGHT_COLOR: Color32 = Color32::WHITE;
/// Default base color for a posed mesh without its own color (e.g. a RobotModel link with no material).
pub const MESH_DEFAULT: Color32 = Color32::from_rgb(0x9F, 0xB3, 0xC7);

/// Third accent, warm: reads as "intent" against the cyan environment and never collides with the map palette.
pub const ACCENT_AMBER: Color32 = Color32::from_rgb(0xFF, 0xB8, 0x4D);
/// Default Path line color (assumes global plan; local plan overrides via color setting).
pub const PATH_DEFAULT: Color32 = ACCENT_AMBER;
/// Default Odometry color (shared by arrow and trail).
pub const ODOM_DEFAULT: Color32 = ACCENT_PURPLE;
/// 3D grid lines.
pub const GRID_3D: Color32 = Color32::from_rgb(0x22, 0x30, 0x41);
/// 3D grid center lines (X=0 / Y=0 emphasized brighter).
pub const GRID_3D_MAJOR: Color32 = Color32::from_rgb(0x3C, 0x4E, 0x66);

/// Tree indent width (egui's default 18.0 makes hierarchy hard to read).
pub const INDENT_WIDTH: f32 = 28.0;

/// Chip background behind 3D frame-name labels (keeps them readable over any scene content).
pub const LABEL_CHIP_BG: Color32 = Color32::from_rgba_premultiplied(0x08, 0x0C, 0x11, 0xC8);
/// Chip border of 3D frame-name labels.
pub const LABEL_CHIP_BORDER: Color32 = Color32::from_rgba_premultiplied(0x1E, 0x2A, 0x38, 0xC8);
/// On-screen instrument marks in the 3D view (scale bar, gizmo hub).
pub const INSTRUMENT: Color32 = Color32::from_rgb(0x8C, 0xA0, 0xB8);

/// Type-identity accent for a Displays card's left stripe/dot: the scene color of that type, or the muted UI text color when there is none.
pub fn display_accent(ros_type: &str) -> Color32 {
    let leaf = ros_type.rsplit('/').next().unwrap_or(ros_type);
    match leaf {
        "LaserScan" | "PointCloud2" => ACCENT_CYAN,
        // Both the ROS type name and the registry label are accepted, since topic items are keyed by label where one exists.
        "OccupancyGrid" | "Map" => MAP_OCCUPIED,
        "Path" => PATH_DEFAULT,
        "Odometry" => ODOM_DEFAULT,
        "Marker" | "MarkerArray" => ACCENT_PURPLE,
        "Image" | "CompressedImage" => ACCENT_PURPLE,
        // Standalone displays are keyed by their registry label rather than a ROS type.
        "RobotModel" => MESH_DEFAULT,
        _ => ui::palette().text_muted,
    }
}

/// Whether the render target applies sRGB encoding on write; set once from viewport::init before any frame is drawn.
static TARGET_SRGB: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Records the target's color space so the conversions below emit values that survive it unchanged.
pub fn set_target_srgb(is_srgb: bool) {
    TARGET_SRGB.store(is_srgb, std::sync::atomic::Ordering::Relaxed);
}

fn target_srgb() -> bool {
    TARGET_SRGB.load(std::sync::atomic::Ordering::Relaxed)
}

/// Premultiplied linear RGBA (what an sRGB target expects, since it encodes on write).
fn linear_rgba(color: Color32) -> [f32; 4] {
    egui::Rgba::from(color).to_array()
}

/// Premultiplied sRGB RGBA (what a non-sRGB target expects, since it stores the value as-is).
fn gamma_rgba(color: Color32) -> [f32; 4] {
    color.to_normalized_gamma_f32()
}

/// Converts Color32 to the RGBA our custom pipelines must write so the on-screen color matches the theme.
pub fn to_linear_rgba(color: Color32) -> [f32; 4] {
    if target_srgb() {
        linear_rgba(color)
    } else {
        gamma_rgba(color)
    }
}

/// Quantizes Color32 to packed RGBA8 in the target's color space (color part of the 16B point-cloud vertex).
pub fn to_linear_rgba8(color: Color32) -> [u8; 4] {
    to_linear_rgba(color).map(|c| (c * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// sRGB byte → target-space 8-bit LUT (packs PointCloud2 RGB fields without a per-point pow).
pub fn srgb_to_linear_u8() -> &'static [u8; 256] {
    static LINEAR: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    static IDENTITY: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    if !target_srgb() {
        return IDENTITY.get_or_init(|| std::array::from_fn(|srgb| srgb as u8));
    }
    LINEAR.get_or_init(|| {
        std::array::from_fn(|srgb| {
            let linear = egui::Rgba::from(Color32::from_gray(srgb as u8)).r();
            (linear * 255.0).round().clamp(0.0, 255.0) as u8
        })
    })
}

/// Mode-dependent UI colors: panels, widgets and their text, which follow the light/dark switch.
pub mod ui {
    use super::{CornerRadius, INDENT_WIDTH, Stroke};
    use egui::Color32;

    /// The UI colors of one mode; field names mirror the roles egui's `Visuals` and our panels ask for.
    pub struct UiPalette {
        /// Backmost layer (dock active tab, dialog background, `extreme_bg_color`).
        pub bg_app: Color32,
        /// Panel / dock-area background.
        pub bg_panel: Color32,
        /// Inactive widget surface.
        pub bg_widget: Color32,
        /// Displays-panel card background (one shade off the panel so items read as cards).
        pub card_bg: Color32,
        /// Hover surface.
        pub bg_hover: Color32,
        /// Semi-transparent panel background for controls floating over the 3D viewport.
        pub overlay_bg: Color32,
        /// Borders and separators.
        pub border: Color32,
        /// Tree indent guide line (more contrast than `border`, which blends into the background).
        pub indent_guide: Color32,
        /// Normal text.
        pub text_primary: Color32,
        /// Secondary / inactive text.
        pub text_muted: Color32,
        /// Machine values shown in the status bar and timeline.
        pub instrument: Color32,
        /// Primary accent (selection, focus, links, active tab).
        pub accent: Color32,
        /// Dimmed accent (selection background, unfocused emphasis).
        pub accent_dim: Color32,
        /// Secondary accent (source-mode chip, TF tree emphasis).
        pub accent_secondary: Color32,
        /// Error display (connection/decode failure).
        pub status_error: Color32,
        /// Warning display (e.g. lost fixed frame).
        pub status_warn: Color32,
        /// Chip background in the status bar.
        pub chip_bg: Color32,
        /// Chip border in the status bar.
        pub chip_border: Color32,
    }

    /// Dark mode: the control-room palette (base deepened and borders lifted for a crisper panel step; other values original).
    static DARK: UiPalette = UiPalette {
        bg_app: Color32::from_rgb(0x06, 0x08, 0x0C),
        bg_panel: Color32::from_rgb(0x11, 0x16, 0x1D),
        bg_widget: Color32::from_rgb(0x1A, 0x21, 0x2B),
        card_bg: Color32::from_rgb(0x1A, 0x21, 0x2B),
        bg_hover: Color32::from_rgb(0x23, 0x2C, 0x38),
        overlay_bg: Color32::from_rgba_premultiplied(0x0D, 0x12, 0x18, 0xDC),
        border: Color32::from_rgb(0x2A, 0x3A, 0x4F),
        indent_guide: Color32::from_rgb(0x3C, 0x4E, 0x66),
        text_primary: Color32::from_rgb(0xD5, 0xE3, 0xEE),
        text_muted: Color32::from_rgb(0x75, 0x87, 0x9B),
        instrument: Color32::from_rgb(0x8C, 0xA0, 0xB8),
        accent: Color32::from_rgb(0x00, 0xE5, 0xFF),
        accent_dim: Color32::from_rgb(0x0A, 0x7E, 0x8C),
        accent_secondary: Color32::from_rgb(0xB3, 0x88, 0xFF),
        status_error: Color32::from_rgb(0xFF, 0x5C, 0x6E),
        status_warn: Color32::from_rgb(0xFF, 0xC4, 0x5C),
        chip_bg: Color32::from_rgba_premultiplied(0x08, 0x0C, 0x11, 0xC8),
        chip_border: Color32::from_rgba_premultiplied(0x1E, 0x2A, 0x38, 0xC8),
    };

    /// Light mode: every text color is pushed to at least the contrast its dark counterpart has, so cyan becomes a deep teal and amber a brown.
    static LIGHT: UiPalette = UiPalette {
        bg_app: Color32::from_rgb(0xE4, 0xEA, 0xF0),
        bg_panel: Color32::from_rgb(0xFF, 0xFF, 0xFF),
        bg_widget: Color32::from_rgb(0xE7, 0xEC, 0xF1),
        card_bg: Color32::from_rgb(0xF5, 0xF8, 0xFA),
        bg_hover: Color32::from_rgb(0xDA, 0xE2, 0xEA),
        overlay_bg: Color32::from_rgba_unmultiplied_const(0xFA, 0xFC, 0xFD, 0xDC),
        border: Color32::from_rgb(0xC9, 0xD4, 0xDF),
        indent_guide: Color32::from_rgb(0xA9, 0xB8, 0xC6),
        text_primary: Color32::from_rgb(0x16, 0x21, 0x2C),
        text_muted: Color32::from_rgb(0x44, 0x56, 0x6A),
        instrument: Color32::from_rgb(0x3B, 0x4C, 0x5E),
        accent: Color32::from_rgb(0x00, 0x68, 0x7A),
        accent_dim: Color32::from_rgb(0xA8, 0xE4, 0xEE),
        accent_secondary: Color32::from_rgb(0x5B, 0x32, 0xB8),
        status_error: Color32::from_rgb(0xA8, 0x1C, 0x1C),
        status_warn: Color32::from_rgb(0x7A, 0x46, 0x00),
        chip_bg: Color32::from_rgba_unmultiplied_const(0xFF, 0xFF, 0xFF, 0xC8),
        chip_border: Color32::from_rgba_unmultiplied_const(0xC9, 0xD4, 0xDF, 0xC8),
    };

    /// Current mode; an atomic rather than a lock because [`palette`] is read thousands of times per frame.
    static LIGHT_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// The palette of a given mode, without consulting the current mode (what tests should use).
    pub fn palette_of(theme: egui::Theme) -> &'static UiPalette {
        match theme {
            egui::Theme::Dark => &DARK,
            egui::Theme::Light => &LIGHT,
        }
    }

    /// The palette of the current mode.
    pub fn palette() -> &'static UiPalette {
        palette_of(mode())
    }

    /// The current mode.
    pub fn mode() -> egui::Theme {
        if LIGHT_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            egui::Theme::Light
        } else {
            egui::Theme::Dark
        }
    }

    /// Switches the palette without touching an egui context (use [`apply`] to also restyle the UI).
    pub fn set_mode(theme: egui::Theme) {
        let light = matches!(theme, egui::Theme::Light);
        LIGHT_MODE.store(light, std::sync::atomic::Ordering::Relaxed);
    }

    /// Switches the palette and writes the matching visuals into the egui context.
    pub fn apply(ctx: &egui::Context, theme: egui::Theme) {
        set_mode(theme);
        let p = palette_of(theme);
        let mut visuals = theme.default_visuals();

        visuals.panel_fill = p.bg_panel;
        visuals.window_fill = p.bg_panel;
        visuals.extreme_bg_color = p.bg_app;
        visuals.faint_bg_color = p.bg_widget;

        visuals.widgets.noninteractive.bg_fill = p.bg_panel;
        visuals.widgets.noninteractive.weak_bg_fill = p.bg_panel;
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, p.border);
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, p.text_primary);

        visuals.widgets.inactive.bg_fill = p.bg_widget;
        visuals.widgets.inactive.weak_bg_fill = p.bg_widget;
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, p.text_primary);

        visuals.widgets.hovered.bg_fill = p.bg_hover;
        visuals.widgets.hovered.weak_bg_fill = p.bg_hover;
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, p.accent_dim);
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.5, p.accent);

        visuals.widgets.active.bg_fill = p.bg_hover;
        visuals.widgets.active.weak_bg_fill = p.bg_hover;
        visuals.widgets.active.bg_stroke = Stroke::new(1.0, p.accent);
        visuals.widgets.active.fg_stroke = Stroke::new(1.5, p.accent);

        visuals.selection.bg_fill = p.accent_dim;
        visuals.selection.stroke = Stroke::new(1.0, p.accent);
        visuals.hyperlink_color = p.accent;

        visuals.window_corner_radius = CornerRadius::same(8);
        visuals.menu_corner_radius = CornerRadius::same(6);
        visuals.window_stroke = Stroke::new(1.0, p.border);
        for widget in [
            &mut visuals.widgets.inactive,
            &mut visuals.widgets.hovered,
            &mut visuals.widgets.active,
            &mut visuals.widgets.open,
        ] {
            widget.corner_radius = CornerRadius::same(4);
        }

        ctx.set_theme(egui::ThemePreference::from(theme));
        ctx.set_visuals_of(theme, visuals);
        apply_metrics(ctx);
        ctx.request_repaint();
    }

    /// Mode-independent spacing and text sizes; headings carry the display face.
    fn apply_metrics(ctx: &egui::Context) {
        ctx.all_styles_mut(|style| {
            style.spacing.indent = INDENT_WIDTH;
            style.spacing.item_spacing = egui::vec2(8.0, 6.0);
            style.spacing.button_padding = egui::vec2(8.0, 4.0);
            style.spacing.scroll.bar_width = 8.0;
            for (text_style, size) in [
                (egui::TextStyle::Heading, 15.0),
                (egui::TextStyle::Body, 12.5),
                (egui::TextStyle::Button, 12.5),
                (egui::TextStyle::Small, 10.5),
                (egui::TextStyle::Monospace, 11.5),
            ] {
                if let Some(font) = style.text_styles.get_mut(&text_style) {
                    font.size = size;
                }
            }
            if let Some(font) = style.text_styles.get_mut(&egui::TextStyle::Heading) {
                font.family = super::display_family();
            }
        });
    }
}

/// Family key of the instrument-panel display face (headings, tabs, chips); Chakra Petch with the default proportional fonts as glyph fallback.
const DISPLAY_FAMILY: &str = "display";

/// Font family for headings, dock tabs and chips.
pub fn display_family() -> egui::FontFamily {
    egui::FontFamily::Name(DISPLAY_FAMILY.into())
}

/// Text in the display face (panel titles, tags — never body copy or machine values).
pub fn display_text(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).family(display_family())
}

/// Embedded font set: IBM Plex Mono leads Monospace (machine values), Chakra Petch leads the display family; egui defaults stay behind both as glyph fallback (icons, ● / ✕).
fn font_definitions() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "ChakraPetch-Medium".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/ChakraPetch-Medium.ttf"))
            .into(),
    );
    fonts.font_data.insert(
        "IBMPlexMono-Regular".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/IBMPlexMono-Regular.ttf"))
            .into(),
    );
    if let Some(mono) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        mono.insert(0, "IBMPlexMono-Regular".to_owned());
    }
    let mut display = fonts
        .families
        .get(&egui::FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    display.insert(0, "ChakraPetch-Medium".to_owned());
    fonts.families.insert(display_family(), display);
    fonts
}

/// Installs the embedded fonts into the context (once at startup; a theme switch does not touch fonts).
pub fn install_fonts(ctx: &egui::Context) {
    ctx.set_fonts(font_definitions());
}

/// Monospace text for machine values (endpoints, ids, rates, resolutions), so columns of digits line up.
pub fn machine_value(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).monospace()
}

/// Small status LED (filled core + faint halo); painted, so it cannot fall victim to font fallback like a text ●.
pub fn status_led(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    let center = rect.center();
    ui.painter().circle_filled(center, 5.5, color.gamma_multiply(0.25));
    ui.painter().circle_filled(center, 3.0, color);
}

/// One status-bar chip holding a tag or machine value (the bordered capsule the mode chip introduced).
pub fn chip(ui: &mut egui::Ui, text: egui::RichText) -> egui::Response {
    let p = self::ui::palette();
    egui::Frame::default()
        .fill(p.chip_bg)
        .stroke(Stroke::new(1.0, p.chip_border))
        .corner_radius(3)
        .inner_margin(egui::Margin::symmetric(5, 1))
        .show(ui, |ui| ui.label(text))
        .inner
}

/// Applies the indent-guide line color to a tree Ui (call in the scope holding CollapsingHeaders).
pub fn apply_indent_guide(ui: &mut egui::Ui) {
    ui.visuals_mut().widgets.noninteractive.bg_stroke =
        Stroke::new(1.0, self::ui::palette().indent_guide);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_conversion_passes_srgb_through_for_a_non_srgb_target() {
        let color = Color32::from_rgb(0x22, 0x30, 0x41);
        let gamma = gamma_rgba(color);
        assert!((gamma[0] - 0x22 as f32 / 255.0).abs() < 1e-6);
        assert!((gamma[1] - 0x30 as f32 / 255.0).abs() < 1e-6);
        assert!((gamma[2] - 0x41 as f32 / 255.0).abs() < 1e-6);
        // Linearizing a dark color crushes it toward black, which is why it must not happen on a non-sRGB target.
        assert!(linear_rgba(color)[0] < gamma[0] * 0.25);
    }

    #[test]
    fn target_conversion_defaults_to_the_non_srgb_swapchain() {
        let color = Color32::from_rgb(0x22, 0x30, 0x41);
        assert_eq!(to_linear_rgba(color), gamma_rgba(color));
        assert_eq!(to_linear_rgba8(color), [0x22, 0x30, 0x41, 0xFF]);
        assert_eq!(srgb_to_linear_u8()[0x22], 0x22);
    }

    /// WCAG relative luminance of an opaque color.
    fn relative_luminance(color: Color32) -> f32 {
        let channel = |c: u8| {
            let c = c as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(color.r()) + 0.7152 * channel(color.g()) + 0.0722 * channel(color.b())
    }

    /// WCAG contrast ratio between two opaque colors.
    fn contrast_ratio(a: Color32, b: Color32) -> f32 {
        let (la, lb) = (relative_luminance(a), relative_luminance(b));
        let (hi, lo) = if la > lb { (la, lb) } else { (lb, la) };
        (hi + 0.05) / (lo + 0.05)
    }

    /// Every color the palette draws text with, paired with its role name for assertion messages.
    fn text_roles(p: &ui::UiPalette) -> [(&'static str, Color32); 6] {
        [
            ("text_muted", p.text_muted),
            ("instrument", p.instrument),
            ("accent", p.accent),
            ("accent_secondary", p.accent_secondary),
            ("status_error", p.status_error),
            ("status_warn", p.status_warn),
        ]
    }

    /// Every surface text is drawn on.
    fn surfaces(p: &ui::UiPalette) -> [(&'static str, Color32); 4] {
        [
            ("bg_panel", p.bg_panel),
            ("bg_widget", p.bg_widget),
            ("bg_hover", p.bg_hover),
            ("bg_app", p.bg_app),
        ]
    }

    #[test]
    fn both_ui_palettes_keep_primary_text_at_aa_on_every_surface() {
        for theme in [egui::Theme::Dark, egui::Theme::Light] {
            let p = ui::palette_of(theme);
            for (bg_name, bg) in surfaces(p) {
                let ratio = contrast_ratio(p.text_primary, bg);
                assert!(ratio >= 4.5, "{theme:?} text_primary on {bg_name}: {ratio}");
            }
        }
    }

    /// The dark palette predates the switch and its muted text bottoms out at 3.83 on bg_hover; that value is the floor, not a target.
    #[test]
    fn the_dark_palette_keeps_secondary_text_above_its_historical_floor() {
        let p = ui::palette_of(egui::Theme::Dark);
        for (role, color) in text_roles(p) {
            for (bg_name, bg) in surfaces(p) {
                let ratio = contrast_ratio(color, bg);
                assert!(ratio >= 3.8, "dark {role} on {bg_name}: {ratio}");
            }
        }
    }

    /// The light palette was tuned after the fact, so it holds full AA everywhere; keep it there rather than drifting back toward pale.
    #[test]
    fn the_light_palette_keeps_every_text_color_at_aa_on_every_surface() {
        let p = ui::palette_of(egui::Theme::Light);
        for (role, color) in text_roles(p) {
            for (bg_name, bg) in surfaces(p) {
                let ratio = contrast_ratio(color, bg);
                assert!(ratio >= 4.5, "light {role} on {bg_name}: {ratio}");
            }
        }
    }

    /// Light's accent_dim is a pale fill under dark text, so it can hold AA; dark's is a saturated fill that predates the switch and sits at 3.67.
    #[test]
    fn a_selected_row_stays_readable_in_light_mode() {
        let p = ui::palette_of(egui::Theme::Light);
        let ratio = contrast_ratio(p.text_primary, p.accent_dim);
        assert!(ratio >= 4.5, "light text_primary on accent_dim: {ratio}");
    }

    #[test]
    fn the_dark_ui_palette_still_holds_the_original_control_room_colors() {
        let p = ui::palette_of(egui::Theme::Dark);
        assert_eq!(p.bg_app, VIEWPORT_BG);
        assert_eq!(p.bg_panel, Color32::from_rgb(0x11, 0x16, 0x1D));
        assert_eq!(p.bg_widget, Color32::from_rgb(0x1A, 0x21, 0x2B));
        assert_eq!(p.card_bg, p.bg_widget);
        assert_eq!(p.text_primary, LABEL_TEXT);
        assert_eq!(p.text_muted, LABEL_LEADER);
        assert_eq!(p.accent, ACCENT_CYAN);
        assert_eq!(p.accent_secondary, ACCENT_PURPLE);
        assert_eq!(p.instrument, INSTRUMENT);
        assert_eq!(p.chip_bg, LABEL_CHIP_BG);
        assert_eq!(p.chip_border, LABEL_CHIP_BORDER);
    }

    #[test]
    fn the_ui_mode_starts_dark() {
        assert_eq!(ui::mode(), egui::Theme::Dark);
    }

    #[test]
    fn embedded_fonts_lead_their_families_with_defaults_behind_them() {
        let fonts = font_definitions();
        let mono = &fonts.families[&egui::FontFamily::Monospace];
        assert_eq!(mono[0], "IBMPlexMono-Regular");
        let display = &fonts.families[&display_family()];
        assert_eq!(display[0], "ChakraPetch-Medium");
        // The egui defaults must stay behind ours so icon glyphs (●, ✕, ▶) keep rendering.
        assert!(mono.len() > 1, "monospace lost its fallback fonts");
        assert!(display.len() > 1, "display family lost its fallback fonts");
    }
}
