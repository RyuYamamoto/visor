//! TF tree panel (frame hierarchy display, fixed-frame selection and the TF-topic picker; wiring done by app.rs).

use std::collections::{BTreeMap, HashSet};

use egui::RichText;

use crate::config::TfConfig;
use crate::tf::buffer::{FrameInfo, TfBuffer};
use crate::theme;

/// The TF-topic picker's inputs: every TFMessage topic in the graph, the user's pins, and what is subscribed right now.
pub struct TfTopicPicker<'a> {
    pub candidates: &'a [String],
    pub pins: &'a mut TfConfig,
    pub active_dynamic: Option<&'a str>,
    pub active_static: Option<&'a str>,
}

/// What the panel changed this frame.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct FramesResponse {
    /// A new fixed-frame selection, if any.
    pub fixed_frame: Option<String>,
    /// A TF-topic pin changed, so the caller resubscribes.
    pub tf_topics_changed: bool,
}

/// Tree draw depth cap (stops even if bad TF forms a cycle).
const MAX_TREE_DEPTH: usize = 64;
/// TF axis-length adjustment range [m] (UI-side constraint for the DragValue).
const TF_AXIS_LEN_RANGE: std::ops::RangeInclusive<f32> = 0.01..=2.0;
/// TF line-width adjustment range [m] (UI-side constraint for the DragValue).
const TF_LINE_WIDTH_RANGE: std::ops::RangeInclusive<f32> = 0.002..=0.1;

/// Draws the frame tree, the TF-topic picker and 3D TF settings (Names/Links, axis length/width, per-frame visibility).
#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    buffer: &TfBuffer,
    fixed_frame: Option<&str>,
    show_names: &mut bool,
    show_links: &mut bool,
    tf_axis_len: &mut f32,
    tf_line_width: &mut f32,
    hidden_frames: &mut HashSet<String>,
    picker: TfTopicPicker<'_>,
) -> FramesResponse {
    let p = theme::ui::palette();
    let mut response = FramesResponse::default();
    ui.horizontal(|ui| {
        ui.label(RichText::new("TF topics").color(p.text_muted));
        response.tf_topics_changed |= tf_topic_combo(
            ui,
            "tf_dynamic_topic",
            &mut picker.pins.dynamic_topic,
            picker.active_dynamic,
            picker.candidates,
        );
        response.tf_topics_changed |= tf_topic_combo(
            ui,
            "tf_static_topic",
            &mut picker.pins.static_topic,
            picker.active_static,
            picker.candidates,
        );
    });
    let mut selected: Option<String> = None;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Fixed frame").color(p.text_muted));
        let missing = fixed_frame.is_some_and(|f| !buffer.contains_frame(f));
        let label = match fixed_frame {
            Some(f) if missing => RichText::new(format!("{f} (missing)")).color(p.status_warn),
            Some(f) => RichText::new(f.to_owned()),
            None => RichText::new("(none)").color(p.text_muted),
        };
        egui::ComboBox::from_id_salt("fixed_frame")
            .selected_text(label)
            .show_ui(ui, |ui| {
                for name in buffer.frame_names() {
                    if ui
                        .selectable_label(fixed_frame == Some(name.as_str()), &name)
                        .clicked()
                    {
                        selected = Some(name);
                    }
                }
            });
    });
    ui.horizontal(|ui| {
        ui.checkbox(show_names, "Names");
        ui.checkbox(show_links, "Links");
        ui.label(RichText::new("Axes").color(p.text_muted));
        ui.add(
            egui::DragValue::new(tf_axis_len)
                .range(TF_AXIS_LEN_RANGE)
                .speed(0.01)
                .suffix(" m"),
        );
        ui.label(RichText::new("Width").color(p.text_muted));
        ui.add(
            egui::DragValue::new(tf_line_width)
                .range(TF_LINE_WIDTH_RANGE)
                .speed(0.001)
                .suffix(" m"),
        );
    });
    ui.separator();
    let frames = buffer.frames();
    if frames.is_empty() {
        ui.colored_label(p.text_muted, "no tf frames received");
        response.fixed_frame = selected;
        return response;
    }
    // Bulk show/hide of all frames (indeterminate state when only some are hidden).
    let names = buffer.frame_names();
    let hidden_count = names.iter().filter(|n| hidden_frames.contains(*n)).count();
    let mut all_visible = hidden_count == 0;
    let indeterminate = hidden_count > 0 && hidden_count < names.len();
    if ui
        .add(egui::Checkbox::new(&mut all_visible, "All").indeterminate(indeterminate))
        .changed()
    {
        if all_visible {
            for name in &names {
                hidden_frames.remove(name);
            }
        } else {
            hidden_frames.extend(names.iter().cloned());
        }
    }
    // frames() is name-ascending, so each parent's child list stays ascending as built.
    let mut children: BTreeMap<&str, Vec<&FrameInfo<'_>>> = BTreeMap::new();
    for frame in &frames {
        children.entry(frame.parent).or_default().push(frame);
    }
    let mut row_state = RowState {
        fixed_frame,
        hidden_frames,
        selected: &mut selected,
    };
    egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
        theme::apply_indent_guide(ui);
        for root in buffer.roots() {
            frame_node(ui, &root, None, &children, &mut row_state, 0);
        }
    });
    response.fixed_frame = selected;
    response
}

/// One TF-topic combo: `auto · <what the rule picked>` or the pinned name; the list offers `auto` plus every TFMessage topic. Returns true when the pin changed.
fn tf_topic_combo(
    ui: &mut egui::Ui,
    id: &str,
    pin: &mut Option<String>,
    active: Option<&str>,
    candidates: &[String],
) -> bool {
    let p = theme::ui::palette();
    let label = match (pin.as_deref(), active) {
        (Some(name), _) => RichText::new(name.to_owned()),
        (None, Some(name)) => RichText::new(format!("auto · {name}")).color(p.text_muted),
        (None, None) => RichText::new("auto · (none)").color(p.text_muted),
    };
    let mut changed = false;
    egui::ComboBox::from_id_salt(id)
        .selected_text(label)
        .show_ui(ui, |ui| {
            if ui.selectable_label(pin.is_none(), "auto").clicked() && pin.is_some() {
                *pin = None;
                changed = true;
            }
            for name in candidates {
                let picked = pin.as_deref() == Some(name.as_str());
                if ui.selectable_label(picked, name).clicked() && !picked {
                    *pin = Some(name.clone());
                    changed = true;
                }
            }
        });
    changed
}

/// Selection/visibility state shared by all tree rows (grouped to keep the recursion's arg list small).
struct RowState<'a> {
    fixed_frame: Option<&'a str>,
    hidden_frames: &'a mut HashSet<String>,
    selected: &'a mut Option<String>,
}

/// Recursively draws one frame node (collapsible if it has children, otherwise a single row).
fn frame_node(
    ui: &mut egui::Ui,
    name: &str,
    info: Option<&FrameInfo<'_>>,
    children: &BTreeMap<&str, Vec<&FrameInfo<'_>>>,
    state: &mut RowState<'_>,
    depth: usize,
) {
    if depth >= MAX_TREE_DEPTH {
        return;
    }
    match children.get(name) {
        Some(kids) => {
            let id = ui.make_persistent_id(("tf_frame", name));
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
                .show_header(ui, |ui| frame_row(ui, name, info, state))
                .body(|ui| {
                    for kid in kids {
                        frame_node(ui, kid.name, Some(kid), children, state, depth + 1);
                    }
                });
        }
        // A leaf is a single row without a collapse toggle (horizontal to match the header row layout).
        None => {
            ui.horizontal(|ui| frame_row(ui, name, info, state));
        }
    }
}

/// One row: visibility checkbox, frame name, static badge, age since last receipt (row click sets the fixed frame).
fn frame_row(
    ui: &mut egui::Ui,
    name: &str,
    info: Option<&FrameInfo<'_>>,
    state: &mut RowState<'_>,
) {
    let p = theme::ui::palette();
    let mut visible = !state.hidden_frames.contains(name);
    if ui.checkbox(&mut visible, "").changed() {
        if visible {
            state.hidden_frames.remove(name);
        } else {
            state.hidden_frames.insert(name.to_owned());
        }
    }
    let is_fixed = state.fixed_frame == Some(name);
    let name_color = if !visible {
        p.text_muted
    } else if is_fixed {
        p.accent
    } else {
        p.text_primary
    };
    // Reserve the draw order now; the background is filled once the row rect is known after cells are drawn.
    let background = ui.painter().add(egui::Shape::Noop);
    let mut cells = ui.label(RichText::new(name).color(name_color)).rect;
    if let Some(info) = info {
        if info.is_static {
            cells = cells.union(
                ui.label(RichText::new("static").color(p.accent_secondary).small())
                    .rect,
            );
        }
        let age = info.last_received.elapsed().as_secs_f32();
        cells = cells.union(
            ui.label(
                RichText::new(format!("{age:.1}s"))
                    .color(p.text_muted)
                    .small(),
            )
            .rect,
        );
    }
    // Keep the row's left edge at the label start (not over the collapse toggle) and extend the right edge to the panel edge.
    let x_range = egui::Rangef::new(cells.left(), ui.max_rect().right().max(cells.right()));
    let row_rect = egui::Rect::from_x_y_ranges(x_range, cells.y_range());
    let response = ui
        .interact(
            row_rect,
            ui.id().with(("tf_row", name)),
            egui::Sense::click(),
        )
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    if response.clicked() {
        *state.selected = Some(name.to_owned());
    }
    let fill = if is_fixed {
        Some(ui.visuals().selection.bg_fill)
    } else if response.hovered() {
        Some(ui.visuals().widgets.hovered.bg_fill)
    } else {
        None
    };
    if let Some(fill) = fill {
        let padded = row_rect.expand2(egui::vec2(0.0, ui.spacing().item_spacing.y / 2.0));
        ui.painter()
            .set(background, egui::Shape::rect_filled(padded, 0, fill));
    }
}
