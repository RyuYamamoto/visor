//! 3D display item panel (cards with visibility toggle, topic switch, removal, status, settings; Add button).

use egui::RichText;

use crate::render::{DisplayItemId, RenderStatus};
use crate::theme;

/// Per-item settings panel, implemented by both 3D renderers and the 2D image view (app.rs delegates).
pub trait ItemContentUi {
    fn settings_ui(&mut self, ui: &mut egui::Ui);
}

/// One panel interaction, wired up by app.rs (at most one per frame).
pub enum DisplayListAction {
    /// The item's ✕ was clicked.
    Remove(DisplayItemId),
    /// The bottom "Add" button was clicked (app.rs opens the Add dialog).
    OpenAddDialog,
    /// A compatible topic was chosen from the item's Topic dropdown.
    SwitchTopic { id: DisplayItemId, topic: String },
}

/// Draw view for one item (app.rs builds it from DisplayItem every frame).
pub struct DisplayRow<'a> {
    pub id: DisplayItemId,
    /// Display name in "topic (short type name)" form.
    pub title: &'a str,
    /// Whether this display type takes a topic; false on a standalone item, which shows no Topic row.
    pub takes_topic: bool,
    /// The item's current topic (selected value of the Topic dropdown); None with `takes_topic` means it was added by display type and none is assigned yet.
    pub topic: Option<&'a str>,
    /// In-graph topics this display type answers for (includes the current topic); the Topic dropdown options.
    pub compatible_topics: &'a [String],
    /// Type-identity accent color for the card stripe (resolved via theme::display_accent in app.rs).
    pub accent: egui::Color32,
    /// True when the topic is absent from the discovery snapshot (subscription is kept).
    pub offline: bool,
    pub visible: &'a mut bool,
    /// Most recent scene() or decode failure (None = drawing normally).
    pub status: Option<&'a RenderStatus>,
    pub content: &'a mut dyn ItemContentUi,
}

/// Draws the item cards plus the bottom Add button; returns the frame's single action, if any.
pub fn show(ui: &mut egui::Ui, rows: &mut [DisplayRow<'_>]) -> Option<DisplayListAction> {
    let p = theme::ui::palette();
    let mut action = None;
    egui::Panel::bottom("displays_add_bar")
        .show_separator_line(true)
        .show(ui, |ui| {
            ui.add_space(2.0);
            if ui
                .button(RichText::new("+  Add display").color(p.text_primary))
                .clicked()
            {
                action = Some(DisplayListAction::OpenAddDialog);
            }
            ui.add_space(2.0);
        });
    if rows.is_empty() {
        ui.colored_label(p.text_muted, "no display items — click Add");
        return action;
    }
    egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
        for row in rows.iter_mut() {
            if let Some(row_action) = show_card(ui, row) {
                action = Some(row_action);
            }
        }
    });
    action
}

/// Draws one item as an outlined card with a type-colored left stripe; returns the row's action if any.
fn show_card(ui: &mut egui::Ui, row: &mut DisplayRow<'_>) -> Option<DisplayListAction> {
    let p = theme::ui::palette();
    let mut action = None;
    let id = ui.make_persistent_id(("display_item", row.id.0));
    let frame = egui::Frame::default()
        .fill(p.card_bg)
        .stroke(egui::Stroke::new(1.0, p.border))
        .corner_radius(6)
        // Extra left padding keeps the header clear of the stripe.
        .inner_margin(egui::Margin {
            left: 12,
            right: 8,
            top: 6,
            bottom: 6,
        })
        .outer_margin(egui::Margin::symmetric(0, 3));
    let inner = frame.show(ui, |ui| {
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                ui.checkbox(row.visible, "");
                let title_color = if row.offline {
                    p.text_muted
                } else {
                    p.text_primary
                };
                // Right-side controls are laid out first so a long topic name elides instead of pushing them off the card.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // U+00D7, which the embedded fonts cover (U+2715 fell back to a tofu box).
                    if ui.small_button("×").clicked() {
                        action = Some(DisplayListAction::Remove(row.id));
                    }
                    if row.offline {
                        theme::chip(ui, RichText::new("offline").color(p.status_warn).small());
                    }
                    // Added by display type and still waiting for a topic, which is the one thing the card asks for.
                    if row.takes_topic && row.topic.is_none() {
                        theme::chip(ui, RichText::new("no topic").color(p.status_warn).small());
                    }
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.add(
                            egui::Label::new(theme::display_text(row.title).color(title_color))
                                .truncate(),
                        )
                        .on_hover_text(row.title);
                    });
                });
            })
            .body(|ui| {
                if row.takes_topic
                    && let Some(topic) = topic_dropdown(ui, id, row.topic, row.compatible_topics)
                {
                    action = Some(DisplayListAction::SwitchTopic { id: row.id, topic });
                }
                // A topic-less item has no subscription, so its renderer status would only ever say "waiting for data".
                match row.topic.is_none() && row.takes_topic {
                    true => {
                        ui.colored_label(p.status_warn, "select a topic to start subscribing");
                    }
                    false => status_line(ui, row.status),
                }
                row.content.settings_ui(ui);
            });
    });
    let rect = inner.response.rect;
    let stripe = egui::Rect::from_min_max(rect.min, egui::pos2(rect.left() + 4.0, rect.bottom()));
    ui.painter().rect_filled(
        stripe,
        egui::CornerRadius {
            nw: 6,
            sw: 6,
            ne: 0,
            se: 0,
        },
        row.accent,
    );
    action
}

/// Topic dropdown over compatible topics; returns the newly chosen topic, or None if unchanged. `current` None = added by display type, nothing assigned yet.
fn topic_dropdown(
    ui: &mut egui::Ui,
    id: egui::Id,
    current: Option<&str>,
    compatible: &[String],
) -> Option<String> {
    let p = theme::ui::palette();
    let mut chosen: Option<String> = None;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Topic").color(p.text_muted));
        egui::ComboBox::from_id_salt((id, "topic"))
            .selected_text(current.unwrap_or("(select)"))
            .show_ui(ui, |ui| {
                if compatible.is_empty() {
                    ui.colored_label(p.text_muted, "no matching topic in the graph");
                }
                for topic in compatible {
                    if ui
                        .selectable_label(current == Some(topic.as_str()), topic)
                        .clicked()
                    {
                        chosen = Some(topic.clone());
                    }
                }
            });
    });
    chosen.filter(|topic| current != Some(topic.as_str()))
}

/// Renders the item's current status line (nothing when drawing normally).
fn status_line(ui: &mut egui::Ui, status: Option<&RenderStatus>) {
    let p = theme::ui::palette();
    match status {
        Some(RenderStatus::NoData) => {
            ui.colored_label(p.text_muted, "waiting for data…");
        }
        Some(RenderStatus::TfUnavailable { frame }) => {
            ui.colored_label(p.status_warn, format!("TF: cannot resolve `{frame}`"));
        }
        Some(RenderStatus::InvalidMessage(error)) => {
            ui.colored_label(p.status_error, format!("message error: {error}"));
        }
        Some(RenderStatus::NoSource(message)) => {
            ui.colored_label(p.text_muted, message);
        }
        Some(RenderStatus::SourceError(error)) => {
            ui.colored_label(p.status_error, format!("source error: {error}"));
        }
        None => {}
    }
}
