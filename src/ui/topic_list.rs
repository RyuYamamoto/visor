//! Topic selector widget for the Add-display dialog (grouped, filterable, single-select).

use std::collections::{BTreeMap, HashSet};

use egui::RichText;

use crate::comm::session::TopicRow;
use crate::theme;

/// How the list is organized (toggled via the combo box at the top of the selector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupBy {
    Flat,
    #[default]
    Namespace,
    Type,
}

impl GroupBy {
    fn label(self) -> &'static str {
        match self {
            Self::Flat => "Flat",
            Self::Namespace => "Namespace",
            Self::Type => "Type",
        }
    }
}

/// Parent namespace of a topic (e.g. `/robot1/sensors/scan` → `/robot1/sensors`; root-level is `/`).
fn namespace_of(topic: &str) -> &str {
    match topic.rsplit_once('/') {
        Some(("", _)) | None => "/",
        Some((ns, _)) => ns,
    }
}

/// Last segment of a topic name (e.g. `/robot1/sensors/scan` → `scan`).
fn leaf_of(topic: &str) -> &str {
    topic
        .rsplit_once('/')
        .map(|(_, leaf)| leaf)
        .unwrap_or(topic)
}

/// Case-insensitive substring match against the topic name or its ROS type.
fn matches_filter(row: &TopicRow, filter: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    let needle = filter.to_lowercase();
    row.name.to_lowercase().contains(&needle) || row.ros_type.to_lowercase().contains(&needle)
}

/// Grouped, filterable single-select topic list; a row click writes its name into `selected` (added rows greyed/inert).
pub fn show_selector(
    ui: &mut egui::Ui,
    topics: &[TopicRow],
    group_by: &mut GroupBy,
    filter: &str,
    added: &HashSet<String>,
    selected: &mut Option<String>,
) {
    let p = theme::ui::palette();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Group by").color(p.text_muted));
        egui::ComboBox::from_id_salt("add_group_by")
            .selected_text(group_by.label())
            .show_ui(ui, |ui| {
                for mode in [GroupBy::Flat, GroupBy::Namespace, GroupBy::Type] {
                    ui.selectable_value(group_by, mode, mode.label());
                }
            });
    });
    ui.separator();
    let filtered: Vec<&TopicRow> = topics
        .iter()
        .filter(|r| matches_filter(r, filter))
        .collect();
    if filtered.is_empty() {
        ui.colored_label(p.text_muted, "no matching topics");
        return;
    }
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            theme::apply_indent_guide(ui);
            match *group_by {
                GroupBy::Flat => {
                    let rows: Vec<(&TopicRow, &str)> =
                        filtered.iter().map(|r| (*r, r.name.as_str())).collect();
                    rows_select(ui, "flat", &rows, true, added, selected);
                }
                GroupBy::Namespace => {
                    let mut groups: BTreeMap<&str, Vec<(&TopicRow, &str)>> = BTreeMap::new();
                    for row in &filtered {
                        groups
                            .entry(namespace_of(&row.name))
                            .or_default()
                            .push((row, leaf_of(&row.name)));
                    }
                    for (ns, rows) in &groups {
                        egui::CollapsingHeader::new(RichText::new(*ns).strong())
                            .id_salt(("add_ns", ns))
                            .default_open(true)
                            .show(ui, |ui| rows_select(ui, ns, rows, true, added, selected));
                    }
                }
                GroupBy::Type => {
                    let mut groups: BTreeMap<&str, Vec<(&TopicRow, &str)>> = BTreeMap::new();
                    for row in &filtered {
                        groups
                            .entry(row.ros_type.as_str())
                            .or_default()
                            .push((row, row.name.as_str()));
                    }
                    for (ros_type, rows) in &groups {
                        egui::CollapsingHeader::new(RichText::new(*ros_type).strong())
                            .id_salt(("add_type", ros_type))
                            .default_open(true)
                            .show(ui, |ui| {
                                rows_select(ui, ros_type, rows, false, added, selected)
                            });
                    }
                }
            }
        });
}

/// A grid of single-select rows; added rows are greyed and inert, `show_type` toggles the type column.
fn rows_select(
    ui: &mut egui::Ui,
    id_salt: &str,
    rows: &[(&TopicRow, &str)],
    show_type: bool,
    added: &HashSet<String>,
    selected: &mut Option<String>,
) {
    let p = theme::ui::palette();
    egui::Grid::new(("add_rows", id_salt))
        .num_columns(if show_type { 2 } else { 1 })
        .show(ui, |ui| {
            for (index, (row, label)) in rows.iter().enumerate() {
                let is_added = added.contains(&row.name);
                let is_selected = selected.as_deref() == Some(row.name.as_str());
                let background = ui.painter().add(egui::Shape::Noop);
                let text_color = if is_added {
                    p.text_muted
                } else {
                    p.text_primary
                };
                let mut cells = ui.label(RichText::new(*label).color(text_color)).rect;
                if show_type {
                    cells = cells.union(
                        ui.label(RichText::new(&row.ros_type).color(p.text_muted))
                            .rect,
                    );
                }
                ui.end_row();
                let x_range = egui::Rangef::new(
                    ui.max_rect().left().min(cells.left()),
                    ui.max_rect().right().max(cells.right()),
                );
                let row_rect = egui::Rect::from_x_y_ranges(x_range, cells.y_range());
                let sense = if is_added {
                    egui::Sense::hover()
                } else {
                    egui::Sense::click()
                };
                let response = ui.interact(row_rect, ui.id().with(&row.name), sense);
                let hovered = response.hovered();
                if is_added {
                    response.on_hover_text("already added");
                } else if response.clicked() {
                    *selected = Some(row.name.clone());
                }
                let fill = if is_selected {
                    Some(ui.visuals().selection.bg_fill)
                } else if hovered && !is_added {
                    Some(ui.visuals().widgets.hovered.bg_fill)
                } else if index % 2 == 1 {
                    Some(ui.visuals().faint_bg_color)
                } else {
                    None
                };
                if let Some(fill) = fill {
                    let padded =
                        row_rect.expand2(egui::vec2(0.0, ui.spacing().item_spacing.y / 2.0));
                    ui.painter()
                        .set(background, egui::Shape::rect_filled(padded, 0, fill));
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_and_leaf_extraction() {
        assert_eq!(namespace_of("/chatter"), "/");
        assert_eq!(leaf_of("/chatter"), "chatter");
        assert_eq!(namespace_of("/robot1/scan"), "/robot1");
        assert_eq!(leaf_of("/robot1/scan"), "scan");
        assert_eq!(namespace_of("/robot1/sensors/scan"), "/robot1/sensors");
        assert_eq!(leaf_of("/robot1/sensors/scan"), "scan");
    }

    fn row(name: &str, ros_type: &str) -> TopicRow {
        TopicRow {
            name: name.to_owned(),
            ros_type: ros_type.to_owned(),
            type_name_dds: String::new(),
            type_hash: String::new(),
            publisher_count: 1,
            subscriber_count: 0,
        }
    }

    #[test]
    fn filter_matches_name_and_type_case_insensitively() {
        let r = row("/robot1/scan", "sensor_msgs/msg/LaserScan");
        assert!(matches_filter(&r, ""));
        assert!(matches_filter(&r, "SCAN"));
        assert!(matches_filter(&r, "laser"));
        assert!(!matches_filter(&r, "cloud"));
    }
}
