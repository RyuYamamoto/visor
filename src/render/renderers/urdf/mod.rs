//! RobotModel display: loads a URDF file, bakes primitives on the spot, loads mesh files on a worker thread, and places every visual by TF each frame (no FK; robot_state_publisher owns that).

pub mod collada;
pub mod geometry;
pub mod loader;
pub mod model;
pub mod primitives;
pub mod resolve;
pub mod xacro;

use std::path::PathBuf;
use std::sync::Arc;

use egui::RichText;
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};

use crate::decode::value::Value;
use crate::render::{FileRequest, MeshBatch, RenderStatus, Renderer, SceneBatch, TfContext};
use crate::theme;

use loader::{MeshLoader, MeshRequest};
use model::LinkVisual;
use resolve::MeshRoots;

/// Per-visual state of the last scene() call (drives the settings_ui list; at most one frame stale).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualState {
    Ok,
    TfUnavailable,
    Unsupported,
    Loading,
    Failed,
}

impl VisualState {
    fn label(self) -> &'static str {
        match self {
            VisualState::Ok => "ok",
            VisualState::TfUnavailable => "tf",
            VisualState::Unsupported => "unsupported",
            VisualState::Loading => "loading",
            VisualState::Failed => "failed",
        }
    }

    fn color(self) -> egui::Color32 {
        let p = theme::ui::palette();
        match self {
            VisualState::Ok => p.text_primary,
            VisualState::TfUnavailable => p.status_warn,
            VisualState::Unsupported => p.text_muted,
            VisualState::Loading => p.text_muted,
            VisualState::Failed => p.status_error,
        }
    }
}

/// Where a visual's vertices are: baked immediately for primitives, or in flight on the loader worker for meshes.
enum VisualLoad {
    /// Primitive baked at load time (None = geometry this renderer cannot draw).
    Primitive(Option<MeshBatch>),
    Pending,
    Ready(MeshBatch),
    /// Resolution or reading failed; the message names the reason and the paths that were tried.
    Failed(String),
}

impl VisualLoad {
    /// Drawable batch, if this visual has one yet.
    fn batch(&self) -> Option<&MeshBatch> {
        match self {
            VisualLoad::Primitive(batch) => batch.as_ref(),
            VisualLoad::Ready(batch) => Some(batch),
            VisualLoad::Pending | VisualLoad::Failed(_) => None,
        }
    }

    /// State to show before scene() has had a chance to place the visual.
    fn initial_state(&self) -> VisualState {
        match self {
            VisualLoad::Primitive(Some(_)) | VisualLoad::Ready(_) => VisualState::TfUnavailable,
            VisualLoad::Primitive(None) => VisualState::Unsupported,
            VisualLoad::Pending => VisualState::Loading,
            VisualLoad::Failed(_) => VisualState::Failed,
        }
    }
}

/// One parsed visual with its vertices (or the reason it has none).
struct LoadedVisual {
    spec: LinkVisual,
    load: VisualLoad,
}

/// Which registry entry built this renderer. Fixed for its lifetime: the two entries share one `label` but have separate `make` functions, so construction is where the entry can say which door the user came through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    /// Added as the standalone RobotModel; the model comes from a file the user picks.
    Standalone,
    /// Added from a `robot_description` topic; the model comes from the messages.
    Topic,
}

/// Where the currently loaded model came from.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelSource {
    /// Nothing loaded (yet, or after a failed load).
    None,
    /// Read from this file, which is also the base directory for relative mesh paths.
    File(String),
    /// Received on the subscribed `robot_description` topic (no base directory exists for it).
    Topic,
}

impl ModelSource {
    /// What to name in a load error, so the message says which source failed.
    fn describe(&self) -> &str {
        match self {
            ModelSource::None => "no source",
            ModelSource::File(path) => path,
            ModelSource::Topic => "robot_description",
        }
    }
}

pub struct UrdfRenderer {
    /// Registry entry that built this renderer (decides the empty-state wording and whether the file row shows).
    origin: Origin,
    /// Path being edited in the UI; read on Load and persisted in the settings.
    path: String,
    /// xacro `name:=value` mappings as typed; parsed on Load / Reload and persisted in the settings.
    mappings: String,
    /// Source of the model in `visuals` (enables Reload and the result line).
    loaded: ModelSource,
    /// Files the xacro expansion read for the loaded model (0 when the source was plain URDF).
    expanded_files: usize,
    /// Macros the xacro defined but never called; non-empty on a description file that is meant to be included by an entry file.
    unused_macros: Vec<String>,
    /// Variables the last xacro expansion read (the rows of the Variables form; kept when the expansion failed).
    variables: Vec<xacro::Variable>,
    /// Text being edited per variable row, parallel to `variables`; committed to the mappings when the field loses focus.
    variable_edits: Vec<String>,
    /// Last XML taken from the topic: the key that skips re-ingesting an identical message, and what Reload re-applies.
    received_xml: Option<String>,
    robot_name: String,
    /// `<link>` count of the loaded model, including links without a visual.
    link_count: usize,
    visuals: Vec<LoadedVisual>,
    /// Reason the last load failed (also reported through source_error for config restore).
    load_error: Option<String>,
    /// Model replacement generation; the high 32 bits of each batch generation.
    epoch: u64,
    /// Overall opacity, applied through the batch uniform so changing it needs no re-bake.
    alpha: f32,
    states: Vec<VisualState>,
    /// Per-visual `N tris, X.X MB, Y ms` line, filled in when a mesh arrives.
    mesh_info: Vec<Option<String>>,
    /// User-editable `package://` search roots, persisted with the item.
    mesh_roots: Vec<String>,
    /// Search roots coming from the environment; listed read-only so they never look lost.
    env_roots: Vec<PathBuf>,
    /// Mesh worker, started on the first mesh a model needs and stopped when this renderer drops.
    mesh_loader: Option<MeshLoader>,
    /// Meshes still in flight for the current epoch (drives both the loading status and the repaint requests).
    pending: usize,
    /// Set by the Browse button, taken by app.rs which owns the native dialog.
    file_request: Option<FileRequest>,
}

impl Default for UrdfRenderer {
    fn default() -> Self {
        Self {
            origin: Origin::Standalone,
            path: String::new(),
            mappings: String::new(),
            loaded: ModelSource::None,
            expanded_files: 0,
            unused_macros: Vec::new(),
            variables: Vec::new(),
            variable_edits: Vec::new(),
            received_xml: None,
            robot_name: String::new(),
            link_count: 0,
            visuals: Vec::new(),
            load_error: None,
            epoch: 0,
            alpha: 1.0,
            states: Vec::new(),
            mesh_info: Vec::new(),
            mesh_roots: Vec::new(),
            env_roots: resolve::env_user_roots(|key| std::env::var(key).ok()),
            mesh_loader: None,
            pending: 0,
            file_request: None,
        }
    }
}

/// Extensions offered by the Browse dialog (`.xacro` is expanded in-app; `.xml` because some robots ship their URDF under that name).
const URDF_EXTENSIONS: &[&str] = &["urdf", "xacro", "xml"];

/// Topic-name condition of the registry's `std_msgs/msg/String` entry: the bare name, or one whose last segment is `robot_description`. A plain `ends_with` would also claim `/my_robot_description`, so the segment boundary is required.
pub fn is_robot_description_topic(topic: &str) -> bool {
    topic == "robot_description"
        || topic
            .strip_suffix("robot_description")
            .is_some_and(|head| head.ends_with('/'))
}

/// Batch generation = (model epoch, visual index), because the viewport keys GPU buffers by (item id, batch index) and only compares generations for equality: skipping a TF-less visual shifts later visuals into earlier slots, so the value must change with the slot's content.
fn generation(epoch: u64, visual_index: usize) -> u64 {
    (epoch << 32) | (visual_index as u64 & 0xFFFF_FFFF)
}

impl UrdfRenderer {
    /// Constructor for the registry's `robot_description` entry (the standalone entry uses Default).
    pub fn from_topic() -> Self {
        Self {
            origin: Origin::Topic,
            ..Default::default()
        }
    }

    /// Drop the model and advance the epoch, so results of the meshes still in flight are discarded when they land.
    fn clear_model(&mut self) {
        self.epoch += 1;
        self.loaded = ModelSource::None;
        self.expanded_files = 0;
        self.unused_macros.clear();
        self.variables.clear();
        self.variable_edits.clear();
        self.robot_name.clear();
        self.link_count = 0;
        self.visuals.clear();
        self.states.clear();
        self.mesh_info.clear();
        self.load_error = None;
        // Counting from zero again is what keeps in-flight results of older epochs from unbalancing this.
        self.pending = 0;
        if let Some(loader) = &self.mesh_loader {
            loader.set_epoch(self.epoch);
        }
    }

    /// Replace the model with the one in `text` (expanding xacro first); `source` names it in errors and decides what relative paths resolve against.
    fn load_text(&mut self, text: &str, source: ModelSource) {
        self.clear_model();
        let roots = Arc::new(MeshRoots::from_env(&self.mesh_roots, |key| {
            std::env::var(key).ok()
        }));
        let expanded: String;
        let text = if model::is_xacro(text) {
            match self.expand_xacro(text, &source, &roots) {
                Ok(result) => {
                    self.expanded_files = result.files;
                    self.unused_macros = result.unused_macros;
                    self.set_variables(result.variables);
                    expanded = result.xml;
                    expanded.as_str()
                }
                Err((message, variables)) => {
                    self.set_variables(variables);
                    self.load_error = Some(message);
                    return;
                }
            }
        } else {
            text
        };
        let parsed = match model::parse(text) {
            Ok(parsed) => parsed,
            Err(e) => {
                self.load_error = Some(format!("{}: {e}", source.describe()));
                return;
            }
        };
        self.robot_name = parsed.robot_name;
        self.link_count = parsed.link_count;
        // A model off the topic has no file behind it, so relative mesh paths have nothing to resolve against.
        let base_dir = match &source {
            ModelSource::File(path) => std::path::Path::new(path)
                .parent()
                .map(std::path::Path::to_path_buf),
            ModelSource::None | ModelSource::Topic => None,
        };
        let mut visuals = Vec::with_capacity(parsed.visuals.len());
        for (index, spec) in parsed.visuals.into_iter().enumerate() {
            let load = match &spec.shape {
                model::Shape::Mesh { uri, scale } => {
                    let request = MeshRequest {
                        epoch: self.epoch,
                        visual_index: index,
                        uri: uri.clone(),
                        urdf_dir: base_dir.clone(),
                        roots: Arc::clone(&roots),
                        scale: *scale,
                        fallback_rgba: spec.color,
                    };
                    match self.worker().request(request) {
                        Ok(()) => {
                            self.pending += 1;
                            VisualLoad::Pending
                        }
                        Err(e) => VisualLoad::Failed(e),
                    }
                }
                shape => VisualLoad::Primitive(primitives::bake(shape, spec.color)),
            };
            visuals.push(LoadedVisual { spec, load });
        }
        // Drawable visuals stay "tf" until the first scene() places them (scene is not called while the fixed frame is unset).
        self.states = visuals.iter().map(|v| v.load.initial_state()).collect();
        self.mesh_info = vec![None; visuals.len()];
        self.visuals = visuals;
        self.loaded = source;
    }

    /// Run the in-app xacro expansion on `text`; the error string already names the file and line when the failure has one.
    fn expand_xacro(
        &self,
        text: &str,
        source: &ModelSource,
        roots: &MeshRoots,
    ) -> Result<xacro::Expanded, (String, Vec<xacro::Variable>)> {
        let mappings = xacro::Mappings::parse(&self.mappings)
            .map_err(|e| (format!("mappings: {e}"), Vec::new()))?;
        let file = match source {
            ModelSource::File(path) => Some(std::path::Path::new(path)),
            ModelSource::None | ModelSource::Topic => None,
        };
        let env = |key: &str| std::env::var(key).ok();
        xacro::expand(
            text,
            &xacro::XacroContext {
                file,
                roots,
                mappings: &mappings,
                env: &env,
            },
        )
        .map_err(|e| {
            let message = match &e.location {
                Some(at) if at.file.is_some() => e.to_string(),
                _ => format!("{}: {e}", source.describe()),
            };
            (message, e.variables)
        })
    }

    /// Show the variables the last expansion read (kept on failure too, so a bad mapping can be fixed from the form).
    fn set_variables(&mut self, variables: Vec<xacro::Variable>) {
        self.variable_edits = variables.iter().map(|v| v.value.clone()).collect();
        self.variables = variables;
    }

    /// Write `name:=value` into the mappings and rebuild (what the variables form does on a change).
    fn set_variable(&mut self, name: &str, value: &str) {
        let mut mappings = xacro::Mappings::parse(&self.mappings).unwrap_or_default();
        mappings.set(name, value);
        self.mappings = mappings.to_text();
        self.reload();
    }

    /// Drop `name` from the mappings so the environment or the file's default answers again, and rebuild.
    fn clear_variable(&mut self, name: &str) {
        let mut mappings = xacro::Mappings::parse(&self.mappings).unwrap_or_default();
        mappings.remove(name);
        self.mappings = mappings.to_text();
        self.reload();
    }

    /// One row per variable the xacro read: a dropdown when the variable picks an include file (the candidates are the files that exist), a text field otherwise; a change lands in Mappings and reloads.
    fn variables_ui(&mut self, ui: &mut egui::Ui) {
        if self.variables.is_empty() {
            return;
        }
        let p = theme::ui::palette();
        let mut change: Option<(String, Option<String>)> = None;
        egui::CollapsingHeader::new("Variables")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("xacro_variables")
                    .num_columns(3)
                    .show(ui, |ui| {
                        for (index, variable) in self.variables.iter().enumerate() {
                            let kind = match variable.kind {
                                xacro::VariableKind::Env => "$(env) / $(optenv)",
                                xacro::VariableKind::Arg => "$(arg)",
                            };
                            ui.label(RichText::new(&variable.name).color(p.text_muted))
                                .on_hover_text(format!(
                                    "read through {kind}; default in the file: {}",
                                    variable.default.as_deref().unwrap_or("(none)")
                                ));
                            if variable.candidates.is_empty() {
                                let edit = &mut self.variable_edits[index];
                                let response =
                                    ui.add(egui::TextEdit::singleline(edit).desired_width(160.0));
                                if response.lost_focus() && *edit != variable.value {
                                    change = Some((variable.name.clone(), Some(edit.clone())));
                                }
                            } else {
                                let mut selected = variable.value.clone();
                                egui::ComboBox::from_id_salt(("xacro_variable", &variable.name))
                                    .selected_text(&selected)
                                    .show_ui(ui, |ui| {
                                        if !variable.candidates.contains(&variable.value) {
                                            ui.selectable_value(
                                                &mut selected,
                                                variable.value.clone(),
                                                &variable.value,
                                            );
                                        }
                                        for candidate in &variable.candidates {
                                            ui.selectable_value(
                                                &mut selected,
                                                candidate.clone(),
                                                candidate,
                                            );
                                        }
                                    });
                                if selected != variable.value {
                                    change = Some((variable.name.clone(), Some(selected)));
                                }
                            }
                            ui.horizontal(|ui| {
                                let (label, color) = match variable.source {
                                    xacro::VariableSource::Mapping => ("mapping", p.accent),
                                    xacro::VariableSource::Environment => ("env", p.text_muted),
                                    xacro::VariableSource::Default => ("default", p.text_muted),
                                    xacro::VariableSource::Undefined => ("unset", p.status_error),
                                };
                                ui.colored_label(color, label);
                                if variable.source == xacro::VariableSource::Mapping
                                    && ui
                                        .small_button("×")
                                        .on_hover_text("Forget this mapping (back to the environment or the file's default)")
                                        .clicked()
                                {
                                    change = Some((variable.name.clone(), None));
                                }
                            });
                            ui.end_row();
                        }
                    });
                ui.colored_label(
                    p.text_muted,
                    "what the xacro reads through $(env) / $(optenv) / $(arg); a change is written to Mappings and reloads",
                );
            });
        match change {
            Some((name, Some(value))) => self.set_variable(&name, &value),
            Some((name, None)) => self.clear_variable(&name),
            None => {}
        }
    }

    /// Read the URDF at `path`, bake its primitives and queue its meshes; on failure the model is left empty and the reason kept for display.
    fn load(&mut self) {
        let path = self.path.trim().to_owned();
        if path.is_empty() {
            self.clear_model();
            return;
        }
        // Read here rather than urdf_rs::read_file so the error message can name the path.
        match std::fs::read_to_string(&path) {
            Ok(text) => self.load_text(&text, ModelSource::File(path)),
            Err(e) => {
                self.clear_model();
                self.load_error = Some(format!("{path}: {e}"));
            }
        }
    }

    /// Rebuild the current model from scratch (the way to apply edited mesh roots). The topic cannot be asked again, so its last XML is re-parsed instead.
    fn reload(&mut self) {
        match (&self.loaded, self.received_xml.clone()) {
            (ModelSource::Topic, Some(xml)) => self.load_text(&xml, ModelSource::Topic),
            _ => self.load(),
        }
    }

    /// Report a message that is not a `std_msgs/String`. Idempotent, so a stuck publisher does not bump the epoch every message.
    fn report_bad_message(&mut self) {
        const REASON: &str =
            "robot_description: expected a std_msgs/msg/String with a `data` field";
        if self.load_error.as_deref() == Some(REASON) {
            return;
        }
        self.clear_model();
        self.load_error = Some(REASON.to_owned());
    }

    /// The mesh worker, started on first use so a primitive-only model never spawns a thread.
    fn worker(&mut self) -> &MeshLoader {
        let epoch = self.epoch;
        let loader = self.mesh_loader.get_or_insert_with(MeshLoader::spawn);
        loader.set_epoch(epoch);
        loader
    }

    /// Summary line for a mesh that arrived, e.g. `116261 tris, 9.8 MB, 412 ms`.
    fn mesh_summary(mesh: &loader::LoadedMesh) -> String {
        let megabytes = mesh.batch.bytes.len() as f32 / (1024.0 * 1024.0);
        let mut line = format!(
            "{} tris, {megabytes:.1} MB, {} ms",
            mesh.triangles, mesh.load_ms
        );
        if mesh.cached {
            line.push_str(" (cached)");
        }
        if mesh.skipped > 0 {
            line.push_str(&format!(", {} degenerate skipped", mesh.skipped));
        }
        line
    }

    /// Ask app.rs to open the native file dialog, starting in the current path's directory when there is one.
    fn request_browse(&mut self) {
        let start_dir = std::path::Path::new(self.path.trim())
            .parent()
            .filter(|dir| dir.is_dir())
            .map(std::path::Path::to_path_buf);
        self.file_request = Some(FileRequest {
            filter_name: "URDF",
            extensions: URDF_EXTENSIONS,
            start_dir,
        });
    }

    /// `package://` search roots, editable here and applied on the next Load or Reload.
    fn mesh_roots_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        egui::CollapsingHeader::new("Mesh roots").show(ui, |ui| {
            let mut remove = None;
            for (index, root) in self.mesh_roots.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    if ui.button("−").on_hover_text("Remove this root").clicked() {
                        remove = Some(index);
                    }
                    ui.add(
                        egui::TextEdit::singleline(root)
                            .hint_text("/path/to/workspace/src")
                            .desired_width(f32::INFINITY),
                    );
                });
            }
            if let Some(index) = remove {
                self.mesh_roots.remove(index);
            }
            if ui.button("Add root").clicked() {
                self.mesh_roots.push(String::new());
            }
            // Environment roots are shown rather than merged into the list, so removing a row here cannot look undone.
            for root in &self.env_roots {
                ui.colored_label(p.text_muted, format!("{} (env)", root.display()));
            }
            ui.colored_label(
                p.text_muted,
                format!(
                    "where package:// meshes and xacro $(find) look, before AMENT_PREFIX_PATH, ROS_PACKAGE_PATH and the URDF's own directory tree; press Reload to apply ({} adds more)",
                    resolve::MESH_ROOTS_ENV
                ),
            );
        });
    }

    /// Header line naming where the model is coming from, so a file item and a topic item never look alike.
    fn source_line(&self) -> String {
        match (&self.loaded, self.origin) {
            (ModelSource::File(path), _) if self.expanded_files > 0 => {
                format!("Source: file {path} (xacro, {} files)", self.expanded_files)
            }
            (ModelSource::File(path), _) => format!("Source: file {path}"),
            (ModelSource::Topic, _) => "Source: robot_description topic".to_owned(),
            (ModelSource::None, Origin::Topic) => {
                "Source: robot_description topic (waiting)".to_owned()
            }
            (ModelSource::None, Origin::Standalone) => "Source: none".to_owned(),
        }
    }

    /// Count line for the settings panel, e.g. `12 visuals: 10 ok, 1 tf, 1 unsupported`.
    fn state_summary(&self) -> String {
        let count = |state: VisualState| self.states.iter().filter(|s| **s == state).count();
        let parts: Vec<String> = [
            VisualState::Ok,
            VisualState::TfUnavailable,
            VisualState::Loading,
            VisualState::Failed,
            VisualState::Unsupported,
        ]
        .into_iter()
        .filter_map(|state| {
            let n = count(state);
            (n > 0).then(|| format!("{n} {}", state.label()))
        })
        .collect();
        format!("{} visuals: {}", self.states.len(), parts.join(", "))
    }
}

impl Renderer for UrdfRenderer {
    /// Take the URDF out of a `robot_description` message (never called on a standalone item, which subscribes to nothing).
    fn on_message(&mut self, value: &Value) {
        let Some(Value::String(xml)) = value.get("data") else {
            self.report_bad_message();
            return;
        };
        // The topic is latched, so the same XML comes back on every resubscribe (visibility toggle, publisher restart, config restore). Rebuilding here would re-read and re-upload every mesh for no new information.
        if self.received_xml.as_deref() == Some(xml.as_str()) {
            return;
        }
        // Stored before parsing so a broken URDF is not re-parsed on every redelivery either.
        self.received_xml = Some(xml.clone());
        self.load_text(xml, ModelSource::Topic);
    }

    fn poll(&mut self) -> bool {
        let mut responses = Vec::new();
        if let Some(loader) = &self.mesh_loader {
            while let Some(response) = loader.try_recv() {
                responses.push(response);
            }
        }
        for response in responses {
            // Results of a replaced model are dropped: their visual index means nothing now.
            if response.epoch != self.epoch {
                continue;
            }
            let Some(visual) = self.visuals.get_mut(response.visual_index) else {
                continue;
            };
            if !matches!(visual.load, VisualLoad::Pending) {
                continue;
            }
            self.pending = self.pending.saturating_sub(1);
            visual.load = match response.result {
                Ok(mesh) => {
                    self.mesh_info[response.visual_index] = Some(Self::mesh_summary(&mesh));
                    VisualLoad::Ready(mesh.batch)
                }
                Err(reason) => VisualLoad::Failed(reason),
            };
            self.states[response.visual_index] = visual.load.initial_state();
        }
        self.pending > 0
    }

    fn scene(&mut self, tf: &TfContext<'_>) -> Result<Vec<SceneBatch>, RenderStatus> {
        if let Some(error) = &self.load_error {
            return Err(RenderStatus::SourceError(error.clone()));
        }
        if self.visuals.is_empty() {
            return Err(match (&self.loaded, self.link_count) {
                // A subscribed item with nothing received yet is NoData, the same status it carries from the moment it is added.
                (ModelSource::None, _) if self.origin == Origin::Topic => RenderStatus::NoData,
                (ModelSource::None, _) => {
                    RenderStatus::NoSource("no URDF loaded — pick a file with Browse…".to_owned())
                }
                (_, 0) if !self.unused_macros.is_empty() => RenderStatus::NoSource(format!(
                    "<robot> has no <link> elements: this xacro only defines the macro(s) {} and never calls them, so it is a part meant to be included — load the entry file that has <xacro:{}/> (and pick the model with Mappings, e.g. ROBOT_MODEL:=…, when the entry selects it by environment)",
                    self.unused_macros
                        .iter()
                        .map(|m| format!("`{m}`"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    self.unused_macros[0]
                )),
                (_, 0) => RenderStatus::NoSource("<robot> has no <link> elements".to_owned()),
                (_, links) => {
                    RenderStatus::NoSource(format!("no <visual> in any of the {links} link(s)"))
                }
            });
        }
        let mut batches = Vec::with_capacity(self.visuals.len());
        let mut states = Vec::with_capacity(self.visuals.len());
        let mut drawable = 0;
        let mut first_missing: Option<&str> = None;
        // Visuals of one link are adjacent (link order -> visual order), so a single memo replaces a map.
        let mut cached: Option<(&str, Option<Isometry3<f64>>)> = None;
        for (index, visual) in self.visuals.iter().enumerate() {
            let Some(batch) = visual.load.batch() else {
                states.push(visual.load.initial_state());
                continue;
            };
            drawable += 1;
            let link = visual.spec.link.as_str();
            let resolved = match cached {
                Some((cached_link, iso)) if cached_link == link => iso,
                _ => {
                    let iso = tf.resolve(link);
                    cached = Some((link, iso));
                    iso
                }
            };
            match resolved {
                Some(fixed_from_link) => {
                    let pose = (fixed_from_link * visual.spec.origin).cast::<f32>();
                    batches.push(SceneBatch::posed_mesh(
                        batch.clone(),
                        generation(self.epoch, index),
                        &pose,
                        self.alpha,
                    ));
                    states.push(VisualState::Ok);
                }
                None => {
                    first_missing = first_missing.or(Some(link));
                    states.push(VisualState::TfUnavailable);
                }
            }
        }
        let missing = first_missing.map(str::to_owned);
        self.states = states;
        if drawable == 0 {
            return Err(RenderStatus::NoSource(match self.pending {
                0 => format!("{} visual(s), none drawable", self.visuals.len()),
                pending => format!("loading {pending} mesh(es)…"),
            }));
        }
        // One drawable visual is enough to call it Ok (same convention as the marker renderer).
        match (batches.is_empty(), missing) {
            (true, Some(frame)) => Err(RenderStatus::TfUnavailable { frame }),
            _ => Ok(batches),
        }
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let p = theme::ui::palette();
        ui.colored_label(p.text_muted, self.source_line());
        let from_file = self.origin == Origin::Standalone;
        if from_file {
            ui.horizontal(|ui| {
                ui.label(RichText::new("URDF").color(p.text_muted));
                ui.add(
                    egui::TextEdit::singleline(&mut self.path)
                        .hint_text("/path/to/robot.urdf or .urdf.xacro"),
                );
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("Mappings").color(p.text_muted));
                ui.add(
                    egui::TextEdit::singleline(&mut self.mappings)
                        .hint_text("ROBOT_MODEL:=model_a name:=value"),
                );
            })
            .response
            .on_hover_text(
                "xacro name:=value pairs: they define $(arg name) and override the environment for $(env) / $(optenv); applied on Load / Reload",
            );
            self.variables_ui(ui);
        }
        ui.horizontal(|ui| {
            if from_file {
                if ui.button("Browse…").clicked() {
                    self.request_browse();
                }
                if ui.button("Load").clicked() {
                    self.load();
                }
            }
            if ui
                .add_enabled(
                    self.loaded != ModelSource::None,
                    egui::Button::new("Reload"),
                )
                .clicked()
            {
                self.reload();
            }
        });
        // Alpha gets its own row: sharing one with the buttons squeezes the slider to a few pixels at the panel's default width.
        ui.horizontal(|ui| {
            ui.label(RichText::new("Alpha").color(p.text_muted));
            ui.add(egui::Slider::new(&mut self.alpha, 0.0..=1.0));
        });
        self.mesh_roots_ui(ui);
        match (&self.load_error, &self.loaded) {
            (Some(error), _) => {
                ui.colored_label(p.status_error, error);
            }
            (None, ModelSource::File(_) | ModelSource::Topic) => {
                ui.colored_label(
                    p.text_muted,
                    format!(
                        "{}: {} links, {} visuals",
                        self.robot_name,
                        self.link_count,
                        self.visuals.len()
                    ),
                );
                ui.colored_label(p.text_muted, self.state_summary());
                egui::CollapsingHeader::new("Visuals").show(ui, |ui| {
                    for (index, (visual, state)) in
                        self.visuals.iter().zip(&self.states).enumerate()
                    {
                        let detail = match (&visual.load, &visual.spec.shape) {
                            (VisualLoad::Failed(reason), _) => reason.clone(),
                            (_, model::Shape::Unsupported { reason, .. }) => reason.clone(),
                            _ => match self.mesh_info.get(index).and_then(Option::as_ref) {
                                Some(info) => format!("{} — {info}", state.label()),
                                None => state.label().to_owned(),
                            },
                        };
                        ui.colored_label(
                            state.color(),
                            format!(
                                "{} / {} — {} — {detail}",
                                visual.spec.link,
                                visual.spec.name,
                                visual.spec.shape.kind()
                            ),
                        );
                    }
                });
            }
            (None, ModelSource::None) => {}
        }
    }

    fn settings(&self) -> Option<toml::Value> {
        toml::Value::try_from(UrdfSettings {
            path: self.path.clone(),
            alpha: self.alpha,
            mesh_roots: self.mesh_roots.clone(),
            mappings: self.mappings.clone(),
        })
        .ok()
    }

    fn apply_settings(&mut self, value: &toml::Value) {
        if let Ok(s) = value.clone().try_into::<UrdfSettings>() {
            self.path = s.path;
            self.alpha = s.alpha;
            self.mesh_roots = s.mesh_roots;
            self.mappings = s.mappings;
            if !self.path.trim().is_empty() {
                self.load();
            }
        }
    }

    fn source_error(&self) -> Option<String> {
        self.load_error.clone()
    }

    fn take_file_request(&mut self) -> Option<FileRequest> {
        self.file_request.take()
    }

    /// Picking a file means "use this one", so it replaces the path and loads immediately.
    fn on_file_picked(&mut self, path: &std::path::Path) {
        self.path = path.display().to_string();
        self.load();
    }
}

/// Persistence DTO for UrdfRenderer's user-editable settings (the URDF path lives here, not in the config schema).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct UrdfSettings {
    /// Saved verbatim as typed; relative paths are left to the OS (no config-relative resolution).
    path: String,
    alpha: f32,
    /// `package://` search roots; absent in configs written before mesh loading existed.
    mesh_roots: Vec<String>,
    /// xacro `name:=value` mappings as typed; absent in configs written before xacro expansion existed.
    mappings: String,
}

impl Default for UrdfSettings {
    fn default() -> Self {
        Self {
            path: String::new(),
            alpha: 1.0,
            mesh_roots: Vec::new(),
            mappings: String::new(),
        }
    }
}

/// Absolute path of a bundled sample URDF (tests read the real file so the I/O path is covered too).
#[cfg(test)]
fn sample_urdf_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/urdf")
        .join(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::BatchData;
    use crate::tf::buffer::{TfBuffer, TfTransform, tf_update};
    use nalgebra::{Translation3, UnitQuaternion};

    /// TF buffer where each named frame is a child of `map`, offset along X by its index.
    fn tf_with(frames: &[&str]) -> TfBuffer {
        let mut buffer = TfBuffer::new();
        buffer.insert(&tf_update(
            frames
                .iter()
                .enumerate()
                .map(|(i, frame)| TfTransform {
                    parent: "map".to_owned(),
                    child: (*frame).to_owned(),
                    stamp: 1_000,
                    transform: Isometry3::from_parts(
                        Translation3::new(i as f64, 0.0, 0.0),
                        UnitQuaternion::identity(),
                    ),
                })
                .collect(),
            true,
        ));
        buffer
    }

    const SAMPLE_FRAMES: [&str; 4] = ["base_link", "laser_link", "arm_link", "hand_link"];

    /// Load a bundled sample by name (exercises the real file I/O path).
    fn loaded(name: &str) -> UrdfRenderer {
        let mut renderer = UrdfRenderer {
            path: sample_urdf_path(name).display().to_string(),
            ..Default::default()
        };
        renderer.load();
        assert_eq!(renderer.load_error, None);
        renderer
    }

    /// Text of a bundled sample, for driving the topic route with content that also has a file test.
    fn sample_text(name: &str) -> String {
        std::fs::read_to_string(sample_urdf_path(name)).expect("read sample urdf")
    }

    /// A decoded std_msgs/msg/String carrying `xml`, as on_message receives it.
    fn urdf_message(xml: &str) -> Value {
        Value::Struct(vec![("data".to_owned(), Value::String(xml.to_owned()))])
    }

    /// Write URDF text to a temp file so tests can drive load() with arbitrary content.
    fn temp_urdf(name: &str, text: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("visor_urdf_{}_{name}", std::process::id()));
        std::fs::write(&path, text).expect("write temp urdf");
        path
    }

    /// Drive poll() until every queued mesh has come back (or the worker is clearly stuck).
    fn poll_until_idle(renderer: &mut UrdfRenderer) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while renderer.poll() {
            assert!(
                std::time::Instant::now() < deadline,
                "mesh loading never finished"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    fn scene_of(renderer: &mut UrdfRenderer, frames: &[&str]) -> Vec<SceneBatch> {
        let buffer = tf_with(frames);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        renderer.scene(&tf).expect("expected batches")
    }

    #[test]
    fn fresh_renderer_reports_no_source() {
        let mut renderer = UrdfRenderer::default();
        let buffer = TfBuffer::new();
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let Err(RenderStatus::NoSource(message)) = renderer.scene(&tf) else {
            panic!("expected NoSource");
        };
        assert!(message.contains("Browse"), "message={message}");
        assert_eq!(renderer.source_error(), None);
    }

    #[test]
    fn loading_the_bundled_sample_produces_one_batch_per_drawable_visual() {
        let mut renderer = loaded("sample_robot.urdf");
        assert_eq!(renderer.robot_name, "visor_sample");
        // 5 links (one collision-only, so no visual) and 4 drawable visuals.
        assert_eq!(renderer.link_count, 5);
        assert_eq!(renderer.visuals.len(), 4);
        let batches = scene_of(&mut renderer, &SAMPLE_FRAMES);
        assert_eq!(batches.len(), 4);
        assert!(
            batches
                .iter()
                .all(|b| matches!(b.data, BatchData::PosedMesh(_)))
        );
        assert!(renderer.states.iter().all(|s| *s == VisualState::Ok));
        // The pose is fixed_from_link * visual origin: arm_link is at x = 2 and its visual origin adds z = 0.15.
        let arm = &batches[2];
        assert!((arm.model[(0, 3)] - 2.0).abs() < 1e-6);
        assert!((arm.model[(2, 3)] - 0.15).abs() < 1e-6);
    }

    #[test]
    fn unresolved_links_are_skipped_and_the_rest_draw() {
        let mut renderer = loaded("sample_robot.urdf");
        let batches = scene_of(&mut renderer, &["base_link", "arm_link", "hand_link"]);
        assert_eq!(batches.len(), 3);
        assert_eq!(
            renderer.states,
            vec![
                VisualState::Ok,
                VisualState::TfUnavailable,
                VisualState::Ok,
                VisualState::Ok
            ]
        );
        assert!(renderer.state_summary().contains("3 ok, 1 tf"));
    }

    #[test]
    fn all_links_unresolved_reports_tf_unavailable() {
        let mut renderer = loaded("sample_robot.urdf");
        let buffer = tf_with(&["unrelated"]);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        assert_eq!(
            renderer.scene(&tf).unwrap_err(),
            RenderStatus::TfUnavailable {
                frame: "base_link".to_owned()
            }
        );
        assert!(
            renderer
                .states
                .iter()
                .all(|s| *s == VisualState::TfUnavailable)
        );
    }

    #[test]
    fn slot_reuse_changes_generation_when_a_visual_is_skipped() {
        let mut renderer = loaded("sample_robot.urdf");
        let all = scene_of(&mut renderer, &SAMPLE_FRAMES);
        let skipped = scene_of(&mut renderer, &["base_link", "arm_link", "hand_link"]);
        // Slot 1 now carries a different visual, so its generation must differ or the GPU would reuse stale vertices.
        assert_ne!(all[1].generation, skipped[1].generation);
        assert_ne!(all[2].generation, skipped[2].generation);
        assert_eq!(all[0].generation, skipped[0].generation);
        assert_eq!(skipped[1].generation, generation(renderer.epoch, 2));
    }

    #[test]
    fn primitives_draw_while_meshes_load_and_a_missing_package_is_reported() {
        let mut renderer = UrdfRenderer {
            path: sample_urdf_path("sample_mesh.urdf").display().to_string(),
            mesh_roots: vec!["/nonexistent/workspace".to_owned()],
            ..Default::default()
        };
        renderer.load();
        assert_eq!(renderer.load_error, None);
        assert_eq!(renderer.visuals.len(), 3);
        assert_eq!(renderer.pending, 2);
        // The box is drawn immediately; the two meshes are still on the worker.
        let early = scene_of(&mut renderer, &["base_link"]);
        assert_eq!(early.len(), 1);
        assert_eq!(
            renderer.states,
            vec![VisualState::Ok, VisualState::Loading, VisualState::Loading]
        );
        poll_until_idle(&mut renderer);
        let batches = scene_of(&mut renderer, &["base_link"]);
        assert_eq!(batches.len(), 2);
        assert_eq!(
            renderer.states,
            vec![VisualState::Ok, VisualState::Ok, VisualState::Failed]
        );
        assert!(renderer.state_summary().contains("2 ok, 1 failed"));
        // The failure names the package and every path that was tried.
        let VisualLoad::Failed(reason) = &renderer.visuals[2].load else {
            panic!("expected a failed visual");
        };
        assert!(reason.contains("visor_nowhere"), "reason={reason}");
        // Built with the same joins as resolve_package so the separators match on every OS.
        let tried = std::path::Path::new("/nonexistent/workspace")
            .join("visor_nowhere")
            .join("meshes/absent.stl");
        assert!(
            reason.contains(&tried.display().to_string()),
            "reason={reason}"
        );
        // The mesh that loaded reports its size in the visuals list.
        let info = renderer.mesh_info[1].as_ref().expect("mesh info");
        assert!(info.starts_with("2 tris"), "info={info}");
    }

    #[test]
    fn a_mesh_only_urdf_reports_loading_and_then_the_failure() {
        let path = temp_urdf(
            "mesh_only.urdf",
            r#"<robot name="mesh_only">
                 <link name="base_link">
                   <visual><geometry><mesh filename="package://x/y.stl"/></geometry></visual>
                 </link>
               </robot>"#,
        );
        let mut renderer = UrdfRenderer {
            path: path.display().to_string(),
            ..Default::default()
        };
        renderer.load();
        let buffer = tf_with(&["base_link"]);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let Err(RenderStatus::NoSource(message)) = renderer.scene(&tf) else {
            panic!("expected NoSource");
        };
        assert!(message.contains("loading 1 mesh"), "message={message}");
        poll_until_idle(&mut renderer);
        let Err(RenderStatus::NoSource(message)) = renderer.scene(&tf) else {
            panic!("expected NoSource");
        };
        assert!(message.contains("none drawable"), "message={message}");
    }

    #[test]
    fn a_mesh_arriving_changes_the_generation_of_the_slots_it_shifts() {
        let mesh = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/meshes/shell.stl");
        let path = temp_urdf(
            "mesh_first.urdf",
            &format!(
                r#"<robot name="mesh_first">
                     <link name="base_link">
                       <visual><geometry><mesh filename="{}"/></geometry></visual>
                       <visual><geometry><box size="1 1 1"/></geometry></visual>
                     </link>
                   </robot>"#,
                mesh.display()
            ),
        );
        let mut renderer = UrdfRenderer {
            path: path.display().to_string(),
            ..Default::default()
        };
        renderer.load();
        // While the mesh is pending the box occupies slot 0.
        let early = scene_of(&mut renderer, &["base_link"]);
        assert_eq!(early.len(), 1);
        assert_eq!(early[0].generation, generation(renderer.epoch, 1));
        poll_until_idle(&mut renderer);
        let late = scene_of(&mut renderer, &["base_link"]);
        assert_eq!(late.len(), 2);
        // Slot 0 now holds different vertices, so its generation must differ or the GPU would keep the old ones.
        assert_ne!(early[0].generation, late[0].generation);
        assert_eq!(late[0].generation, generation(renderer.epoch, 0));
        assert_eq!(late[1].generation, generation(renderer.epoch, 1));
    }

    #[test]
    fn reload_restarts_mesh_loading_and_ignores_the_previous_results() {
        let mut renderer = loaded("sample_mesh.urdf");
        poll_until_idle(&mut renderer);
        let before = scene_of(&mut renderer, &["base_link"]);
        let epoch = renderer.epoch;
        renderer.load();
        // Nothing of the old model survives the reload; the meshes are queued again under the new epoch.
        assert_eq!(renderer.epoch, epoch + 1);
        assert_eq!(renderer.pending, 2);
        assert!(renderer.mesh_info.iter().all(Option::is_none));
        poll_until_idle(&mut renderer);
        let after = scene_of(&mut renderer, &["base_link"]);
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert_ne!(a.generation, b.generation);
        }
        // The reloaded meshes come from the worker's cache, so the second load does no disk I/O.
        assert!(
            renderer.mesh_info[1]
                .as_ref()
                .is_some_and(|info| info.contains("cached")),
            "{:?}",
            renderer.mesh_info[1]
        );
    }

    #[test]
    fn poll_stays_quiet_when_there_is_nothing_to_load() {
        let mut fresh = UrdfRenderer::default();
        assert!(!fresh.poll());
        // A primitive-only model never starts the worker, so idle frames request no repaints.
        let mut primitives = loaded("sample_robot.urdf");
        assert!(!primitives.poll());
        assert!(primitives.mesh_loader.is_none());
        let mut meshes = loaded("sample_mesh.urdf");
        assert!(meshes.mesh_loader.is_some());
        poll_until_idle(&mut meshes);
        assert!(!meshes.poll());
    }

    #[test]
    fn link_less_urdf_reports_no_link_elements() {
        let path = temp_urdf("no_link.urdf", "<robot name=\"empty\"/>");
        let mut renderer = UrdfRenderer {
            path: path.display().to_string(),
            ..Default::default()
        };
        renderer.load();
        let buffer = tf_with(&["base_link"]);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let Err(RenderStatus::NoSource(message)) = renderer.scene(&tf) else {
            panic!("expected NoSource");
        };
        assert!(message.contains("<link>"), "message={message}");
        // A description file that only defines macros (e.g. a `*_description.urdf.xacro`) gets told which entry file to load instead.
        let mut defs_only = UrdfRenderer {
            path: xacro_fixture_path("demo_model.xacro").display().to_string(),
            ..Default::default()
        };
        defs_only.load();
        assert_eq!(defs_only.load_error, None);
        let Err(RenderStatus::NoSource(message)) = defs_only.scene(&tf) else {
            panic!("expected NoSource");
        };
        assert!(
            message.contains("only defines the macro(s) `robot`"),
            "message={message}"
        );
        assert!(message.contains("<xacro:robot/>"), "message={message}");
    }

    #[test]
    fn visual_less_urdf_reports_no_source() {
        let path = temp_urdf(
            "no_visual.urdf",
            "<robot name=\"bare\"><link name=\"base_link\"/></robot>",
        );
        let mut renderer = UrdfRenderer {
            path: path.display().to_string(),
            ..Default::default()
        };
        renderer.load();
        let buffer = tf_with(&["base_link"]);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let Err(RenderStatus::NoSource(message)) = renderer.scene(&tf) else {
            panic!("expected NoSource");
        };
        assert!(message.contains("<visual>"), "message={message}");
    }

    #[test]
    fn alpha_change_keeps_generation_and_applies_to_batches() {
        let mut renderer = loaded("sample_robot.urdf");
        let before = scene_of(&mut renderer, &SAMPLE_FRAMES);
        assert!(before.iter().all(|b| b.alpha == 1.0));
        renderer.alpha = 0.4;
        let after = scene_of(&mut renderer, &SAMPLE_FRAMES);
        assert!(after.iter().all(|b| b.alpha == 0.4));
        let gens_before: Vec<u64> = before.iter().map(|b| b.generation).collect();
        let gens_after: Vec<u64> = after.iter().map(|b| b.generation).collect();
        assert_eq!(gens_before, gens_after);
    }

    #[test]
    fn reload_after_a_bad_path_recovers() {
        let mut renderer = UrdfRenderer {
            path: "/nonexistent/robot.urdf".to_owned(),
            ..Default::default()
        };
        renderer.load();
        let buffer = tf_with(&SAMPLE_FRAMES);
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        let Err(RenderStatus::SourceError(error)) = renderer.scene(&tf) else {
            panic!("expected SourceError");
        };
        assert!(error.contains("/nonexistent/robot.urdf"), "error={error}");
        assert_eq!(renderer.source_error(), Some(error));
        assert_eq!(renderer.loaded, ModelSource::None);
        // Invalid XML at a readable path also reports SourceError rather than panicking.
        let bad = temp_urdf("broken.urdf", "<robot name=\"x\"><link></robot>");
        renderer.path = bad.display().to_string();
        renderer.load();
        assert!(matches!(
            renderer.scene(&tf),
            Err(RenderStatus::SourceError(_))
        ));
        // A valid path recovers, and the epoch advanced with every load attempt.
        renderer.path = sample_urdf_path("sample_robot.urdf").display().to_string();
        renderer.load();
        assert_eq!(renderer.source_error(), None);
        assert_eq!(renderer.epoch, 3);
        assert_eq!(renderer.scene(&tf).expect("recovered").len(), 4);
    }

    #[test]
    fn model_swap_bumps_generation_for_every_slot() {
        let mut renderer = loaded("sample_robot.urdf");
        let first = scene_of(&mut renderer, &SAMPLE_FRAMES);
        renderer.load();
        let second = scene_of(&mut renderer, &SAMPLE_FRAMES);
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(&second) {
            assert_ne!(a.generation, b.generation);
        }
    }

    #[test]
    fn settings_roundtrip_reloads_the_model() {
        let mut renderer = loaded("sample_robot.urdf");
        renderer.alpha = 0.6;
        renderer.mesh_roots = vec!["/ws/src".to_owned(), "~/other".to_owned()];
        let value = renderer.settings().expect("urdf has settings");
        let mut restored = UrdfRenderer::default();
        restored.apply_settings(&value);
        assert_eq!(restored.path, renderer.path);
        assert_eq!(restored.alpha, 0.6);
        assert_eq!(restored.mesh_roots, renderer.mesh_roots);
        // apply_settings loads the file, so the restored item draws without any further action.
        assert_eq!(restored.visuals.len(), 4);
        assert_eq!(scene_of(&mut restored, &SAMPLE_FRAMES).len(), 4);
        // An empty path restores as "not loaded" instead of attempting I/O.
        let mut empty = UrdfRenderer::default();
        empty.apply_settings(&toml::Value::try_from(UrdfSettings::default()).unwrap());
        assert_eq!(empty.source_error(), None);
        assert!(empty.visuals.is_empty());
        // A config written before mesh roots existed still restores.
        let old: toml::Value = toml::from_str(&format!(
            "path = {:?}\nalpha = 0.5\n",
            sample_urdf_path("sample_robot.urdf").display().to_string()
        ))
        .expect("old config parses");
        let mut legacy = UrdfRenderer::default();
        legacy.apply_settings(&old);
        assert_eq!(legacy.alpha, 0.5);
        assert!(legacy.mesh_roots.is_empty());
        assert_eq!(legacy.visuals.len(), 4);
    }

    #[test]
    fn browse_raises_one_request_and_the_picked_file_is_loaded() {
        let mut renderer = UrdfRenderer::default();
        assert_eq!(renderer.take_file_request(), None);
        renderer.request_browse();
        let request = renderer
            .take_file_request()
            .expect("browse raises a request");
        assert_eq!(request.filter_name, "URDF");
        assert!(request.extensions.contains(&"urdf"));
        // No path typed yet, so there is no directory to start in.
        assert_eq!(request.start_dir, None);
        // Taking it clears it: app.rs must not reopen the dialog every frame.
        assert_eq!(renderer.take_file_request(), None);
        // Picking a file replaces the path and loads it right away.
        let path = sample_urdf_path("sample_robot.urdf");
        renderer.on_file_picked(&path);
        assert_eq!(renderer.path, path.display().to_string());
        assert_eq!(renderer.source_error(), None);
        assert_eq!(renderer.visuals.len(), 4);
        assert_eq!(scene_of(&mut renderer, &SAMPLE_FRAMES).len(), 4);
        // With a loaded path the dialog starts in that file's directory.
        renderer.request_browse();
        let request = renderer.take_file_request().expect("second request");
        assert_eq!(request.start_dir.as_deref(), path.parent());
    }

    /// Absolute path of a file in the xacro fixture package (`tests/fixtures/xacro/robot_description/urdf`).
    fn xacro_fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/xacro/robot_description/urdf")
            .join(name)
    }

    #[test]
    fn a_xacro_file_expands_and_its_meshes_resolve_through_its_workspace() {
        let mut renderer = UrdfRenderer {
            path: xacro_fixture_path("robot.urdf.xacro").display().to_string(),
            ..Default::default()
        };
        renderer.load();
        assert_eq!(renderer.load_error, None);
        assert_eq!(renderer.robot_name, "fixture");
        assert_eq!((renderer.link_count, renderer.visuals.len()), (3, 3));
        assert_eq!(renderer.expanded_files, 3);
        assert!(
            renderer
                .source_line()
                .ends_with("robot.urdf.xacro (xacro, 3 files)"),
            "{}",
            renderer.source_line()
        );
        // The mesh is package://mesh/..., found through the fixture directory's ancestors with no root configured.
        poll_until_idle(&mut renderer);
        assert!(
            matches!(renderer.visuals[0].load, VisualLoad::Ready(_)),
            "{:?}",
            renderer.states
        );
        let batches = scene_of(
            &mut renderer,
            &["base_link", "left_wheel_link", "right_wheel_link"],
        );
        assert_eq!(batches.len(), 3);
        // Browse offers .xacro files.
        renderer.request_browse();
        let request = renderer
            .take_file_request()
            .expect("browse raises a request");
        assert!(request.extensions.contains(&"xacro"));
    }

    #[test]
    fn mappings_change_the_expanded_model_and_persist() {
        let mut renderer = UrdfRenderer {
            path: xacro_fixture_path("robot.urdf.xacro").display().to_string(),
            mappings: "ROBOT_MODEL:=alt robot_name:=mapped".to_owned(),
            ..Default::default()
        };
        renderer.load();
        assert_eq!(renderer.load_error, None);
        assert_eq!(renderer.robot_name, "mapped");
        assert_eq!(renderer.link_count, 1);
        // Reload re-reads the mappings, so editing them and pressing Reload switches the model.
        renderer.mappings.clear();
        renderer.reload();
        assert_eq!(renderer.link_count, 3);
        // Mappings survive a settings roundtrip and a broken mapping is reported, not ignored.
        renderer.mappings = "USE_MESH:=false".to_owned();
        let value = renderer.settings().expect("urdf has settings");
        let mut restored = UrdfRenderer::default();
        restored.apply_settings(&value);
        assert_eq!(restored.mappings, "USE_MESH:=false");
        assert_eq!(restored.visuals.len(), 3);
        assert!(matches!(
            restored.visuals[0].spec.shape,
            model::Shape::Box { .. }
        ));
        renderer.mappings = "oops".to_owned();
        renderer.reload();
        let error = renderer.source_error().expect("bad mapping fails the load");
        assert!(error.contains("mappings"), "{error}");
        assert_eq!(renderer.loaded, ModelSource::None);
    }

    #[test]
    fn the_variables_form_writes_mappings_and_reloads() {
        let mut renderer = UrdfRenderer {
            path: xacro_fixture_path("robot.urdf.xacro").display().to_string(),
            ..Default::default()
        };
        renderer.load();
        let names: Vec<&str> = renderer.variables.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["robot_name", "ROBOT_MODEL", "USE_MESH"]);
        assert_eq!(renderer.variable_edits, ["fixture", "demo", "true"]);
        let model = &renderer.variables[1];
        assert_eq!(model.candidates, ["alt", "demo"]);
        assert_eq!(model.source, xacro::VariableSource::Default);
        // Choosing a candidate lands in the mappings text and swaps the model.
        renderer.set_variable("ROBOT_MODEL", "alt");
        assert_eq!(renderer.mappings, "ROBOT_MODEL:=alt");
        assert_eq!(renderer.link_count, 1);
        assert_eq!(renderer.variables[1].source, xacro::VariableSource::Mapping);
        // A second variable joins the same text; clearing one leaves the other.
        renderer.set_variable("USE_MESH", "false");
        assert_eq!(renderer.mappings, "ROBOT_MODEL:=alt USE_MESH:=false");
        renderer.clear_variable("ROBOT_MODEL");
        assert_eq!(renderer.mappings, "USE_MESH:=false");
        assert_eq!(renderer.link_count, 3);
        assert!(matches!(
            renderer.visuals[0].spec.shape,
            model::Shape::Box { .. }
        ));
        // A value that names no model file fails the load but keeps the form (with the candidates) so it can be fixed.
        renderer.set_variable("ROBOT_MODEL", "nope");
        assert!(renderer.source_error().is_some());
        assert_eq!(renderer.variables[1].candidates, ["alt", "demo"]);
        assert_eq!(renderer.variables[1].value, "nope");
        renderer.set_variable("ROBOT_MODEL", "demo");
        assert_eq!(renderer.source_error(), None);
        assert_eq!(renderer.link_count, 3);
        // A plain URDF has no variables, so the form stays hidden.
        let plain = loaded("sample_robot.urdf");
        assert!(plain.variables.is_empty());
    }

    #[test]
    fn xacro_errors_surface_as_source_error_with_file_and_line() {
        let mut renderer = UrdfRenderer {
            path: xacro_fixture_path("bad_insert_block.xacro")
                .display()
                .to_string(),
            ..Default::default()
        };
        renderer.load();
        let tf = tf_with(&[]);
        let tf = TfContext {
            buffer: &tf,
            fixed_frame: "map",
        };
        let Err(RenderStatus::SourceError(error)) = renderer.scene(&tf) else {
            panic!("expected SourceError");
        };
        assert!(error.contains("bad_insert_block.xacro:4:"), "{error}");
        assert!(error.contains("insert_block"), "{error}");
        // Xacro text off the topic is expanded too; without a file its includes cannot resolve.
        let mut topic = UrdfRenderer::from_topic();
        topic.on_message(&urdf_message(
            r#"<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"><xacro:include filename="x.xacro"/></robot>"#,
        ));
        let error = topic.source_error().expect("include without a file fails");
        assert!(error.starts_with("robot_description: "), "{error}");
        assert!(error.contains("did not come from a file"), "{error}");
        // Self-contained xacro off the topic works.
        let mut topic = UrdfRenderer::from_topic();
        topic.on_message(&urdf_message(
            r#"<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"><xacro:property name="s" value="0.5"/><link name="a"><visual><geometry><sphere radius="${s}"/></geometry></visual></link></robot>"#,
        ));
        assert_eq!(topic.source_error(), None);
        assert_eq!(topic.visuals.len(), 1);
        assert_eq!(topic.loaded, ModelSource::Topic);
    }

    #[test]
    fn robot_description_filter_requires_a_segment_boundary() {
        for topic in [
            "robot_description",
            "/robot_description",
            "/ns/robot_description",
            "/a/b/robot_description",
        ] {
            assert!(is_robot_description_topic(topic), "{topic}");
        }
        for topic in [
            "/my_robot_description",
            "/robot_description_raw",
            "/chatter",
            "robot_description/x",
            "",
        ] {
            assert!(!is_robot_description_topic(topic), "{topic}");
        }
    }

    #[test]
    fn a_urdf_message_builds_the_model() {
        let mut renderer = UrdfRenderer::from_topic();
        renderer.on_message(&urdf_message(&sample_text("sample_robot.urdf")));
        assert_eq!(renderer.load_error, None);
        assert_eq!(renderer.loaded, ModelSource::Topic);
        assert_eq!(renderer.robot_name, "visor_sample");
        assert_eq!(renderer.visuals.len(), 4);
        assert_eq!(scene_of(&mut renderer, &SAMPLE_FRAMES).len(), 4);
        // No file was involved, so the file path stays empty and gets persisted that way.
        assert_eq!(renderer.path, "");
    }

    #[test]
    fn the_same_xml_arriving_again_changes_nothing() {
        let text = sample_text("sample_mesh.urdf");
        let mut renderer = UrdfRenderer::from_topic();
        renderer.on_message(&urdf_message(&text));
        poll_until_idle(&mut renderer);
        let before = scene_of(&mut renderer, &["base_link"]);
        let epoch = renderer.epoch;
        let info = renderer.mesh_info.clone();
        // A latched redelivery (visibility toggle, publisher restart, restore) must not rebuild anything.
        renderer.on_message(&urdf_message(&text));
        assert_eq!(renderer.epoch, epoch);
        assert_eq!(renderer.pending, 0);
        assert_eq!(renderer.mesh_info, info);
        let after = scene_of(&mut renderer, &["base_link"]);
        let gens: Vec<u64> = before.iter().map(|b| b.generation).collect();
        assert_eq!(gens, after.iter().map(|b| b.generation).collect::<Vec<_>>());
    }

    #[test]
    fn a_different_xml_replaces_the_model() {
        let mut renderer = UrdfRenderer::from_topic();
        renderer.on_message(&urdf_message(&sample_text("sample_robot.urdf")));
        let before = scene_of(&mut renderer, &SAMPLE_FRAMES);
        let epoch = renderer.epoch;
        renderer.on_message(&urdf_message(
            r#"<robot name="other">
                 <link name="base_link">
                   <visual><geometry><box size="1 1 1"/></geometry></visual>
                 </link>
               </robot>"#,
        ));
        assert_eq!(renderer.epoch, epoch + 1);
        assert_eq!(renderer.robot_name, "other");
        let after = scene_of(&mut renderer, &SAMPLE_FRAMES);
        assert_eq!(after.len(), 1);
        assert_ne!(before[0].generation, after[0].generation);
    }

    #[test]
    fn a_message_without_string_data_reports_a_source_error() {
        let mut renderer = UrdfRenderer::from_topic();
        renderer.on_message(&Value::Struct(vec![("data".to_owned(), Value::U32(7))]));
        let error = renderer.source_error().expect("expected a source error");
        assert!(error.contains("std_msgs/msg/String"), "error={error}");
        let epoch = renderer.epoch;
        // A publisher stuck on the wrong type must not bump the epoch on every message.
        renderer.on_message(&Value::Struct(Vec::new()));
        assert_eq!(renderer.epoch, epoch);
        assert_eq!(renderer.source_error(), Some(error));
    }

    #[test]
    fn a_topic_item_waits_for_data_before_anything_arrives() {
        let buffer = TfBuffer::new();
        let tf = TfContext {
            buffer: &buffer,
            fixed_frame: "map",
        };
        // Same status the item already carries from the moment it is added, so the wording never flickers.
        assert_eq!(
            UrdfRenderer::from_topic().scene(&tf).unwrap_err(),
            RenderStatus::NoData
        );
        let Err(RenderStatus::NoSource(message)) = UrdfRenderer::default().scene(&tf) else {
            panic!("a standalone item has no subscription to wait on");
        };
        assert!(message.contains("Browse"), "message={message}");
        assert!(
            UrdfRenderer::from_topic()
                .source_line()
                .contains("(waiting)")
        );
    }

    #[test]
    fn reload_reapplies_the_last_received_xml() {
        let mut renderer = UrdfRenderer::from_topic();
        renderer.on_message(&urdf_message(&sample_text("sample_robot.urdf")));
        let before = scene_of(&mut renderer, &SAMPLE_FRAMES);
        let epoch = renderer.epoch;
        // The topic cannot be asked again, so Reload (the way to apply edited mesh roots) re-parses what was kept.
        renderer.reload();
        assert_eq!(renderer.epoch, epoch + 1);
        assert_eq!(renderer.loaded, ModelSource::Topic);
        let after = scene_of(&mut renderer, &SAMPLE_FRAMES);
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert_ne!(a.generation, b.generation);
        }
    }

    #[test]
    fn a_broken_xml_from_the_topic_is_not_reparsed() {
        let mut renderer = UrdfRenderer::from_topic();
        let broken = "<robot name=\"x\"><link></robot>";
        renderer.on_message(&urdf_message(broken));
        let error = renderer.source_error().expect("expected a source error");
        let epoch = renderer.epoch;
        renderer.on_message(&urdf_message(broken));
        assert_eq!(renderer.epoch, epoch);
        assert_eq!(renderer.source_error(), Some(error));
    }

    #[test]
    fn a_relative_mesh_path_has_no_base_directory_on_the_topic_route() {
        let mut renderer = UrdfRenderer::from_topic();
        renderer.on_message(&urdf_message(
            r#"<robot name="relative">
                 <link name="base_link">
                   <visual><geometry><mesh filename="meshes/shell.stl"/></geometry></visual>
                 </link>
               </robot>"#,
        ));
        poll_until_idle(&mut renderer);
        let VisualLoad::Failed(reason) = &renderer.visuals[0].load else {
            panic!("a relative path cannot resolve without a URDF file to sit next to");
        };
        assert!(reason.contains("no URDF directory"), "reason={reason}");
    }

    #[test]
    fn generation_packs_epoch_and_index_without_collision() {
        assert_eq!(generation(0, 0), 0);
        assert_ne!(generation(1, 0), generation(0, 1));
        assert_ne!(generation(0, 1), generation(0, 2));
        assert_eq!(generation(2, 5), (2 << 32) | 5);
    }
}
