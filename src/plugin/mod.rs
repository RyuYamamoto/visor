//! Plugin API facade. A plugin depends on `visor::plugin::*` and nothing else; every other module is visor's own internals.

pub mod ids;
pub mod panel;
pub mod registry;
pub mod types;
pub mod view2d;

/// Plugin API version. Bumped whenever anything re-exported here changes shape; mismatched plugins are rejected at startup.
pub const PLUGIN_API_VERSION: u32 = 5;

/// Identity a plugin reports about itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginInfo {
    /// Namespace for everything this plugin registers; `[a-z0-9_]+`.
    pub id: &'static str,
    /// Human-readable name shown in the Plugins dialog.
    pub name: &'static str,
    /// The plugin's own version, conventionally `env!("CARGO_PKG_VERSION")`.
    pub version: &'static str,
    /// Must be `PLUGIN_API_VERSION`.
    pub api_version: u32,
}

/// One plugin: reports who it is, then registers its extensions into the namespaced registrar.
pub trait Plugin {
    fn info(&self) -> PluginInfo;
    fn register(&self, registrar: &mut Registrar<'_>);
}

pub use ids::PluginId;
pub use panel::{PanelContext, PanelPlugin};
pub use registry::{
    PanelDescriptor, Problem, ProblemKind, Registrar, Registry, RendererDescriptor, RendererEntry,
    SourceDescriptor, View2dDescriptor,
};
pub use view2d::View2d;

pub use crate::comm::session::{LIVE_EPOCH, TopicRow};
/// serde helper for `Color32` settings fields: `#[serde(with = "visor::plugin::color_hex")]`, so plugin config matches the built-in renderers.
pub use crate::config::color_hex;
/// The app's merged `.msg` registry, handed to a `SourceDescriptor`'s factory as the decoding fallback.
pub use crate::decode::msg_parser::TypeRegistry;
pub use crate::decode::value::Value;
pub use crate::render::{
    BatchData, Companion, DisplayItemId, FileRequest, GridPalette, GridTexture, IntensityScale,
    LINE_WIDTH_PX_DEFAULT, Label, MAX_TEXTURE_DIM, MeshBatch, MeshBatchBuilder, OccupancyScheme,
    PointBatch, PointBatchBuilder, PointStyle, PointStyleSettings, RenderStatus, Renderer,
    SceneBatch, SizeSpec, TfContext, Vertex, arrow_mesh_vertex_count, colormap, extract_header,
    extract_pose, intensity_color, prism_vertex_count, push_arrow_mesh, push_prism, push_triad,
    push_triangle,
};
pub use crate::tf::buffer::{TfBuffer, TfTransform, TfUpdate, TimeNs};
pub use crate::theme;
