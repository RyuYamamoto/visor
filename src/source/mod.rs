//! The app's single data source: owns the UI-facing channels and the tokio runtime, and routes commands to whichever backend is running.

pub mod launch;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};

use crate::bag::{BagEvent, BagHandle, PlayerChannels, reader, rosbag2};
use crate::comm::CommConfig;
use crate::comm::session::{
    ConnectionStatus, LiveHandle, Notify, TopicMessage, TopicRow, UiSenders,
};
use crate::decode::msg_parser::TypeRegistry;
use crate::plugin::registry::{Registrar, Registry, SourceDescriptor};
use crate::tf::buffer::{TfUpdate, TimeNs};

/// display channel capacity; full means dropped, and the next tick or flush carries the newer value.
const DISPLAY_CHANNEL_CAPACITY: usize = 32;

/// Channels a backend writes to. The same set in every mode, so the UI drains them without knowing which source is running.
pub struct SourceChannels {
    pub graph_tx: Sender<Vec<TopicRow>>,
    pub display_tx: Sender<TopicMessage>,
    pub tf_tx: Sender<TfUpdate>,
    /// Open / status / notice / failure events; a source with no transport only ever sends Notice and Failed.
    pub event_tx: Sender<BagEvent>,
}

/// Transport controls of a source that replays recorded data; a live connection has none.
pub trait Playback {
    fn play(&self);
    fn pause(&self);
    fn seek(&self, to: TimeNs);
    /// Advance to the next message on any subscribed topic.
    fn step(&self);
    /// Go back to the previous message on any subscribed topic.
    fn step_back(&self);
    fn set_speed(&self, speed: f32);
    fn set_loop(&self, enabled: bool);
}

/// A running data producer. Implement this plus a SourceDescriptor to add a new source (ROS 2 bag, MCAP, a network replay).
pub trait SourceBackend {
    /// Start delivering a topic; the returned counter is incremented per received message.
    fn subscribe_display(&self, row: &TopicRow) -> Arc<AtomicU64>;
    fn unsubscribe_display(&self, topic: &str);
    fn subscribe_tf(&self, row: &TopicRow, is_static: bool);
    /// Stop one kind of TF (the user pinned another topic, or one that is absent); a no-op for a backend that never subscribed.
    fn unsubscribe_tf(&self, is_static: bool);
    fn refresh_tf_static(&self);
    /// Transport controls, if this source replays recorded data.
    fn playback(&self) -> Option<&dyn Playback> {
        None
    }
}

/// Register the data sources visor ships with (the live zenoh connection needs no files, so it is not a descriptor).
pub fn register_builtin(reg: &mut Registrar<'_>) {
    // A ROS 1 bag's own definitions stay the only authority: no fallback, exactly as before rosbag2 support.
    reg.source(SourceDescriptor::new(
        "bag",
        "ROS 1 bag",
        &["bag"],
        |paths, generation, channels, notify, _types| {
            Ok(Box::new(BagHandle::spawn(
                paths,
                generation,
                player_channels(channels),
                notify,
                reader::open_storage,
                None,
            )))
        },
    ));
    // The menu picks bag directories, which `start` turns into `metadata.yaml`; `yaml` is what routes that (and `--bag <dir>`) here.
    reg.source(
        SourceDescriptor::new(
            "rosbag2",
            "ROS 2 bag",
            &["mcap", "db3", "yaml"],
            |paths, generation, channels, notify, types| {
                let files = rosbag2::expand_paths(paths).map_err(|e| e.to_string())?;
                Ok(Box::new(BagHandle::spawn(
                    &files,
                    generation,
                    player_channels(channels),
                    notify,
                    rosbag2::open_storage,
                    Some(types),
                )))
            },
        )
        .folders(),
    );
}

fn player_channels(channels: SourceChannels) -> PlayerChannels {
    PlayerChannels {
        graph_tx: channels.graph_tx,
        display_tx: channels.display_tx,
        tf_tx: channels.tf_tx,
        bag_tx: channels.event_tx,
    }
}

/// Which data source the process is serving. The alternatives are exclusive by design (requirements §0-2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    Live(CommConfig),
    /// One or more files handled by a registered source descriptor, merged in file order (as `rosbag play a.bag b.bag` does).
    Files(Vec<PathBuf>),
}

impl Mode {
    /// Whether this mode reads files rather than a live connection.
    pub fn is_files(&self) -> bool {
        matches!(self, Mode::Files(_))
    }
}

/// Data source: the UI reads the four channels regardless of mode and only consults `playback()` for transport controls.
pub struct Source {
    /// Alive in every mode: rfd's xdg-portal backend needs a reactor to block_on (requirements G2).
    runtime: Arc<tokio::runtime::Runtime>,
    /// None while swapping sources, and after a source failed to start.
    backend: Option<Box<dyn SourceBackend>>,
    mode: Mode,
    /// Incremented per source, and stamped into replayed rows' type_hash so reopening forces a resubscribe.
    generation: u64,
    /// Merged .msg definitions used to decode live traffic (a file source carries its own definitions).
    types: Arc<TypeRegistry>,
    registry: Arc<Registry>,
    /// Uppercased id of the descriptor that opened a file source, shown as the status bar's mode chip; None in live mode.
    tag: Option<String>,
    pub conn_rx: Receiver<ConnectionStatus>,
    pub graph_rx: Receiver<Vec<TopicRow>>,
    pub display_rx: Receiver<TopicMessage>,
    pub tf_rx: Receiver<TfUpdate>,
    /// Only present for a file source: playback state, source info and read failures.
    pub event_rx: Option<Receiver<BagEvent>>,
}

/// The receiving ends the UI keeps, produced together with the senders handed to a backend.
struct Wiring {
    conn_rx: Receiver<ConnectionStatus>,
    graph_rx: Receiver<Vec<TopicRow>>,
    display_rx: Receiver<TopicMessage>,
    tf_rx: Receiver<TfUpdate>,
    event_rx: Option<Receiver<BagEvent>>,
    tag: Option<String>,
}

impl Source {
    /// Build the runtime and start the requested source (called once from the UI thread).
    pub fn spawn(
        mode: Mode,
        notify: Notify,
        types: Arc<TypeRegistry>,
        registry: Arc<Registry>,
    ) -> Self {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("failed to build tokio runtime"),
        );
        let generation = 1;
        let (backend, wiring) = start(&runtime, &mode, generation, notify, &types, &registry);
        Self {
            runtime,
            backend,
            mode,
            generation,
            types,
            registry,
            tag: wiring.tag,
            conn_rx: wiring.conn_rx,
            graph_rx: wiring.graph_rx,
            display_rx: wiring.display_rx,
            tf_rx: wiring.tf_rx,
            event_rx: wiring.event_rx,
        }
    }

    /// Switch to another source in place, keeping the runtime; the old backend is stopped before the new one starts.
    pub fn respawn(&mut self, mode: Mode, notify: Notify) {
        self.backend = None;
        self.generation += 1;
        let (backend, wiring) = start(
            &self.runtime,
            &mode,
            self.generation,
            notify,
            &self.types,
            &self.registry,
        );
        self.backend = backend;
        self.mode = mode;
        self.tag = wiring.tag;
        self.conn_rx = wiring.conn_rx;
        self.graph_rx = wiring.graph_rx;
        self.display_rx = wiring.display_rx;
        self.tf_rx = wiring.tf_rx;
        self.event_rx = wiring.event_rx;
    }

    /// Handle to the tokio runtime, used to block_on native dialogs from the UI thread.
    pub fn handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
    }

    pub fn mode(&self) -> &Mode {
        &self.mode
    }

    /// Whether a file source is open rather than a live connection.
    pub fn is_files(&self) -> bool {
        self.mode.is_files()
    }

    /// Short tag for the status bar's mode chip: the file source's id, or LIVE. FILE covers a file no source claimed.
    pub fn mode_tag(&self) -> &str {
        match (&self.tag, self.mode.is_files()) {
            (Some(tag), _) => tag,
            (None, true) => "FILE",
            (None, false) => "LIVE",
        }
    }

    /// Transport controls, present only for a source that replays recorded data.
    pub fn playback(&self) -> Option<&dyn Playback> {
        self.backend.as_ref()?.playback()
    }

    /// Start delivering a topic; the returned counter is incremented per received message.
    pub fn subscribe_display(&self, row: &TopicRow) -> Arc<AtomicU64> {
        match &self.backend {
            Some(backend) => backend.subscribe_display(row),
            None => Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn unsubscribe_display(&self, topic: &str) {
        if let Some(backend) = &self.backend {
            backend.unsubscribe_display(topic);
        }
    }

    pub fn subscribe_tf(&self, row: &TopicRow, is_static: bool) {
        if let Some(backend) = &self.backend {
            backend.subscribe_tf(row, is_static);
        }
    }

    pub fn unsubscribe_tf(&self, is_static: bool) {
        if let Some(backend) = &self.backend {
            backend.unsubscribe_tf(is_static);
        }
    }

    pub fn refresh_tf_static(&self) {
        if let Some(backend) = &self.backend {
            backend.refresh_tf_static();
        }
    }
}

/// Create the channels and start the backend for `mode`; a file with no registered handler yields no backend and one Failed event.
fn start(
    runtime: &Arc<tokio::runtime::Runtime>,
    mode: &Mode,
    generation: u64,
    notify: Notify,
    types: &Arc<TypeRegistry>,
    registry: &Registry,
) -> (Option<Box<dyn SourceBackend>>, Wiring) {
    let (conn_tx, conn_rx) = unbounded();
    let (graph_tx, graph_rx) = unbounded();
    let (display_tx, display_rx) = bounded(DISPLAY_CHANNEL_CAPACITY);
    let (tf_tx, tf_rx) = unbounded();
    let (backend, event_rx, tag) = match mode {
        Mode::Live(config) => {
            let senders = UiSenders {
                conn_tx,
                graph_tx,
                display_tx,
                tf_tx,
            };
            let live: Box<dyn SourceBackend> = Box::new(LiveHandle::spawn(
                runtime.handle(),
                config.clone(),
                senders,
                notify,
                Arc::clone(types),
            ));
            (Some(live), None, None)
        }
        Mode::Files(paths) => {
            let (event_tx, event_rx) = unbounded();
            let channels = SourceChannels {
                graph_tx,
                display_tx,
                tf_tx,
                event_tx: event_tx.clone(),
            };
            // conn_tx has no counterpart for a file source; dropping it leaves conn_rx permanently empty, which is what the status bar wants.
            drop(conn_tx);
            // A folder picked in the menu has no extension to route on, so directories become their files here, the same way `--bag <dir>` does.
            let expanded = launch::expand_directories(paths.clone());
            let entry = expanded
                .as_ref()
                .ok()
                .and_then(|paths| paths.first())
                .and_then(|p| registry.find_source_for_path(p));
            let (backend, tag) = match (&expanded, entry) {
                (Err(e), _) => {
                    let _ = event_tx.send(BagEvent::Failed(e.clone()));
                    (None, None)
                }
                (Ok(paths), Some(entry)) => {
                    let tag = entry.descriptor.id.to_uppercase();
                    match (entry.descriptor.make)(
                        paths,
                        generation,
                        channels,
                        notify,
                        Arc::clone(types),
                    ) {
                        Ok(backend) => (Some(backend), Some(tag)),
                        Err(e) => {
                            let _ = event_tx.send(BagEvent::Failed(e));
                            (None, Some(tag))
                        }
                    }
                }
                (Ok(paths), None) => {
                    let name = paths
                        .first()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default();
                    let _ =
                        event_tx.send(BagEvent::Failed(format!("no data source handles `{name}`")));
                    (None, None)
                }
            };
            (backend, Some(event_rx), tag)
        }
    };
    (
        backend,
        Wiring {
            conn_rx,
            graph_rx,
            display_rx,
            tf_rx,
            event_rx,
            tag,
        },
    )
}
