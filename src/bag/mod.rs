//! ROS 1 / ROS 2 bag playback: storage readers, self-describing type definitions, and the player thread that feeds the existing UI channels.

pub mod index;
pub mod mcap;
pub mod msgdef;
pub mod naming;
pub mod player;
pub mod reader;
pub mod rosbag2;
pub mod set;
pub mod sqlite3;
pub mod storage;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::Sender;

use crate::comm::session::{Notify, TopicMessage, TopicRow};
use crate::decode::msg_parser::TypeRegistry;
use crate::tf::buffer::{TfUpdate, TimeNs};
use storage::OpenStorage;

/// Speeds offered in the timeline's selector (FR-5).
pub const SPEEDS: [f32; 7] = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];

/// What was found in an opened bag; shown in the status bar and used to scale the timeline.
#[derive(Debug, Clone)]
pub struct BagInfo {
    /// Every file of the set, in play order.
    pub paths: Vec<PathBuf>,
    /// File names alone, which is what the status bar has room for.
    pub file_names: Vec<String>,
    pub size_bytes: u64,
    pub start: TimeNs,
    pub end: TimeNs,
    pub message_count: u64,
    pub chunk_count: usize,
    pub compression: String,
    /// Topics that will not be offered, with why (unreadable definition); shown once as a notice.
    pub skipped_topics: Vec<String>,
}

impl BagInfo {
    /// Status-bar label: the file name, or the first plus a count when several were opened.
    pub fn label(&self) -> String {
        match self.file_names.as_slice() {
            [] => "no bag".to_owned(),
            [one] => one.clone(),
            [first, rest @ ..] => format!("{first} +{}", rest.len()),
        }
    }

    /// Recorded length in seconds.
    pub fn duration_secs(&self) -> f64 {
        (self.end - self.start).max(0) as f64 / 1e9
    }

    /// Seconds from the start of the bag to `time` (the timeline's own coordinate).
    pub fn offset_secs(&self, time: TimeNs) -> f64 {
        (time - self.start) as f64 / 1e9
    }

    /// Absolute bag time for an offset in seconds from the start.
    pub fn time_at_offset(&self, secs: f64) -> TimeNs {
        self.start + (secs * 1e9) as TimeNs
    }
}

/// Transport state of playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    /// Parked at the start, nothing delivered yet.
    Stopped,
    Playing,
    Paused,
}

impl PlayState {
    pub fn is_playing(self) -> bool {
        matches!(self, PlayState::Playing)
    }
}

/// Where playback is; sent after every delivery so the timeline reflects what was actually shown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackStatus {
    /// Bumped on every seek, loop and reopen; the UI resets accumulated state when it grows.
    pub epoch: u64,
    pub state: PlayState,
    /// Time actually delivered, not the clock's target, so the bar never runs ahead of the picture.
    pub playhead: TimeNs,
    pub speed: f32,
    pub looping: bool,
}

/// Player-to-UI notifications that do not fit the shared comm channels.
#[derive(Debug, Clone)]
pub enum BagEvent {
    Opened(Box<BagInfo>),
    /// The bag could not be opened; the app keeps running and shows this (FR-1).
    Failed(String),
    /// Something partial worth telling the user about (a dropped connection, a message that would not decode).
    Notice(String),
    Status(PlaybackStatus),
}

/// Commands the UI sends to the player thread.
#[derive(Debug)]
pub enum PlayerCmd {
    SubscribeDisplay {
        topic: String,
        counter: Arc<AtomicU64>,
    },
    UnsubscribeDisplay {
        topic: String,
    },
    SubscribeTf {
        topic: String,
        is_static: bool,
    },
    UnsubscribeTf {
        is_static: bool,
    },
    RefreshTfStatic,
    Play,
    Pause,
    Seek(TimeNs),
    SetSpeed(f32),
    SetLoop(bool),
    /// Advance to the next message on any subscribed topic (no gap, so no reset).
    Step,
    /// Go back to the previous message on any subscribed topic (a backward jump, so it resets).
    StepBack,
}

/// Channels the player writes to; the same ones the live path uses, plus `bag_tx`.
pub struct PlayerChannels {
    pub graph_tx: Sender<Vec<TopicRow>>,
    pub display_tx: Sender<TopicMessage>,
    pub tf_tx: Sender<TfUpdate>,
    pub bag_tx: Sender<BagEvent>,
}

/// UI-side handle to the player thread. Dropping it stops the thread and joins it.
pub struct BagHandle {
    /// None only while dropping: the sender has to go before the join or the player never sees the disconnect.
    cmd_tx: Option<Sender<PlayerCmd>>,
    thread: Option<JoinHandle<()>>,
    /// Raised on drop so an open or index build still in progress returns at the next chunk instead of finishing the file.
    cancel: Arc<AtomicBool>,
}

impl BagHandle {
    /// Start a player thread for `paths` (merged by record time) opened with `open`; `fallback` supplies type definitions for connections whose own definition is missing or unreadable. Opening happens on that thread, so the UI never waits on file I/O.
    pub fn spawn(
        paths: &[PathBuf],
        generation: u64,
        channels: PlayerChannels,
        notify: Notify,
        open: OpenStorage,
        fallback: Option<Arc<TypeRegistry>>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let paths = paths.to_vec();
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        let thread = std::thread::Builder::new()
            .name("visor-bag-player".to_owned())
            .spawn(move || {
                player::run(
                    &paths,
                    generation,
                    channels,
                    cmd_rx,
                    notify,
                    open,
                    fallback,
                    &thread_cancel,
                )
            })
            .expect("failed to spawn the bag player thread");
        Self {
            cmd_tx: Some(cmd_tx),
            thread: Some(thread),
            cancel,
        }
    }

    fn send(&self, cmd: PlayerCmd) {
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.send(cmd);
        }
    }

    /// Start delivering a topic and hand back the counter the player increments per message.
    pub fn subscribe_display(&self, topic: &str) -> Arc<AtomicU64> {
        let counter = Arc::new(AtomicU64::new(0));
        self.send(PlayerCmd::SubscribeDisplay {
            topic: topic.to_owned(),
            counter: counter.clone(),
        });
        counter
    }

    pub fn unsubscribe_display(&self, topic: &str) {
        self.send(PlayerCmd::UnsubscribeDisplay {
            topic: topic.to_owned(),
        });
    }

    pub fn subscribe_tf(&self, topic: &str, is_static: bool) {
        self.send(PlayerCmd::SubscribeTf {
            topic: topic.to_owned(),
            is_static,
        });
    }

    pub fn unsubscribe_tf(&self, is_static: bool) {
        self.send(PlayerCmd::UnsubscribeTf { is_static });
    }

    pub fn refresh_tf_static(&self) {
        self.send(PlayerCmd::RefreshTfStatic);
    }

    pub fn play(&self) {
        self.send(PlayerCmd::Play);
    }

    pub fn pause(&self) {
        self.send(PlayerCmd::Pause);
    }

    pub fn seek(&self, to: TimeNs) {
        self.send(PlayerCmd::Seek(to));
    }

    pub fn set_speed(&self, speed: f32) {
        self.send(PlayerCmd::SetSpeed(speed));
    }

    pub fn set_loop(&self, enabled: bool) {
        self.send(PlayerCmd::SetLoop(enabled));
    }

    pub fn step(&self) {
        self.send(PlayerCmd::Step);
    }

    pub fn step_back(&self) {
        self.send(PlayerCmd::StepBack);
    }
}

impl crate::source::SourceBackend for BagHandle {
    fn subscribe_display(&self, row: &TopicRow) -> Arc<AtomicU64> {
        BagHandle::subscribe_display(self, &row.name)
    }

    fn unsubscribe_display(&self, topic: &str) {
        BagHandle::unsubscribe_display(self, topic);
    }

    fn subscribe_tf(&self, row: &TopicRow, is_static: bool) {
        BagHandle::subscribe_tf(self, &row.name, is_static);
    }

    fn unsubscribe_tf(&self, is_static: bool) {
        BagHandle::unsubscribe_tf(self, is_static);
    }

    fn refresh_tf_static(&self) {
        BagHandle::refresh_tf_static(self);
    }

    fn playback(&self) -> Option<&dyn crate::source::Playback> {
        Some(self)
    }
}

impl crate::source::Playback for BagHandle {
    fn play(&self) {
        BagHandle::play(self);
    }

    fn pause(&self) {
        BagHandle::pause(self);
    }

    fn seek(&self, to: TimeNs) {
        BagHandle::seek(self, to);
    }

    fn step(&self) {
        BagHandle::step(self);
    }

    fn step_back(&self) {
        BagHandle::step_back(self);
    }

    fn set_speed(&self, speed: f32) {
        BagHandle::set_speed(self, speed);
    }

    fn set_loop(&self, enabled: bool) {
        BagHandle::set_loop(self, enabled);
    }
}

impl Drop for BagHandle {
    fn drop(&mut self) {
        // Cancel first so a player still opening the file returns at the next chunk; then dropping the sender ends the recv loop, and joining makes "the old bag stopped" a fact rather than a hope.
        self.cancel.store(true, Ordering::Relaxed);
        self.cmd_tx = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
