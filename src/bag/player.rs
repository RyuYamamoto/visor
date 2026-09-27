//! Playback clock, tick delivery, seeking, snapshots and TF handling; runs on its own thread.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError};

use super::index::MessageIndex;
use super::msgdef::RegistrySet;
use super::naming::{normalize_topic, ros1_type_to_ros2};
use super::reader::{BagError, Connection, IndexEntry};
use super::set::BagSet;
use super::storage::OpenStorage;
use super::{BagEvent, BagInfo, PlayState, PlaybackStatus, PlayerChannels, PlayerCmd};
use crate::comm::session::{DecodeFailure, Notify, TopicMessage, TopicRow};
use crate::decode::DecodeError;
use crate::decode::msg_parser::TypeRegistry;
use crate::decode::value::Value;
use crate::tf::buffer::{TfUpdate, TimeNs, transforms_from_value};

/// Delivery interval; matches the live path's display throttle so both sources thin identically.
const TICK: Duration = Duration::from_millis(33);
/// Dynamic TF replayed before a seek target so lookup_transform can interpolate immediately (plan §8 (c)).
const BACKFILL_WINDOW: TimeNs = 2_000_000_000;
/// Snapshots are worth waiting for (the screen is empty until they land), unlike a tick that the next one supersedes.
const SNAPSHOT_SEND_TIMEOUT: Duration = Duration::from_millis(100);
/// TF transforms batched into one channel message, so a fast-forward tick does not send thousands of tiny updates.
const TF_BATCH: usize = 200;
/// Static TF entries examined per loop iteration, keeping the loop responsive to commands while scanning.
const STATIC_SCAN_STEP: usize = 32;
/// Consecutive already-known static transforms after which scanning stops (plan §5.5).
const STATIC_SCAN_STREAK: usize = 200;
/// Repeated decode failures on one topic are reported once, then counted silently.
const MAX_DECODE_NOTICES: usize = 3;

/// Bag time only advances by wall time times speed; kept separate so the arithmetic is testable without a clock.
#[derive(Debug, Clone, Copy)]
struct Clock {
    base_bag: TimeNs,
    base_wall: Instant,
    speed: f32,
}

impl Clock {
    fn new(base_bag: TimeNs, speed: f32) -> Self {
        Self {
            base_bag,
            base_wall: Instant::now(),
            speed,
        }
    }

    fn rebase(&mut self, base_bag: TimeNs, now: Instant) {
        self.base_bag = base_bag;
        self.base_wall = now;
    }

    fn target(&self, now: Instant) -> TimeNs {
        advance(
            self.base_bag,
            now.saturating_duration_since(self.base_wall),
            self.speed,
        )
    }
}

/// Bag time after `elapsed` of wall time at `speed`; saturates so a long pause cannot overflow.
fn advance(base_bag: TimeNs, elapsed: Duration, speed: f32) -> TimeNs {
    let delta = elapsed.as_secs_f64() * f64::from(speed) * 1e9;
    base_bag.saturating_add(delta as TimeNs)
}

/// One subscribed display topic: which connections feed it and what was last delivered from each.
struct DisplaySub {
    conns: Vec<u32>,
    counter: Arc<AtomicU64>,
    /// Connection -> time of the entry last sent, so an unchanged snapshot is not resent every tick.
    last: HashMap<u32, TimeNs>,
}

/// Progressive scan for static transforms that appear later in the bag than its first chunk.
struct StaticScan {
    entries: Vec<IndexEntry>,
    at: usize,
    known: HashSet<(String, String)>,
    streak: usize,
}

/// Everything the player owns once the bag is open.
struct Player {
    bags: BagSet,
    index: MessageIndex,
    registries: RegistrySet,
    info: BagInfo,
    channels: PlayerChannels,
    notify: Notify,
    /// Normalized topic -> connections carrying it (only those with a usable definition).
    topic_conns: HashMap<String, Vec<u32>>,
    subs: HashMap<String, DisplaySub>,
    tf_dyn: Vec<u32>,
    tf_static: Vec<u32>,
    /// Every static transform seen so far, replayed after each seek since inserts are idempotent (FR-8).
    static_cache: Vec<crate::tf::buffer::TfTransform>,
    static_scan: Option<StaticScan>,
    clock: Clock,
    state: PlayState,
    /// Bag time already delivered; the playhead the UI is told about.
    delivered: TimeNs,
    epoch: u64,
    looping: bool,
    /// Topic -> decode failures reported, so a systematically undecodable topic does not flood the notice line.
    decode_notices: HashMap<String, usize>,
}

/// Player thread body: open the bag, publish what is in it, then serve commands and ticks until the handle is dropped.
#[allow(clippy::too_many_arguments)]
pub fn run(
    paths: &[PathBuf],
    generation: u64,
    channels: PlayerChannels,
    cmd_rx: Receiver<PlayerCmd>,
    notify: Notify,
    open: OpenStorage,
    fallback: Option<Arc<TypeRegistry>>,
    cancel: &AtomicBool,
) {
    let Some(mut player) =
        Player::open(paths, generation, channels, notify, open, fallback, cancel)
    else {
        return;
    };
    loop {
        let wait = match (player.state, player.static_scan.is_some()) {
            // Scanning still has work, so come back promptly instead of sleeping on the channel.
            (_, true) => Duration::from_millis(1),
            (PlayState::Playing, false) => player.until_next_tick(),
            // Idle means genuinely idle: no CPU and no repaints while paused (AC-15).
            (_, false) => Duration::MAX,
        };
        match cmd_rx.recv_timeout(wait) {
            Ok(cmd) => player.handle(cmd),
            Err(RecvTimeoutError::Timeout) => {
                if player.static_scan.is_some() {
                    player.advance_static_scan();
                } else if player.state.is_playing() {
                    player.tick();
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

impl Player {
    /// Open the bag, build the type registries and the time index, and publish the topic list.
    fn open(
        paths: &[PathBuf],
        generation: u64,
        channels: PlayerChannels,
        notify: Notify,
        open: OpenStorage,
        fallback: Option<Arc<TypeRegistry>>,
        cancel: &AtomicBool,
    ) -> Option<Self> {
        let started = Instant::now();
        let fail = |reason: String| {
            let _ = channels.bag_tx.send(BagEvent::Failed(reason));
            notify();
        };
        let describe = |paths: &[PathBuf]| match paths {
            [one] => one.display().to_string(),
            many => format!("{} bags", many.len()),
        };
        let mut bags = match BagSet::open(paths, open, cancel) {
            Ok(bags) => bags,
            // Cancelled means the handle is gone and nobody is listening for a Failed event any more.
            Err(BagError::Cancelled) => return None,
            Err(e) => {
                fail(format!("{}: {e}", describe(paths)));
                return None;
            }
        };
        let index = match bags.read_message_index(cancel) {
            Ok(entries) => MessageIndex::build(entries),
            Err(BagError::Cancelled) => return None,
            Err(e) => {
                fail(format!("{}: {e}", describe(paths)));
                return None;
            }
        };
        if index.is_empty() {
            fail(format!("{}: no messages", describe(paths)));
            return None;
        }
        let registries = RegistrySet::build(bags.connections(), fallback.as_ref());
        let (topics, topic_conns, skipped_topics) =
            build_graph(bags.connections(), &registries, generation);
        let info = BagInfo {
            paths: paths.to_vec(),
            file_names: bags.file_names(),
            size_bytes: bags.size_bytes(),
            start: index.start(),
            end: index.end(),
            message_count: bags.message_count(),
            chunk_count: bags.chunk_count(),
            compression: bags.compression(),
            skipped_topics,
        };
        eprintln!(
            "visor: opened {} ({} messages, {} topics, {:.1}s) in {} ms",
            info.label(),
            info.message_count,
            topics.len(),
            info.duration_secs(),
            started.elapsed().as_millis()
        );
        let start = index.start();
        let player = Self {
            bags,
            index,
            registries,
            info,
            channels,
            notify,
            topic_conns,
            subs: HashMap::new(),
            tf_dyn: Vec::new(),
            tf_static: Vec::new(),
            static_cache: Vec::new(),
            static_scan: None,
            clock: Clock::new(start, 1.0),
            state: PlayState::Stopped,
            delivered: start,
            epoch: 1,
            looping: false,
            decode_notices: HashMap::new(),
        };
        player.emit(BagEvent::Opened(Box::new(player.info.clone())));
        for warning in player.bags.warnings() {
            player.emit(BagEvent::Notice(warning));
        }
        for (conn, reason) in player.registries.fallback_notices() {
            let topic = player
                .bags
                .connections()
                .iter()
                .find(|c| c.id == *conn)
                .map_or_else(|| format!("conn {conn}"), |c| normalize_topic(&c.topic_raw));
            player.emit(BagEvent::Notice(format!("{topic}: {reason}")));
        }
        if !player.info.skipped_topics.is_empty() {
            let list = player.info.skipped_topics.join(", ");
            player.emit(BagEvent::Notice(format!("topics not shown: {list}")));
        }
        let _ = player.channels.graph_tx.send(topics);
        player.emit_status();
        Some(player)
    }

    fn emit(&self, event: BagEvent) {
        let _ = self.channels.bag_tx.send(event);
        (self.notify)();
    }

    fn emit_status(&self) {
        let _ = self.channels.bag_tx.send(BagEvent::Status(PlaybackStatus {
            epoch: self.epoch,
            state: self.state,
            playhead: self.delivered,
            speed: self.clock.speed,
            looping: self.looping,
        }));
        (self.notify)();
    }

    /// Time left until the next delivery; zero once the tick is already due.
    fn until_next_tick(&self) -> Duration {
        let now = Instant::now();
        let target = self.clock.target(now);
        if target >= self.delivered + TICK.as_nanos() as TimeNs {
            return Duration::ZERO;
        }
        TICK
    }

    fn handle(&mut self, cmd: PlayerCmd) {
        match cmd {
            PlayerCmd::SubscribeDisplay { topic, counter } => {
                self.subscribe_display(topic, counter)
            }
            PlayerCmd::UnsubscribeDisplay { topic } => {
                self.subs.remove(&topic);
            }
            PlayerCmd::SubscribeTf { topic, is_static } => self.subscribe_tf(&topic, is_static),
            PlayerCmd::UnsubscribeTf { is_static } => self.unsubscribe_tf(is_static),
            PlayerCmd::RefreshTfStatic => self.send_static_cache(),
            PlayerCmd::Play => self.play(),
            PlayerCmd::Pause => {
                self.state = PlayState::Paused;
                self.emit_status();
            }
            PlayerCmd::Seek(to) => self.seek(to),
            PlayerCmd::SetSpeed(speed) => {
                self.clock.speed = speed;
                self.clock.rebase(self.delivered, Instant::now());
                self.emit_status();
            }
            PlayerCmd::SetLoop(enabled) => {
                self.looping = enabled;
                self.emit_status();
            }
            PlayerCmd::Step => self.step(),
            PlayerCmd::StepBack => self.step_back(),
        }
    }

    fn play(&mut self) {
        // Restart from the beginning when play is pressed at the very end and looping is off.
        if self.delivered >= self.index.end() {
            self.seek(self.index.start());
        }
        self.state = PlayState::Playing;
        self.clock.rebase(self.delivered, Instant::now());
        self.emit_status();
    }

    /// Start feeding a topic and immediately show its current state, so adding a display while paused is not blank.
    fn subscribe_display(&mut self, topic: String, counter: Arc<AtomicU64>) {
        let conns = self.topic_conns.get(&topic).cloned().unwrap_or_default();
        self.subs.insert(
            topic.clone(),
            DisplaySub {
                conns,
                counter,
                last: HashMap::new(),
            },
        );
        self.send_snapshot(&topic, self.delivered);
    }

    fn subscribe_tf(&mut self, topic: &str, is_static: bool) {
        let conns = self.topic_conns.get(topic).cloned().unwrap_or_default();
        if is_static {
            // The cache is per topic: transforms only the previous static topic carried must not be replayed under the new one.
            self.unsubscribe_tf(true);
            self.tf_static = conns;
            self.collect_static_seed();
            self.send_static_cache();
        } else {
            self.tf_dyn = conns;
            self.backfill_tf(self.delivered);
        }
    }

    /// Stop feeding one kind of TF; for static that also drops the replayed cache and the background scan.
    fn unsubscribe_tf(&mut self, is_static: bool) {
        if is_static {
            self.tf_static.clear();
            self.static_cache.clear();
            self.static_scan = None;
        } else {
            self.tf_dyn.clear();
        }
    }

    /// Deliver everything due up to the clock's target; the same path a seek's continuation uses.
    fn tick(&mut self) {
        let target = self.clock.target(Instant::now());
        let end = self.index.end();
        let reached_end = target >= end;
        self.deliver_until(target.min(end));
        if !reached_end {
            self.emit_status();
            return;
        }
        if self.looping {
            self.seek(self.index.start());
        } else {
            self.state = PlayState::Paused;
            self.emit_status();
        }
    }

    /// Connections that stepping considers: every subscribed display topic plus dynamic TF.
    fn stepping_conns(&self) -> HashSet<u32> {
        let mut conns: HashSet<u32> = self
            .subs
            .values()
            .flat_map(|s| s.conns.iter().copied())
            .collect();
        conns.extend(self.tf_dyn.iter().copied());
        conns
    }

    /// Go back one message. Any backward move is a jump, so it goes through seek (epoch bump + snapshot).
    fn step_back(&mut self) {
        let conns = self.stepping_conns();
        let to = self
            .index
            .prev_before(&conns, self.delivered)
            .map_or(self.index.start(), |entry| entry.time);
        self.state = PlayState::Paused;
        self.seek(to);
    }

    /// Advance to the next message on any subscribed topic; contiguous, so no reset and no snapshot.
    fn step(&mut self) {
        let conns = self.stepping_conns();
        match self.index.next_after(&conns, self.delivered) {
            Some(entry) => {
                let to = entry.time;
                self.deliver_until(to);
            }
            // Already at the last subscribed message: park there rather than pretending to move.
            None => self.delivered = self.index.end(),
        }
        self.state = PlayState::Paused;
        self.emit_status();
    }

    /// Jump anywhere: bump the epoch (the UI clears TF and accumulated renderer state), then repaint the target time.
    fn seek(&mut self, to: TimeNs) {
        let to = to.clamp(self.index.start(), self.index.end());
        self.epoch += 1;
        self.delivered = to;
        self.clock.rebase(to, Instant::now());
        for sub in self.subs.values_mut() {
            sub.last.clear();
        }
        // TF before displays: a renderer that cannot resolve its frame draws nothing, so the transforms have to be in place first.
        self.send_static_cache();
        self.backfill_tf(to);
        let topics: Vec<String> = self.subs.keys().cloned().collect();
        for topic in topics {
            self.send_snapshot(&topic, to);
        }
        self.emit_status();
    }

    /// Deliver dynamic TF for `(delivered, to]` in full and the newest message per subscribed topic.
    fn deliver_until(&mut self, to: TimeNs) {
        if to < self.delivered {
            return;
        }
        let from = self.delivered;
        self.deliver_tf_range(from, to);
        let topics: Vec<String> = self.subs.keys().cloned().collect();
        for topic in topics {
            self.deliver_topic(&topic, to);
        }
        self.delivered = to;
    }

    /// Every dynamic TF message in `(from, to]`, batched; TF is never thinned because interpolation needs the samples.
    fn deliver_tf_range(&mut self, from: TimeNs, to: TimeNs) {
        if self.tf_dyn.is_empty() {
            return;
        }
        let mut entries: Vec<IndexEntry> = self
            .tf_dyn
            .iter()
            .flat_map(|conn| {
                self.index
                    .range(*conn, from, to)
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        entries.sort_by_key(|e| e.time);
        self.send_tf_entries(&entries);
    }

    /// Replay dynamic TF just before a seek target so transforms are available the moment the picture is drawn.
    fn backfill_tf(&mut self, to: TimeNs) {
        if self.tf_dyn.is_empty() {
            return;
        }
        let from = (to - BACKFILL_WINDOW).max(self.index.start() - 1);
        let mut entries: Vec<IndexEntry> = self
            .tf_dyn
            .iter()
            .flat_map(|conn| {
                self.index
                    .range(*conn, from, to)
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        entries.sort_by_key(|e| e.time);
        self.send_tf_entries(&entries);
    }

    /// Decode TF entries and push them out in batches; `wait` blocks briefly so a seek's transforms are not dropped.
    fn send_tf_entries(&mut self, entries: &[IndexEntry]) {
        let mut batch = Vec::new();
        for entry in entries {
            let Some(value) = self.decode_entry(entry, "/tf") else {
                continue;
            };
            match transforms_from_value(&value) {
                Ok(mut transforms) => batch.append(&mut transforms),
                Err(e) => self.note_decode_failure("/tf", &e.to_string()),
            }
            if batch.len() >= TF_BATCH {
                self.send_tf(std::mem::take(&mut batch), false);
            }
        }
        if !batch.is_empty() {
            self.send_tf(batch, false);
        }
    }

    fn send_tf(&self, transforms: Vec<crate::tf::buffer::TfTransform>, is_static: bool) {
        // The tf channel is unbounded, so this never blocks and never drops.
        let _ = self.channels.tf_tx.send(TfUpdate {
            transforms,
            is_static,
            epoch: self.epoch,
        });
        (self.notify)();
    }

    /// Newest message on a subscribed topic at `to`, skipped when it is the one already delivered (33ms last-wins).
    fn deliver_topic(&mut self, topic: &str, to: TimeNs) {
        let Some(sub) = self.subs.get(topic) else {
            return;
        };
        let pending: Vec<IndexEntry> = sub
            .conns
            .iter()
            .filter_map(|conn| {
                let entry = self.index.at_or_before(*conn, to)?;
                (sub.last.get(conn) != Some(&entry.time)).then_some(*entry)
            })
            .collect();
        // A topic recorded across several files has one connection per file; delivering oldest first lets the newest win.
        let mut pending = pending;
        pending.sort_by_key(|e| e.time);
        for entry in pending {
            self.send_message(topic, &entry, false);
        }
    }

    /// The state of a topic at `at`, delivered even if it was already sent (used after a seek and on subscribe).
    fn send_snapshot(&mut self, topic: &str, at: TimeNs) {
        let Some(sub) = self.subs.get(topic) else {
            return;
        };
        let pending: Vec<IndexEntry> = sub
            .conns
            .iter()
            .filter_map(|conn| self.index.at_or_before(*conn, at).copied())
            .collect();
        let mut pending = pending;
        pending.sort_by_key(|e| e.time);
        for entry in pending {
            self.send_message(topic, &entry, true);
        }
    }

    /// Decode one entry and hand it to the UI, remembering what was sent per connection.
    fn send_message(&mut self, topic: &str, entry: &IndexEntry, snapshot: bool) {
        let types = match self.registries.get(entry.conn) {
            Some(types) => types.clone(),
            None => return,
        };
        let payload = match self.bags.message_at(entry) {
            Ok(msg) => msg.data.to_vec(),
            Err(e) => {
                let reason = e.to_string();
                self.note_decode_failure(topic, &reason);
                return;
            }
        };
        let result = types
            .decode(&payload)
            .map_err(|error| decode_failure(error, &payload));
        if let Err(failure) = &result {
            let reason = failure.error.to_string();
            self.note_decode_failure(topic, &reason);
        }
        let message = TopicMessage {
            topic: topic.to_owned(),
            result,
            received_at: Instant::now(),
            epoch: self.epoch,
        };
        if snapshot {
            // Nothing else will resend this one, so give the bounded channel a moment to drain.
            let _ = self
                .channels
                .display_tx
                .send_timeout(message, SNAPSHOT_SEND_TIMEOUT);
        } else {
            let _ = self.channels.display_tx.try_send(message);
        }
        if let Some(sub) = self.subs.get_mut(topic) {
            sub.last.insert(entry.conn, entry.time);
            sub.counter.fetch_add(1, Ordering::Relaxed);
        }
        (self.notify)();
    }

    /// Decode one entry to a Value, reporting (and swallowing) failures.
    fn decode_entry(&mut self, entry: &IndexEntry, topic: &str) -> Option<Value> {
        let types = self.registries.get(entry.conn)?.clone();
        let payload = match self.bags.message_at(entry) {
            Ok(msg) => msg.data.to_vec(),
            Err(e) => {
                let reason = e.to_string();
                self.note_decode_failure(topic, &reason);
                return None;
            }
        };
        match types.decode(&payload) {
            Ok(value) => Some(value),
            Err(e) => {
                let reason = e.to_string();
                self.note_decode_failure(topic, &reason);
                None
            }
        }
    }

    /// Report the first few failures on a topic, then stay quiet so a broken topic cannot drown the notice line.
    fn note_decode_failure(&mut self, topic: &str, reason: &str) {
        let seen = self.decode_notices.entry(topic.to_owned()).or_insert(0);
        *seen += 1;
        if *seen <= MAX_DECODE_NOTICES {
            self.emit(BagEvent::Notice(format!("{topic}: {reason}")));
        }
    }

    /// Read the first static TF message on each static connection: the latched publish that covers most bags.
    fn collect_static_seed(&mut self) {
        let mut entries: Vec<IndexEntry> = Vec::new();
        let mut all: Vec<IndexEntry> = Vec::new();
        for conn in self.tf_static.clone() {
            let mut per_conn: Vec<IndexEntry> = self
                .index
                .range(conn, self.index.start() - 1, self.index.end())
                .copied()
                .collect();
            if let Some(first) = per_conn.first() {
                entries.push(*first);
            }
            all.append(&mut per_conn);
        }
        for entry in &entries {
            if let Some(value) = self.decode_entry(entry, "/tf_static")
                && let Ok(transforms) = transforms_from_value(&value)
            {
                self.merge_static(transforms);
            }
        }
        // The rest is scanned in the background: /tf_static can appear hundreds of times, which would blow the open budget.
        all.sort_by_key(|e| e.time);
        if !all.is_empty() {
            let known = self
                .static_cache
                .iter()
                .map(|t| (t.parent.clone(), t.child.clone()))
                .collect();
            self.static_scan = Some(StaticScan {
                entries: all,
                at: 0,
                known,
                streak: 0,
            });
        }
    }

    /// Examine a bounded number of static entries, keeping only transforms for pairs not seen yet.
    fn advance_static_scan(&mut self) {
        let Some(mut scan) = self.static_scan.take() else {
            return;
        };
        let mut fresh = Vec::new();
        for _ in 0..STATIC_SCAN_STEP {
            let Some(entry) = scan.entries.get(scan.at).copied() else {
                break;
            };
            scan.at += 1;
            let Some(value) = self.decode_entry(&entry, "/tf_static") else {
                continue;
            };
            let Ok(transforms) = transforms_from_value(&value) else {
                continue;
            };
            let mut any_new = false;
            for transform in transforms {
                if scan
                    .known
                    .insert((transform.parent.clone(), transform.child.clone()))
                {
                    fresh.push(transform);
                    any_new = true;
                }
            }
            scan.streak = if any_new { 0 } else { scan.streak + 1 };
            if scan.streak >= STATIC_SCAN_STREAK {
                scan.at = scan.entries.len();
                break;
            }
        }
        if !fresh.is_empty() {
            self.merge_static(fresh.clone());
            self.send_tf(fresh, true);
        }
        if scan.at < scan.entries.len() {
            self.static_scan = Some(scan);
        }
    }

    /// Keep one transform per parent/child pair in the cache that is replayed after every seek.
    fn merge_static(&mut self, transforms: Vec<crate::tf::buffer::TfTransform>) {
        for transform in transforms {
            match self
                .static_cache
                .iter_mut()
                .find(|t| t.parent == transform.parent && t.child == transform.child)
            {
                Some(existing) => *existing = transform,
                None => self.static_cache.push(transform),
            }
        }
    }

    /// Re-insert every known static transform (idempotent, so this is safe to repeat after each seek).
    fn send_static_cache(&self) {
        if self.static_cache.is_empty() {
            return;
        }
        self.send_tf(self.static_cache.clone(), true);
    }
}

/// Trim a payload for the hex fallback view, exactly as the live path does.
fn decode_failure(error: DecodeError, payload: &[u8]) -> DecodeFailure {
    const HEAD: usize = 256;
    DecodeFailure {
        error,
        payload_head: payload[..payload.len().min(HEAD)].to_vec(),
        payload_len: payload.len(),
    }
}

/// Aggregate connections into the graph rows the UI already understands, plus the topic -> connections map.
fn build_graph(
    connections: &[Connection],
    registries: &RegistrySet,
    generation: u64,
) -> (Vec<TopicRow>, HashMap<String, Vec<u32>>, Vec<String>) {
    let type_hash = format!("bag{generation}");
    let mut rows: Vec<TopicRow> = Vec::new();
    let mut topic_conns: HashMap<String, Vec<u32>> = HashMap::new();
    let mut skipped: Vec<String> = Vec::new();
    for conn in connections {
        let topic = normalize_topic(&conn.topic_raw);
        if topic.is_empty() {
            continue;
        }
        let ros_type = ros1_type_to_ros2(&conn.type_raw);
        if registries.get(conn.id).is_none() {
            let label = format!("{topic} ({})", conn.type_raw);
            if !skipped.contains(&label) {
                skipped.push(label);
            }
            continue;
        }
        topic_conns.entry(topic.clone()).or_default().push(conn.id);
        // A topic with several publishers has one connection each, but the UI wants one row per topic and type (requirements f1).
        match rows
            .iter_mut()
            .find(|r| r.name == topic && r.ros_type == ros_type)
        {
            Some(row) => row.publisher_count += 1,
            None => rows.push(TopicRow {
                name: topic,
                ros_type,
                // A bag builds no zenoh key, so there is no DDS name; the hash is per-bag so reopening triggers a resubscribe.
                type_name_dds: String::new(),
                type_hash: type_hash.clone(),
                publisher_count: 1,
                subscriber_count: 0,
            }),
        }
    }
    rows.sort_by(|a, b| (&a.name, &a.ros_type).cmp(&(&b.name, &b.ros_type)));
    // Topics whose only connections failed are genuinely unavailable; ones with a surviving connection are not "skipped".
    skipped.retain(|label| {
        let topic = label.split_whitespace().next().unwrap_or_default();
        !topic_conns.contains_key(topic)
    });
    skipped.sort();
    (rows, topic_conns, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conn(id: u32, topic: &str, type_raw: &str, definition: &str) -> Connection {
        Connection {
            id,
            topic_raw: topic.to_owned(),
            type_raw: type_raw.to_owned(),
            type_hash: "0".to_owned(),
            definition: definition.to_owned(),
            message_encoding: "ros1".to_owned(),
            definition_encoding: "ros1msg".to_owned(),
        }
    }

    const SCAN_DEF: &str = "float32 angle_min";
    const TF_DEF: &str = concat!(
        "geometry_msgs/TransformStamped[] transforms\n",
        "================================================================================\n",
        "MSG: geometry_msgs/TransformStamped\n",
        "Header header\nstring child_frame_id\nTransform transform\n",
        "================================================================================\n",
        "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n",
        "================================================================================\n",
        "MSG: geometry_msgs/Transform\nVector3 translation\nQuaternion rotation\n",
        "================================================================================\n",
        "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n",
        "================================================================================\n",
        "MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w"
    );

    #[test]
    fn t15_clock_advances_with_wall_time_scaled_by_speed() {
        let base = 1_000_000_000;
        assert_eq!(advance(base, Duration::ZERO, 1.0), base);
        assert_eq!(
            advance(base, Duration::from_secs(1), 1.0),
            base + 1_000_000_000
        );
        assert_eq!(
            advance(base, Duration::from_secs(1), 0.25),
            base + 250_000_000
        );
        assert_eq!(
            advance(base, Duration::from_secs(2), 5.0),
            base + 10_000_000_000
        );
        // A pause long enough to overflow saturates instead of wrapping into the past.
        assert_eq!(
            advance(TimeNs::MAX - 5, Duration::from_secs(3600), 10.0),
            TimeNs::MAX
        );
    }

    #[test]
    fn t15_rebasing_makes_the_clock_resume_from_the_playhead() {
        let mut clock = Clock::new(0, 1.0);
        let now = Instant::now();
        clock.rebase(500, now);
        assert_eq!(clock.target(now), 500);
        // A target from before the base cannot run the clock backwards.
        assert_eq!(clock.target(now - Duration::from_secs(1)), 500);
    }

    #[test]
    fn t10_connections_aggregate_into_one_row_per_topic_and_type() {
        // /tf on three publishers, /scan on one; relative names and the legacy tf type both get normalized.
        let connections = vec![
            conn(0, "tf", "tf/tfMessage", TF_DEF),
            conn(1, "/tf", "tf2_msgs/TFMessage", TF_DEF),
            conn(2, "/tf", "tf2_msgs/TFMessage", TF_DEF),
            conn(3, "scan", "sensor_msgs/LaserScan", SCAN_DEF),
        ];
        let registries = RegistrySet::build(&connections, None);
        let (rows, topic_conns, skipped) = build_graph(&connections, &registries, 7);
        assert!(skipped.is_empty());
        assert_eq!(rows.len(), 2);
        let tf = rows.iter().find(|r| r.name == "/tf").unwrap();
        assert_eq!(tf.ros_type, "tf2_msgs/msg/TFMessage");
        assert_eq!(tf.publisher_count, 3);
        assert_eq!(tf.type_hash, "bag7");
        assert!(tf.type_name_dds.is_empty());
        let scan = rows.iter().find(|r| r.name == "/scan").unwrap();
        assert_eq!(scan.ros_type, "sensor_msgs/msg/LaserScan");
        // All three /tf connections feed the one subscription.
        assert_eq!(topic_conns["/tf"], vec![0, 1, 2]);
        assert_eq!(topic_conns["/scan"], vec![3]);
    }

    #[test]
    fn t8_a_topic_survives_when_only_one_of_its_connections_is_broken() {
        let connections = vec![
            conn(0, "/scan", "sensor_msgs/LaserScan", SCAN_DEF),
            conn(1, "/scan", "sensor_msgs/LaserScan", "pkg/Missing dep"),
            conn(2, "/junk", "pkg/Junk", "pkg/Missing dep"),
        ];
        let registries = RegistrySet::build(&connections, None);
        let (rows, topic_conns, skipped) = build_graph(&connections, &registries, 0);
        // /scan is offered through its healthy connection; /junk has none left, so it is reported instead.
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "/scan");
        assert_eq!(rows[0].publisher_count, 1);
        assert_eq!(topic_conns["/scan"], vec![0]);
        assert_eq!(skipped, vec!["/junk (pkg/Junk)".to_owned()]);
    }

    #[test]
    fn rows_are_sorted_so_the_add_dialog_order_is_stable() {
        let connections = vec![
            conn(0, "/scan", "sensor_msgs/LaserScan", SCAN_DEF),
            conn(1, "/map", "sensor_msgs/LaserScan", SCAN_DEF),
            conn(2, "/odom", "sensor_msgs/LaserScan", SCAN_DEF),
        ];
        let registries = RegistrySet::build(&connections, None);
        let (rows, _, _) = build_graph(&connections, &registries, 0);
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["/map", "/odom", "/scan"]);
    }
}
