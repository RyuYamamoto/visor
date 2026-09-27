//! zenoh connection, subscription management, and feeding crossbeam channels (zenoh receive runs on tokio tasks, UI on the main thread).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;
use tokio::sync::mpsc;
use zenoh::Session;

use super::CommConfig;
use super::discovery::run_discovery;
use super::keyexpr::{
    COMPATIBLE_SUBSCRIPTION_QOS, VIEWER_NODE_NAME, build_subscription_token, build_topic_keyexpr,
};
use crate::decode::cdr::{DecodeError, decode_message};
use crate::decode::msg_parser::TypeRegistry;
use crate::decode::value::Value;
use crate::tf::buffer::{TfUpdate, transforms_from_value};

/// Epoch stamped on every live message; playback bumps its own epoch past this, and live never rewinds (plan §5.4).
pub const LIVE_EPOCH: u64 = 0;

/// Retry interval after a failed connection.
const RETRY_INTERVAL: Duration = Duration::from_secs(3);
/// Minimum interval for the display-item path (matches View3d's 33ms repaint).
const DISPLAY_MIN_INTERVAL: Duration = Duration::from_millis(33);
/// Leading bytes attached to the UI on decode failure (avoids sending full PointCloud2-sized payloads).
const PAYLOAD_HEAD_LEN: usize = 256;
/// Timeout for the transient_local history get (`@adv/**`) (same as probe's liveliness get).
const HISTORY_GET_TIMEOUT: Duration = Duration::from_secs(3);
/// nid of the single node we advertise in subscriber liveliness tokens (only eid increments per subscription).
const VIEWER_NID: u64 = 0;

/// Monotonic counter assigning subscriber-token eids, unique within a session.
static SUB_EID: AtomicU64 = AtomicU64::new(0);

/// Declare a subscriber (MS) liveliness token when a subscription starts (works around the publisher's subscriber-count guard).
async fn declare_subscription_token(
    session: &Session,
    domain_id: u32,
    row: &TopicRow,
) -> Option<zenoh::liveliness::LivelinessToken> {
    let key = build_subscription_token(
        domain_id,
        &session.zid().to_string(),
        VIEWER_NID,
        SUB_EID.fetch_add(1, Ordering::Relaxed),
        VIEWER_NODE_NAME,
        &row.name,
        &row.type_name_dds,
        &row.type_hash,
        COMPATIBLE_SUBSCRIPTION_QOS,
    );
    match session.liveliness().declare_token(&key).await {
        Ok(token) => Some(token),
        Err(e) => {
            eprintln!("visor: failed to declare subscription token `{key}`: {e}");
            None
        }
    }
}

/// Closure the comm side uses to request a UI repaint (backed by `egui::Context::request_repaint`).
pub type Notify = Arc<dyn Fn() + Send + Sync>;

/// Connection status (for the status bar).
#[derive(Debug, Clone)]
pub enum ConnectionStatus {
    Connecting,
    Connected { zid: String },
    Failed { error: String, retry_in: Duration },
}

/// One row of a discovery snapshot (aggregated by topic x type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicRow {
    pub name: String,
    /// ROS type name (falls back to the DDS name if conversion fails).
    pub ros_type: String,
    /// DDS mangled type name used to build the subscription key.
    pub type_name_dds: String,
    pub type_hash: String,
    pub publisher_count: usize,
    pub subscriber_count: usize,
}

/// Decode-failure details (leading fragment for hex fallback display, plus full length).
#[derive(Debug)]
pub struct DecodeFailure {
    pub error: DecodeError,
    pub payload_head: Vec<u8>,
    pub payload_len: usize,
}

/// One decoded message passed from a source to the UI.
#[derive(Debug)]
pub struct TopicMessage {
    pub topic: String,
    pub result: Result<Value, DecodeFailure>,
    pub received_at: Instant,
    /// Which playback generation produced this; the UI drops anything older than the epoch it has applied.
    pub epoch: u64,
}

enum Command {
    SubscribeDisplay {
        row: TopicRow,
        counter: Arc<AtomicU64>,
    },
    UnsubscribeDisplay {
        topic: String,
    },
    SubscribeTf {
        row: TopicRow,
        is_static: bool,
    },
    UnsubscribeTf {
        is_static: bool,
    },
    RefreshTfStatic,
}

/// UI-side handle to the zenoh source. Dropping it ends comm_task, which aborts every subscription it spawned.
pub struct LiveHandle {
    cmd_tx: mpsc::UnboundedSender<Command>,
}

/// All comm_task to UI send channels, grouped to avoid argument bloat.
pub struct UiSenders {
    pub conn_tx: Sender<ConnectionStatus>,
    pub graph_tx: Sender<Vec<TopicRow>>,
    pub display_tx: Sender<TopicMessage>,
    pub tf_tx: Sender<TfUpdate>,
}

/// Destination for a subscribe task's decoded messages (display items; throttled to the display update interval).
struct DataSink {
    tx: Sender<TopicMessage>,
    min_interval: Duration,
}

impl LiveHandle {
    /// Run the comm task as a resident on the shared runtime; the Source owns both the runtime and the channels.
    pub fn spawn(
        handle: &tokio::runtime::Handle,
        config: CommConfig,
        senders: UiSenders,
        notify: Notify,
        types: Arc<TypeRegistry>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        handle.spawn(comm_task(config, senders, cmd_rx, notify, types));
        Self { cmd_tx }
    }

    /// Start a subscription for a display item (raw_view/click subscriptions were removed).
    pub fn subscribe_display(&self, row: &TopicRow) -> Arc<AtomicU64> {
        let counter = Arc::new(AtomicU64::new(0));
        let _ = self.cmd_tx.send(Command::SubscribeDisplay {
            row: row.clone(),
            counter: counter.clone(),
        });
        counter
    }

    pub fn unsubscribe_display(&self, topic: &str) {
        let _ = self.cmd_tx.send(Command::UnsubscribeDisplay {
            topic: topic.to_owned(),
        });
    }

    /// Start a persistent subscription to /tf and /tf_static (managed independently of display subscriptions).
    pub fn subscribe_tf(&self, row: &TopicRow, is_static: bool) {
        let _ = self.cmd_tx.send(Command::SubscribeTf {
            row: row.clone(),
            is_static,
        });
    }

    /// Stop one kind of TF subscription (the user pinned another topic, or none).
    pub fn unsubscribe_tf(&self, is_static: bool) {
        let _ = self.cmd_tx.send(Command::UnsubscribeTf { is_static });
    }

    /// Re-run the /tf_static history get (when a publisher increase signals a late publisher).
    pub fn refresh_tf_static(&self) {
        let _ = self.cmd_tx.send(Command::RefreshTfStatic);
    }
}

impl crate::source::SourceBackend for LiveHandle {
    fn subscribe_display(&self, row: &TopicRow) -> Arc<AtomicU64> {
        LiveHandle::subscribe_display(self, row)
    }

    fn unsubscribe_display(&self, topic: &str) {
        LiveHandle::unsubscribe_display(self, topic);
    }

    fn subscribe_tf(&self, row: &TopicRow, is_static: bool) {
        LiveHandle::subscribe_tf(self, row, is_static);
    }

    fn unsubscribe_tf(&self, is_static: bool) {
        LiveHandle::unsubscribe_tf(self, is_static);
    }

    fn refresh_tf_static(&self) {
        LiveHandle::refresh_tf_static(self);
    }
}

async fn comm_task(
    config: CommConfig,
    senders: UiSenders,
    mut cmd_rx: mpsc::UnboundedReceiver<Command>,
    notify: Notify,
    registry: Arc<TypeRegistry>,
) {
    // Commands that arrive while connecting; replayed once the session is up.
    let mut pending: std::collections::VecDeque<Command> = std::collections::VecDeque::new();
    // Retrying forever would keep this source alive after the UI switched to a bag, and it would then open a
    // session the moment the router appears. Racing the connect against the command channel closing prevents that.
    let session = tokio::select! {
        session = connect_with_retry(&config, &senders.conn_tx, &notify) => session,
        () = drain_until_closed(&mut cmd_rx, &mut pending) => return,
    };
    let discovery = tokio::spawn(run_discovery(
        session.clone(),
        config.domain_id,
        senders.graph_tx.clone(),
        notify.clone(),
    ));
    let tf_tx = senders.tf_tx.clone();
    // Start a subscription, replacing any existing task for the same topic.
    let spawn_subscribe = |subs: &mut HashMap<String, tokio::task::JoinHandle<()>>,
                           row: TopicRow,
                           counter: Arc<AtomicU64>,
                           sink: DataSink| {
        if let Some(handle) = subs.remove(&row.name) {
            handle.abort();
        }
        let topic = row.name.clone();
        let handle = tokio::spawn(subscribe_task(
            session.clone(),
            config.domain_id,
            row,
            counter,
            registry.clone(),
            sink,
            notify.clone(),
        ));
        subs.insert(topic, handle);
    };
    // Display-item subscriptions (raw_view/click subscriptions were removed; use the ROS 2 CLI for echo/hz).
    let mut display_subs: HashMap<String, tokio::task::JoinHandle<()>> = HashMap::new();
    // Persistent TF subscriptions live in fields separate from display subs, structurally decoupled from subscribe/unsubscribe.
    let mut tf_dyn: Option<tokio::task::JoinHandle<()>> = None;
    let mut tf_static: Option<(tokio::task::JoinHandle<()>, TopicRow)> = None;
    loop {
        let cmd = match pending.pop_front() {
            Some(cmd) => cmd,
            None => match cmd_rx.recv().await {
                Some(cmd) => cmd,
                None => break,
            },
        };
        match cmd {
            Command::SubscribeDisplay { row, counter } => {
                let sink = DataSink {
                    tx: senders.display_tx.clone(),
                    min_interval: DISPLAY_MIN_INTERVAL,
                };
                spawn_subscribe(&mut display_subs, row, counter, sink);
            }
            Command::UnsubscribeDisplay { topic } => {
                if let Some(handle) = display_subs.remove(&topic) {
                    handle.abort();
                }
            }
            Command::SubscribeTf { row, is_static } => {
                let handle = tokio::spawn(subscribe_tf_task(
                    session.clone(),
                    config.domain_id,
                    row.clone(),
                    is_static,
                    registry.clone(),
                    tf_tx.clone(),
                    notify.clone(),
                ));
                if is_static {
                    if let Some((old, _)) = tf_static.replace((handle, row)) {
                        old.abort();
                    }
                } else if let Some(old) = tf_dyn.replace(handle) {
                    old.abort();
                }
            }
            Command::UnsubscribeTf { is_static } => {
                if is_static {
                    if let Some((old, _)) = tf_static.take() {
                        old.abort();
                    }
                } else if let Some(old) = tf_dyn.take() {
                    old.abort();
                }
            }
            Command::RefreshTfStatic => {
                if let Some((_, row)) = tf_static.as_ref() {
                    let key = build_topic_keyexpr(
                        config.domain_id,
                        &row.name,
                        &row.type_name_dds,
                        &row.type_hash,
                    );
                    tokio::spawn(query_tf_static_history(
                        session.clone(),
                        key,
                        row.ros_type.clone(),
                        registry.clone(),
                        tf_tx.clone(),
                        notify.clone(),
                    ));
                }
            }
        }
    }
    // The runtime outlives this task now, so spawned subscriptions would otherwise stay alive and keep the session (and its liveliness tokens) open after a source switch.
    discovery.abort();
    for handle in display_subs.into_values() {
        handle.abort();
    }
    if let Some(handle) = tf_dyn {
        handle.abort();
    }
    if let Some((handle, _)) = tf_static {
        handle.abort();
    }
    drop(session);
}

/// Buffers commands until the UI drops this source; returns only on closure, so it is the losing side of the connect race.
async fn drain_until_closed(
    cmd_rx: &mut mpsc::UnboundedReceiver<Command>,
    pending: &mut std::collections::VecDeque<Command>,
) {
    while let Some(cmd) = cmd_rx.recv().await {
        pending.push_back(cmd);
    }
}

/// Persistent TF subscribe task. Unlike raw_view, forwards every sample to the tf channel with no throttling or dropping.
async fn subscribe_tf_task(
    session: Session,
    domain_id: u32,
    row: TopicRow,
    is_static: bool,
    registry: Arc<TypeRegistry>,
    tf_tx: Sender<TfUpdate>,
    notify: Notify,
) {
    let key = build_topic_keyexpr(domain_id, &row.name, &row.type_name_dds, &row.type_hash);
    let sub = match session.declare_subscriber(&key).await {
        Ok(sub) => sub,
        Err(e) => {
            eprintln!("visor: failed to subscribe tf `{key}`: {e}");
            return;
        }
    };
    let _token = declare_subscription_token(&session, domain_id, &row).await;
    // Getting history after declaring the subscriber closes the miss window (duplicate receives are harmless since inserts are idempotent).
    if is_static {
        query_tf_static_history(
            session.clone(),
            key.clone(),
            row.ros_type.clone(),
            registry.clone(),
            tf_tx.clone(),
            notify.clone(),
        )
        .await;
    }
    // notify() paces itself to ~30fps UI-side (request_repaint_after), so forward every sample and let that coalesce.
    while let Ok(sample) = sub.recv_async().await {
        forward_tf_payload(
            &sample.payload().to_bytes(),
            &row.ros_type,
            is_static,
            &registry,
            &tf_tx,
        );
        notify();
    }
}

/// Fetch already-published static TF from the AdvancedPublisher cache (`<topic>/@adv/**` queryable).
async fn query_tf_static_history(
    session: Session,
    topic_key: String,
    ros_type: String,
    registry: Arc<TypeRegistry>,
    tf_tx: Sender<TfUpdate>,
    notify: Notify,
) {
    let selector = format!("{topic_key}/@adv/**");
    // Reply keyexpr doesn't intersect the selector so Any is required; None consolidation to receive all publishers' replies.
    let replies = match session
        .get(&selector)
        .consolidation(zenoh::query::ConsolidationMode::None)
        .accept_replies(zenoh::query::ReplyKeyExpr::Any)
        .timeout(HISTORY_GET_TIMEOUT)
        .await
    {
        Ok(replies) => replies,
        Err(e) => {
            eprintln!("visor: tf_static history get `{selector}` failed: {e}");
            return;
        }
    };
    let mut received = false;
    while let Ok(reply) = replies.recv_async().await {
        match reply.result() {
            Ok(sample) => {
                forward_tf_payload(
                    &sample.payload().to_bytes(),
                    &ros_type,
                    true,
                    &registry,
                    &tf_tx,
                );
                received = true;
            }
            Err(e) => eprintln!("visor: tf_static history reply error: {e:?}"),
        }
    }
    // Static TF arrives once in a burst; repaint after draining so the view reflects it (no per-sample notify needed).
    if received {
        notify();
    }
}

/// Convert a CDR payload to a TfUpdate and send it to the tf channel (skip per-message on failure).
fn forward_tf_payload(
    payload: &[u8],
    ros_type: &str,
    is_static: bool,
    registry: &TypeRegistry,
    tf_tx: &Sender<TfUpdate>,
) {
    let value = match decode_message(registry, ros_type, payload) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("visor: failed to decode tf message: {e}");
            return;
        }
    };
    match transforms_from_value(&value) {
        Ok(transforms) => {
            let _ = tf_tx.send(TfUpdate {
                transforms,
                is_static,
                epoch: LIVE_EPOCH,
            });
        }
        Err(e) => eprintln!("visor: {e}"),
    }
}

/// The client default is fail-fast (timeout_ms: 0), so retry ourselves every 3s until it succeeds.
async fn connect_with_retry(
    config: &CommConfig,
    conn_tx: &Sender<ConnectionStatus>,
    notify: &Notify,
) -> Session {
    loop {
        send_status(conn_tx, notify, ConnectionStatus::Connecting);
        match open_session(config).await {
            Ok(session) => {
                send_status(
                    conn_tx,
                    notify,
                    ConnectionStatus::Connected {
                        zid: session.zid().to_string(),
                    },
                );
                return session;
            }
            Err(e) => {
                send_status(
                    conn_tx,
                    notify,
                    ConnectionStatus::Failed {
                        error: e.to_string(),
                        retry_in: RETRY_INTERVAL,
                    },
                );
                tokio::time::sleep(RETRY_INTERVAL).await;
            }
        }
    }
}

/// Same single direct-router-connection setup as probe (client / explicit endpoints / multicast scouting disabled).
async fn open_session(
    config: &CommConfig,
) -> Result<Session, Box<dyn std::error::Error + Send + Sync>> {
    let mut zc = zenoh::Config::default();
    zc.insert_json5("mode", r#""client""#)?;
    zc.insert_json5("connect/endpoints", &format!(r#"["{}"]"#, config.endpoint))?;
    zc.insert_json5("scouting/multicast/enabled", "false")?;
    zenoh::open(zc).await
}

/// Drain and count every sample immediately, throttling only decode and UI send in time (latest value + minimum interval).
async fn subscribe_task(
    session: Session,
    domain_id: u32,
    row: TopicRow,
    counter: Arc<AtomicU64>,
    registry: Arc<TypeRegistry>,
    sink: DataSink,
    notify: Notify,
) {
    let key = build_topic_keyexpr(domain_id, &row.name, &row.type_name_dds, &row.type_hash);
    let sub = match session.declare_subscriber(&key).await {
        Ok(sub) => sub,
        Err(e) => {
            eprintln!("visor: failed to subscribe `{key}`: {e}");
            return;
        }
    };
    let _token = declare_subscription_token(&session, domain_id, &row).await;
    // Fetch already-published transient_local messages (harmless for volatile, which just yields no reply).
    let mut history = match session
        .get(format!("{key}/@adv/**"))
        .consolidation(zenoh::query::ConsolidationMode::None)
        .accept_replies(zenoh::query::ReplyKeyExpr::Any)
        .timeout(HISTORY_GET_TIMEOUT)
        .await
    {
        Ok(replies) => Some(replies),
        Err(e) => {
            eprintln!("visor: history get `{key}/@adv/**` failed: {e}");
            None
        }
    };
    let mut pending: Option<(zenoh::sample::Sample, Instant)> = None;
    let mut next_flush = tokio::time::Instant::now();
    loop {
        tokio::select! {
            received = sub.recv_async() => match received {
                Ok(sample) => {
                    counter.fetch_add(1, Ordering::Relaxed);
                    pending = Some((sample, Instant::now()));
                }
                Err(_) => break,
            },
            // Route history replies through the same last-wins pending path as live samples; disable this arm when the channel ends.
            reply = async { history.as_ref().expect("guarded by is_some").recv_async().await }, if history.is_some() => {
                match reply {
                    Ok(reply) => match reply.into_result() {
                        Ok(sample) => {
                            counter.fetch_add(1, Ordering::Relaxed);
                            pending = Some((sample, Instant::now()));
                        }
                        Err(e) => eprintln!("visor: history reply error on `{key}`: {e:?}"),
                    },
                    Err(_) => history = None,
                }
            },
            _ = tokio::time::sleep_until(next_flush), if pending.is_some() => {
                let (sample, received_at) = pending.take().expect("guarded by is_some");
                let payload = sample.payload().to_bytes();
                let result = decode_message(&registry, &row.ros_type, &payload).map_err(|error| {
                    DecodeFailure {
                        error,
                        payload_head: payload[..payload.len().min(PAYLOAD_HEAD_LEN)].to_vec(),
                        payload_len: payload.len(),
                    }
                });
                let _ = sink.tx.try_send(TopicMessage {
                    topic: row.name.clone(),
                    result,
                    received_at,
                    epoch: LIVE_EPOCH,
                });
                notify();
                next_flush = tokio::time::Instant::now() + sink.min_interval;
            }
        }
    }
}

fn send_status(conn_tx: &Sender<ConnectionStatus>, notify: &Notify, status: ConnectionStatus) {
    let _ = conn_tx.send(status);
    notify();
}
