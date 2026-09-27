//! ROS 1 / ROS 2 bag inspection CLI (no GUI): topics, timing, per-stage timings, first-message decode, index verification, seek latency.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use visor::bag::index::MessageIndex;
use visor::bag::msgdef::RegistrySet;
use visor::bag::naming::{normalize_topic, ros1_type_to_ros2, short_type};
use visor::bag::set::BagSet;
use visor::bag::storage::OpenStorage;
use visor::bag::{BagEvent, BagHandle, PlayerChannels, reader, rosbag2};
use visor::comm::session::Notify;
use visor::decode::msg_parser::TypeRegistry;
use visor::decode::value::{FormatLimits, format_compact};
use visor::plugin::Registry;
use visor::source::launch::expand_directories;
use visor::tf::buffer::TimeNs;

fn usage() -> &'static str {
    "usage: baginfo <bag> [<bag>…] [--merge] [--decode] [--verify-index] [--play <secs>] [--help]\n\
     \n\
     <bag> is a ROS 1 .bag, or a ROS 2 bag: its directory, its metadata.yaml, or .mcap / .db3 files\n\
     \n\
     options:\n\
       --merge           treat every bag as one set merged by record time (as playback does)\n\
       --decode          decode the first message of every topic and print it\n\
       --verify-index     read every indexed message and check op/conn/time against the index\n\
       --play <secs>     run the real player for <secs>: playback rate, seek latency, message counts\n\
       --help            show this help\n\
     \n\
     ROS 2 bags fall back to the bundled .msg definitions (plus VISOR_MSG_PATHS) for types the bag does not\n\
     describe; definitions from plugins linked into the GUI binary are not available here"
}

/// What the user asked for.
struct Args {
    paths: Vec<String>,
    /// Report every bag as one merged set instead of one report per file.
    merge: bool,
    decode: bool,
    verify: bool,
    /// Seconds to exercise the player for; None skips the playback check.
    play: Option<f64>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        paths: Vec::new(),
        merge: false,
        decode: false,
        verify: false,
        play: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            "--merge" => args.merge = true,
            "--decode" => args.decode = true,
            "--verify-index" => args.verify = true,
            "--play" => {
                let value = it.next().ok_or("--play requires a value")?;
                args.play = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --play seconds `{value}`"))?,
                );
            }
            other if other.starts_with('-') => return Err(format!("unknown argument `{other}`")),
            other => args.paths.push(other.to_owned()),
        }
    }
    if args.paths.is_empty() {
        return Err("a bag path is required".to_owned());
    }
    Ok(args)
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("baginfo: {e}");
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    };
    // The same merged registry the GUI decodes live traffic with (bundled + VISOR_MSG_PATHS); plugin `.msg` files are linked into apps/visor only.
    let (fallback, problems) = Registry::builtin().build_type_registry();
    for problem in &problems {
        eprintln!("baginfo: {problem}");
    }
    let mut failed = false;
    let groups: Vec<Vec<PathBuf>> = if args.merge {
        vec![args.paths.iter().map(PathBuf::from).collect()]
    } else {
        args.paths.iter().map(|p| vec![PathBuf::from(p)]).collect()
    };
    for paths in &groups {
        println!("== {}", describe(paths));
        if let Err(e) = report(paths, &args, &fallback) {
            eprintln!("baginfo: {e}");
            failed = true;
        }
        println!();
    }
    if failed {
        std::process::exit(1);
    }
}

/// How a group of paths is named in the report header.
fn describe(paths: &[PathBuf]) -> String {
    match paths {
        [one] => one.display().to_string(),
        many => format!(
            "{} bags merged ({} … {})",
            many.len(),
            many[0].display(),
            many[many.len() - 1].display()
        ),
    }
}

/// Which storage family a group belongs to, decided the way the GUI decides it: directories and metadata.yaml expand first, then the first file's extension picks the opener.
struct Opened {
    files: Vec<PathBuf>,
    open: OpenStorage,
    /// None for ROS 1 bags, whose own definitions stay the only authority (as in the GUI).
    fallback: Option<Arc<TypeRegistry>>,
}

fn resolve(paths: &[PathBuf], fallback: &Arc<TypeRegistry>) -> Result<Opened, String> {
    let files = expand_directories(paths.to_vec())?;
    let ext = files
        .first()
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if ext == "bag" {
        return Ok(Opened {
            files,
            open: reader::open_storage,
            fallback: None,
        });
    }
    let files = rosbag2::expand_paths(&files).map_err(|e| e.to_string())?;
    Ok(Opened {
        files,
        open: rosbag2::open_storage,
        fallback: Some(Arc::clone(fallback)),
    })
}

/// Inspect a bag (or a merged set) and print what was found.
fn report(paths: &[PathBuf], args: &Args, fallback: &Arc<TypeRegistry>) -> Result<(), String> {
    let opened = resolve(paths, fallback)?;
    let cancel = AtomicBool::new(false);
    let open_at = Instant::now();
    let mut bags = BagSet::open(&opened.files, opened.open, &cancel).map_err(|e| e.to_string())?;
    let open_ms = open_at.elapsed().as_secs_f64() * 1e3;

    let defs_at = Instant::now();
    let registries = RegistrySet::build(bags.connections(), opened.fallback.as_ref());
    let defs_ms = defs_at.elapsed().as_secs_f64() * 1e3;

    let index_at = Instant::now();
    let entries = bags
        .read_message_index(&cancel)
        .map_err(|e| e.to_string())?;
    let index = MessageIndex::build(entries);
    let index_ms = index_at.elapsed().as_secs_f64() * 1e3;

    println!(
        "size {:.2} MB  files {}  chunks {}  connections {}  messages {}  compression {}",
        bags.size_bytes() as f64 / 1e6,
        bags.len(),
        bags.chunk_count(),
        bags.connections().len(),
        bags.message_count(),
        bags.compression()
    );
    println!(
        "time {}..{}  ({:.2}s)",
        stamp(index.start()),
        stamp(index.end()),
        (index.end() - index.start()) as f64 / 1e9
    );
    println!(
        "open {open_ms:.1} ms  definitions {defs_ms:.1} ms ({} distinct)  index {index_ms:.1} ms ({} entries, {:.2} MB)",
        registries.distinct_registries(),
        index.len(),
        index.memory_bytes() as f64 / 1e6
    );
    for warning in bags.warnings() {
        println!("warning: {warning}");
    }

    // One line per (topic, type), the same aggregation the viewer's topic list uses.
    let mut rows: Vec<TopicRow> = Vec::new();
    for conn in bags.connections() {
        let topic = normalize_topic(&conn.topic_raw);
        let ros_type = ros1_type_to_ros2(&conn.type_raw);
        let count = index.count(conn.id);
        let supported = registries.get(conn.id).is_some();
        match rows
            .iter_mut()
            .find(|(t, ty, _, _, _)| *t == topic && *ty == ros_type)
        {
            Some(row) => {
                row.2 += count;
                row.3 += 1;
                if row.4.is_none() && supported {
                    row.4 = Some(conn.id);
                }
            }
            None => rows.push((topic, ros_type, count, 1, supported.then_some(conn.id))),
        }
    }
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    println!("{} topics:", rows.len());
    for (topic, ros_type, count, conns, first) in &rows {
        let mark = if first.is_some() { ' ' } else { '!' };
        println!(
            "{mark} {count:>8}  {topic:<44} {:<28} conns {conns}",
            short_type(ros_type)
        );
    }
    let notices = registries.fallback_notices();
    if !notices.is_empty() {
        println!(
            "{} connection(s) decoded with bundled definitions:",
            notices.len()
        );
        for (id, why) in notices {
            println!("  conn {id}: {why}");
        }
    }
    let unsupported = registries.unsupported();
    if !unsupported.is_empty() {
        println!(
            "{} connection(s) with unreadable definitions:",
            unsupported.len()
        );
        for (id, why) in unsupported {
            println!("  conn {id}: {why}");
        }
    }

    if args.decode {
        println!("first message per topic:");
        let limits = FormatLimits::default();
        for (topic, _, _, _, first) in &rows {
            let Some(conn) = first else { continue };
            let Some(types) = registries.get(*conn) else {
                continue;
            };
            let types = types.clone();
            let Some(entry) = index.at_or_before(*conn, index.end()).copied() else {
                continue;
            };
            let first_entry = index
                .range(*conn, index.start() - 1, index.end())
                .next()
                .copied()
                .unwrap_or(entry);
            let payload = match bags.message_at(&first_entry) {
                Ok(msg) => msg.data.to_vec(),
                Err(e) => {
                    println!("  {topic}: read failed: {e}");
                    continue;
                }
            };
            match types.decode(&payload) {
                Ok(value) => println!(
                    "  {topic} ({} B): {}",
                    payload.len(),
                    format_compact(&value, &limits)
                ),
                Err(e) => println!("  {topic}: decode failed: {e}"),
            }
        }
    }

    if args.verify {
        verify_index(&mut bags, &index)?;
    }
    if let Some(secs) = args.play {
        play_check(&opened, &rows, secs);
    }
    Ok(())
}

/// Drive the real player (the same thread the GUI uses) and report playback rate, seek latency and delivery counts.
fn play_check(opened: &Opened, rows: &[TopicRow], secs: f64) {
    let (graph_tx, graph_rx) = crossbeam_channel::unbounded();
    let (display_tx, display_rx) = crossbeam_channel::bounded(32);
    let (tf_tx, tf_rx) = crossbeam_channel::unbounded();
    let (bag_tx, bag_rx) = crossbeam_channel::unbounded();
    let notify: Notify = Arc::new(|| {});
    let handle = BagHandle::spawn(
        &opened.files,
        1,
        PlayerChannels {
            graph_tx,
            display_tx,
            tf_tx,
            bag_tx,
        },
        notify,
        opened.open,
        opened.fallback.clone(),
    );
    // Wait for the graph, which the player sends once the bag is open.
    let topics = match graph_rx.recv_timeout(Duration::from_secs(10)) {
        Ok(topics) => topics,
        Err(_) => {
            println!("play: the player never published a topic list");
            return;
        }
    };
    let mut info = None;
    for event in bag_rx.try_iter() {
        match event {
            BagEvent::Opened(opened) => info = Some(*opened),
            BagEvent::Notice(notice) => println!("play: notice: {notice}"),
            _ => {}
        }
    }
    let Some(info) = info else {
        println!("play: no Opened event");
        return;
    };
    // Subscribe to what the viewer would: TF plus the three busiest drawable topics.
    for name in ["/tf", "/tf_static"] {
        if let Some(row) = topics.iter().find(|r| r.name == name) {
            handle.subscribe_tf(&row.name, name == "/tf_static");
        }
    }
    let subscribed: Vec<String> = rows
        .iter()
        .filter(|(topic, _, _, _, ok)| ok.is_some() && topic != "/tf" && topic != "/tf_static")
        .take(3)
        .map(|(topic, _, _, _, _)| topic.clone())
        .collect();
    for topic in &subscribed {
        handle.subscribe_display(topic);
    }
    println!("play: subscribed to {}", subscribed.join(", "));

    let started = Instant::now();
    handle.play();
    let mut messages = 0usize;
    let mut transforms = 0usize;
    let mut playhead = info.start;
    let mut epoch = 0;
    while started.elapsed().as_secs_f64() < secs {
        crossbeam_channel::select! {
            recv(display_rx) -> msg => if let Ok(msg) = msg {
                messages += 1;
                if msg.result.is_err() {
                    println!("play: decode failure on {}", msg.topic);
                }
            },
            recv(tf_rx) -> update => if let Ok(update) = update {
                transforms += update.transforms.len();
            },
            recv(bag_rx) -> event => if let Ok(BagEvent::Status(status)) = event {
                playhead = status.playhead;
                epoch = status.epoch;
            },
            default(Duration::from_millis(50)) => {}
        }
    }
    let wall = started.elapsed().as_secs_f64();
    let advanced = (playhead - info.start) as f64 / 1e9;
    println!(
        "play: {advanced:.2}s of bag in {wall:.2}s wall (rate ×{:.2}), {messages} messages, {transforms} transforms",
        advanced / wall
    );
    // A faster rate must actually deliver faster, not just move the playhead (FR-5).
    for speed in [5.0f32, 0.25] {
        handle.set_speed(speed);
        // Let the new speed take effect and take the baseline from a status produced under it, or the tail of the previous (faster) phase would be counted against this one.
        std::thread::sleep(Duration::from_millis(60));
        for event in bag_rx.try_iter() {
            if let BagEvent::Status(status) = event {
                playhead = status.playhead;
            }
        }
        let from = playhead;
        let at = Instant::now();
        while at.elapsed().as_secs_f64() < 2.0 {
            crossbeam_channel::select! {
                recv(display_rx) -> _ => {},
                recv(tf_rx) -> _ => {},
                recv(bag_rx) -> event => if let Ok(BagEvent::Status(status)) = event {
                    playhead = status.playhead;
                    epoch = status.epoch;
                },
                default(Duration::from_millis(20)) => {}
            }
        }
        println!(
            "play: at ×{speed} the playhead advanced {:.2}s in {:.2}s wall (rate ×{:.2})",
            (playhead - from) as f64 / 1e9,
            at.elapsed().as_secs_f64(),
            (playhead - from) as f64 / 1e9 / at.elapsed().as_secs_f64()
        );
    }
    handle.set_speed(1.0);
    handle.pause();

    // Seek latency at ten points spread over the bag. The chunk cache is not cleared between points (the player has no such command), so a point landing in a cached chunk is a warm read; the spread makes misses likely on a long bag, not certain. "complete" is the Status whose epoch advanced: `Player::seek` emits it after the static cache, the TF backfill and every snapshot went out, so it marks the end of the whole seek. The first display message is reported alongside.
    let mut completions: Vec<f64> = Vec::new();
    for point in 0..10 {
        let fraction = 0.05 + 0.1 * f64::from(point);
        let offset = info.duration_secs() * fraction;
        let target = info.time_at_offset(offset);
        while display_rx.try_recv().is_ok() {}
        while tf_rx.try_recv().is_ok() {}
        while bag_rx.try_recv().is_ok() {}
        let at = Instant::now();
        handle.seek(target);
        let mut first = None;
        let mut complete = None;
        let mut count = 0;
        while at.elapsed() < Duration::from_secs(2) && complete.is_none() {
            crossbeam_channel::select! {
                recv(display_rx) -> msg => if msg.is_ok() {
                    first = first.or(Some(at.elapsed()));
                    count += 1;
                },
                recv(tf_rx) -> _ => {},
                recv(bag_rx) -> event => if let Ok(BagEvent::Status(status)) = event
                    && status.epoch > epoch
                {
                    epoch = status.epoch;
                    complete = Some(at.elapsed());
                },
                default(Duration::from_millis(20)) => {}
            }
        }
        let first_text = match first {
            Some(latency) => format!("first message {:.0} ms", latency.as_secs_f64() * 1e3),
            None => "nothing subscribed has a message at or before it".to_owned(),
        };
        match complete {
            Some(latency) => {
                completions.push(latency.as_secs_f64() * 1e3);
                println!(
                    "play: seek to {offset:.1}s ({:.0}%): complete in {:.0} ms, {first_text}, {count} messages (epoch {epoch})",
                    fraction * 100.0,
                    latency.as_secs_f64() * 1e3
                );
            }
            None => println!(
                "play: seek to {offset:.1}s ({:.0}%): EPOCH NOT BUMPED within 2 s, {first_text}",
                fraction * 100.0
            ),
        }
    }
    if !completions.is_empty() {
        let mut sorted = completions.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "play: seek complete over {} points (chunk cache kept between them): max {:.0} ms, median {:.0} ms",
            sorted.len(),
            sorted[sorted.len() - 1],
            sorted[sorted.len() / 2]
        );
    }

    // Message stepping is the finest positioning available, and it must not depend on the bar's pixel width.
    {
        handle.seek(info.time_at_offset(info.duration_secs() * 0.5));
        std::thread::sleep(Duration::from_millis(150));
        let mut marks = Vec::new();
        for _ in 0..4 {
            handle.step();
            std::thread::sleep(Duration::from_millis(120));
            for event in bag_rx.try_iter() {
                if let BagEvent::Status(status) = event {
                    playhead = status.playhead;
                    epoch = status.epoch;
                }
            }
            marks.push(playhead);
        }
        let deltas: Vec<String> = marks
            .windows(2)
            .map(|w| format!("{:.4}", (w[1] - w[0]) as f64 / 1e9))
            .collect();
        println!("play: forward steps advanced by {} s", deltas.join(", "));
        let before = playhead;
        handle.step_back();
        std::thread::sleep(Duration::from_millis(150));
        for event in bag_rx.try_iter() {
            if let BagEvent::Status(status) = event {
                playhead = status.playhead;
                epoch = status.epoch;
            }
        }
        println!(
            "play: one step back moved {:.4} s (epoch now {epoch})",
            (before - playhead) as f64 / 1e9
        );
    }

    // Looping must wrap to the start and bump the epoch, so accumulated renderer state does not carry over.
    handle.set_loop(true);
    handle.seek(info.time_at_offset((info.duration_secs() - 0.5).max(0.0)));
    handle.play();
    let before = epoch;
    let at = Instant::now();
    let mut wrapped = None;
    while at.elapsed() < Duration::from_secs(3) {
        for event in bag_rx.try_iter() {
            if let BagEvent::Status(status) = event
                && status.epoch > before + 1
                && status.playhead < info.time_at_offset(1.0)
            {
                wrapped = wrapped.or(Some((status.epoch, at.elapsed())));
            }
        }
        while display_rx.try_recv().is_ok() {}
        while tf_rx.try_recv().is_ok() {}
        std::thread::sleep(Duration::from_millis(20));
    }
    match wrapped {
        Some((epoch, elapsed)) => println!(
            "play: loop wrapped to the start after {:.1}s with epoch {epoch}",
            elapsed.as_secs_f64()
        ),
        None => println!("play: loop did not wrap within 3s"),
    }
}

/// Read every indexed message and confirm the index's connection and time match the record itself (plan §3.1 V1–V3).
fn verify_index(bags: &mut BagSet, index: &MessageIndex) -> Result<(), String> {
    let started = Instant::now();
    let mut by_conn: HashMap<u32, Vec<visor::bag::reader::IndexEntry>> = HashMap::new();
    // The index only exposes per-connection views, which together cover every entry.
    for conn in bags.connections().iter().map(|c| c.id).collect::<Vec<_>>() {
        let entries: Vec<_> = index
            .range(conn, index.start() - 1, index.end())
            .copied()
            .collect();
        if !entries.is_empty() {
            by_conn.insert(conn, entries);
        }
    }
    let mut checked = 0usize;
    let mut bytes = 0usize;
    for (conn, entries) in &by_conn {
        for entry in entries {
            let message = bags.message_at(entry).map_err(|e| e.to_string())?;
            if message.conn != *conn {
                return Err(format!(
                    "index says conn {conn} at chunk {} offset {} but the record says {}",
                    entry.chunk, entry.offset, message.conn
                ));
            }
            if message.time != entry.time {
                return Err(format!(
                    "index says {} at chunk {} offset {} but the record says {}",
                    entry.time, entry.chunk, entry.offset, message.time
                ));
            }
            bytes += message.data.len();
            checked += 1;
        }
    }
    println!(
        "verify-index: {checked} messages ok ({:.2} MB payload) in {:.0} ms",
        bytes as f64 / 1e6,
        started.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}

/// One aggregated line: topic, ROS 2 type name, message count, connection count, and a decodable connection if any.
type TopicRow = (String, String, usize, usize, Option<u32>);

/// A record time as `secs.nanos`, matching how rosbag prints stamps.
fn stamp(time: TimeNs) -> String {
    format!(
        "{}.{:09}",
        time.div_euclid(1_000_000_000),
        time.rem_euclid(1_000_000_000)
    )
}
