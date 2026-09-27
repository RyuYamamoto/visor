//! zenoh connectivity-check CLI (router connect, graph enumeration via liveliness, hex dump of wildcard subscription).

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use visor::comm::CommConfig;
use visor::comm::keyexpr::{
    EntityKind, LivelinessToken, dds_type_to_ros, parse_liveliness_token, parse_topic_keyexpr,
};
use visor::decode::cdr::decode_message;
use visor::decode::msg_parser::TypeRegistry;
use visor::decode::value::{FormatLimits, format_compact};
use visor::plugin::registry::Registry;
use zenoh::Wait;
use zenoh::sample::{Sample, SampleKind};

const LIVELINESS_GET_TIMEOUT: Duration = Duration::from_secs(3);
const HEX_DUMP_LEN: usize = 32;
const THROTTLE_INTERVAL: Duration = Duration::from_secs(1);

fn usage() -> &'static str {
    "usage: probe [--endpoint <ep>] [--domain-id <n>] [--help]\n\
     \n\
     options:\n\
       --endpoint <ep>   zenoh router endpoint (default: env ZENOH_ENDPOINT or tcp/localhost:7447)\n\
       --domain-id <n>   ROS domain id (default: env ROS_DOMAIN_ID or 0)\n\
       --help            show this help\n\
     \n\
     the type registry is built the way the GUI builds it, so VISOR_MSG_PATHS roots are picked up here too"
}

fn parse_args() -> Result<CommConfig, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", usage());
        std::process::exit(0);
    }
    CommConfig::resolve(args, |key| std::env::var(key).ok())
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("probe: {e}\n{}", usage());
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("probe: error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &CommConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Same merge the GUI performs (builtin + VISOR_MSG_PATHS), so a custom type can be checked without the GUI.
    let (registry, problems) = Registry::builtin()
        .finish(&|key| std::env::var(key).ok())
        .build_type_registry();
    for problem in &problems {
        eprintln!("probe: {problem}");
    }
    println!("probe: type registry loaded ({} types)", registry.len());
    println!(
        "probe: connecting to {} (domain_id={}) ...",
        args.endpoint, args.domain_id
    );
    let mut config = zenoh::Config::default();
    config.insert_json5("mode", r#""client""#)?;
    config.insert_json5("connect/endpoints", &format!(r#"["{}"]"#, args.endpoint))?;
    config.insert_json5("scouting/multicast/enabled", "false")?;
    let session = zenoh::open(config).wait()?;
    println!("probe: connected. zid={}", session.zid());

    let liveliness_key = format!("@ros2_lv/{}/**", args.domain_id);
    let graph_sub = session
        .liveliness()
        .declare_subscriber(&liveliness_key)
        .wait()?;

    let mut seen: HashSet<String> = HashSet::new();
    let replies = session
        .liveliness()
        .get(&liveliness_key)
        .timeout(LIVELINESS_GET_TIMEOUT)
        .wait()?;
    println!("\n--- graph (initial) ---");
    while let Ok(reply) = replies.recv() {
        match reply.result() {
            Ok(sample) => {
                let key = sample.key_expr().as_str();
                if seen.insert(key.to_owned()) {
                    println!("{}", format_graph_entry(key));
                }
            }
            Err(e) => eprintln!("[graph] reply error: {:?}", e.payload()),
        }
    }
    println!("-----------------------\n");

    let data_key = format!("{}/**", args.domain_id);
    let data_sub = session.declare_subscriber(&data_key).wait()?;
    println!("probe: subscribed to `{liveliness_key}` and `{data_key}` (Ctrl-C to quit)\n");

    std::thread::spawn(move || {
        while let Ok(sample) = graph_sub.recv() {
            let key = sample.key_expr().as_str().to_owned();
            match sample.kind() {
                SampleKind::Put => {
                    if seen.insert(key.clone()) {
                        println!("[graph] + {}", format_graph_entry(&key));
                    }
                }
                SampleKind::Delete => {
                    if seen.remove(&key) {
                        println!("[graph] - {}", format_graph_entry(&key));
                    }
                }
            }
        }
    });

    let mut throttle: HashMap<String, (Instant, u64)> = HashMap::new();
    while let Ok(sample) = data_sub.recv() {
        print_data_sample(&sample, &mut throttle, &registry);
    }
    Ok(())
}

/// Format a liveliness token into a one-line graph entry (show the raw key on parse failure rather than dropping it).
fn format_graph_entry(key: &str) -> String {
    match parse_liveliness_token(key) {
        Ok(token) => format_token(&token),
        Err(e) => format!("[graph:unparsed] {key} ({e})"),
    }
}

fn format_token(token: &LivelinessToken) -> String {
    let label = match token.kind {
        EntityKind::Node => "node",
        EntityKind::Publisher => "pub ",
        EntityKind::Subscription => "sub ",
        EntityKind::Service => "srv ",
        EntityKind::Client => "cli ",
    };
    match &token.topic {
        None => format!(
            "{label}    {:<40} [zid={}, nid={}]",
            token.full_node_name(),
            token.zid,
            token.nid
        ),
        Some(t) => {
            let ros_type = dds_type_to_ros(&t.type_name).unwrap_or_else(|_| t.type_name.clone());
            format!(
                "{label}    {:<40} {:<30} {}  (node {})",
                t.name,
                ros_type,
                t.type_hash,
                token.full_node_name()
            )
        }
    }
}

/// Print received samples with throttling (one line per second per key, noting the suppressed count).
fn print_data_sample(
    sample: &Sample,
    throttle: &mut HashMap<String, (Instant, u64)>,
    registry: &TypeRegistry,
) {
    let key = sample.key_expr().as_str();
    let now = Instant::now();
    if let Some((last, suppressed)) = throttle.get_mut(key)
        && now.duration_since(*last) < THROTTLE_INTERVAL
    {
        *suppressed += 1;
        return;
    }
    let suppressed = match throttle.insert(key.to_owned(), (now, 0)) {
        Some((_, n)) => n,
        None => 0,
    };
    let payload = sample.payload().to_bytes();
    let suffix = if suppressed > 0 {
        format!(
            "  (+{suppressed} msgs suppressed in last {:.1}s)",
            THROTTLE_INTERVAL.as_secs_f32()
        )
    } else {
        String::new()
    };
    // Derive the ROS type name from the keyexpr; on successful decode, show one-line JSON-ish text instead of a hex dump.
    let ros_type = parse_topic_keyexpr(key)
        .ok()
        .and_then(|t| dds_type_to_ros(&t.type_name).ok());
    let topic_note = match parse_topic_keyexpr(key) {
        Ok(t) => format!("  topic={}", t.topic),
        Err(_) => "  [data:unparsed]".to_owned(),
    };
    println!("[data] {key}  {} bytes{topic_note}{suffix}", payload.len());
    match ros_type.map(|ty| decode_message(registry, &ty, &payload)) {
        Some(Ok(value)) => {
            println!(
                "       {}",
                format_compact(&value, &FormatLimits::default())
            );
        }
        Some(Err(e)) => {
            println!("       decode failed: {e}");
            println!("       {}", hex_dump(&payload, HEX_DUMP_LEN));
        }
        None => println!("       {}", hex_dump(&payload, HEX_DUMP_LEN)),
    }
}

fn hex_dump(bytes: &[u8], max_len: usize) -> String {
    let shown: Vec<String> = bytes
        .iter()
        .take(max_len)
        .map(|b| format!("{b:02x}"))
        .collect();
    let ellipsis = if bytes.len() > max_len { " ..." } else { "" };
    format!("{}{}", shown.join(" "), ellipsis)
}
