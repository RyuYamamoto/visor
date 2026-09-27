//! Regression tests over small bags written by rosbag2 0.26 (Jazzy) itself: mcap `zstd_fast`, a two-file split mcap, and sqlite3. Regeneration: tests/fixtures/rosbag2/gen_fixtures.py.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use visor::bag::index::MessageIndex;
use visor::bag::mcap::McapReader;
use visor::bag::msgdef::{Decoder, RegistrySet};
use visor::bag::naming::normalize_topic;
use visor::bag::reader::Connection;
use visor::bag::rosbag2;
use visor::bag::set::BagSet;
use visor::decode::value::Value;
use visor::source::launch::expand_directories;
use visor::tf::buffer::{TimeNs, transforms_from_value};

/// Every fixture was written with these stamps: `/chatter` at BASE + i·STEP, `/tf` one nanosecond later, `/scan` two.
const BASE: TimeNs = 1_700_000_000_000_000_000;
const STEP: TimeNs = 100_000_000;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rosbag2")
        .join(name)
}

fn no_cancel() -> AtomicBool {
    AtomicBool::new(false)
}

/// Open a bag directory the way the GUI does: directory → metadata.yaml → storage files → `BagSet`.
fn open_dir(name: &str) -> (BagSet, MessageIndex, RegistrySet) {
    let paths = expand_directories(vec![fixture(name)]).unwrap();
    assert_eq!(paths, vec![fixture(name).join("metadata.yaml")]);
    let files = rosbag2::expand_paths(&paths).unwrap();
    let mut bags = BagSet::open(&files, rosbag2::open_storage, &no_cancel()).unwrap();
    let index = MessageIndex::build(bags.read_message_index(&no_cancel()).unwrap());
    // No fallback: a Jazzy bag describes every type itself, and that alone has to be enough.
    let registries = RegistrySet::build(bags.connections(), None);
    assert!(
        registries.unsupported().is_empty(),
        "{:?}",
        registries.unsupported()
    );
    assert!(registries.fallback_notices().is_empty());
    (bags, index, registries)
}

fn conn_named(bags: &BagSet, topic: &str) -> Connection {
    bags.connections()
        .iter()
        .find(|c| normalize_topic(&c.topic_raw) == topic)
        .cloned()
        .unwrap_or_else(|| panic!("no connection for {topic}"))
}

/// The assertions every single-file fixture has to pass, whatever its storage plugin.
fn check_single_file(name: &str, compression: &str) {
    let (mut bags, index, registries) = open_dir(name);
    assert_eq!(bags.len(), 1);
    assert_eq!(bags.compression(), compression);
    assert_eq!(bags.message_count(), 51);
    assert_eq!(index.len(), 51);
    assert_eq!(index.start(), BASE);
    assert_eq!(index.end(), BASE + 19 * STEP + 1);
    assert!(bags.warnings().is_empty(), "{:?}", bags.warnings());
    let topics: Vec<String> = bags
        .connections()
        .iter()
        .map(|c| normalize_topic(&c.topic_raw))
        .collect();
    assert_eq!(topics.len(), 4);
    for topic in ["/chatter", "/tf", "/tf_static", "/scan"] {
        assert!(topics.contains(&topic.to_owned()), "{topics:?}");
    }
    let chatter = conn_named(&bags, "/chatter");
    assert_eq!(chatter.type_raw, "std_msgs/msg/String");
    assert_eq!(chatter.message_encoding, "cdr");
    assert_eq!(chatter.definition_encoding, "ros2msg");
    assert!(
        chatter.definition.contains("string data"),
        "{}",
        chatter.definition
    );
    // rosbag2_py's writer was given no hash, so the channel carries the key with an empty value (a real recorder fills in `RIHS01_…`).
    assert_eq!(chatter.type_hash, "");
    assert_eq!(index.count(chatter.id), 20);
    // Every message the bag holds is decodable with the bag's own definitions.
    let mut decoded = 0;
    for conn in bags.connections().to_vec() {
        let types = registries.get(conn.id).unwrap().clone();
        assert_eq!(types.decoder, Decoder::Cdr);
        let entries: Vec<_> = index
            .range(conn.id, index.start() - 1, index.end())
            .copied()
            .collect();
        for entry in entries {
            let message = bags.message_at(&entry).unwrap();
            assert_eq!(message.conn, entry.conn);
            assert_eq!(message.time, entry.time);
            types.decode(message.data).unwrap();
            decoded += 1;
        }
    }
    assert_eq!(decoded, 51);
    // The first chatter message is "hello 0" at BASE; the tenth scan carries the stamp the writer gave it.
    let first = *index.at_or_before(chatter.id, BASE).unwrap();
    let value = registries
        .get(chatter.id)
        .unwrap()
        .decode(bags.message_at(&first).unwrap().data)
        .unwrap();
    assert_eq!(
        value.get("data"),
        Some(&Value::String("hello 0".to_owned()))
    );
    let scan = conn_named(&bags, "/scan");
    let entry = *index.at_or_before(scan.id, index.end()).unwrap();
    assert_eq!(entry.time, BASE + 18 * STEP + 2);
    let value = registries
        .get(scan.id)
        .unwrap()
        .decode(bags.message_at(&entry).unwrap().data)
        .unwrap();
    let header = value.get("header").unwrap();
    assert_eq!(
        header.get("frame_id"),
        Some(&Value::String("laser".to_owned()))
    );
    assert_eq!(
        header.get("stamp").unwrap().get("sec"),
        Some(&Value::I32(1_700_000_001))
    );
    assert_eq!(
        header.get("stamp").unwrap().get("nanosec"),
        Some(&Value::U32(800_000_000))
    );
    let Some(Value::Array(ranges)) = value.get("ranges") else {
        panic!("ranges missing");
    };
    assert_eq!(ranges.len(), 9);
    assert_eq!(ranges[0], Value::F32(19.0));
    // TF goes through the same path the player uses.
    let tf = conn_named(&bags, "/tf");
    let entry = *index.at_or_before(tf.id, BASE + 1).unwrap();
    let value = registries
        .get(tf.id)
        .unwrap()
        .decode(bags.message_at(&entry).unwrap().data)
        .unwrap();
    let transforms = transforms_from_value(&value).unwrap();
    assert_eq!(transforms.len(), 1);
    assert_eq!(transforms[0].parent, "map");
    assert_eq!(transforms[0].child, "base_link");
}

#[test]
fn mcap_zstd_fast_bag_written_by_rosbag2_reads_and_decodes() {
    check_single_file("mini_mcap", "zstd");
}

#[test]
fn sqlite3_bag_written_by_rosbag2_reads_and_decodes() {
    check_single_file("mini_sqlite3", "none");
}

#[test]
fn a_split_mcap_recording_merges_into_one_timeline() {
    let (mut bags, index, registries) = open_dir("mini_mcap_split");
    assert_eq!(bags.len(), 2);
    assert_eq!(
        bags.file_names(),
        vec![
            "mini_mcap_split_0.mcap".to_owned(),
            "mini_mcap_split_1.mcap".to_owned()
        ]
    );
    // rosbag2 restarts channel ids in every file, so both halves use the same local ids...
    let files = rosbag2::expand_paths(&[fixture("mini_mcap_split").join("metadata.yaml")]).unwrap();
    let local_ids = |path: &Path| -> Vec<u32> {
        let reader = McapReader::open(path, 1, &no_cancel()).unwrap();
        reader.connections().iter().map(|c| c.id).collect()
    };
    assert_eq!(local_ids(&files[0]), local_ids(&files[1]));
    // ...and the set makes them distinct: one connection per topic per file.
    assert_eq!(bags.connections().len(), 8);
    let mut ids: Vec<u32> = bags.connections().iter().map(|c| c.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 8);
    assert_eq!(index.len(), 151);
    assert_eq!(bags.message_count(), 151);
    assert_eq!(index.start(), BASE);
    assert_eq!(index.end(), BASE + 59 * STEP + 1);
    // The index is one time-ordered list across the file boundary.
    let mut all: Vec<_> = bags
        .connections()
        .iter()
        .flat_map(|c| {
            index
                .range(c.id, index.start() - 1, index.end())
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    all.sort_by_key(|e| (e.time, e.conn));
    assert!(all.windows(2).all(|w| w[0].time <= w[1].time));
    let boundary = all.iter().position(|e| e.file == 1).unwrap();
    assert!(all[..boundary].iter().all(|e| e.file == 0));
    assert!(all[boundary..].iter().all(|e| e.file == 1));
    // Seeking just past the boundary: the first file's /chatter still answers with its last message, the second file's has nothing yet.
    let chatters: Vec<Connection> = bags
        .connections()
        .iter()
        .filter(|c| normalize_topic(&c.topic_raw) == "/chatter")
        .cloned()
        .collect();
    assert_eq!(chatters.len(), 2);
    let last_in_first = index
        .at_or_before(chatters[0].id, all[boundary].time)
        .unwrap();
    assert_eq!(last_in_first.file, 0);
    assert_eq!(last_in_first.time, BASE + 51 * STEP);
    assert!(
        index
            .at_or_before(chatters[1].id, all[boundary].time)
            .is_none()
    );
    let first_in_second = index
        .at_or_before(chatters[1].id, BASE + 52 * STEP + STEP / 2)
        .unwrap();
    assert_eq!(first_in_second.file, 1);
    assert_eq!(first_in_second.time, BASE + 52 * STEP);
    // Both files decode with their own definitions, and payloads route to the right file.
    for (n, entry) in [last_in_first, first_in_second].into_iter().enumerate() {
        let value = registries
            .get(entry.conn)
            .unwrap()
            .decode(bags.message_at(entry).unwrap().data)
            .unwrap();
        assert_eq!(
            value.get("data"),
            Some(&Value::String(format!("hello {}", 51 + n)))
        );
    }
    assert!(bags.warnings().is_empty(), "{:?}", bags.warnings());
}

#[test]
fn the_storage_files_can_be_opened_without_the_metadata() {
    // `visor --bag dir/*.mcap` has to see the same bag as `visor --bag dir`.
    let dir = fixture("mini_mcap_split");
    let files = vec![
        dir.join("mini_mcap_split_1.mcap"),
        dir.join("mini_mcap_split_0.mcap"),
    ];
    let mut bags = BagSet::open(&files, rosbag2::open_storage, &no_cancel()).unwrap();
    let index = MessageIndex::build(bags.read_message_index(&no_cancel()).unwrap());
    assert_eq!(index.len(), 151);
    assert_eq!(bags.file_names()[0], "mini_mcap_split_0.mcap");
}
