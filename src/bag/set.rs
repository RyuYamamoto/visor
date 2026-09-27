//! One or more storage files presented as a single source, merged by record time (the same semantics as `rosbag play a.bag b.bag`).

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use super::reader::{BagError, CHUNK_CACHE_TOTAL, Connection, IndexEntry, RawMessage};
use super::storage::{OpenStorage, Storage};

/// One opened file of the set.
struct Member {
    reader: Box<dyn Storage>,
    /// Set-wide id of this file's connection 0; local ids are offset by it.
    conn_base: u32,
}

/// A set of files read as one. Files are ordered by first record time, and connection ids are made set-wide.
pub struct BagSet {
    members: Vec<Member>,
    /// Connections of every file, with `id` already rewritten to the set-wide id.
    connections: Vec<Connection>,
}

impl BagSet {
    /// Open every path with `open` and order them by first record time; one unreadable file fails the whole set.
    pub fn open(
        paths: &[PathBuf],
        open: OpenStorage,
        cancel: &AtomicBool,
    ) -> Result<Self, BagError> {
        if paths.is_empty() {
            return Err(BagError::Malformed("no bag file given".to_owned()));
        }
        // Splitting the cache budget keeps a 20-part recording from holding 20 times the chunks in memory.
        let per_file = CHUNK_CACHE_TOTAL / paths.len();
        let mut readers = Vec::with_capacity(paths.len());
        for path in paths {
            readers.push(open(path, per_file, cancel)?);
        }
        // Time order, so set-wide connection ids also ascend with time and the newest sample of a topic is delivered last.
        readers.sort_by_key(|r| r.start());
        let mut members = Vec::with_capacity(readers.len());
        let mut connections = Vec::new();
        let mut conn_base = 0u32;
        for reader in readers {
            for conn in reader.connections() {
                connections.push(Connection {
                    id: conn_base + conn.id,
                    ..conn.clone()
                });
            }
            // Ids are dense enough in practice, but base on the maximum so a sparse file cannot collide with the next.
            let span = reader
                .connections()
                .iter()
                .map(|c| c.id + 1)
                .max()
                .unwrap_or(0);
            members.push(Member { reader, conn_base });
            conn_base += span;
        }
        Ok(Self {
            members,
            connections,
        })
    }

    /// Connections of every file, keyed by set-wide id.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// File names in play order, for the status bar.
    pub fn file_names(&self) -> Vec<String> {
        self.members
            .iter()
            .map(|m| file_name(m.reader.path()))
            .collect()
    }

    /// Directory of the first file, used as the next dialog's starting point.
    pub fn first_dir(&self) -> Option<&Path> {
        self.members.first()?.reader.path().parent()
    }

    pub fn size_bytes(&self) -> u64 {
        self.members.iter().map(|m| m.reader.size_bytes()).sum()
    }

    pub fn message_count(&self) -> u64 {
        self.members.iter().map(|m| m.reader.message_count()).sum()
    }

    pub fn chunk_count(&self) -> usize {
        self.members.iter().map(|m| m.reader.chunk_count()).sum()
    }

    /// Compression across the set; distinct values are joined so a mixed selection is visible.
    pub fn compression(&self) -> String {
        let mut seen: Vec<&str> = Vec::new();
        for member in &self.members {
            let value = member.reader.compression().unwrap_or("unknown");
            if !seen.contains(&value) {
                seen.push(value);
            }
        }
        seen.join("+")
    }

    /// Every file's warnings, each prefixed with the file name so a set's notices stay attributable.
    pub fn warnings(&self) -> Vec<String> {
        self.members
            .iter()
            .flat_map(|m| {
                let name = file_name(m.reader.path());
                m.reader
                    .warnings()
                    .iter()
                    .map(move |w| format!("{name}: {w}"))
            })
            .collect()
    }

    /// Read every file's message index into one list, with set-wide connection ids and file tags.
    pub fn read_message_index(&mut self, cancel: &AtomicBool) -> Result<Vec<IndexEntry>, BagError> {
        let mut all = Vec::new();
        for (file, member) in self.members.iter_mut().enumerate() {
            let base = member.conn_base;
            for mut entry in member.reader.read_message_index(cancel)? {
                entry.conn += base;
                entry.file = file as u16;
                all.push(entry);
            }
        }
        Ok(all)
    }

    /// Fetch one message, routing to the file the entry came from. The returned `conn` is the set-wide id.
    pub fn message_at(&mut self, entry: &IndexEntry) -> Result<RawMessage<'_>, BagError> {
        let member = self.members.get_mut(entry.file as usize).ok_or_else(|| {
            BagError::Malformed(format!("file index {} out of range", entry.file))
        })?;
        let base = member.conn_base;
        // The reader knows nothing of set-wide ids, so hand it the local one and translate the answer back.
        let local = IndexEntry {
            conn: entry.conn - base,
            ..*entry
        };
        let message = member.reader.message_at(&local)?;
        Ok(RawMessage {
            conn: message.conn + base,
            ..message
        })
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bag::mcap::test_support::{SyntheticMcap, synthetic_mcap};
    use crate::bag::reader::test_support::{synthetic_bag, write_temp};
    use crate::bag::{mcap, reader};
    use crate::tf::buffer::TimeNs;

    fn no_cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    /// Two bags whose time ranges follow one another, like a `--split` recording.
    fn split_pair(tag: &str) -> (PathBuf, PathBuf) {
        let first = write_temp(
            &format!("{tag}_0.bag"),
            &synthetic_bag(false, &[(4, 100, b"a0"), (4, 101, b"a1")]),
        );
        let second = write_temp(
            &format!("{tag}_1.bag"),
            &synthetic_bag(false, &[(4, 102, b"b0"), (4, 103, b"b1")]),
        );
        (first, second)
    }

    #[test]
    fn merges_two_bags_into_one_time_ordered_index() {
        let (first, second) = split_pair("merge");
        // Given out of order, the set still plays in time order.
        let mut set = BagSet::open(
            &[second.clone(), first.clone()],
            reader::open_storage,
            &no_cancel(),
        )
        .unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set.message_count(), 4);
        assert_eq!(
            set.file_names(),
            vec![
                first.file_name().unwrap().to_string_lossy().into_owned(),
                second.file_name().unwrap().to_string_lossy().into_owned(),
            ]
        );
        let index = set.read_message_index(&no_cancel()).unwrap();
        let mut times: Vec<TimeNs> = index.iter().map(|e| e.time).collect();
        times.sort_unstable();
        assert_eq!(
            times,
            vec![
                100_000_000_000,
                101_000_000_000,
                102_000_000_000,
                103_000_000_000
            ]
        );
        // Each file keeps its own connection, so a topic recorded in both appears as two connections.
        assert_eq!(set.connections().len(), 2);
        assert_ne!(set.connections()[0].id, set.connections()[1].id);
        // Messages report the set-wide id, so callers never see a file's local numbering.
        for entry in &index {
            assert_eq!(set.message_at(entry).unwrap().conn, entry.conn);
        }
        let payloads: Vec<Vec<u8>> = {
            let mut sorted = index.clone();
            sorted.sort_by_key(|e| e.time);
            sorted
                .iter()
                .map(|e| set.message_at(e).unwrap().data.to_vec())
                .collect()
        };
        // Routing by file tag and the local id rewrite both work: payloads come back in time order.
        assert_eq!(
            payloads,
            vec![
                b"a0".to_vec(),
                b"a1".to_vec(),
                b"b0".to_vec(),
                b"b1".to_vec()
            ]
        );
        std::fs::remove_file(&first).ok();
        std::fs::remove_file(&second).ok();
    }

    #[test]
    fn a_single_bag_behaves_exactly_as_before() {
        let path = write_temp("single.bag", &synthetic_bag(true, &[(2, 7, b"only")]));
        let mut set = BagSet::open(
            std::slice::from_ref(&path),
            reader::open_storage,
            &no_cancel(),
        )
        .unwrap();
        assert_eq!(set.len(), 1);
        let index = set.read_message_index(&no_cancel()).unwrap();
        // Compression is read off chunk headers, so it is known once the index has been walked.
        assert_eq!(set.compression(), "lz4");
        assert_eq!(index.len(), 1);
        assert_eq!(index[0].file, 0);
        // With one file the connection ids are untouched.
        assert_eq!(index[0].conn, 2);
        assert_eq!(set.message_at(&index[0]).unwrap().data, b"only");
        assert!(set.warnings().is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn overlapping_bags_interleave_by_record_time() {
        // Not a split set: two recordings covering the same span, which rosbag play would also interleave.
        let first = write_temp(
            "overlap_a.bag",
            &synthetic_bag(false, &[(1, 10, b"a10"), (1, 30, b"a30")]),
        );
        let second = write_temp(
            "overlap_b.bag",
            &synthetic_bag(false, &[(1, 20, b"b20"), (1, 40, b"b40")]),
        );
        let mut set = BagSet::open(
            &[first.clone(), second.clone()],
            reader::open_storage,
            &no_cancel(),
        )
        .unwrap();
        let mut index = set.read_message_index(&no_cancel()).unwrap();
        index.sort_by_key(|e| e.time);
        let payloads: Vec<Vec<u8>> = index
            .iter()
            .map(|e| set.message_at(e).unwrap().data.to_vec())
            .collect();
        assert_eq!(
            payloads,
            vec![
                b"a10".to_vec(),
                b"b20".to_vec(),
                b"a30".to_vec(),
                b"b40".to_vec()
            ]
        );
        std::fs::remove_file(&first).ok();
        std::fs::remove_file(&second).ok();
    }

    #[test]
    fn an_empty_selection_and_an_unreadable_member_both_fail() {
        assert!(matches!(
            BagSet::open(&[], reader::open_storage, &no_cancel()),
            Err(BagError::Malformed(_))
        ));
        let good = write_temp("good.bag", &synthetic_bag(false, &[(1, 1, b"x")]));
        let bad = write_temp("bad.bag", b"#ROSBAG V1.2\nnope");
        assert!(matches!(
            BagSet::open(
                &[good.clone(), bad.clone()],
                reader::open_storage,
                &no_cancel()
            ),
            Err(BagError::UnsupportedVersion(_))
        ));
        std::fs::remove_file(&good).ok();
        std::fs::remove_file(&bad).ok();
    }

    #[test]
    fn two_mcap_files_merge_like_a_split_rosbag2_recording() {
        // rosbag2 restarts channel ids in every split file, so both halves use channel 1 for the same topic.
        let first = write_temp(
            "split_0.mcap",
            &synthetic_mcap(&SyntheticMcap {
                chunks: vec![vec![(1, 100, b"a0".to_vec()), (1, 101, b"a1".to_vec())]],
                ..SyntheticMcap::default()
            }),
        );
        let second = write_temp(
            "split_1.mcap",
            &synthetic_mcap(&SyntheticMcap {
                chunks: vec![vec![(1, 102, b"b0".to_vec()), (1, 103, b"b1".to_vec())]],
                ..SyntheticMcap::default()
            }),
        );
        let mut set = BagSet::open(
            &[second.clone(), first.clone()],
            mcap::open_storage,
            &no_cancel(),
        )
        .unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set.connections().len(), 2);
        assert_ne!(set.connections()[0].id, set.connections()[1].id);
        assert_eq!(set.compression(), "none");
        let mut index = set.read_message_index(&no_cancel()).unwrap();
        index.sort_by_key(|e| e.time);
        assert_eq!(
            index.iter().map(|e| e.file).collect::<Vec<_>>(),
            vec![0, 0, 1, 1]
        );
        let payloads: Vec<Vec<u8>> = index
            .iter()
            .map(|e| set.message_at(e).unwrap().data.to_vec())
            .collect();
        assert_eq!(
            payloads,
            vec![
                b"a0".to_vec(),
                b"a1".to_vec(),
                b"b0".to_vec(),
                b"b1".to_vec()
            ]
        );
        for entry in &index {
            assert_eq!(set.message_at(entry).unwrap().conn, entry.conn);
        }
        assert!(set.warnings().is_empty());
        std::fs::remove_file(&first).ok();
        std::fs::remove_file(&second).ok();
    }

    #[test]
    fn warnings_carry_the_file_name() {
        let mut bytes = synthetic_mcap(&SyntheticMcap {
            chunks: vec![vec![(1, 100, b"a0".to_vec())]],
            ..SyntheticMcap::default()
        });
        // Cutting the summary and footer off leaves a readable file that has something to say about itself.
        let end = bytes.len() - mcap::test_support::summary_len(&bytes);
        bytes.truncate(end);
        let path = write_temp("torn_set.mcap", &bytes);
        let set = BagSet::open(
            std::slice::from_ref(&path),
            mcap::open_storage,
            &no_cancel(),
        )
        .unwrap();
        let warnings = set.warnings();
        assert!(!warnings.is_empty());
        assert!(
            warnings.iter().all(|w| w.contains("torn_set.mcap: ")),
            "{warnings:?}"
        );
        std::fs::remove_file(&path).ok();
    }
}
