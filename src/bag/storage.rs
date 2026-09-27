//! Format-independent view of one recorded file (`rosbag2_storage`'s vocabulary): what `BagSet` needs from a ROS 1 bag, an MCAP file or a sqlite3 database.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use super::reader::{BagError, Connection, IndexEntry, RawMessage};
use crate::tf::buffer::TimeNs;

/// One opened storage file. `BagSet` merges several of these into one timeline; the player never sees the format.
pub trait Storage {
    fn path(&self) -> &Path;
    fn size_bytes(&self) -> u64;
    /// Connections with the file's own (local) ids; `BagSet` rewrites them to set-wide ids.
    fn connections(&self) -> &[Connection];
    fn chunk_count(&self) -> usize;
    /// Chunk compression as the file declares it (`none` / `lz4` / `zstd`), or None until it is known.
    fn compression(&self) -> Option<&str>;
    fn message_count(&self) -> u64;
    /// Earliest record time, used to order the files of a set.
    fn start(&self) -> TimeNs;
    /// Build the full message index; checked against `cancel` at chunk granularity so a dropped source stops promptly.
    fn read_message_index(&mut self, cancel: &AtomicBool) -> Result<Vec<IndexEntry>, BagError>;
    /// Fetch one message by index entry (local connection id in and out).
    fn message_at(&mut self, entry: &IndexEntry) -> Result<RawMessage<'_>, BagError>;
    /// Things worth telling the user after a successful open (a torn tail, a missing summary); empty for a clean file.
    fn warnings(&self) -> &[String];
}

/// How a file is opened: `(path, chunk cache capacity, cancel)`. One per format family, chosen by the source descriptor.
pub type OpenStorage = fn(&Path, usize, &AtomicBool) -> Result<Box<dyn Storage>, BagError>;

/// Decompressed-chunk LRU; `Vec<Vec<u8>>` in most-recently-used order (tiny N, so a linear scan beats a map).
pub(crate) struct ChunkCache {
    pub(crate) entries: Vec<(u32, Vec<u8>)>,
    capacity: usize,
}

impl ChunkCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Move a hit to the front and return its index, or None on a miss.
    pub(crate) fn touch(&mut self, chunk: u32) -> Option<usize> {
        let at = self.entries.iter().position(|(id, _)| *id == chunk)?;
        if at != 0 {
            let entry = self.entries.remove(at);
            self.entries.insert(0, entry);
        }
        Some(0)
    }

    pub(crate) fn insert(&mut self, chunk: u32, body: Vec<u8>) {
        if self.entries.len() >= self.capacity {
            self.entries.pop();
        }
        self.entries.insert(0, (chunk, body));
    }
}

/// True once the owner of `cancel` has gone away; readers poll this between chunks.
pub(crate) fn is_cancelled(cancel: &AtomicBool) -> bool {
    cancel.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_cache_evicts_the_least_recently_used() {
        let mut cache = ChunkCache::new(2);
        cache.insert(0, vec![0]);
        cache.insert(1, vec![1]);
        assert_eq!(cache.touch(0), Some(0));
        // 1 is now the oldest, so inserting 2 evicts it and leaves 0 resident.
        cache.insert(2, vec![2]);
        assert_eq!(cache.touch(1), None);
        assert_eq!(cache.touch(0), Some(0));
        assert_eq!(cache.entries.len(), 2);
    }
}
