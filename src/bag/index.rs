//! Time index over a bag's messages: the playback cursor, seek snapshots and TF backfill all resolve through it.

use std::collections::{HashMap, HashSet};

use super::reader::IndexEntry;
use crate::tf::buffer::TimeNs;

/// All messages in time order, plus a per-connection view for "the one message at or before T".
#[derive(Debug, Default)]
pub struct MessageIndex {
    all: Vec<IndexEntry>,
    /// Connection id -> positions in `all`, ascending in time (so the sub-list stays binary-searchable).
    per_conn: HashMap<u32, Vec<u32>>,
}

impl MessageIndex {
    /// Build from raw index entries; sorting by time is what makes every later lookup a binary search.
    pub fn build(mut entries: Vec<IndexEntry>) -> Self {
        entries.sort_by_key(|e| (e.time, e.conn));
        let mut per_conn: HashMap<u32, Vec<u32>> = HashMap::new();
        for (at, entry) in entries.iter().enumerate() {
            per_conn.entry(entry.conn).or_default().push(at as u32);
        }
        Self {
            all: entries,
            per_conn,
        }
    }

    /// Record time of the first message, or 0 for an empty bag.
    pub fn start(&self) -> TimeNs {
        self.all.first().map_or(0, |e| e.time)
    }

    /// Record time of the last message, or 0 for an empty bag.
    pub fn end(&self) -> TimeNs {
        self.all.last().map_or(0, |e| e.time)
    }

    pub fn len(&self) -> usize {
        self.all.len()
    }

    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }

    /// Messages recorded on one connection.
    pub fn count(&self, conn: u32) -> usize {
        self.per_conn.get(&conn).map_or(0, Vec::len)
    }

    /// Approximate resident size of the index, for the `baginfo` report.
    pub fn memory_bytes(&self) -> usize {
        self.all.len() * std::mem::size_of::<IndexEntry>()
            + self.all.len() * std::mem::size_of::<u32>()
    }

    /// The latest message on `conn` with `time <= at`; the workhorse of both playback ticks and seek snapshots.
    pub fn at_or_before(&self, conn: u32, at: TimeNs) -> Option<&IndexEntry> {
        let positions = self.per_conn.get(&conn)?;
        let upper = positions.partition_point(|p| self.all[*p as usize].time <= at);
        if upper == 0 {
            return None;
        }
        Some(&self.all[positions[upper - 1] as usize])
    }

    /// Messages on `conn` in `(from, to]`, ascending; used for TF, which is never thinned.
    pub fn range(
        &self,
        conn: u32,
        from: TimeNs,
        to: TimeNs,
    ) -> impl Iterator<Item = &IndexEntry> + '_ {
        let positions = self.per_conn.get(&conn).map_or(&[][..], Vec::as_slice);
        let begin = positions.partition_point(|p| self.all[*p as usize].time <= from);
        let end = positions.partition_point(|p| self.all[*p as usize].time <= to);
        positions[begin..end.max(begin)]
            .iter()
            .map(move |p| &self.all[*p as usize])
    }

    /// Latest message on any of `conns` strictly before `before`; drives stepping backwards.
    pub fn prev_before(&self, conns: &HashSet<u32>, before: TimeNs) -> Option<&IndexEntry> {
        conns
            .iter()
            .filter_map(|conn| {
                let positions = self.per_conn.get(conn)?;
                let at = positions.partition_point(|p| self.all[*p as usize].time < before);
                (at > 0).then(|| &self.all[positions[at - 1] as usize])
            })
            .max_by_key(|e| (e.time, e.conn))
    }

    /// Earliest message on any of `conns` strictly after `after`; drives single-message stepping.
    pub fn next_after(&self, conns: &HashSet<u32>, after: TimeNs) -> Option<&IndexEntry> {
        conns
            .iter()
            .filter_map(|conn| {
                let positions = self.per_conn.get(conn)?;
                let at = positions.partition_point(|p| self.all[*p as usize].time <= after);
                positions.get(at).map(|p| &self.all[*p as usize])
            })
            .min_by_key(|e| (e.time, e.conn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(time: TimeNs, conn: u32) -> IndexEntry {
        IndexEntry {
            time,
            conn,
            file: 0,
            chunk: 0,
            offset: time as u32,
        }
    }

    /// Two connections interleaved: conn 1 at 10/20/30, conn 2 at 15/25.
    fn index() -> MessageIndex {
        MessageIndex::build(vec![
            entry(30, 1),
            entry(15, 2),
            entry(10, 1),
            entry(25, 2),
            entry(20, 1),
        ])
    }

    #[test]
    fn t14_bounds_and_counts_come_from_the_sorted_index() {
        let index = index();
        assert_eq!(index.start(), 10);
        assert_eq!(index.end(), 30);
        assert_eq!(index.len(), 5);
        assert_eq!(index.count(1), 3);
        assert_eq!(index.count(2), 2);
        assert_eq!(index.count(9), 0);
        assert!(MessageIndex::build(Vec::new()).is_empty());
    }

    #[test]
    fn t14_at_or_before_is_inclusive_and_per_connection() {
        let index = index();
        // Exactly on a sample.
        assert_eq!(index.at_or_before(1, 20).unwrap().time, 20);
        // Between samples returns the earlier one, and conn 2's samples never leak into conn 1.
        assert_eq!(index.at_or_before(1, 24).unwrap().time, 20);
        assert_eq!(index.at_or_before(2, 24).unwrap().time, 15);
        // Before the first sample there is nothing to show.
        assert!(index.at_or_before(1, 9).is_none());
        assert!(index.at_or_before(2, 14).is_none());
        // After the last sample the newest one stands.
        assert_eq!(index.at_or_before(1, 1_000).unwrap().time, 30);
        assert!(index.at_or_before(42, 1_000).is_none());
    }

    #[test]
    fn t14_range_is_open_on_the_left_and_closed_on_the_right() {
        let index = index();
        let times: Vec<TimeNs> = index.range(1, 10, 30).map(|e| e.time).collect();
        // 10 is excluded (already delivered), 30 included.
        assert_eq!(times, vec![20, 30]);
        let times: Vec<TimeNs> = index.range(1, 0, 10).map(|e| e.time).collect();
        assert_eq!(times, vec![10]);
        // An empty window and an unknown connection both yield nothing rather than panicking.
        assert_eq!(index.range(1, 20, 20).count(), 0);
        assert_eq!(index.range(1, 30, 10).count(), 0);
        assert_eq!(index.range(7, 0, 100).count(), 0);
    }

    #[test]
    fn t14_prev_before_picks_the_latest_subscribed_message() {
        let index = index();
        let both: HashSet<u32> = [1, 2].into_iter().collect();
        assert_eq!(index.prev_before(&both, 30).unwrap().time, 25);
        assert_eq!(index.prev_before(&both, 25).unwrap().time, 20);
        // Only conn 1 subscribed, so conn 2's 25 is skipped.
        let one: HashSet<u32> = [1].into_iter().collect();
        assert_eq!(index.prev_before(&one, 30).unwrap().time, 20);
        // Nothing before the first sample, and no subscriptions means nothing to step to.
        assert!(index.prev_before(&both, 10).is_none());
        assert!(index.prev_before(&HashSet::new(), 100).is_none());
        // Strictly before: a time equal to an existing sample does not return that sample.
        assert_eq!(index.prev_before(&both, 31).unwrap().time, 30);
    }

    #[test]
    fn t14_next_after_picks_the_earliest_subscribed_message() {
        let index = index();
        let both: HashSet<u32> = [1, 2].into_iter().collect();
        assert_eq!(index.next_after(&both, 10).unwrap().time, 15);
        assert_eq!(index.next_after(&both, 15).unwrap().time, 20);
        // Only conn 1 subscribed, so conn 2's 15 is skipped.
        let one: HashSet<u32> = [1].into_iter().collect();
        assert_eq!(index.next_after(&one, 10).unwrap().time, 20);
        // Nothing left after the end, and no subscriptions means nothing to step to.
        assert!(index.next_after(&both, 30).is_none());
        assert!(index.next_after(&HashSet::new(), 0).is_none());
        // Strictly after: a time equal to an existing sample does not return that sample.
        assert_eq!(index.next_after(&both, 9).unwrap().time, 10);
    }
}
