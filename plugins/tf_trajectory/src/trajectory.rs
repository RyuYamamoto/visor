//! Sample bookkeeping for the trail: what gets appended, what expires, what the cap drops. Pure logic (no TF, no egui, no wall clock).

use std::collections::VecDeque;

use nalgebra::Point3;
use visor::plugin::TimeNs;

/// One trail sample: the target frame's origin in fixed-frame coords, plus the TF time it was resolved at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub position: Point3<f32>,
    /// None when the path to the fixed frame is static only, which leaves nothing to age the sample against.
    pub stamp: Option<TimeNs>,
}

/// Rules deciding what the trail holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// A sample is appended only once the frame has moved at least this far [m].
    pub min_step: f32,
    /// Samples older than this are dropped, measured against the newest TF time seen [s].
    pub hold_sec: f32,
    /// Hard cap, oldest first. Bounds the re-bake cost; not a substitute for hold_sec.
    pub max_samples: usize,
}

fn hold_ns(limits: &Limits) -> TimeNs {
    (f64::from(limits.hold_sec.max(0.0)) * 1.0e9) as TimeNs
}

/// Sample list in fixed-frame coordinates, oldest at the front.
#[derive(Debug, Default)]
pub struct Trajectory {
    samples: VecDeque<Sample>,
    now: Option<TimeNs>,
}

impl Trajectory {
    /// Feed one resolved TF sample; true means the sample set changed and a re-bake is due.
    pub fn observe(
        &mut self,
        position: Point3<f32>,
        stamp: Option<TimeNs>,
        limits: &Limits,
    ) -> bool {
        let mut changed = self.rewind_if_time_went_back(stamp);
        if self.should_append(position, stamp, limits) {
            self.samples.push_back(Sample { position, stamp });
            changed = true;
        }
        if let Some(stamp) = stamp {
            self.now = Some(self.now.map_or(stamp, |now| now.max(stamp)));
        }
        changed |= self.expire(limits);
        changed |= self.enforce_cap(limits);
        changed
    }

    /// Drop everything and forget the expiry clock (fixed-frame change, target-frame change, playback jump).
    pub fn clear(&mut self) -> bool {
        let had_anything = !self.samples.is_empty() || self.now.is_some();
        self.samples.clear();
        self.now = None;
        had_anything
    }

    pub fn samples(&self) -> &VecDeque<Sample> {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Newest TF time seen; the reference both the expiry and the fade measure age against.
    pub fn now(&self) -> Option<TimeNs> {
        self.now
    }

    /// TF time the tail should be cut at: samples older than this are outside the window.
    pub fn cutoff(&self, limits: &Limits) -> Option<TimeNs> {
        self.now.map(|now| now.saturating_sub(hold_ns(limits)))
    }

    /// Time went backwards (seek, sim restart, or an out-of-order edge): drop the samples that are now in the future.
    fn rewind_if_time_went_back(&mut self, stamp: Option<TimeNs>) -> bool {
        let (Some(stamp), Some(now)) = (stamp, self.now) else {
            return false;
        };
        if stamp >= now {
            return false;
        }
        let before = self.samples.len();
        while let Some(last) = self.samples.back() {
            match last.stamp {
                Some(last_stamp) if last_stamp > stamp => self.samples.pop_back(),
                _ => break,
            };
        }
        self.now = Some(stamp);
        self.samples.len() != before
    }

    /// Append only once the TF time has advanced and the frame has moved far enough to be worth a segment.
    fn should_append(&self, position: Point3<f32>, stamp: Option<TimeNs>, limits: &Limits) -> bool {
        let Some(last) = self.samples.back() else {
            return true;
        };
        let advanced = match (stamp, last.stamp) {
            (Some(stamp), Some(last_stamp)) => stamp > last_stamp,
            _ => true,
        };
        advanced && (position - last.position).norm() >= limits.min_step
    }

    /// Drop expired samples, but stop at the one straddling the cutoff so the tail can be cut exactly there (a continuous retract instead of one sample popping off at a time).
    fn expire(&mut self, limits: &Limits) -> bool {
        let Some(now) = self.now else {
            return false;
        };
        let cutoff = now.saturating_sub(hold_ns(limits));
        let before = self.samples.len();
        while self.samples.len() > 1 {
            match self.samples.get(1).and_then(|s| s.stamp) {
                Some(stamp) if stamp < cutoff => self.samples.pop_front(),
                _ => break,
            };
        }
        self.samples.len() != before
    }

    fn enforce_cap(&mut self, limits: &Limits) -> bool {
        let cap = limits.max_samples.max(1);
        let before = self.samples.len();
        while self.samples.len() > cap {
            self.samples.pop_front();
        }
        self.samples.len() != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEC: TimeNs = 1_000_000_000;

    fn limits() -> Limits {
        Limits {
            min_step: 0.05,
            hold_sec: 10.0,
            max_samples: 1000,
        }
    }

    fn at(x: f32) -> Point3<f32> {
        Point3::new(x, 0.0, 0.0)
    }

    fn positions(t: &Trajectory) -> Vec<f32> {
        t.samples().iter().map(|s| s.position.x).collect()
    }

    #[test]
    fn appends_only_after_moving_min_step() {
        let mut t = Trajectory::default();
        assert!(t.observe(at(0.0), Some(SEC), &limits()));
        assert!(!t.observe(at(0.049), Some(2 * SEC), &limits()));
        assert_eq!(t.len(), 1);
        // The threshold itself counts as moved (>=).
        assert!(t.observe(at(0.05), Some(3 * SEC), &limits()));
        assert_eq!(positions(&t), vec![0.0, 0.05]);
    }

    #[test]
    fn zero_min_step_appends_every_advancing_sample() {
        let mut t = Trajectory::default();
        let limits = Limits {
            min_step: 0.0,
            ..limits()
        };
        for i in 0..4 {
            t.observe(at(0.0), Some(i * SEC), &limits);
        }
        assert_eq!(t.len(), 4);
    }

    #[test]
    fn does_not_append_when_tf_time_has_not_advanced() {
        let mut t = Trajectory::default();
        t.observe(at(0.0), Some(SEC), &limits());
        assert!(!t.observe(at(5.0), Some(SEC), &limits()));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn samples_older_than_hold_expire_against_the_newest_tf_time() {
        let mut t = Trajectory::default();
        let limits = Limits {
            min_step: 0.0,
            hold_sec: 10.0,
            max_samples: 1000,
        };
        for sec in [89, 91, 95] {
            t.observe(at(sec as f32), Some(sec * SEC), &limits);
        }
        t.observe(at(100.0), Some(100 * SEC), &limits);
        // Cutoff is 90s: 89s straddles it and is kept so the tail can be cut between 89s and 91s.
        assert_eq!(positions(&t), vec![89.0, 91.0, 95.0, 100.0]);
        assert_eq!(t.cutoff(&limits), Some(90 * SEC));
        t.observe(at(102.0), Some(102 * SEC), &limits);
        // Cutoff is now 92s, so 91s has fallen outside and 89s is no longer needed to straddle it.
        assert_eq!(positions(&t), vec![91.0, 95.0, 100.0, 102.0]);
    }

    #[test]
    fn expiry_always_keeps_the_newest_sample() {
        let mut t = Trajectory::default();
        let limits = Limits {
            min_step: 0.0,
            hold_sec: 1.0,
            max_samples: 1000,
        };
        t.observe(at(0.0), Some(SEC), &limits);
        // Standing still for far longer than hold: at most the straddling sample and the current one remain.
        t.observe(at(0.0), Some(100 * SEC), &limits);
        assert_eq!(t.len(), 2);
        assert_eq!(t.samples().back().map(|s| s.stamp), Some(Some(100 * SEC)));
    }

    #[test]
    fn a_path_entirely_older_than_hold_collapses_to_the_straddler_and_the_newest() {
        let mut t = Trajectory::default();
        let limits = Limits {
            min_step: 0.0,
            hold_sec: 10.0,
            max_samples: 1000,
        };
        t.observe(at(0.0), Some(SEC), &limits);
        t.observe(at(1.0), Some(2 * SEC), &limits);
        t.observe(at(1.0), Some(100 * SEC), &limits);
        // Only the straddler and the current position survive, and the cut lands between them.
        assert_eq!(positions(&t), vec![1.0, 1.0]);
        assert_eq!(t.cutoff(&limits), Some(90 * SEC));
    }

    #[test]
    fn samples_without_a_stamp_never_expire() {
        let mut t = Trajectory::default();
        let limits = Limits {
            min_step: 0.0,
            hold_sec: 0.0,
            max_samples: 1000,
        };
        t.observe(at(1.0), None, &limits);
        t.observe(at(2.0), None, &limits);
        assert_eq!(t.len(), 2);
        assert_eq!(t.now(), None);
    }

    #[test]
    fn cap_drops_the_oldest() {
        let mut t = Trajectory::default();
        let limits = Limits {
            min_step: 0.0,
            hold_sec: 3600.0,
            max_samples: 3,
        };
        for i in 0..5 {
            t.observe(at(i as f32), Some(i * SEC), &limits);
        }
        assert_eq!(positions(&t), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn time_going_backwards_drops_the_samples_that_became_future() {
        let mut t = Trajectory::default();
        let limits = Limits {
            min_step: 0.0,
            hold_sec: 3600.0,
            max_samples: 1000,
        };
        for i in 1..=5 {
            t.observe(at(i as f32), Some(i * SEC), &limits);
        }
        // Small step back (a slow edge reordering): only the samples past the new time go.
        assert!(t.observe(at(3.5), Some(3 * SEC + SEC / 2), &limits));
        assert_eq!(positions(&t), vec![1.0, 2.0, 3.0, 3.5]);
        assert_eq!(t.now(), Some(3 * SEC + SEC / 2));
        // A big jump back (a seek) empties it and starts over from the new sample.
        t.observe(at(9.0), Some(0), &limits);
        assert_eq!(positions(&t), vec![9.0]);
        assert_eq!(t.now(), Some(0));
    }

    #[test]
    fn clear_empties_and_forgets_the_clock() {
        let mut t = Trajectory::default();
        t.observe(at(0.0), Some(SEC), &limits());
        assert!(t.clear());
        assert!(t.is_empty());
        assert_eq!(t.now(), None);
        assert!(!t.clear());
    }

    #[test]
    fn observe_reports_no_change_when_nothing_moved() {
        let mut t = Trajectory::default();
        t.observe(at(0.0), Some(SEC), &limits());
        // Same position, later time, nothing to expire yet: the bake stays valid.
        assert!(!t.observe(at(0.0), Some(2 * SEC), &limits()));
    }
}
