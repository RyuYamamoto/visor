//! TF ring buffer and lookup_transform (pure logic, no zenoh/egui deps; unit-tested).

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

use crate::decode::value::Value;

/// Internal time: nanoseconds since the UNIX epoch (i64 spans ±292 years, ample for ROS time).
pub type TimeNs = i64;

/// Dynamic TF retention window (matches tf2's default of 10s).
const RETENTION_NS: TimeNs = 10_000_000_000;
/// Per-frame sample cap; combined with the time window it bounds memory.
const MAX_SAMPLES_PER_FRAME: usize = 10_000;
/// Parent-chain walk depth cap (stops even if bad data forms a cycle).
const MAX_CHAIN_DEPTH: usize = 64;
/// slerp degeneracy epsilon; try_slerp flips sign on dot<0 so None is unreachable in practice, but guard it.
const SLERP_EPSILON: f64 = 1.0e-9;

/// One transform from a TFMessage (also the unit sent over the comm → UI channel).
#[derive(Debug, Clone)]
pub struct TfTransform {
    /// header.frame_id (leading `/` stripped).
    pub parent: String,
    /// child_frame_id (leading `/` stripped).
    pub child: String,
    pub stamp: TimeNs,
    /// parent_T_child (maps a point in child coords into parent coords).
    pub transform: Isometry3<f64>,
}

/// One message on the source → UI channel.
#[derive(Debug, Clone)]
pub struct TfUpdate {
    pub transforms: Vec<TfTransform>,
    pub is_static: bool,
    /// Playback generation that produced this; the UI clears the buffer when it grows (plan §5.4).
    pub epoch: u64,
}

/// Test helper: an update at the live epoch, since unit tests are never about playback generations.
#[cfg(test)]
pub fn tf_update(transforms: Vec<TfTransform>, is_static: bool) -> TfUpdate {
    TfUpdate {
        transforms,
        is_static,
        epoch: crate::comm::session::LIVE_EPOCH,
    }
}

/// Why lookup_transform failed (never panics; caller decides to display or skip).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TfError {
    UnknownFrame(String),
    Disconnected {
        target: String,
        source: String,
    },
    /// Extrapolation forbidden (tf2 semantics); a time exactly on a boundary succeeds.
    OutOfRange {
        frame: String,
        requested: TimeNs,
        oldest: TimeNs,
        newest: TimeNs,
    },
}

impl std::fmt::Display for TfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFrame(frame) => write!(f, "unknown frame `{frame}`"),
            Self::Disconnected { target, source } => {
                write!(f, "frames `{target}` and `{source}` are not connected")
            }
            Self::OutOfRange {
                frame,
                requested,
                oldest,
                newest,
            } => write!(
                f,
                "time {requested} out of range for frame `{frame}` (have {oldest}..{newest})"
            ),
        }
    }
}

/// History for one dynamic TF frame (time-ascending ring buffer).
struct FrameHistory {
    parent: String,
    samples: VecDeque<(TimeNs, Isometry3<f64>)>,
    last_received: Instant,
}

/// Static TF is time-independent (latest value only, kept indefinitely).
struct StaticEntry {
    parent: String,
    transform: Isometry3<f64>,
    last_received: Instant,
}

/// Read-only view of one frame for the Frames panel.
pub struct FrameInfo<'a> {
    pub name: &'a str,
    pub parent: &'a str,
    pub is_static: bool,
    pub last_received: Instant,
}

/// TF buffer keyed by child_frame_id. Read and written on the UI thread (no lock).
#[derive(Default)]
pub struct TfBuffer {
    dynamic: HashMap<String, FrameHistory>,
    statics: HashMap<String, StaticEntry>,
    /// Newest stamp across all dynamic samples (detects backward time jump = sim restart).
    newest_dynamic_stamp: Option<TimeNs>,
}

/// tf2 semantics: ignore a leading `/` (legacy tf1 form) on frame_id.
fn normalize(frame: &str) -> &str {
    frame.strip_prefix('/').unwrap_or(frame)
}

impl TfBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.dynamic.is_empty() && self.statics.is_empty()
    }

    pub fn insert(&mut self, update: &TfUpdate) {
        for t in &update.transforms {
            let parent = normalize(&t.parent);
            let child = normalize(&t.child);
            if parent.is_empty() || child.is_empty() {
                continue;
            }
            if update.is_static {
                self.statics.insert(
                    child.to_owned(),
                    StaticEntry {
                        parent: parent.to_owned(),
                        transform: t.transform,
                        last_received: Instant::now(),
                    },
                );
            } else {
                self.insert_dynamic(parent, child, t.stamp, t.transform);
            }
        }
    }

    fn insert_dynamic(&mut self, parent: &str, child: &str, stamp: TimeNs, tf: Isometry3<f64>) {
        // Jump past the retention window into the past => treat as sim restart: rebuild dynamic buffer (keep static).
        if let Some(newest) = self.newest_dynamic_stamp
            && stamp < newest - RETENTION_NS
        {
            eprintln!(
                "visor: tf time jumped backwards ({newest} -> {stamp}), clearing dynamic tf buffer"
            );
            self.dynamic.clear();
            self.newest_dynamic_stamp = None;
        }
        let history = self
            .dynamic
            .entry(child.to_owned())
            .or_insert_with(|| FrameHistory {
                parent: parent.to_owned(),
                samples: VecDeque::new(),
                last_received: Instant::now(),
            });
        // Parent change is last-writer-wins (tf2 behavior); old-parent history is useless for interpolation, so drop it.
        if history.parent != parent {
            history.parent = parent.to_owned();
            history.samples.clear();
        }
        history.last_received = Instant::now();
        // Usually push to the back; out-of-order arrivals insert via reverse scan; same stamp overwrites.
        match history.samples.iter().rposition(|(t, _)| *t <= stamp) {
            Some(i) if history.samples[i].0 == stamp => history.samples[i].1 = tf,
            Some(i) => history.samples.insert(i + 1, (stamp, tf)),
            None => history.samples.push_front((stamp, tf)),
        }
        let newest = history.samples.back().map(|(t, _)| *t).unwrap_or(stamp);
        while let Some(&(oldest, _)) = history.samples.front()
            && (oldest < newest - RETENTION_NS || history.samples.len() > MAX_SAMPLES_PER_FRAME)
        {
            history.samples.pop_front();
        }
        self.newest_dynamic_stamp = Some(self.newest_dynamic_stamp.unwrap_or(stamp).max(stamp));
    }

    /// Returns the transform mapping a point in `source` frame into `target` coords (tf2 semantics).
    pub fn lookup_transform(
        &self,
        target: &str,
        source: &str,
        time: TimeNs,
    ) -> Result<Isometry3<f64>, TfError> {
        let (source_chain, target_chain, si, ti) = self.find_chains(target, source)?;
        if si == 0 && ti == 0 {
            return Ok(Isometry3::identity());
        }
        // Left-multiply parent_T_child from source up to the common ancestor (same for target side).
        let mut common_t_source = Isometry3::identity();
        for child in &source_chain[..si] {
            common_t_source = self.edge_at(child, time)? * common_t_source;
        }
        let mut common_t_target = Isometry3::identity();
        for child in &target_chain[..ti] {
            common_t_target = self.edge_at(child, time)? * common_t_target;
        }
        Ok(common_t_target.inverse() * common_t_source)
    }

    /// Lookup at the "latest" time (tf2 time=0): uses the min of the newest stamps of dynamic edges on the path.
    pub fn lookup_transform_latest(
        &self,
        target: &str,
        source: &str,
    ) -> Result<Isometry3<f64>, TfError> {
        self.lookup_transform_latest_stamped(target, source)
            .map(|(transform, _)| transform)
    }

    /// lookup_transform_latest plus the time it resolved at (None = static-only path); one chain walk, so pose and time always agree.
    pub fn lookup_transform_latest_stamped(
        &self,
        target: &str,
        source: &str,
    ) -> Result<(Isometry3<f64>, Option<TimeNs>), TfError> {
        let (source_chain, target_chain, si, ti) = self.find_chains(target, source)?;
        let mut latest: Option<TimeNs> = None;
        for child in source_chain[..si].iter().chain(target_chain[..ti].iter()) {
            if let Some(history) = self.dynamic.get(child.as_str())
                && let Some(&(newest, _)) = history.samples.back()
            {
                latest = Some(latest.map_or(newest, |t| t.min(newest)));
            }
        }
        // If no dynamic edge exists, compose statics only (static ignores time, so 0 is fine).
        let transform = self.lookup_transform(target, source, latest.unwrap_or(0))?;
        Ok((transform, latest))
    }

    /// Finds both frames' chains to root and the common-ancestor positions (si on source, ti on target).
    #[allow(clippy::type_complexity)]
    fn find_chains(
        &self,
        target: &str,
        source: &str,
    ) -> Result<(Vec<String>, Vec<String>, usize, usize), TfError> {
        let target = normalize(target);
        let source = normalize(source);
        if target != source {
            if !self.frame_known(source) {
                return Err(TfError::UnknownFrame(source.to_owned()));
            }
            if !self.frame_known(target) {
                return Err(TfError::UnknownFrame(target.to_owned()));
            }
        }
        let source_chain = self.chain_to_root(source);
        let target_chain = self.chain_to_root(target);
        for (ti, frame) in target_chain.iter().enumerate() {
            if let Some(si) = source_chain.iter().position(|s| s == frame) {
                return Ok((source_chain, target_chain, si, ti));
            }
        }
        Err(TfError::Disconnected {
            target: target.to_owned(),
            source: source.to_owned(),
        })
    }

    /// Frame names from `frame` up to root: [frame, parent, …, root].
    fn chain_to_root(&self, frame: &str) -> Vec<String> {
        let mut chain = vec![frame.to_owned()];
        let mut current = frame.to_owned();
        for _ in 0..MAX_CHAIN_DEPTH {
            match self.edge_parent(&current) {
                Some(parent) => {
                    current = parent.to_owned();
                    chain.push(current.clone());
                }
                None => break,
            }
        }
        chain
    }

    /// child → parent frame name (if a child exists in both dynamic and static, dynamic wins).
    fn edge_parent(&self, child: &str) -> Option<&str> {
        if let Some(history) = self.dynamic.get(child) {
            return Some(&history.parent);
        }
        self.statics.get(child).map(|s| s.parent.as_str())
    }

    /// Resolves edge parent_T_child at `time` (static ignores time; dynamic interpolates).
    fn edge_at(&self, child: &str, time: TimeNs) -> Result<Isometry3<f64>, TfError> {
        if let Some(history) = self.dynamic.get(child) {
            return Self::interpolate(child, &history.samples, time);
        }
        match self.statics.get(child) {
            Some(entry) => Ok(entry.transform),
            None => Err(TfError::UnknownFrame(child.to_owned())),
        }
    }

    fn interpolate(
        child: &str,
        samples: &VecDeque<(TimeNs, Isometry3<f64>)>,
        time: TimeNs,
    ) -> Result<Isometry3<f64>, TfError> {
        let (Some(&(oldest, _)), Some(&(newest, _))) = (samples.front(), samples.back()) else {
            return Err(TfError::UnknownFrame(child.to_owned()));
        };
        let idx = samples.partition_point(|(t, _)| *t < time);
        if idx < samples.len() && samples[idx].0 == time {
            return Ok(samples[idx].1);
        }
        if idx == 0 || idx == samples.len() {
            return Err(TfError::OutOfRange {
                frame: child.to_owned(),
                requested: time,
                oldest,
                newest,
            });
        }
        let (t0, tf0) = samples[idx - 1];
        let (t1, tf1) = samples[idx];
        let ratio = (time - t0) as f64 / (t1 - t0) as f64;
        let translation = tf0.translation.vector.lerp(&tf1.translation.vector, ratio);
        // If try_slerp returns None (degenerate), fall back to the nearer sample in time (closes the panic path).
        let rotation = tf0
            .rotation
            .try_slerp(&tf1.rotation, ratio, SLERP_EPSILON)
            .unwrap_or(if ratio < 0.5 {
                tf0.rotation
            } else {
                tf1.rotation
            });
        Ok(Isometry3::from_parts(
            Translation3::from(translation),
            rotation,
        ))
    }

    fn frame_known(&self, frame: &str) -> bool {
        self.dynamic.contains_key(frame)
            || self.statics.contains_key(frame)
            || self.dynamic.values().any(|h| h.parent == frame)
            || self.statics.values().any(|s| s.parent == frame)
    }

    /// For the Frames panel: read view of all edges (child name ascending; dynamic wins on dynamic/static overlap).
    pub fn frames(&self) -> Vec<FrameInfo<'_>> {
        let mut out: Vec<FrameInfo<'_>> = self
            .dynamic
            .iter()
            .map(|(child, h)| FrameInfo {
                name: child,
                parent: &h.parent,
                is_static: false,
                last_received: h.last_received,
            })
            .chain(
                self.statics
                    .iter()
                    .filter(|(child, _)| !self.dynamic.contains_key(*child))
                    .map(|(child, s)| FrameInfo {
                        name: child,
                        parent: &s.parent,
                        is_static: true,
                        last_received: s.last_received,
                    }),
            )
            .collect();
        out.sort_by(|a, b| a.name.cmp(b.name));
        out
    }

    /// All known frame names (ascending, deduped; fixed-frame candidates).
    pub fn frame_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .frames()
            .iter()
            .flat_map(|f| [f.name.to_owned(), f.parent.to_owned()])
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Root frames (those appearing only as a parent), ascending.
    pub fn roots(&self) -> Vec<String> {
        let frames = self.frames();
        let mut roots: Vec<String> = frames
            .iter()
            .map(|f| f.parent)
            .filter(|p| !frames.iter().any(|f| f.name == *p))
            .map(str::to_owned)
            .collect();
        roots.sort();
        roots.dedup();
        roots
    }

    pub fn contains_frame(&self, frame: &str) -> bool {
        self.frame_known(normalize(frame))
    }

    /// Root with the most descendant frames (ties broken by name); used to auto-follow so a stray small tree isn't picked as fixed frame.
    pub fn primary_root(&self) -> Option<String> {
        let frames = self.frames();
        let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
        for frame in &frames {
            children.entry(frame.parent).or_default().push(frame.name);
        }
        self.roots()
            .into_iter()
            .map(|root| {
                let mut count = 0usize;
                let mut stack = vec![root.as_str()];
                let mut visited: HashSet<&str> = HashSet::new();
                while let Some(name) = stack.pop() {
                    if !visited.insert(name) {
                        continue;
                    }
                    if let Some(kids) = children.get(name) {
                        count += kids.len();
                        stack.extend(kids);
                    }
                }
                (count, root)
            })
            // roots() is ascending; on a tie keep the first (avoids max_by_key's last-wins).
            .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
            .map(|(_, root)| root)
    }
}

/// transforms_from_value failure (TFMessage structure mismatch; caller skips per message).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TfConvertError(pub String);

impl std::fmt::Display for TfConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid TFMessage structure: {}", self.0)
    }
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, TfConvertError> {
    value
        .get(name)
        .ok_or_else(|| TfConvertError(format!("missing field `{name}`")))
}

fn get_f64(value: &Value, name: &str) -> Result<f64, TfConvertError> {
    match field(value, name)? {
        Value::F64(v) => Ok(*v),
        other => Err(TfConvertError(format!(
            "field `{name}` is not float64: {other:?}"
        ))),
    }
}

/// Extracts TfTransforms from a decoded TFMessage (`Value`), skipping entries with empty frame_id.
pub fn transforms_from_value(value: &Value) -> Result<Vec<TfTransform>, TfConvertError> {
    let Some(Value::Array(items)) = value.get("transforms") else {
        return Err(TfConvertError("missing `transforms` array".to_owned()));
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let header = field(item, "header")?;
        let stamp = field(header, "stamp")?;
        let sec = match field(stamp, "sec")? {
            Value::I32(v) => *v,
            other => return Err(TfConvertError(format!("stamp.sec is not int32: {other:?}"))),
        };
        let nanosec = match field(stamp, "nanosec")? {
            Value::U32(v) => *v,
            other => {
                return Err(TfConvertError(format!(
                    "stamp.nanosec is not uint32: {other:?}"
                )));
            }
        };
        let parent = match field(header, "frame_id")? {
            Value::String(s) => normalize(s),
            other => return Err(TfConvertError(format!("frame_id is not string: {other:?}"))),
        };
        let child = match field(item, "child_frame_id")? {
            Value::String(s) => normalize(s),
            other => {
                return Err(TfConvertError(format!(
                    "child_frame_id is not string: {other:?}"
                )));
            }
        };
        if parent.is_empty() || child.is_empty() {
            continue;
        }
        let transform = field(item, "transform")?;
        let translation = field(transform, "translation")?;
        let rotation = field(transform, "rotation")?;
        let translation = Translation3::new(
            get_f64(translation, "x")?,
            get_f64(translation, "y")?,
            get_f64(translation, "z")?,
        );
        // Quaternion::new argument order is (w, x, y, z); from_quaternion normalizes.
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(
            get_f64(rotation, "w")?,
            get_f64(rotation, "x")?,
            get_f64(rotation, "y")?,
            get_f64(rotation, "z")?,
        ));
        out.push(TfTransform {
            parent: parent.to_owned(),
            child: child.to_owned(),
            stamp: TimeNs::from(sec) * 1_000_000_000 + TimeNs::from(nanosec),
            transform: Isometry3::from_parts(translation, rotation),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use std::f64::consts::FRAC_PI_2;

    const SEC: TimeNs = 1_000_000_000;

    fn rot_z(angle: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
    }

    fn tf(
        parent: &str,
        child: &str,
        stamp: TimeNs,
        xyz: (f64, f64, f64),
        rot: UnitQuaternion<f64>,
    ) -> TfTransform {
        TfTransform {
            parent: parent.to_owned(),
            child: child.to_owned(),
            stamp,
            transform: Isometry3::from_parts(Translation3::new(xyz.0, xyz.1, xyz.2), rot),
        }
    }

    fn insert_dynamic(buffer: &mut TfBuffer, transforms: Vec<TfTransform>) {
        buffer.insert(&TfUpdate {
            transforms,
            is_static: false,
            epoch: 0,
        });
    }

    fn insert_static(buffer: &mut TfBuffer, transforms: Vec<TfTransform>) {
        buffer.insert(&TfUpdate {
            transforms,
            is_static: true,
            epoch: 0,
        });
    }

    fn assert_isometry_eq(actual: &Isometry3<f64>, expected: &Isometry3<f64>) {
        assert!(
            (actual.translation.vector - expected.translation.vector).norm() < 1e-9
                && actual.rotation.angle_to(&expected.rotation) < 1e-9,
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn lookup_at_sample_boundaries_returns_exact_values() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "base", SEC, (2.0, 0.0, 0.0), rot_z(FRAC_PI_2)),
            ],
        );
        assert_isometry_eq(
            &buffer.lookup_transform("map", "base", 0).unwrap(),
            &Isometry3::identity(),
        );
        assert_isometry_eq(
            &buffer.lookup_transform("map", "base", SEC).unwrap(),
            &Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), rot_z(FRAC_PI_2)),
        );
    }

    #[test]
    fn lookup_between_samples_interpolates_lerp_and_slerp() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "base", SEC, (2.0, -1.0, 0.0), rot_z(FRAC_PI_2)),
            ],
        );
        let result = buffer.lookup_transform("map", "base", SEC / 2).unwrap();
        assert_isometry_eq(
            &result,
            &Isometry3::from_parts(Translation3::new(1.0, -0.5, 0.0), rot_z(FRAC_PI_2 / 2.0)),
        );
    }

    #[test]
    fn slerp_across_antipodal_quaternion_representation_does_not_panic() {
        // Interpolating between q and -q (same rotation, dot<0 sign-flip path) stays identity.
        let mut buffer = TfBuffer::new();
        let q_neg = UnitQuaternion::from_quaternion(Quaternion::new(-1.0, 0.0, 0.0, 0.0));
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "base", SEC, (0.0, 0.0, 0.0), q_neg),
            ],
        );
        let result = buffer.lookup_transform("map", "base", SEC / 2).unwrap();
        assert!(result.rotation.angle_to(&rot_z(0.0)) < 1e-9);
    }

    #[test]
    fn lookup_composes_two_edge_chain() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "odom", 0, (1.0, 0.0, 0.0), rot_z(0.0)),
                tf("odom", "base_link", 0, (0.0, 1.0, 0.0), rot_z(FRAC_PI_2)),
            ],
        );
        let map_t_base = buffer.lookup_transform("map", "base_link", 0).unwrap();
        // Point (1,0,0) in base_link is (0,2,0) in odom and (1,2,0) in map.
        let p = map_t_base.transform_point(&nalgebra::Point3::new(1.0, 0.0, 0.0));
        assert!(
            (p - nalgebra::Point3::new(1.0, 2.0, 0.0)).norm() < 1e-9,
            "p={p}"
        );
        let base_t_map = buffer.lookup_transform("base_link", "map", 0).unwrap();
        assert_isometry_eq(&base_t_map, &map_t_base.inverse());
    }

    #[test]
    fn lookup_between_sibling_branches_goes_through_common_ancestor() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("odom", "left", 0, (0.0, 1.0, 0.0), rot_z(0.0)),
                tf("odom", "right", 0, (0.0, -1.0, 0.0), rot_z(0.0)),
            ],
        );
        let left_t_right = buffer.lookup_transform("left", "right", 0).unwrap();
        assert_isometry_eq(
            &left_t_right,
            &Isometry3::from_parts(Translation3::new(0.0, -2.0, 0.0), rot_z(0.0)),
        );
    }

    #[test]
    fn lookup_mixes_static_and_dynamic_edges() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![tf("map", "base_link", SEC, (1.0, 0.0, 0.0), rot_z(0.0))],
        );
        insert_static(
            &mut buffer,
            vec![tf("base_link", "laser", 0, (0.5, 0.0, 0.0), rot_z(0.0))],
        );
        // Static edges are usable regardless of the lookup time.
        let map_t_laser = buffer.lookup_transform("map", "laser", SEC).unwrap();
        assert_isometry_eq(
            &map_t_laser,
            &Isometry3::from_parts(Translation3::new(1.5, 0.0, 0.0), rot_z(0.0)),
        );
    }

    #[test]
    fn latest_lookup_uses_minimum_of_newest_dynamic_stamps() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "odom", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "odom", 2 * SEC, (2.0, 0.0, 0.0), rot_z(0.0)),
                tf("odom", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("odom", "base", SEC, (0.0, 1.0, 0.0), rot_z(0.0)),
            ],
        );
        insert_static(
            &mut buffer,
            vec![tf("base", "laser", 0, (0.1, 0.0, 0.0), rot_z(0.0))],
        );
        // Newest stamps: map→odom 2s, odom→base 1s => common time is 1s.
        let latest = buffer.lookup_transform_latest("map", "laser").unwrap();
        let at_1s = buffer.lookup_transform("map", "laser", SEC).unwrap();
        assert_isometry_eq(&latest, &at_1s);
        assert_isometry_eq(
            &latest,
            &Isometry3::from_parts(Translation3::new(1.1, 1.0, 0.0), rot_z(0.0)),
        );
    }

    #[test]
    fn stamped_latest_lookup_reports_the_time_it_resolved_at() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "odom", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "odom", 2 * SEC, (2.0, 0.0, 0.0), rot_z(0.0)),
            ],
        );
        let (pose, stamp) = buffer
            .lookup_transform_latest_stamped("map", "odom")
            .unwrap();
        assert_eq!(stamp, Some(2 * SEC));
        assert_isometry_eq(
            &pose,
            &buffer.lookup_transform_latest("map", "odom").unwrap(),
        );
        insert_dynamic(
            &mut buffer,
            vec![tf("odom", "base", SEC, (0.0, 1.0, 0.0), rot_z(0.0))],
        );
        // Two dynamic edges: the reported time is the minimum of their newest stamps, as the pose is.
        let (pose, stamp) = buffer
            .lookup_transform_latest_stamped("map", "base")
            .unwrap();
        assert_eq!(stamp, Some(SEC));
        assert_isometry_eq(&pose, &buffer.lookup_transform("map", "base", SEC).unwrap());
    }

    #[test]
    fn stamped_latest_lookup_has_no_time_without_a_dynamic_edge() {
        let mut buffer = TfBuffer::new();
        insert_static(
            &mut buffer,
            vec![tf("base", "laser", 0, (0.5, 0.0, 0.0), rot_z(0.0))],
        );
        let (_, stamp) = buffer
            .lookup_transform_latest_stamped("base", "laser")
            .unwrap();
        assert_eq!(stamp, None);
        assert!(
            buffer
                .lookup_transform_latest_stamped("base", "nope")
                .is_err()
        );
    }

    #[test]
    fn latest_lookup_on_static_only_chain_succeeds() {
        let mut buffer = TfBuffer::new();
        insert_static(
            &mut buffer,
            vec![tf("base", "laser", 0, (0.5, 0.0, 0.0), rot_z(0.0))],
        );
        let base_t_laser = buffer.lookup_transform_latest("base", "laser").unwrap();
        assert_isometry_eq(
            &base_t_laser,
            &Isometry3::from_parts(Translation3::new(0.5, 0.0, 0.0), rot_z(0.0)),
        );
    }

    #[test]
    fn unknown_frame_is_reported() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![tf("map", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0))],
        );
        assert_eq!(
            buffer.lookup_transform("map", "nope", 0),
            Err(TfError::UnknownFrame("nope".to_owned()))
        );
        assert_eq!(
            buffer.lookup_transform("nope", "base", 0),
            Err(TfError::UnknownFrame("nope".to_owned()))
        );
    }

    #[test]
    fn disconnected_trees_are_reported() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("world", "cam", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
            ],
        );
        assert_eq!(
            buffer.lookup_transform("base", "cam", 0),
            Err(TfError::Disconnected {
                target: "base".to_owned(),
                source: "cam".to_owned(),
            })
        );
    }

    #[test]
    fn extrapolation_is_rejected() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "base", SEC, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "base", 2 * SEC, (1.0, 0.0, 0.0), rot_z(0.0)),
            ],
        );
        let expected_range = |requested| TfError::OutOfRange {
            frame: "base".to_owned(),
            requested,
            oldest: SEC,
            newest: 2 * SEC,
        };
        // Both past and future extrapolation are rejected (exact boundary succeeds).
        assert_eq!(
            buffer.lookup_transform("map", "base", SEC / 2),
            Err(expected_range(SEC / 2))
        );
        assert_eq!(
            buffer.lookup_transform("map", "base", 3 * SEC),
            Err(expected_range(3 * SEC))
        );
        assert!(buffer.lookup_transform("map", "base", SEC).is_ok());
    }

    #[test]
    fn single_sample_frame_matches_exact_time_only() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![tf("map", "base", SEC, (1.0, 0.0, 0.0), rot_z(0.0))],
        );
        assert!(buffer.lookup_transform("map", "base", SEC).is_ok());
        assert_eq!(
            buffer.lookup_transform("map", "base", SEC + 1),
            Err(TfError::OutOfRange {
                frame: "base".to_owned(),
                requested: SEC + 1,
                oldest: SEC,
                newest: SEC,
            })
        );
    }

    #[test]
    fn samples_older_than_retention_window_are_evicted() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![tf("map", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0))],
        );
        assert!(buffer.lookup_transform("map", "base", 0).is_ok());
        // A sample 11s later pushes the old one out of the 10s window.
        insert_dynamic(
            &mut buffer,
            vec![tf("map", "base", 11 * SEC, (1.0, 0.0, 0.0), rot_z(0.0))],
        );
        assert_eq!(
            buffer.lookup_transform("map", "base", 0),
            Err(TfError::OutOfRange {
                frame: "base".to_owned(),
                requested: 0,
                oldest: 11 * SEC,
                newest: 11 * SEC,
            })
        );
    }

    #[test]
    fn sample_count_limit_evicts_oldest() {
        let mut buffer = TfBuffer::new();
        // Insert cap+1 samples at 1ns spacing (below the time window) => only the first is dropped.
        for stamp in 0..=(MAX_SAMPLES_PER_FRAME as TimeNs) {
            insert_dynamic(
                &mut buffer,
                vec![tf("map", "base", stamp, (0.0, 0.0, 0.0), rot_z(0.0))],
            );
        }
        assert_eq!(
            buffer.lookup_transform("map", "base", 0),
            Err(TfError::OutOfRange {
                frame: "base".to_owned(),
                requested: 0,
                oldest: 1,
                newest: MAX_SAMPLES_PER_FRAME as TimeNs,
            })
        );
        assert!(buffer.lookup_transform("map", "base", 1).is_ok());
    }

    #[test]
    fn leading_slash_is_stripped_everywhere() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![tf("/map", "/base", 0, (1.0, 0.0, 0.0), rot_z(0.0))],
        );
        assert!(buffer.lookup_transform("map", "base", 0).is_ok());
        assert!(buffer.lookup_transform("/map", "/base", 0).is_ok());
    }

    #[test]
    fn parent_change_wins_and_clears_old_samples() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![tf("odom", "base", 0, (1.0, 0.0, 0.0), rot_z(0.0))],
        );
        insert_dynamic(
            &mut buffer,
            vec![tf("map", "base", SEC, (2.0, 0.0, 0.0), rot_z(0.0))],
        );
        // Parent becomes map (last-wins), and the old-parent sample (t=0) is gone.
        assert_isometry_eq(
            &buffer.lookup_transform("map", "base", SEC).unwrap(),
            &Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), rot_z(0.0)),
        );
        assert!(matches!(
            buffer.lookup_transform("map", "base", 0),
            Err(TfError::OutOfRange { .. })
        ));
    }

    #[test]
    fn time_jump_backwards_clears_dynamic_but_keeps_static() {
        let mut buffer = TfBuffer::new();
        insert_static(
            &mut buffer,
            vec![tf("base", "laser", 0, (0.5, 0.0, 0.0), rot_z(0.0))],
        );
        insert_dynamic(
            &mut buffer,
            vec![tf("map", "base", 100 * SEC, (1.0, 0.0, 0.0), rot_z(0.0))],
        );
        // Sim restart: a stamp past the retention window into the past clears dynamic only.
        insert_dynamic(
            &mut buffer,
            vec![tf("map", "base", 5 * SEC, (2.0, 0.0, 0.0), rot_z(0.0))],
        );
        assert!(matches!(
            buffer.lookup_transform("map", "base", 100 * SEC),
            Err(TfError::OutOfRange { .. })
        ));
        assert!(buffer.lookup_transform("map", "base", 5 * SEC).is_ok());
        assert!(buffer.lookup_transform_latest("base", "laser").is_ok());
    }

    #[test]
    fn same_frame_lookup_is_identity() {
        let buffer = TfBuffer::new();
        assert_isometry_eq(
            &buffer.lookup_transform("map", "map", 0).unwrap(),
            &Isometry3::identity(),
        );
    }

    #[test]
    fn out_of_order_arrival_is_sorted_and_same_stamp_overwrites() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "base", 2 * SEC, (2.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "base", SEC, (1.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "base", SEC, (5.0, 0.0, 0.0), rot_z(0.0)),
            ],
        );
        assert_isometry_eq(
            &buffer.lookup_transform("map", "base", SEC).unwrap(),
            &Isometry3::from_parts(Translation3::new(5.0, 0.0, 0.0), rot_z(0.0)),
        );
        // 0.5s is midway between t=0 and the (overwritten) t=1s sample = 2.5.
        assert_isometry_eq(
            &buffer.lookup_transform("map", "base", SEC / 2).unwrap(),
            &Isometry3::from_parts(Translation3::new(2.5, 0.0, 0.0), rot_z(0.0)),
        );
    }

    #[test]
    fn frames_and_roots_reflect_edges() {
        let mut buffer = TfBuffer::new();
        insert_dynamic(
            &mut buffer,
            vec![
                tf("map", "odom", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("odom", "base", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("world", "cam", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
            ],
        );
        insert_static(
            &mut buffer,
            vec![tf("base", "laser", 0, (0.0, 0.0, 0.0), rot_z(0.0))],
        );
        let frames = buffer.frames();
        let names: Vec<&str> = frames.iter().map(|f| f.name).collect();
        assert_eq!(names, vec!["base", "cam", "laser", "odom"]);
        assert!(frames.iter().find(|f| f.name == "laser").unwrap().is_static);
        assert!(!frames.iter().find(|f| f.name == "base").unwrap().is_static);
        assert_eq!(buffer.roots(), vec!["map".to_owned(), "world".to_owned()]);
        assert_eq!(
            buffer.frame_names(),
            vec!["base", "cam", "laser", "map", "odom", "world"]
        );
        assert!(buffer.contains_frame("map"));
        assert!(!buffer.contains_frame("nope"));
    }

    #[test]
    fn real_capture_flows_into_buffer() {
        use crate::decode::cdr::decode_message;
        use crate::decode::msg_parser::TypeRegistry;

        let payload = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tf_message_le.bin"
        ));
        let registry = TypeRegistry::with_embedded().unwrap();
        let value = decode_message(&registry, "tf2_msgs/msg/TFMessage", payload).unwrap();
        let transforms = transforms_from_value(&value).unwrap();
        assert_eq!(transforms.len(), 1);
        assert_eq!(transforms[0].parent, "map");
        assert_eq!(transforms[0].child, "base_link");
        assert_eq!(transforms[0].stamp, 100 * SEC + 5);

        let mut buffer = TfBuffer::new();
        buffer.insert(&TfUpdate {
            transforms,
            is_static: false,
            epoch: 0,
        });
        let map_t_base = buffer.lookup_transform_latest("map", "base_link").unwrap();
        assert!((map_t_base.translation.vector - Vector3::new(1.5, -0.5, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn transforms_from_value_rejects_wrong_structure_and_skips_empty_frames() {
        assert!(transforms_from_value(&Value::Struct(vec![])).is_err());
        assert!(transforms_from_value(&Value::I32(0)).is_err());
        let entry = |parent: &str, child: &str| {
            Value::Struct(vec![
                (
                    "header".to_owned(),
                    Value::Struct(vec![
                        (
                            "stamp".to_owned(),
                            Value::Struct(vec![
                                ("sec".to_owned(), Value::I32(1)),
                                ("nanosec".to_owned(), Value::U32(0)),
                            ]),
                        ),
                        ("frame_id".to_owned(), Value::String(parent.to_owned())),
                    ]),
                ),
                ("child_frame_id".to_owned(), Value::String(child.to_owned())),
                (
                    "transform".to_owned(),
                    Value::Struct(vec![
                        (
                            "translation".to_owned(),
                            Value::Struct(vec![
                                ("x".to_owned(), Value::F64(0.0)),
                                ("y".to_owned(), Value::F64(0.0)),
                                ("z".to_owned(), Value::F64(0.0)),
                            ]),
                        ),
                        (
                            "rotation".to_owned(),
                            Value::Struct(vec![
                                ("x".to_owned(), Value::F64(0.0)),
                                ("y".to_owned(), Value::F64(0.0)),
                                ("z".to_owned(), Value::F64(0.0)),
                                ("w".to_owned(), Value::F64(1.0)),
                            ]),
                        ),
                    ]),
                ),
            ])
        };
        // Empty-frame_id entries are skipped; only valid ones remain (tf2 behavior).
        let message = Value::Struct(vec![(
            "transforms".to_owned(),
            Value::Array(vec![entry("", "base"), entry("map", "base")]),
        )]);
        let transforms = transforms_from_value(&message).unwrap();
        assert_eq!(transforms.len(), 1);
        assert_eq!(transforms[0].parent, "map");
    }

    #[test]
    fn primary_root_prefers_largest_tree_over_alphabetical() {
        let mut buffer = TfBuffer::new();
        // A stray small tree (first alphabetically) versus the 3-frame main tree.
        insert_static(
            &mut buffer,
            vec![
                tf("camera_tof_link", "camera_tof", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("map", "odom", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("odom", "base_link", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("base_link", "lidar_link", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
            ],
        );
        assert_eq!(buffer.roots(), vec!["camera_tof_link", "map"]);
        assert_eq!(buffer.primary_root(), Some("map".to_owned()));
    }

    #[test]
    fn primary_root_ties_break_alphabetically_and_empty_is_none() {
        let mut buffer = TfBuffer::new();
        assert_eq!(buffer.primary_root(), None);
        insert_static(
            &mut buffer,
            vec![
                tf("zeta", "z_child", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
                tf("alpha", "a_child", 0, (0.0, 0.0, 0.0), rot_z(0.0)),
            ],
        );
        assert_eq!(buffer.primary_root(), Some("alpha".to_owned()));
    }
}
