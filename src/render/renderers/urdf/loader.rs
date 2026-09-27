//! Background mesh loading: one worker thread per RobotModel item does resolution, disk I/O, parsing and vertex expansion, so the UI thread only ever receives a finished MeshBatch.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};

use crate::render::MeshBatch;

use super::collada;
use super::geometry::{self, MeshParams};
use super::resolve::{self, MeshRoots};

/// Refuse to read anything larger than this; a pathological file would otherwise be expanded into memory before failing.
pub const MAX_MESH_FILE_BYTES: u64 = 256 * 1024 * 1024;
/// Vertex bytes the per-item cache keeps before evicting the least recently used entry.
pub const CACHE_BUDGET_BYTES: usize = 128 * 1024 * 1024;

/// One visual's mesh to load, tagged with the model generation that asked for it.
#[derive(Debug, Clone)]
pub struct MeshRequest {
    pub epoch: u64,
    pub visual_index: usize,
    /// URDF `<mesh filename>` verbatim; resolution happens on the worker because it probes the filesystem.
    pub uri: String,
    pub urdf_dir: Option<PathBuf>,
    pub roots: Arc<MeshRoots>,
    pub scale: [f64; 3],
    pub fallback_rgba: [u8; 4],
}

/// A finished mesh, plus the numbers the settings_ui list reports.
#[derive(Debug, Clone)]
pub struct LoadedMesh {
    pub batch: MeshBatch,
    pub path: PathBuf,
    pub triangles: usize,
    pub skipped: usize,
    /// Wall time spent in the worker, including a cache lookup that hit.
    pub load_ms: u32,
    pub cached: bool,
}

/// Worker result for one request. `Err` is the message shown in the visuals list, including the paths that were tried.
#[derive(Debug, Clone)]
pub struct MeshResponse {
    pub epoch: u64,
    pub visual_index: usize,
    pub result: Result<LoadedMesh, String>,
}

/// Handle to the worker thread. Dropping it closes the request channel, which ends the thread without a join.
pub struct MeshLoader {
    requests: Sender<MeshRequest>,
    responses: Receiver<MeshResponse>,
    epoch: Arc<AtomicU64>,
}

impl MeshLoader {
    pub fn spawn() -> Self {
        Self::spawn_with_budget(CACHE_BUDGET_BYTES)
    }

    fn spawn_with_budget(budget: usize) -> Self {
        let (requests, request_rx) = crossbeam_channel::unbounded::<MeshRequest>();
        let (response_tx, responses) = crossbeam_channel::unbounded::<MeshResponse>();
        let epoch = Arc::new(AtomicU64::new(0));
        let worker_epoch = Arc::clone(&epoch);
        let spawned = std::thread::Builder::new()
            .name("visor-mesh-loader".to_owned())
            .spawn(move || worker(&request_rx, &response_tx, &worker_epoch, budget));
        if let Err(e) = spawned {
            // Without a worker every send fails, which surfaces per visual instead of taking the app down.
            eprintln!("visor: could not start the mesh loader thread: {e}");
        }
        Self {
            requests,
            responses,
            epoch,
        }
    }

    /// Tell the worker which model generation is current; older requests are dropped before any I/O starts.
    pub fn set_epoch(&self, epoch: u64) {
        self.epoch.store(epoch, Ordering::Relaxed);
    }

    pub fn request(&self, request: MeshRequest) -> Result<(), String> {
        self.requests
            .send(request)
            .map_err(|_| "mesh loader thread is not running".to_owned())
    }

    /// Non-blocking result pickup, called once per frame from poll(); a dead worker reads the same as an empty queue.
    pub fn try_recv(&self) -> Option<MeshResponse> {
        self.responses.try_recv().ok()
    }
}

fn worker(
    requests: &Receiver<MeshRequest>,
    responses: &Sender<MeshResponse>,
    epoch: &AtomicU64,
    budget: usize,
) {
    let mut cache = MeshCache::with_budget(budget);
    while let Ok(request) = requests.recv() {
        if request.epoch < epoch.load(Ordering::Relaxed) {
            continue;
        }
        let job = std::panic::AssertUnwindSafe(|| run(&mut cache, &request));
        let result = std::panic::catch_unwind(job).unwrap_or_else(|_| {
            Err(format!(
                "internal error while loading {} (the mesh loader recovered)",
                request.uri
            ))
        });
        let response = MeshResponse {
            epoch: request.epoch,
            visual_index: request.visual_index,
            result,
        };
        if responses.send(response).is_err() {
            break;
        }
    }
}

/// Resolve, load and bake one request, reusing the cached batch when the same file was already built the same way.
fn run(cache: &mut MeshCache, request: &MeshRequest) -> Result<LoadedMesh, String> {
    let path = resolve::resolve(
        &request.uri,
        request.urdf_dir.as_deref(),
        &request.roots,
        |candidate| candidate.exists(),
    )
    .map_err(|e| e.to_string())?;
    let key = CacheKey {
        path: path.clone(),
        scale: request.scale.map(f64::to_bits),
        fallback_rgba: request.fallback_rgba,
    };
    let started = Instant::now();
    let mut cached = true;
    let entry = cache.get_or_insert(key, || {
        cached = false;
        let params = MeshParams {
            scale: request.scale,
            fallback_rgba: request.fallback_rgba,
            up_axis: geometry::UpAxis::Z,
            material_groups: Vec::new(),
        };
        build_from_disk(&path, params, MAX_MESH_FILE_BYTES)
    })?;
    Ok(LoadedMesh {
        batch: entry.batch,
        path,
        triangles: entry.triangles,
        skipped: entry.skipped,
        load_ms: started.elapsed().as_millis().min(u32::MAX as u128) as u32,
        cached,
    })
}

/// Read one mesh file and expand it into vertices. `params.up_axis` is overwritten from the file's own `<up_axis>`.
fn build_from_disk(
    path: &Path,
    mut params: MeshParams,
    size_limit: u64,
) -> Result<CachedMesh, String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if size > size_limit {
        return Err(format!(
            "{}: {size} bytes exceeds the {size_limit} byte mesh size limit",
            path.display()
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    params.up_axis = geometry::detect_up_axis(&bytes);
    params.material_groups = collada::material_groups(&bytes).unwrap_or_default();
    let scene = mesh_loader::Loader::default()
        .load_from_slice(&bytes, path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let built = geometry::build(&scene, &params).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(CachedMesh {
        batch: built.batch,
        triangles: built.triangles,
        skipped: built.skipped,
    })
}

/// Identity of a baked mesh: same file, same URDF scale, same fallback color, because all three are baked into the vertices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    path: PathBuf,
    scale: [u64; 3],
    fallback_rgba: [u8; 4],
}

/// Cached bake result. Cloning shares the vertex bytes through the batch's Arc.
#[derive(Debug, Clone)]
struct CachedMesh {
    batch: MeshBatch,
    triangles: usize,
    skipped: usize,
}

/// Worker-local LRU cache (no locking: only the worker thread touches it).
struct MeshCache {
    entries: Vec<(CacheKey, CachedMesh)>,
    bytes: usize,
    budget: usize,
}

impl MeshCache {
    fn with_budget(budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            budget,
        }
    }

    /// Cached bake for `key`, otherwise `build` (errors are not cached, so fixing the file and reloading works).
    fn get_or_insert(
        &mut self,
        key: CacheKey,
        build: impl FnOnce() -> Result<CachedMesh, String>,
    ) -> Result<CachedMesh, String> {
        if let Some(at) = self.entries.iter().position(|(k, _)| *k == key) {
            let entry = self.entries.remove(at);
            self.entries.push(entry);
            return Ok(self.entries.last().expect("just pushed").1.clone());
        }
        let built = build()?;
        self.bytes += built.batch.bytes.len();
        self.entries.push((key, built.clone()));
        while self.bytes > self.budget && self.entries.len() > 1 {
            let (_, evicted) = self.entries.remove(0);
            self.bytes -= evicted.batch.bytes.len();
        }
        Ok(built)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::MeshBatchBuilder;
    use std::time::Duration;

    fn mesh_of(vertices: usize) -> CachedMesh {
        let mut builder = MeshBatchBuilder::with_capacity(vertices);
        for _ in 0..vertices {
            builder.push_vertex([0.0; 3], [0.0, 0.0, 1.0], [1, 2, 3, 4]);
        }
        CachedMesh {
            batch: builder.build(),
            triangles: vertices / 3,
            skipped: 0,
        }
    }

    fn key(path: &str, scale: [f64; 3], rgba: [u8; 4]) -> CacheKey {
        CacheKey {
            path: PathBuf::from(path),
            scale: scale.map(f64::to_bits),
            fallback_rgba: rgba,
        }
    }

    /// Request built against the mesh fixture directory as a user mesh root.
    fn request(uri: &str, epoch: u64) -> MeshRequest {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        MeshRequest {
            epoch,
            visual_index: 0,
            uri: uri.to_owned(),
            urdf_dir: None,
            roots: Arc::new(MeshRoots {
                user: vec![root],
                ..Default::default()
            }),
            scale: [1.0; 3],
            fallback_rgba: [10, 20, 30, 255],
        }
    }

    /// Wait for one response, failing the test rather than hanging if the worker never answers.
    fn recv(loader: &MeshLoader) -> MeshResponse {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if let Some(response) = loader.try_recv() {
                return response;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("no response from the mesh loader");
    }

    #[test]
    fn the_same_path_scale_and_color_is_built_once() {
        let mut cache = MeshCache::with_budget(CACHE_BUDGET_BYTES);
        let mut builds = 0;
        let ask = |cache: &mut MeshCache, key: CacheKey, builds: &mut usize| {
            cache
                .get_or_insert(key, || {
                    *builds += 1;
                    Ok(mesh_of(3))
                })
                .expect("builds")
        };
        let first = key("/a.stl", [1.0; 3], [1, 1, 1, 255]);
        ask(&mut cache, first.clone(), &mut builds);
        ask(&mut cache, first.clone(), &mut builds);
        assert_eq!(builds, 1);
        // Scale and fallback color are baked into the vertices, so either one changing is a different entry.
        ask(&mut cache, key("/a.stl", [2.0; 3], [1, 1, 1, 255]), &mut builds);
        ask(&mut cache, key("/a.stl", [1.0; 3], [9, 9, 9, 255]), &mut builds);
        ask(&mut cache, key("/b.stl", [1.0; 3], [1, 1, 1, 255]), &mut builds);
        assert_eq!(builds, 4);
        ask(&mut cache, first, &mut builds);
        assert_eq!(builds, 4);
        // An error is not remembered, so a later attempt runs again.
        let mut attempts = 0;
        for _ in 0..2 {
            let result = cache.get_or_insert(key("/bad.stl", [1.0; 3], [0; 4]), || {
                attempts += 1;
                Err("boom".to_owned())
            });
            assert_eq!(result.unwrap_err(), "boom");
        }
        assert_eq!(attempts, 2);
    }

    #[test]
    fn the_least_recently_used_entry_is_evicted_when_over_budget() {
        let vertex_bytes = mesh_of(3).batch.bytes.len();
        let mut cache = MeshCache::with_budget(vertex_bytes * 2);
        let mut builds = 0;
        let ask = |cache: &mut MeshCache, name: &str, builds: &mut usize| {
            cache
                .get_or_insert(key(name, [1.0; 3], [0; 4]), || {
                    *builds += 1;
                    Ok(mesh_of(3))
                })
                .expect("builds")
        };
        ask(&mut cache, "/a.stl", &mut builds);
        ask(&mut cache, "/b.stl", &mut builds);
        // Touching /a.stl makes /b.stl the oldest, so adding /c.stl evicts /b.stl.
        ask(&mut cache, "/a.stl", &mut builds);
        ask(&mut cache, "/c.stl", &mut builds);
        assert_eq!(builds, 3);
        ask(&mut cache, "/a.stl", &mut builds);
        assert_eq!(builds, 3);
        ask(&mut cache, "/b.stl", &mut builds);
        assert_eq!(builds, 4);
        // A single mesh larger than the whole budget still stays usable rather than evicting itself.
        let mut tiny = MeshCache::with_budget(1);
        tiny.get_or_insert(key("/big.stl", [1.0; 3], [0; 4]), || Ok(mesh_of(30)))
            .expect("builds");
        assert_eq!(tiny.entries.len(), 1);
    }

    #[test]
    fn the_worker_returns_a_baked_mesh_for_a_package_uri() {
        let loader = MeshLoader::spawn();
        loader
            .request(request("package://mesh/triangle_ascii.stl", 0))
            .expect("queued");
        let response = recv(&loader);
        assert_eq!((response.epoch, response.visual_index), (0, 0));
        let mesh = response.result.expect("loads");
        assert_eq!((mesh.triangles, mesh.skipped), (2, 0));
        assert_eq!(mesh.batch.count, 6);
        assert!(mesh.path.ends_with("mesh/triangle_ascii.stl"));
        assert!(!mesh.cached);
        // The second request for the same key comes back from the worker's cache.
        loader
            .request(request("package://mesh/triangle_ascii.stl", 0))
            .expect("queued");
        let again = recv(&loader).result.expect("loads");
        assert!(again.cached);
        assert!(Arc::ptr_eq(&mesh.batch.bytes, &again.batch.bytes));
    }

    #[test]
    fn a_missing_package_reports_every_path_it_tried() {
        let loader = MeshLoader::spawn();
        loader
            .request(request("package://nowhere/robot.stl", 0))
            .expect("queued");
        let error = recv(&loader).result.unwrap_err();
        assert!(error.contains("nowhere"), "{error}");
        assert!(error.contains("tried:"), "{error}");
        // The failure is per visual: the worker is still there for the next request.
        loader
            .request(request("package://mesh/triangle_ascii.stl", 0))
            .expect("queued");
        assert!(recv(&loader).result.is_ok());
    }

    #[test]
    fn requests_from_an_older_epoch_are_dropped_before_any_work() {
        let loader = MeshLoader::spawn();
        loader.set_epoch(5);
        loader
            .request(request("package://mesh/triangle_ascii.stl", 1))
            .expect("queued");
        loader
            .request(request("package://mesh/triangle_ascii.stl", 5))
            .expect("queued");
        // Only the current generation answers, and it is the first thing that arrives.
        let response = recv(&loader);
        assert_eq!(response.epoch, 5);
        assert!(response.result.is_ok());
        std::thread::sleep(Duration::from_millis(50));
        assert!(loader.try_recv().is_none());
    }

    #[test]
    fn a_file_over_the_size_limit_is_not_read() {
        let path = geometry::fixture_path("triangle_ascii.stl");
        let error = build_from_disk(&path, MeshParams::default(), 1).unwrap_err();
        assert!(error.contains("size limit"), "{error}");
        assert!(build_from_disk(&path, MeshParams::default(), MAX_MESH_FILE_BYTES).is_ok());
        // A missing file reports the OS reason with the path.
        let missing = build_from_disk(
            Path::new("/nonexistent/mesh.stl"),
            MeshParams::default(),
            MAX_MESH_FILE_BYTES,
        )
        .unwrap_err();
        assert!(missing.contains("/nonexistent/mesh.stl"), "{missing}");
    }
}
