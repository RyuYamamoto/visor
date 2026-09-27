// Screen-space quad expansion for line segments (1 segment = 1 instance, 56B) with a feathered edge for analytic AA.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    // xy = viewport physical pixel size (screen-space width conversion)
    viewport: vec4<f32>,
};

// Same group(1) layout points.wgsl uses; only half_size / size_mode are read here (model is unused: lines are pre-baked into the fixed frame).
struct BatchUniforms {
    model: mat4x4<f32>,
    // Half the line width ([m] when size_mode is 0, [px] when 1)
    half_size: f32,
    // 0 = world-fixed [m] (camera-facing ribbon, RViz Billboards), 1 = screen-fixed [px]
    size_mode: u32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> batch: BatchUniforms;

struct VsIn {
    @builtin(vertex_index) corner: u32,
    @location(0) p0: vec3<f32>,
    @location(1) c0: vec4<f32>,
    @location(2) p1: vec3<f32>,
    @location(3) c1: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Signed position across the ribbon (-1..1 at the expanded edges)
    @location(1) across: f32,
    // |across| where the solid core ends and the feather begins
    @location(2) core: f32,
    // World XY, used by fs_grid to measure the on-screen cell size from its derivatives
    @location(3) world_xy: vec2<f32>,
};

// Feather band outside the requested width (edge AA / faint glow): [px] for screen-fixed lines, a fraction of the width for world-fixed ones.
const FEATHER_PX: f32 = 0.55;
const FEATHER_RATIO: f32 = 0.25;
// Fallback offset direction for a segment pointing straight at the camera, where the cross product degenerates.
const DEGENERATE_EPSILON: f32 = 1.0e-6;

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let clip0 = frame.view_proj * vec4<f32>(in.p0, 1.0);
    let clip1 = frame.view_proj * vec4<f32>(in.p1, 1.0);
    // A segment crossing the camera plane has no screen-space direction; collapse it outside the depth range.
    if clip0.w <= 0.0 || clip1.w <= 0.0 {
        out.clip_position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.across = 0.0;
        out.core = 1.0;
        out.world_xy = vec2<f32>(0.0);
        return out;
    }
    // Expand the TriangleStrip's 4 vertices from vertex_index: bit0 = which endpoint, bit1 = which side.
    let at_end = (in.corner & 1u) == 1u;
    let across = select(-1.0, 1.0, (in.corner & 2u) == 2u);
    let clip = select(clip0, clip1, at_end);
    if batch.size_mode == 1u {
        // Screen-fixed width: take the perpendicular in pixel space so the width stays constant with distance.
        let half_px = batch.half_size + FEATHER_PX;
        let half_viewport = frame.viewport.xy * 0.5;
        let px0 = (clip0.xy / clip0.w) * half_viewport;
        let px1 = (clip1.xy / clip1.w) * half_viewport;
        let delta = px1 - px0;
        let len = length(delta);
        let dir = select(vec2<f32>(1.0, 0.0), delta / len, len > 1e-6);
        let normal = vec2<f32>(-dir.y, dir.x);
        let offset_ndc = (normal * across * half_px) / half_viewport;
        out.clip_position = vec4<f32>(clip.xy + offset_ndc * clip.w, clip.z, clip.w);
        out.core = batch.half_size / half_px;
    } else {
        // World-fixed width: offset in [m] perpendicular to both the segment and the view direction, so the ribbon always faces the camera.
        let half = batch.half_size * (1.0 + FEATHER_RATIO);
        let point = select(in.p0, in.p1, at_end);
        let segment = in.p1 - in.p0;
        let forward = cross(frame.cam_right.xyz, frame.cam_up.xyz);
        var offset_dir = cross(segment, forward);
        // Looking down the segment leaves no perpendicular; any camera-plane direction will do, and the ribbon is edge-on anyway.
        if length(offset_dir) < DEGENERATE_EPSILON {
            offset_dir = frame.cam_right.xyz;
        }
        let world = point + normalize(offset_dir) * across * half;
        out.clip_position = frame.view_proj * vec4<f32>(world, 1.0);
        out.core = 1.0 / (1.0 + FEATHER_RATIO);
    }
    out.color = select(in.c0, in.c1, at_end);
    out.across = across;
    out.world_xy = select(in.p0.xy, in.p1.xy, at_end);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let edge = 1.0 - smoothstep(in.core, 1.0, abs(in.across));
    return vec4<f32>(in.color.rgb, in.color.a * edge);
}

// Grid cell size [m]; must match viewport::GRID_STEP.
const GRID_CELL_M: f32 = 1.0;
// On-screen cell size [px] where the grid is gone / at full strength (kills the moire wash when zoomed out).
const MIN_CELL_PX: f32 = 5.0;
const FULL_CELL_PX: f32 = 16.0;

@fragment
fn fs_grid(in: VsOut) -> @location(0) vec4<f32> {
    let edge = 1.0 - smoothstep(in.core, 1.0, abs(in.across));
    let meters_per_px = max(length(fwidth(in.world_xy)), 1e-6);
    let density = smoothstep(MIN_CELL_PX, FULL_CELL_PX, GRID_CELL_M / meters_per_px);
    return vec4<f32>(in.color.rgb, in.color.a * edge * density);
}
