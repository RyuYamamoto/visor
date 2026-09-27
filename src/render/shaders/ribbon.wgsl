// Polyline ribbon with mitered joins: adjacent segments share their edge, so a curve reads as one continuous band instead of separate quads meeting at their corners.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    // xy = viewport physical pixel size (screen-space width conversion)
    viewport: vec4<f32>,
};

// Same group(1) layout points.wgsl uses; only half_size / size_mode are read (the points are pre-baked into the fixed frame).
struct BatchUniforms {
    model: mat4x4<f32>,
    half_size: f32,
    size_mode: u32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> batch: BatchUniforms;

// One instance = one segment, reading its two points plus the neighbour on each side (the buffer is bound four times at one-point offsets, with the first and last point duplicated so the ends have neighbours).
struct VsIn {
    @builtin(vertex_index) corner: u32,
    @location(0) prev: vec3<f32>,
    @location(1) prev_color: vec4<f32>,
    @location(2) p0: vec3<f32>,
    @location(3) c0: vec4<f32>,
    @location(4) p1: vec3<f32>,
    @location(5) c1: vec4<f32>,
    @location(6) next: vec3<f32>,
    @location(7) next_color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Signed position across the ribbon (-1..1 at the expanded edges)
    @location(1) across: f32,
    // |across| where the solid core ends and the feather begins
    @location(2) core: f32,
};

const FEATHER_PX: f32 = 0.55;
const FEATHER_RATIO: f32 = 0.25;
const EPSILON: f32 = 1.0e-6;
// Lower bound on the miter shortening factor: a sharper corner than this stops extending instead of growing a spike.
const MITER_MIN: f32 = 0.35;

fn collapsed() -> VsOut {
    var out: VsOut;
    out.clip_position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.color = vec4<f32>(0.0);
    out.across = 0.0;
    out.core = 1.0;
    return out;
}

/// Offset direction of a segment in world space: perpendicular to both the segment and the view, so the ribbon faces the camera.
fn world_offset_dir(segment: vec3<f32>, forward: vec3<f32>) -> vec3<f32> {
    let dir = cross(segment, forward);
    if length(dir) < EPSILON {
        return frame.cam_right.xyz;
    }
    return normalize(dir);
}

/// Miter direction and length at one joint, given the offset directions of the two segments meeting there.
fn miter(into: vec3<f32>, out_of: vec3<f32>, half: f32) -> vec3<f32> {
    let sum = into + out_of;
    if length(sum) < EPSILON {
        return out_of * half;
    }
    let dir = normalize(sum);
    return dir * (half / max(dot(dir, out_of), MITER_MIN));
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let segment = in.p1 - in.p0;
    if length(segment) < EPSILON {
        return collapsed();
    }
    // bit0 = which endpoint of the segment, bit1 = which side of the ribbon.
    let at_end = (in.corner & 1u) == 1u;
    let across = select(-1.0, 1.0, (in.corner & 2u) == 2u);
    let point = select(in.p0, in.p1, at_end);
    var out: VsOut;
    if batch.size_mode == 1u {
        let clip0 = frame.view_proj * vec4<f32>(in.p0, 1.0);
        let clip1 = frame.view_proj * vec4<f32>(in.p1, 1.0);
        if clip0.w <= 0.0 || clip1.w <= 0.0 {
            return collapsed();
        }
        let half_px = batch.half_size + FEATHER_PX;
        let half_viewport = frame.viewport.xy * 0.5;
        let px0 = (clip0.xy / clip0.w) * half_viewport;
        let px1 = (clip1.xy / clip1.w) * half_viewport;
        let dir = normalize(px1 - px0);
        let normal = vec2<f32>(-dir.y, dir.x);
        // The neighbour on this end decides the joint; a duplicated end point falls back to this segment's own normal.
        let neighbour = select(in.prev, in.next, at_end);
        let clip_n = frame.view_proj * vec4<f32>(neighbour, 1.0);
        var offset = normal * half_px;
        if clip_n.w > 0.0 {
            let px_n = (clip_n.xy / clip_n.w) * half_viewport;
            let other = select(px0 - px_n, px_n - px1, at_end);
            if length(other) > EPSILON {
                let dir_other = normalize(other);
                let normal_other = vec2<f32>(-dir_other.y, dir_other.x);
                let into = select(normal_other, normal, at_end);
                let out_of = select(normal, normal_other, at_end);
                let sum = into + out_of;
                if length(sum) > EPSILON {
                    let bisector = normalize(sum);
                    offset = bisector * (half_px / max(dot(bisector, out_of), MITER_MIN));
                }
            }
        }
        let clip = select(clip0, clip1, at_end);
        let offset_ndc = (offset * across) / half_viewport;
        out.clip_position = vec4<f32>(clip.xy + offset_ndc * clip.w, clip.z, clip.w);
        out.core = batch.half_size / half_px;
    } else {
        let half = batch.half_size * (1.0 + FEATHER_RATIO);
        let forward = cross(frame.cam_right.xyz, frame.cam_up.xyz);
        let own = world_offset_dir(segment, forward);
        // The neighbour on this end decides the joint; a duplicated end point falls back to this segment's own offset.
        let neighbour = select(in.prev, in.next, at_end);
        let other_segment = select(in.p0 - neighbour, neighbour - in.p1, at_end);
        var offset = own * half;
        if length(other_segment) > EPSILON {
            let other = world_offset_dir(other_segment, forward);
            let into = select(other, own, at_end);
            let out_of = select(own, other, at_end);
            offset = miter(into, out_of, half);
        }
        out.clip_position = frame.view_proj * vec4<f32>(point + offset * across, 1.0);
        out.core = 1.0 / (1.0 + FEATHER_RATIO);
    }
    out.color = select(in.c0, in.c1, at_end);
    out.across = across;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let edge = 1.0 - smoothstep(in.core, 1.0, abs(in.across));
    return vec4<f32>(in.color.rgb, in.color.a * edge);
}
