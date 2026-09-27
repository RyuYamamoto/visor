// Instanced quad shader for point clouds (1 point = 1 instance, 16B; billboarded in the vertex shader)

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    // Camera right/up directions (world-space unit vectors; w is unused padding)
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    // xy = viewport physical pixel size (for the Points style's screen-fixed size conversion)
    viewport: vec4<f32>,
};

struct BatchUniforms {
    // Local coords -> fixed-frame model matrix (points follow via uniform update only, no re-bake)
    model: mat4x4<f32>,
    // Half of the quad's side length ([m] or [px] depending on size_mode)
    half_size: f32,
    // 0 = world-fixed [m] (Squares), 1 = screen-fixed [px] (Points)
    size_mode: u32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> batch: BatchUniforms;

struct VsIn {
    @builtin(vertex_index) corner: u32,
    @location(0) position: vec3<f32>,
    @location(1) color: u32,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Quad-local offset (-1..1 at the expanded edges), used to round off the corners
    @location(1) offset: vec2<f32>,
    // Radius where the solid core ends and the feather begins
    @location(2) core: f32,
};

// Feather band outside the requested size: [px] for screen-fixed points, a fraction of the size for world-fixed ones.
const FEATHER_PX: f32 = 0.75;
const FEATHER_RATIO: f32 = 0.25;

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let center = (batch.model * vec4<f32>(in.position, 1.0)).xyz;
    // Expand the TriangleStrip's 4 vertices from vertex_index to (±1, ±1)
    let ox = select(-1.0, 1.0, (in.corner & 1u) == 1u);
    let oy = select(-1.0, 1.0, (in.corner & 2u) == 2u);
    var clip: vec4<f32>;
    var core: f32;
    if batch.size_mode == 1u {
        // Screen-fixed size: expand in clip space by multiplying the NDC offset by w (same pixel count regardless of distance)
        let half = batch.half_size + FEATHER_PX;
        core = batch.half_size / half;
        clip = frame.view_proj * vec4<f32>(center, 1.0);
        let ndc_per_px = 2.0 / frame.viewport.xy;
        clip.x += ox * half * ndc_per_px.x * clip.w;
        clip.y += oy * half * ndc_per_px.y * clip.w;
    } else {
        // World-fixed size: expand along the camera basis directions in [m]
        let half = batch.half_size * (1.0 + FEATHER_RATIO);
        core = 1.0 / (1.0 + FEATHER_RATIO);
        let world = center + (frame.cam_right.xyz * ox + frame.cam_up.xyz * oy) * half;
        clip = frame.view_proj * vec4<f32>(world, 1.0);
    }
    var out: VsOut;
    out.clip_position = clip;
    // Vertex color is linear-encoded RGBA8 (converted CPU-side; same "shader emits linear values" convention as lines)
    out.color = unpack4x8unorm(in.color);
    out.offset = vec2<f32>(ox, oy);
    out.core = core;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Round the quad into a soft disc: solid core, feathered rim (analytic AA that also reads as a faint glow).
    let edge = 1.0 - smoothstep(in.core, 1.0, length(in.offset));
    return vec4<f32>(in.color.rgb, in.color.a * edge);
}
