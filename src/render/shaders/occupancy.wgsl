// Occupancy-grid tile shader (reads the R8Uint texture directly via textureLoad and colors it through a 256-entry LUT)

struct FrameUniforms {
    view_proj: mat4x4<f32>,
};

struct Lut {
    // Cell value byte -> RGBA, per batch (array stride is 16B)
    colors: array<vec4<f32>, 256>,
};

struct BatchUniforms {
    // Local coords -> fixed-frame model matrix (= fixed_from_frame x origin Pose)
    model: mat4x4<f32>,
    // Quad local XY size [m] (= resolution x width / height)
    size_m: vec2<f32>,
    // Overall tile opacity (multiplied into the LUT alpha)
    alpha: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> batch: BatchUniforms;
// Uint textures are non-filterable, so read directly at integer coords = no sampler and no blurred cell edges
@group(1) @binding(1) var cells: texture_2d<u32>;
@group(1) @binding(2) var<uniform> lut: Lut;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) corner: u32) -> VsOut {
    // Expand the TriangleStrip's 4 vertices to (0,0)(1,0)(0,1)(1,1); u->x (cell column), v->y (cell row), no vertical flip
    let u = f32(corner & 1u);
    let v = f32((corner >> 1u) & 1u);
    var out: VsOut;
    let local = vec4<f32>(u * batch.size_m.x, v * batch.size_m.y, 0.0, 1.0);
    out.clip_position = frame.view_proj * (batch.model * local);
    out.uv = vec2<f32>(u, v);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = textureDimensions(cells);
    // Clamp to dims-1 so uv=1.0 at the edge does not go out of range
    let x = min(u32(in.uv.x * f32(dims.x)), dims.x - 1u);
    let y = min(u32(in.uv.y * f32(dims.y)), dims.y - 1u);
    let value = textureLoad(cells, vec2<u32>(x, y), 0).r;
    var color = lut.colors[value];
    color.a = color.a * batch.alpha;
    return color;
}
