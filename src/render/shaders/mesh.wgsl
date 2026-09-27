// Posed-mesh shader: local-coordinate triangles placed by a rigid model matrix, lambert-shaded in-shader by a world-fixed light

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    // xyz = world-fixed light direction (normalized here), w = ambient coefficient
    light_dir: vec4<f32>,
    // rgb = linear light color, w = diffuse coefficient
    light_color: vec4<f32>,
};

struct BatchUniforms {
    // Local coords -> fixed-frame model matrix (rigid, so normals rotate with it; scale is baked into the vertices)
    model: mat4x4<f32>,
    // Overall opacity of the batch
    alpha: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> batch: BatchUniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = (batch.model * vec4<f32>(in.position, 1.0)).xyz;
    var out: VsOut;
    out.clip_position = frame.view_proj * vec4<f32>(world, 1.0);
    // Rigid model matrix: rotating the normal suffices (no inverse transpose needed)
    out.world_normal = (batch.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.color = vec4<f32>(in.color.rgb, in.color.a * batch.alpha);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light = normalize(frame.light_dir.xyz);
    // Ambient + diffuse coefficients and light color come from theme.rs via the uniform (no color literals here)
    let shade = frame.light_dir.w + frame.light_color.w * max(dot(normal, light), 0.0);
    // Vertex color is linear-encoded RGBA8 (same "shader emits linear values" convention as lines/points)
    return vec4<f32>(in.color.rgb * shade * frame.light_color.rgb, in.color.a);
}
