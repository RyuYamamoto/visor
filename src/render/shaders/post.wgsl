// Post-processing over the offscreen scene target: bright-pass, separable blur, and the final composite.

struct PostUniforms {
    // 1 / texture size of the source being sampled
    texel: vec2<f32>,
    // Luminance above which a pixel feeds the glow
    threshold: f32,
    // How much blurred light is added back in the composite
    intensity: f32,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> post: PostUniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle: 3 vertices covering the viewport, no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    var out: VsOut;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_bright(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(source, source_sampler, in.uv).rgb;
    // Keep the part of each pixel that is brighter than the threshold, so only accents glow.
    let excess = max(luminance(color) - post.threshold, 0.0) / max(1.0 - post.threshold, 1e-4);
    return vec4<f32>(color * excess, 1.0);
}

// 9-tap Gaussian, run once per axis.
const WEIGHTS: array<f32, 5> = array<f32, 5>(0.227027, 0.194594, 0.121621, 0.054054, 0.016216);

fn blur(uv: vec2<f32>, step: vec2<f32>) -> vec4<f32> {
    var sum = textureSample(source, source_sampler, uv).rgb * WEIGHTS[0];
    for (var i = 1; i < 5; i++) {
        let offset = step * f32(i);
        sum += textureSample(source, source_sampler, uv + offset).rgb * WEIGHTS[i];
        sum += textureSample(source, source_sampler, uv - offset).rgb * WEIGHTS[i];
    }
    return vec4<f32>(sum, 1.0);
}

@fragment
fn fs_blur_h(in: VsOut) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2<f32>(post.texel.x, 0.0));
}

@fragment
fn fs_blur_v(in: VsOut) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2<f32>(0.0, post.texel.y));
}
