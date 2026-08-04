@group(0) @binding(0) var normals_texture: texture_2d<f32>;
@group(0) @binding(1) var thickness_texture: texture_2d<f32>;

const N1 = 1.0;
const N2 = 1.33; // approx water

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    return vec4(pos[vi], 0.0, 1.0);
}

fn frensel(angle: f32) -> f32 {
    var f0: f32;
    f0 = (N1 - N2) / (N1 + N2);
    f0 = f0 * f0;
    return f0 + (1 - f0) * pow(1 - cos(angle), 5);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let normal = textureLoad(normals_texture, vec2<i32>(frag_pos.xy), 0);
    let thickness = textureLoad(thickness_texture, vec2<i32>(frag_pos.xy), 0).r;
    return vec4(thickness, 0.0, thickness, 1.0);
}
