@group(0) @binding(0) var normals_texture: texture_2d<f32>;
@group(0) @binding(1) var thickness_texture: texture_2d<f32>;
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    return vec4(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // let d = textureLoad(normals_texture, vec2<i32>(frag_pos.xy), 0);
    // return d;
    let d = textureLoad(thickness_texture, vec2<i32>(frag_pos.xy), 0).r;
    return vec4(d, 0.0, d, 1.0);
}
