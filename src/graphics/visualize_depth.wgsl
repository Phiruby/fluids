struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    particle_radius: f32,
    screen_height: f32,
}

@group(0) @binding(0) var normals_texture: texture_2d<f32>;
@group(0) @binding(1) var thickness_texture: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;
@group(1) @binding(0) var<uniform> ubo: CameraUniform;
const N1 = 1.0;
const N2 = 1.33; // approx water
const LIGHT_POS = vec3<f32>(0.0, -1.0, 2.0);
const FLUID_COLOR = vec3<f32>(0.0, 0.5, 0.99);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    return vec4(pos[vi], 0.0, 1.0);
}

fn fresnel(angle: f32) -> f32 {
    var f0: f32;
    f0 = (N1 - N2) / (N1 + N2);
    f0 = f0 * f0;
    return f0 + (1 - f0) * pow(1 - cos(angle), 5);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(normals_texture));
    let uv = frag_pos.xy / dim;

    let normal_sample = textureLoad(normals_texture, vec2<i32>(frag_pos.xy), 0);
    if normal_sample.a == 0.0 {
        discard;
    }
    let n = normalize(normal_sample.xyz * 2.0 - 1.0);
    let thickness = textureLoad(thickness_texture, vec2<i32>(frag_pos.xy), 0).r;
    let depth = textureLoad(depth_texture, vec2<i32>(frag_pos.xy), 0).r;
    // background where depth is not updated
    if depth >= 1.0 {
        discard;
    }

    // need to reconstruct world pos for lighting
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    var world_pos_4 = ubo.inv_view_proj * ndc;
    let world_pos = world_pos_4.xyz / world_pos_4.w;

    let v = normalize(ubo.camera_pos - world_pos);
    let l = normalize(LIGHT_POS - world_pos);
    let h = normalize(v + l);

    let n_dot_v = saturate(dot(n, v));
    let n_dot_h = saturate(dot(n, h));
    let f_val = fresnel(n_dot_v);

    let a = mix(FLUID_COLOR, vec3<f32>(0.0), exp(-thickness));
    let b = vec3<f32>(0.1, 0.1, 0.1); // scene color
    let ks = vec3<f32>(0.9);
    let alpha = 3.0;

    let term1 = a * (1.0 - f_val);
    let term2 = b * f_val;
    let term3 = ks * pow(n_dot_h, alpha);

    let cout = term1 + term2 + term3;
    return vec4(cout, 1.0);
}

fn saturate(val: f32) -> f32 {
    return clamp(val, 0.0, 1.0);
}
