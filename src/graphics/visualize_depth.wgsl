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
@group(0) @binding(3) var background_texture: texture_2d<f32>;
@group(1) @binding(0) var<uniform> ubo: CameraUniform;

const N1 = 1.0;
const N2 = 1.33; // water
const LIGHT_POS_WORLD = vec3<f32>(1.1, 2.6, 2.0);
const FLUID_COLOR = vec3<f32>(0.05, 0.42, 0.72);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    return vec4(pos[vi], 0.0, 1.0);
}

fn fresnel_schlick(cos_theta: f32) -> f32 {
    var f0 = (N1 - N2) / (N1 + N2);
    f0 = f0 * f0;
    let c = saturate(cos_theta);
    return f0 + (1.0 - f0) * pow(1.0 - c, 5.0);
}

fn saturate(val: f32) -> f32 {
    return clamp(val, 0.0, 1.0);
}

fn uv_depth_to_view(uv: vec2<f32>, eye_depth: f32) -> vec3<f32> {
    // frag_pos is top-left origin; ndc_y is bottom-up after the viewport flip.
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let fx = ubo.projection[0][0];
    let fy = ubo.projection[1][1];
    return vec3<f32>(
        ndc_x * eye_depth / fx,
        ndc_y * eye_depth / fy,
        -eye_depth,
    );
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(normals_texture));
    let uv = frag_pos.xy / dim;
    let pixel = vec2<i32>(frag_pos.xy);

    let normal_sample = textureLoad(normals_texture, pixel, 0);
    let eye_depth = textureLoad(depth_texture, pixel, 0).r;
    let thickness = textureLoad(thickness_texture, pixel, 0).r;
    let background = textureLoad(background_texture, pixel, 0).rgb;
    if normal_sample.a < 0.5 || eye_depth <= 1e-5 {
        return vec4(background, 1.0);
    }

    // computed normals are already in view space
    let n = normalize(normal_sample.xyz);
    let view_pos = uv_depth_to_view(uv, eye_depth);
    // camera is at origin in view space; so just taking negative
    let v = normalize(-view_pos);

    let light_view = (ubo.view * vec4<f32>(LIGHT_POS_WORLD, 1.0)).xyz;
    let l = normalize(light_view - view_pos);
    let h = normalize(v + l);

    let n_dot_v = saturate(dot(n, v));
    let n_dot_l = saturate(dot(n, l));
    let n_dot_h = saturate(dot(n, h));
    let f_val = fresnel_schlick(n_dot_v);

    // eq (14) in "screen space fluid rendering with curvature flow"
    let transmittance = exp(-thickness);
    // TODO: can add defraction as discussed in paper as well
    let a = mix(FLUID_COLOR, background, transmittance);
    let specular = vec3<f32>(0.85) * pow(n_dot_h, 128.0);

    // Eq (13)
    // NOTE: reflected is placeholder env reflection
    let reflected = vec3<f32>(0.82, 0.74, 0.62);
    let cout = (a) * (1.0 - f_val) + reflected * f_val + specular;

    return vec4(cout, 1.0);
}
