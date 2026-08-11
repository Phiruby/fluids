struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    particle_radius: f32,
    screen_height: f32,
}

struct Particle {
    ${PARTICLE_FIELDS}
}

@group(0) @binding(0)
var<uniform> ubo: CameraUniform;
@group(1) @binding(0) var<storage, read> particles: array<Particle>;

struct VertexInput {
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_space: vec3<f32>,
};

struct FragmentOutput {
    // eye_depth is depth from view space
    @location(0) eye_depth: f32,
    @builtin(frag_depth) frag_depth: f32,
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = vec4<f32>(particles[input.instance_idx].position, 1.0);
    let view_pos = ubo.view * world_pos;
    out.view_space = view_pos.xyz;
    let uv_coords = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    let uv = uv_coords[input.vertex_idx];
    out.uv = uv;
    let official_view_pos = vec4<f32>(
        view_pos.xy + uv * ubo.particle_radius,
        view_pos.zw
    );
    out.clip_position = ubo.projection * official_view_pos;
    return out;
}

@fragment
fn frag_main(input: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let r = dot(input.uv, input.uv);
    if r > 1.0 { discard; }
    let normal_view = vec3<f32>(input.uv, sqrt(1.0 - r));
    let view_pos = input.view_space + normal_view * ubo.particle_radius;
    // NOTE: depth for fluid rendering is in view space (everything in screen space rendering is)
    let eye_depth = -view_pos.z;

    let proj_pos = ubo.projection * vec4<f32>(view_pos, 1.0);
    out.eye_depth = eye_depth;
    out.frag_depth = proj_pos.z / proj_pos.w;
    return out;
}

@fragment
fn frag_thickness(input: VertexOutput) -> @location(0) vec4<f32> {
    let r = dot(input.uv, input.uv);
    if r > 1.0 { discard; }
    let thickness = 2.0 * sqrt(1.0 - r) * ubo.particle_radius;
    return vec4<f32>(thickness, 0.0, 0.0, 1.0);
}
