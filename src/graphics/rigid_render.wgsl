struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    particle_radius: f32,
    screen_height: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) local_position: vec3<f32>,
    @location(1) local_normal: vec3<f32>,
    @location(2) instance_position: vec3<f32>,
    @location(3) orientation: vec4<f32>,
    @location(4) params: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_pos = quat_rotate(input.orientation, input.local_position * input.params.xyz)
        + input.instance_position;
    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * vec4<f32>(world_pos, 1.0);
    out.world_normal = quat_rotate(input.orientation, input.local_normal);
    out.color = vec3<f32>(0.72, 0.48, 0.28);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(0.35, 0.7, 0.6));
    let lighting = 0.24 + 0.76 * max(dot(n, light_dir), 0.0);
    return vec4<f32>(in.color * lighting, 1.0);
}
