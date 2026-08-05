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
    @location(3) color: vec3<f32>,
    @location(4) boundary: u32,
}

struct McVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct LineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) alpha: f32,
};

fn compute_lighting(normal: vec3<f32>) -> f32 {
    let n = normalize(normal);
    let light_dir = normalize(vec3<f32>(0.35, 0.7, 0.6));
    let ambient = 0.24;
    let diffuse = max(dot(n, light_dir), 0.0);
    return ambient + (1.0 - ambient) * diffuse;
}

fn culled() -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return out;
}

fn particle_out(input: VertexInput, color: vec3<f32>, alpha: f32) -> VertexOutput {
    var out: VertexOutput;
    out.color = color;
    out.world_normal = input.local_normal;
    out.alpha = alpha;
    out.clip_position = camera.projection * camera.view * vec4<f32>(input.instance_position + input.local_position, 1.0);
    return out;
}

// for fluid particles (stays opauque)
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    if input.boundary == 0u { return culled(); }
    return particle_out(input, input.color, 1.0);
}

@vertex
fn vs_line_main(input: LineVertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = input.color;
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.alpha = 1.0;
    out.clip_position = camera.projection * camera.view * vec4<f32>(input.position, 1.0);
    return out;
}

@vertex
fn vs_mc_main(input: McVertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = vec3<f32>(0.2, 0.5, 0.95);
    out.world_normal = input.normal;
    out.alpha = 1.0;
    out.clip_position = camera.projection * camera.view * vec4<f32>(input.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let lighting = compute_lighting(in.world_normal);
    return vec4<f32>(in.color * lighting, in.alpha);
}
