struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    particle_radius: f32,
    screen_height: f32,
}

struct RigidBody {
    object_type: u32,
    invert: u32,
    pinned: u32,
    operand: u32,
    position: vec3<f32>,
    scale: f32,
    orientation: vec4<f32>,
    params: vec4<f32>,
    v_cm: vec3<f32>,
    mass: f32,
    omega: vec3<f32>,
    inv_mass: f32,
    inv_inertia: vec3<f32>,
    force_x: u32,
    force_y: u32,
    force_z: u32,
    torque_x: u32,
    torque_y: u32,
    torque_z: u32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(1) @binding(0)
var<storage, read> rigid_bodies: array<RigidBody>;

struct VertexInput {
    @location(0) local_position: vec3<f32>,
    @location(1) local_normal: vec3<f32>,
    @builtin(instance_index) instance_idx: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) extent: f32,
    @location(3) @interpolate(flat) body: u32,
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn sd_primitive(ty: u32, p: vec3<f32>, scale: f32, params: vec4<f32>) -> f32 {
    if ty == 0u {
        return length(p) - scale;
    }
    if ty == 1u {
        let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(params.x, params.y);
        return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
    }
    let q = abs(p) - params.xyz;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_body_local(i: u32, p: vec3<f32>) -> f32 {
    return sd_primitive(
        rigid_bodies[i].object_type,
        p,
        rigid_bodies[i].scale,
        rigid_bodies[i].params,
    );
}

fn body_extent(i: u32) -> vec3<f32> {
    if rigid_bodies[i].object_type == 3u {
        return rigid_bodies[u32(rigid_bodies[i].params.x)].params.xyz;
    }
    return rigid_bodies[i].params.xyz;
}

fn sdf_at(i: u32, world_p: vec3<f32>) -> f32 {
    let rb = rigid_bodies[i];
    let p = quat_rotate(quat_conj(rb.orientation), world_p - rb.position);
    var d: f32;
    if rb.object_type == 3u {
        let a = u32(rb.params.x);
        let b = u32(rb.params.y);
        d = max(sd_body_local(a, p), -sd_body_local(b, p));
    } else {
        d = sd_body_local(i, p);
    }
    if rb.invert != 0u {
        d = -d;
    }
    return d;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.body = input.instance_idx;
    let rb = rigid_bodies[input.instance_idx];
    if rb.operand != 0u {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.world_pos = vec3<f32>(0.0);
        out.center = vec3<f32>(0.0);
        out.extent = 0.0;
        return out;
    }
    let extent = body_extent(input.instance_idx);
    let world_pos = quat_rotate(rb.orientation, input.local_position * extent) + rb.position;
    out.clip_position = camera.projection * camera.view * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.center = rb.position;
    out.extent = length(extent);
    return out;
}

// render an entity using just its sdf (like ray marching)
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.extent < 1e-8 {
        discard;
    }
    let rd = normalize(in.world_pos - camera.camera_pos);
    var p = in.world_pos;
    var hit = false;
    for (var i = 0; i < 48; i++) {
        let d = sdf_at(in.body, p);
        if d < 8e-4 {
            hit = true;
            break;
        }
        p += rd * max(d, 8e-4);
        if length(p - in.center) > in.extent * 2.2 {
            break;
        }
    }
    if !hit {
        discard;
    }
    let e = 1e-3;
    let n = normalize(vec3<f32>(
        sdf_at(in.body, p + vec3<f32>(e, 0.0, 0.0)) - sdf_at(in.body, p - vec3<f32>(e, 0.0, 0.0)),
        sdf_at(in.body, p + vec3<f32>(0.0, e, 0.0)) - sdf_at(in.body, p - vec3<f32>(0.0, e, 0.0)),
        sdf_at(in.body, p + vec3<f32>(0.0, 0.0, e)) - sdf_at(in.body, p - vec3<f32>(0.0, 0.0, e)),
    ));
    let light_dir = normalize(vec3<f32>(0.35, 0.85, 0.45));
    let fill = normalize(vec3<f32>(-0.45, 0.25, -0.35));
    let lighting = 0.16 + 0.72 * max(dot(n, light_dir), 0.0) + 0.20 * max(dot(n, fill), 0.0);
    let albedo = vec3<f32>(
        rigid_bodies[in.body].color_r,
        rigid_bodies[in.body].color_g,
        rigid_bodies[in.body].color_b,
    );
    return vec4<f32>(albedo * lighting, 1.0);
}
