const DT = 0.0001;
const MU = 0.01;
const SDF_SPHERE = 0u;
const SDF_CYLINDER = 1u;
const SDF_BOX = 2u;
const SDF_DIFFERENCE = 3u;

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
    force_x: atomic<u32>,
    force_y: atomic<u32>,
    force_z: atomic<u32>,
    torque_x: atomic<u32>,
    torque_y: atomic<u32>,
    torque_z: atomic<u32>,
    color_r: f32,
    color_g: f32,
    color_b: f32,
}

struct InflowSource {
    body: u32,
    flow: vec3<f32>,
}

struct CollideResult {
    v: vec3<f32>,
    force: vec3<f32>,
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz),
    );
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_cylinder(p: vec3<f32>, r: f32, h: f32) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sd_primitive(ty: u32, p: vec3<f32>, scale: f32, params: vec4<f32>) -> f32 {
    if ty == SDF_SPHERE {
        return sd_sphere(p, scale);
    }
    if ty == SDF_CYLINDER {
        return sd_cylinder(p, params.x, params.y);
    }
    return sd_box(p, params.xyz);
}
