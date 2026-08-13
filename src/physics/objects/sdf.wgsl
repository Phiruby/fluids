const DT = 0.0001;
const MU = 0.4;
const SDF_SPHERE = 0u;
const SDF_CYLINDER = 1u;
const SDF_BOX = 2u;

struct RigidBody {
    object_type: u32,
    invert: u32,
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
}

struct RigidState {
    object_type: u32,
    invert: u32,
    position: vec3<f32>,
    scale: f32,
    orientation: vec4<f32>,
    params: vec4<f32>,
    v_cm: vec3<f32>,
    mass: f32,
    omega: vec3<f32>,
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

fn rigid_local(rb: RigidState, world_p: vec3<f32>) -> vec3<f32> {
    return quat_rotate(quat_conj(rb.orientation), world_p - rb.position);
}

fn rigid_sdf(rb: RigidState, world_p: vec3<f32>) -> f32 {
    let p = rigid_local(rb, world_p);
    var d = 0.0;
    if rb.object_type == SDF_SPHERE {
        d = sd_sphere(p, rb.scale);
    } else if rb.object_type == SDF_CYLINDER {
        d = sd_cylinder(p, rb.params.x, rb.params.y);
    } else {
        d = sd_box(p, rb.params.xyz);
    }
    if rb.invert != 0u {
        d = -d;
    }
    return d;
}

fn rigid_sdf_normal(rb: RigidState, world_p: vec3<f32>) -> vec3<f32> {
    let e = 1e-3;
    let n = vec3<f32>(
        rigid_sdf(rb, world_p + vec3<f32>(e, 0.0, 0.0)) - rigid_sdf(rb, world_p - vec3<f32>(e, 0.0, 0.0)),
        rigid_sdf(rb, world_p + vec3<f32>(0.0, e, 0.0)) - rigid_sdf(rb, world_p - vec3<f32>(0.0, e, 0.0)),
        rigid_sdf(rb, world_p + vec3<f32>(0.0, 0.0, e)) - rigid_sdf(rb, world_p - vec3<f32>(0.0, 0.0, e)),
    );
    let len = length(n);
    if len < 1e-8 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return n / len;
}

fn rigid_point_velocity(rb: RigidState, world_p: vec3<f32>) -> vec3<f32> {
    return rb.v_cm + cross(rb.omega, world_p - rb.position);
}

// chapter 12.1 https://alexey.stomakhin.com/research/siggraph2016_mpm.pdf
fn collide_rigid(rb: RigidState, pos: vec3<f32>, v: vec3<f32>, node_mass: f32) -> CollideResult {
    var result: CollideResult;
    result.v = v;
    result.force = vec3<f32>(0.0);
    if rb.mass <= 1e-8 {
        return result;
    }
    if rigid_sdf(rb, pos) > 0.0 {
        return result;
    }
    let n = rigid_sdf_normal(rb, pos);
    let v_body = rigid_point_velocity(rb, pos);
    var v_rel = v - v_body;
    let vn = dot(v_rel, n);
    if vn >= 0.0 {
        return result;
    }
    let vt = v_rel - vn * n;
    let vt_len = length(vt);
    if vt_len <= -MU * vn {
        v_rel = vec3<f32>(0.0);
    } else {
        v_rel = vt + MU * vn * vt / vt_len;
    }
    let v_new = v_rel + v_body;
    result.v = v_new;
    result.force = -node_mass * (v_new - v) / DT;
    return result;
}
