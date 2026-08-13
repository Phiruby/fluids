#include <../objects/sdf.wgsl>

struct ColliderUniform {
    min: vec3<f32>,
    max: vec3<f32>,
}

@group(0) @binding(0) var<storage, read_write> rigid_bodies: array<RigidBody>;
@group(0) @binding(1) var<uniform> collider: ColliderUniform;

#include <../objects/sdf_eval.wgsl>

const WALL_MU = 0.4;

fn apply_inv_inertia(q: vec4<f32>, inv_i: vec3<f32>, torque: vec3<f32>) -> vec3<f32> {
    let t_body = quat_rotate(quat_conj(q), torque);
    return quat_rotate(q, t_body * inv_i);
}

fn invert3(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let r0 = cross(b, c);
    let det = dot(a, r0);
    if abs(det) < 1e-8 {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    let r1 = cross(c, a);
    let r2 = cross(a, b);
    let inv_det = 1.0 / det;
    return transpose(mat3x3<f32>(r0 * inv_det, r1 * inv_det, r2 * inv_det));
}

fn apply_k(J: vec3<f32>, r: vec3<f32>, q: vec4<f32>, inv_mass: f32, inv_i: vec3<f32>) -> vec3<f32> {
    return J * inv_mass + cross(apply_inv_inertia(q, inv_i, cross(r, J)), r);
}

fn impulse_from_dv(dv: vec3<f32>, r: vec3<f32>, q: vec4<f32>, inv_mass: f32, inv_i: vec3<f32>) -> vec3<f32> {
    let k = mat3x3<f32>(
        apply_k(vec3<f32>(1.0, 0.0, 0.0), r, q, inv_mass, inv_i),
        apply_k(vec3<f32>(0.0, 1.0, 0.0), r, q, inv_mass, inv_i),
        apply_k(vec3<f32>(0.0, 0.0, 1.0), r, q, inv_mass, inv_i),
    );
    return invert3(k) * dv;
}

fn sample_local(h: vec3<f32>, k: u32) -> vec3<f32> {
    // 8 AABB corners, then 6 face centers — contact at the body's edges/faces
    switch k {
        case 0u: { return vec3<f32>(-h.x, -h.y, -h.z); }
        case 1u: { return vec3<f32>( h.x, -h.y, -h.z); }
        case 2u: { return vec3<f32>(-h.x,  h.y, -h.z); }
        case 3u: { return vec3<f32>( h.x,  h.y, -h.z); }
        case 4u: { return vec3<f32>(-h.x, -h.y,  h.z); }
        case 5u: { return vec3<f32>( h.x, -h.y,  h.z); }
        case 6u: { return vec3<f32>(-h.x,  h.y,  h.z); }
        case 7u: { return vec3<f32>( h.x,  h.y,  h.z); }
        case 8u: { return vec3<f32>( h.x, 0.0, 0.0); }
        case 9u: { return vec3<f32>(-h.x, 0.0, 0.0); }
        case 10u: { return vec3<f32>(0.0,  h.y, 0.0); }
        case 11u: { return vec3<f32>(0.0, -h.y, 0.0); }
        case 12u: { return vec3<f32>(0.0, 0.0,  h.z); }
        default: { return vec3<f32>(0.0, 0.0, -h.z); }
    }
}

fn collide_sample(
    p: vec3<f32>,
    r: vec3<f32>,
    q: vec4<f32>,
    inv_mass: f32,
    inv_i: vec3<f32>,
    v: ptr<function, vec3<f32>>,
    w: ptr<function, vec3<f32>>,
) {
    let clamped = clamp(p, collider.min, collider.max);
    let delta = p - clamped;
    let d2 = dot(delta, delta);
    if d2 < 1e-12 {
        return;
    }
    let n = -delta / sqrt(d2);
    let v_p = *v + cross(*w, r);
    let vn = dot(v_p, n);
    if vn >= 0.0 {
        return;
    }
    let vt = v_p - vn * n;
    let vt_len = length(vt);
    var v_new: vec3<f32>;
    if vt_len <= -WALL_MU * vn {
        v_new = vec3<f32>(0.0);
    } else {
        v_new = vt + WALL_MU * vn * vt / vt_len;
    }
    let J = impulse_from_dv(v_new - v_p, r, q, inv_mass, inv_i);
    *v += J * inv_mass;
    *w += apply_inv_inertia(q, inv_i, cross(r, J));
}

// TODO: no need for atomics here. Can make redefinition of RigidBody struct here
// and remove atomics
@compute
@workgroup_size(64)
fn update_rigid_bodies(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&rigid_bodies) { return; }
    if rigid_bodies[i].operand != 0u { return; }
    if rigid_bodies[i].mass <= 1e-8 { return; }

    if rigid_bodies[i].pinned != 0u {
        rigid_bodies[i].v_cm = vec3<f32>(0.0);
        rigid_bodies[i].omega = vec3<f32>(0.0);
        atomicStore(&rigid_bodies[i].force_x, bitcast<u32>(0.0));
        atomicStore(&rigid_bodies[i].force_y, bitcast<u32>(0.0));
        atomicStore(&rigid_bodies[i].force_z, bitcast<u32>(0.0));
        atomicStore(&rigid_bodies[i].torque_x, bitcast<u32>(0.0));
        atomicStore(&rigid_bodies[i].torque_y, bitcast<u32>(0.0));
        atomicStore(&rigid_bodies[i].torque_z, bitcast<u32>(0.0));
        return;
    }

    var pos = rigid_bodies[i].position;
    var q = rigid_bodies[i].orientation;
    var v = rigid_bodies[i].v_cm;
    var w = rigid_bodies[i].omega;
    let mass = rigid_bodies[i].mass;
    let inv_mass = rigid_bodies[i].inv_mass;
    let inv_i = rigid_bodies[i].inv_inertia;
    let h = body_extent(i);

    var f = vec3<f32>(
        bitcast<f32>(atomicLoad(&rigid_bodies[i].force_x)),
        bitcast<f32>(atomicLoad(&rigid_bodies[i].force_y)),
        bitcast<f32>(atomicLoad(&rigid_bodies[i].force_z)),
    );
    let tau = vec3<f32>(
        bitcast<f32>(atomicLoad(&rigid_bodies[i].torque_x)),
        bitcast<f32>(atomicLoad(&rigid_bodies[i].torque_y)),
        bitcast<f32>(atomicLoad(&rigid_bodies[i].torque_z)),
    );

    f += mass * vec3<f32>(0.0, -9.81, 0.0);
    v += f * inv_mass * DT;
    w += apply_inv_inertia(q, inv_i, tau) * DT;

    pos += v * DT;
    let wlen = length(w);
    var dq: vec4<f32>;
    if wlen > 1e-8 {
        dq = vec4<f32>(w / wlen * sin(wlen * DT * 0.5), cos(wlen * DT * 0.5));
    } else {
        dq = vec4<f32>(w * DT * 0.5, 1.0);
    }
    q = normalize(quat_mul(dq, q));

    var corr_lo = vec3<f32>(0.0);
    var corr_hi = vec3<f32>(0.0);
    for (var k = 0u; k < 14u; k++) {
        let local = sample_local(h, k);
        let r = quat_rotate(q, local);
        let p = pos + r;
        corr_lo = max(corr_lo, collider.min - p);
        corr_hi = max(corr_hi, p - collider.max);
        collide_sample(p, r, q, inv_mass, inv_i, &v, &w);
    }
    pos += max(corr_lo, vec3<f32>(0.0)) - max(corr_hi, vec3<f32>(0.0));

    rigid_bodies[i].position = pos;
    rigid_bodies[i].orientation = q;
    rigid_bodies[i].v_cm = v;
    rigid_bodies[i].omega = w;
    atomicStore(&rigid_bodies[i].force_x, bitcast<u32>(0.0));
    atomicStore(&rigid_bodies[i].force_y, bitcast<u32>(0.0));
    atomicStore(&rigid_bodies[i].force_z, bitcast<u32>(0.0));
    atomicStore(&rigid_bodies[i].torque_x, bitcast<u32>(0.0));
    atomicStore(&rigid_bodies[i].torque_y, bitcast<u32>(0.0));
    atomicStore(&rigid_bodies[i].torque_z, bitcast<u32>(0.0));
}
