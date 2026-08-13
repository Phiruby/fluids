#include <../objects/sdf.wgsl>

struct ColliderUniform {
    min: vec3<f32>,
    max: vec3<f32>,
}

@group(0) @binding(0) var<storage, read_write> rigid_bodies: array<RigidBody>;
@group(0) @binding(1) var<uniform> collider: ColliderUniform;

#include <../objects/sdf_eval.wgsl>

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        quat_rotate(q, vec3<f32>(1.0, 0.0, 0.0)),
        quat_rotate(q, vec3<f32>(0.0, 1.0, 0.0)),
        quat_rotate(q, vec3<f32>(0.0, 0.0, 1.0)),
    );
}

fn world_aabb_extent(q: vec4<f32>, h: vec3<f32>) -> vec3<f32> {
    let m = quat_to_mat3(q);
    return vec3<f32>(
        abs(m[0].x) * h.x + abs(m[1].x) * h.y + abs(m[2].x) * h.z,
        abs(m[0].y) * h.x + abs(m[1].y) * h.y + abs(m[2].y) * h.z,
        abs(m[0].z) * h.x + abs(m[1].z) * h.y + abs(m[2].z) * h.z,
    );
}

fn apply_inv_inertia(q: vec4<f32>, inv_i: vec3<f32>, torque: vec3<f32>) -> vec3<f32> {
    // inv_i is in body frame. need to convert
    let t_body = quat_rotate(quat_conj(q), torque);
    return quat_rotate(q, t_body * inv_i);
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

    let extent = world_aabb_extent(q, h);
    let lo = collider.min + extent;
    let hi = collider.max - extent;
    // NOTE: this could be better (e.g transferring momentum for bounce?)
    if pos.x < lo.x { pos.x = lo.x; if v.x < 0.0 { v.x = 0.0; } }
    if pos.x > hi.x { pos.x = hi.x; if v.x > 0.0 { v.x = 0.0; } }
    if pos.y < lo.y { pos.y = lo.y; if v.y < 0.0 { v.y = 0.0; } }
    if pos.y > hi.y { pos.y = hi.y; if v.y > 0.0 { v.y = 0.0; } }
    if pos.z < lo.z { pos.z = lo.z; if v.z < 0.0 { v.z = 0.0; } }
    if pos.z > hi.z { pos.z = hi.z; if v.z > 0.0 { v.z = 0.0; } }

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
