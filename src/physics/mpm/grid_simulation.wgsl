#include <../objects/sdf.wgsl>
#include <constants.wgsl>

struct GridParticle {
    momentum_x: f32,
    momentum_y: f32,
    momentum_z: f32,
    mass: f32,
    force_x: f32,
    force_y: f32,
    force_z: f32,
}

struct ColliderUniform {
    min: vec3<f32>,
    max: vec3<f32>,
}


@group(0) @binding(0) var<storage, read_write> grid: array<GridParticle>;
@group(0) @binding(1) var<uniform> collider: ColliderUniform;
@group(0) @binding(2) var<storage, read_write> rigid_bodies: array<RigidBody>;

#include <../objects/sdf_eval.wgsl>

fn get_grid_coord(idx: u32) -> vec3<u32> {
    let z = idx % N;
    let y = (idx / N) % N;
    let x = idx / (N * N);
    return vec3<u32>(x, y, z);
}

fn sd_domain(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
    let c = 0.5 * (bmin + bmax);
    let b = 0.5 * (bmax - bmin);
    let q = abs(p - c) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_domain_normal(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
    let e = 1e-3;
    let n = vec3<f32>(
        sd_domain(p + vec3<f32>(e, 0.0, 0.0), bmin, bmax) - sd_domain(p - vec3<f32>(e, 0.0, 0.0), bmin, bmax),
        sd_domain(p + vec3<f32>(0.0, e, 0.0), bmin, bmax) - sd_domain(p - vec3<f32>(0.0, e, 0.0), bmin, bmax),
        sd_domain(p + vec3<f32>(0.0, 0.0, e), bmin, bmax) - sd_domain(p - vec3<f32>(0.0, 0.0, e), bmin, bmax),
    );
    return normalize(n);
}

// collision with bounding box (deomain of simulation)
fn collide_domain(pos: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let phi = sd_domain(pos, collider.min, collider.max);
    if phi <= 0.0 {
        return v;
    }
    let n = -sd_domain_normal(pos, collider.min, collider.max);
    let vn = dot(v, n);
    if vn >= 0.0 {
        return v;
    }
    let vt = v - vn * n;
    let vt_len = length(vt);
    if vt_len <= -MU * vn {
        return vec3<f32>(0.0);
    }
    return vt + MU * vn * vt / vt_len;
}

fn atomic_add_rb(i: u32, force: vec3<f32>, torque: vec3<f32>) {
    var old_bits = atomicLoad(&rigid_bodies[i].force_x);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + force.x);
        let result = atomicCompareExchangeWeak(&rigid_bodies[i].force_x, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
    old_bits = atomicLoad(&rigid_bodies[i].force_y);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + force.y);
        let result = atomicCompareExchangeWeak(&rigid_bodies[i].force_y, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
    old_bits = atomicLoad(&rigid_bodies[i].force_z);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + force.z);
        let result = atomicCompareExchangeWeak(&rigid_bodies[i].force_z, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
    old_bits = atomicLoad(&rigid_bodies[i].torque_x);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + torque.x);
        let result = atomicCompareExchangeWeak(&rigid_bodies[i].torque_x, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
    old_bits = atomicLoad(&rigid_bodies[i].torque_y);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + torque.y);
        let result = atomicCompareExchangeWeak(&rigid_bodies[i].torque_y, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
    old_bits = atomicLoad(&rigid_bodies[i].torque_z);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + torque.z);
        let result = atomicCompareExchangeWeak(&rigid_bodies[i].torque_z, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

fn collide_rigids(pos: vec3<f32>, v: vec3<f32>, mass: f32) -> vec3<f32> {
    var v_out = v;
    let n = arrayLength(&rigid_bodies);
    for (var i = 0u; i < n; i++) {
        let hit = collide_rigid(i, pos, v_out, mass);
        if any(hit.force != vec3<f32>(0.0)) {
            atomic_add_rb(i, hit.force, cross(pos - rigid_bodies[i].position, hit.force));
        }
        v_out = hit.v;
    }
    return v_out;
}

@compute
@workgroup_size(256)
fn update_momentum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&grid) { return; }
    var p = grid[idx];
    if p.mass <= 1e-8 { return; }

    var mom = vec3<f32>(
        p.momentum_x + p.force_x * DT,
        p.momentum_y + (p.force_y - 9.81 * p.mass) * DT,
        p.momentum_z + p.force_z * DT,
    );

    let coord = get_grid_coord(idx);
    let node_pos = BOTTOM_LEFT + vec3<f32>(coord) * DX;
    var v = mom / p.mass;
    v = collide_domain(node_pos, v);
    v = collide_rigids(node_pos, v, p.mass);
    mom = v * p.mass;

    grid[idx].momentum_x = mom.x;
    grid[idx].momentum_y = mom.y;
    grid[idx].momentum_z = mom.z;
}
