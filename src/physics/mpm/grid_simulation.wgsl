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

const DX: f32 = 0.02;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const N: u32 = 32u;
const DT = 0.0001;
const MU = 0.4;

@group(0) @binding(0) var<storage, read_write> grid: array<GridParticle>;
@group(0) @binding(1) var<uniform> collider: ColliderUniform;

fn get_grid_coord(idx: u32) -> vec3<u32> {
    let z = idx % N;
    let y = (idx / N) % N;
    let x = idx / (N * N);
    return vec3<u32>(x, y, z);
}

fn sd_box(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
    let c = 0.5 * (bmin + bmax);
    let b = 0.5 * (bmax - bmin);
    let q = abs(p - c) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_box_normal(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
    let e = 1e-3;
    let n = vec3<f32>(
        sd_box(p + vec3<f32>(e, 0.0, 0.0), bmin, bmax) - sd_box(p - vec3<f32>(e, 0.0, 0.0), bmin, bmax),
        sd_box(p + vec3<f32>(0.0, e, 0.0), bmin, bmax) - sd_box(p - vec3<f32>(0.0, e, 0.0), bmin, bmax),
        sd_box(p + vec3<f32>(0.0, 0.0, e), bmin, bmax) - sd_box(p - vec3<f32>(0.0, 0.0, e), bmin, bmax),
    );
    return normalize(n);
}

fn collide_velocity(pos: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let phi = sd_box(pos, collider.min, collider.max);
    if phi <= 0.0 {
        return v;
    }
    let n = -sd_box_normal(pos, collider.min, collider.max);
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

@compute
@workgroup_size(256)
fn update_momentum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&grid) { return; }
    var p = grid[idx];
    if p.mass <= 1e-8 { return; }

    var mom = vec3<f32>(
        p.momentum_x + p.force_x * DT,
        p.momentum_y + p.force_y * DT,
        p.momentum_z + p.force_z * DT,
    );

    let coord = get_grid_coord(idx);
    let node_pos = BOTTOM_LEFT + vec3<f32>(coord) * DX;
    var v = mom / p.mass;
    v = collide_velocity(node_pos, v);
    mom = v * p.mass;

    grid[idx].momentum_x = mom.x;
    grid[idx].momentum_y = mom.y;
    grid[idx].momentum_z = mom.z;
}
