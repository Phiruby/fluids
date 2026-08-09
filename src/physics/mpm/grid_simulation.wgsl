struct GridParticle {
    momentum_x: i32,
    momentum_y: i32,
    momentum_z: i32,
    mass: i32,
    force_x: i32,
    force_y: i32,
    force_z: i32,
}

struct ColliderUniform {
    min: vec3<f32>,
    max: vec3<f32>,
}

const DX: f32 = 0.01;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(-1.0, 1.0, -1.0);
const N: u32 = 128u;
const D_INV = 4.0 / (DX * DX);
const DT = 0.0001;

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
    if vn < 0.0 {
        return v - vn * n;
    }
    return v;
}

@compute
@workgroup_size(256)
fn update_momentum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&grid) { return; }
    var p = grid[idx];
    if p.mass <= 0 { return; }

    var mom = vec3<f32>(
        f32(p.momentum_x) + f32(p.force_x) * DT,
        f32(p.momentum_y) + f32(p.force_y) * DT,
        f32(p.momentum_z) + f32(p.force_z) * DT,
    );

    let coord = get_grid_coord(idx);
    let node_pos = BOTTOM_LEFT + vec3<f32>(coord) * DX;
    let mass = f32(p.mass);
    var v = mom / mass;
    v = collide_velocity(node_pos, v);
    mom = v * mass;

    grid[idx].momentum_x = i32(round(mom.x));
    grid[idx].momentum_y = i32(round(mom.y));
    grid[idx].momentum_z = i32(round(mom.z));
}
