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

const DX: f32 = 0.01;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const N: u32 = 192u;
const DT = 0.0001;
const MU = 0.4; // friction

@group(0) @binding(0) var<storage, read_write> grid: array<GridParticle>;
@group(0) @binding(1) var<uniform> collider: ColliderUniform;

fn get_grid_coord(idx: u32) -> vec2<u32> {
    let y = idx % N;
    let x = idx / N;
    return vec2<u32>(x, y);
}

fn sd_box2(p: vec2<f32>, bmin: vec2<f32>, bmax: vec2<f32>) -> f32 {
    let c = 0.5 * (bmin + bmax);
    let b = 0.5 * (bmax - bmin);
    let q = abs(p - c) - b;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

fn sd_box2_normal(p: vec2<f32>, bmin: vec2<f32>, bmax: vec2<f32>) -> vec2<f32> {
    let e = 1e-3;
    let n = vec2<f32>(
        sd_box2(p + vec2<f32>(e, 0.0), bmin, bmax) - sd_box2(p - vec2<f32>(e, 0.0), bmin, bmax),
        sd_box2(p + vec2<f32>(0.0, e), bmin, bmax) - sd_box2(p - vec2<f32>(0.0, e), bmin, bmax),
    );
    return normalize(n);
}

fn collide_velocity(pos: vec2<f32>, v: vec2<f32>) -> vec2<f32> {
    let bmin = collider.min.xy;
    let bmax = collider.max.xy;
    let phi = sd_box2(pos, bmin, bmax);
    if phi <= 0.0 {
        return v;
    }
    let n = -sd_box2_normal(pos, bmin, bmax);
    let vn = dot(v, n);
    if vn >= 0.0 {
        return v;
    }
    let vt = v - vn * n;
    if length(vt) <= -MU * vn {
        return vec2<f32>(0.0);
    }
    return vt + MU * vn * vt / length(vt);
}

@compute
@workgroup_size(256)
fn update_momentum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&grid) { return; }
    var p = grid[idx];
    if p.mass <= 1e-8 { return; }

    var mom = vec2<f32>(
        f32(p.momentum_x) + f32(p.force_x) * DT,
        f32(p.momentum_y) + f32(p.force_y) * DT,
    );

    let coord = get_grid_coord(idx);
    let node_pos = vec2<f32>(
        BOTTOM_LEFT.x + f32(coord.x) * DX,
        BOTTOM_LEFT.y + f32(coord.y) * DX,
    );
    let mass = f32(p.mass);
    var v = mom / mass;

    v = collide_velocity(node_pos, v);
    mom = v * mass;
    grid[idx].momentum_x = mom.x;
    grid[idx].momentum_y = mom.y;
    grid[idx].momentum_z = 0;
}
