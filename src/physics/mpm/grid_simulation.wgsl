struct GridParticle {
    momentum_x: i32,
    momentum_y: i32,
    momentum_z: i32,
    mass: i32,
    force_x: i32,
    force_y: i32,
    force_z: i32,
}

const DX: f32 = 0.1;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(-1.0, 1.0, -1.0);
const N: u32 = 64u;
const FIXED_POINT_SCALE: f32 = 4096.0;
const D_INV = 3.0 / (DX * DX);
const GAMMA = 7.0;
const DT = 0.001;

@group(0) @binding(0) var<storage, read_write> grid: array<GridParticle>;

@compute
@workgroup_size(256)
fn update_momentum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&grid) { return; }
    var p = grid[idx];
    if p.mass <= 0 { return; }
    let mom_x = f32(p.momentum_x) + f32(p.force_x) * DT;
    let mom_y = f32(p.momentum_y) + f32(p.force_y) * DT;
    let mom_z = f32(p.momentum_z) + f32(p.force_z) * DT;

    grid[idx].momentum_x = i32(mom_x);
    grid[idx].momentum_y = i32(mom_y);
    grid[idx].momentum_z = i32(mom_z);
}

