#include <../objects/sdf.wgsl>

struct GridParticle {
    momentum_x: f32,
    momentum_y: f32,
    momentum_z: f32,
    mass: f32,
    force_x: f32,
    force_y: f32,
    force_z: f32,
}

const DX: f32 = 0.02;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const N: u32 = 32u;

@group(0) @binding(0) var<storage, read_write> grid: array<GridParticle>;
@group(0) @binding(1) var<storage, read_write> rigid_bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> sources: array<InflowSource>;

#include <../objects/sdf_eval.wgsl>

fn get_grid_coord(idx: u32) -> vec3<u32> {
    let z = idx % N;
    let y = (idx / N) % N;
    let x = idx / (N * N);
    return vec3<u32>(x, y, z);
}

@compute
@workgroup_size(256)
fn apply_inflow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&grid) { return; }
    let coord = get_grid_coord(idx);
    let node_pos = BOTTOM_LEFT + vec3<f32>(coord) * DX;
    let n = arrayLength(&sources);
    for (var s = 0u; s < n; s++) {
        let i = sources[s].body;
        let flow = sources[s].flow;
        if length(flow) < 1e-8 { continue; }
        if source_sdf(i, node_pos) > 0.0 { continue; }
        let local = local_of(i, node_pos);
        let axis = flow / length(flow);
        let h = body_extent(i).y;
        let t = dot(local, axis);
        let mass = grid[idx].mass;
        if mass <= 1e-8 { continue; }
        let v = quat_rotate(rigid_bodies[i].orientation, flow);
        grid[idx].momentum_x = mass * v.x;
        grid[idx].momentum_y = mass * v.y;
        grid[idx].momentum_z = mass * v.z;
    }
}
