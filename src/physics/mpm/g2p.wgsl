
struct Particle {
    deformation_gradient: mat3x3<f32>,
    B: mat3x3<f32>,
    position: vec3<f32>,
    velocity: vec3<f32>,
    rest_volume: f32,
    mass: f32,
}

struct GridParticle {
    momentum_x: i32,
    momentum_y: i32,
    momentum_z: i32,
    mass: i32,
    force_x: i32,
    force_y: i32,
    force_z: i32,
}

const DX: f32 = 0.02;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(-1.0, 1.0, -1.0);
const N: u32 = 32u;
const FIXED_POINT_SCALE: f32 = 4096.0;
const D_INV = 3.0 / (DX * DX);
const GAMMA = 7.0;
const DT = 0.001;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> grid: array<GridParticle>;

fn quadratic_kernel(r: f32) -> f32 {
    let ar = abs(r);
    if ar < 0.5 {
        return 0.75 - ar * ar;
    }
    if ar < 1.5 {
        return 0.5 * (1.5 - ar) * (1.5 - ar);
    }
    return 0.0;
}

fn get_grid_index(coord: vec3<u32>) -> u32 {
    return coord.x * N * N + coord.y * N + coord.z;
}

fn outer(v1: vec3<f32>, v2: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        v1.x * v2.x, v1.x * v2.y, v1.x * v2.z,
        v1.y * v2.x, v1.y * v2.y, v1.y * v2.z,
        v1.z * v2.x, v1.z * v2.y, v1.z * v2.z,
    );
}

@compute
@workgroup_size(256)
fn grid_to_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&particles) { return; }
    var p = particles[p_idx];

    let relative_pos = p.position - BOTTOM_LEFT;
    let base_coord_f = relative_pos / DX;
    let base_coord = vec3<i32>(floor(base_coord_f - 0.5));

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            for (var k = 0; k < 3; k++) {
                let cell_offset = vec3<i32>(i, j, k);
                let current_coord = base_coord + cell_offset;

                if current_coord.x < 0 || current_coord.x >= i32(N) ||
                    current_coord.y < 0 || current_coord.y >= i32(N) ||
                    current_coord.z < 0 || current_coord.z >= i32(N) {
                    continue;
                }
                let node_pos = BOTTOM_LEFT + vec3<f32>(current_coord) * DX;
                let xi_xp = node_pos - p.position;
                let r = xi_xp / DX;
                let weight = quadratic_kernel(r.x) * quadratic_kernel(r.y) * quadratic_kernel(r.z);
                let linear_grid_idx = get_grid_index(vec3<u32>(current_coord));

                if grid[linear_grid_idx].mass <= 0 { continue; }
                // no need to rescale by 1/FIFIXED_POINT_SCALE since dividing by mass cancels it out
                let vi = vec3<f32>(
                    f32(grid[linear_grid_idx].momentum_x),
                    f32(grid[linear_grid_idx].momentum_y),
                    f32(grid[linear_grid_idx].momentum_z),
                ) / f32(grid[linear_grid_idx].mass);
                let updated_grid_loc = node_pos + DT * vi;
                p.velocity = p.velocity + vi * weight;
                let newpos = p.position + node_pos * weight;
                let d = node_pos - p.position + updated_grid_loc - newpos;
                p.B = p.B + 0.5 * weight * (outer(vi, d) + outer(d, vi));
                p.position = newpos;
            }
        }
    }
    // eq (17)
    let C = D_INV * p.B;
    let identity = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    p.deformation_gradient = (identity + DT * C) * p.deformation_gradient;

    // step forward (can move to separate pass once we pass plasticity)
    p.position += p.velocity * DT;
    particles[p_idx] = p;
}
