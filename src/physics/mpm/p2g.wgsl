struct Particle {
    deformation_gradient: mat3x3<f32>,
    B: mat3x3<f32>,
    position: vec3<f32>,
    velocity: vec3<f32>,
    rest_volume: f32,
    mass: f32,
}

struct GridParticle {
    // using i32 with scaling to preserve precision for now
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
    mass: atomic<i32>,
    force_x: atomic<i32>,
    force_y: atomic<i32>,
    force_z: atomic<i32>,
}

const DX: f32 = 0.01;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(-1.0, 1.0, -1.0);
const N: u32 = 128u;
const FIXED_POINT_SCALE: f32 = 1e8;
const D_INV = 4.0 / (DX * DX);
const GAMMA = 7.0;
const K = 10000.0;

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid: array<GridParticle>;

fn fluid_stress(idx: u32) -> mat3x3<f32> {
    let j: f32 = determinant(particles[idx].deformation_gradient);
    let p = K * (pow(j, -GAMMA) - 1.0);
    let s = j * p;
    return mat3x3<f32>(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s);
}
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

@compute
@workgroup_size(256)
fn particles_to_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&particles) { return; }

    let p = particles[p_idx];

    let relative_pos = p.position - BOTTOM_LEFT;
    let base_coord_f = relative_pos / DX;
    let base_coord = vec3<i32>(floor(base_coord_f - 0.5));

    let v_p = p.velocity;
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
                // eq (18) of https://arxiv.org/pdf/1603.06188
                let affine_contribution = D_INV * p.B * xi_xp;
                let total_velocity = v_p + affine_contribution;
                let mass_contribution = weight * p.mass;
                let momentum_contribution = mass_contribution * total_velocity;
                let force_contribution = weight * p.rest_volume
                  * D_INV * fluid_stress(p_idx) * xi_xp
                  - weight * p.mass * vec3<f32>(0.0, 9.81, 0.0);

                let linear_grid_idx = get_grid_index(vec3<u32>(current_coord));

                // TODO: would using atomics be faster than using spatial hash
                // and performing P2G per grid node?
                atomicAdd(&grid[linear_grid_idx].mass, i32(round(mass_contribution * FIXED_POINT_SCALE)));
                atomicAdd(&grid[linear_grid_idx].momentum_x, i32(round(momentum_contribution.x * FIXED_POINT_SCALE)));
                atomicAdd(&grid[linear_grid_idx].momentum_y, i32(round(momentum_contribution.y * FIXED_POINT_SCALE)));
                atomicAdd(&grid[linear_grid_idx].momentum_z, i32(round(momentum_contribution.z * FIXED_POINT_SCALE)));
                atomicAdd(&grid[linear_grid_idx].force_x, i32(round(force_contribution.x * FIXED_POINT_SCALE)));
                atomicAdd(&grid[linear_grid_idx].force_y, i32(round(force_contribution.y * FIXED_POINT_SCALE)));
                atomicAdd(&grid[linear_grid_idx].force_z, i32(round(force_contribution.z * FIXED_POINT_SCALE)));
            }
        }
    }
}
