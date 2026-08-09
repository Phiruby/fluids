struct Particle {
    deformation_gradient: mat3x3<f32>,
    B: mat3x3<f32>,
    position: vec3<f32>,
    velocity: vec3<f32>,
    rest_volume: f32,
    mass: f32,
}

struct GridParticle {
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
    mass: atomic<i32>,
    force_x: atomic<i32>,
    force_y: atomic<i32>,
    force_z: atomic<i32>,
}

const DX: f32 = 0.02;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const N: u32 = 64u;
const FIXED_POINT_SCALE: f32 = 1e8;
const D_INV = 4.0 / (DX * DX);
const GAMMA = 7.0;
const K = 10000.0;

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid: array<GridParticle>;

fn fluid_stress(idx: u32) -> mat3x3<f32> {
    let f = particles[idx].deformation_gradient;
    let j = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let p = max(-0.1, K * (pow(j, -GAMMA) - 1.0));
    let s = j * p;
    return mat3x3<f32>(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0);
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

fn get_grid_index(coord: vec2<u32>) -> u32 {
    return coord.x * N + coord.y;
}

@compute
@workgroup_size(256)
fn particles_to_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&particles) { return; }

    let p = particles[p_idx];
    let relative = (p.position.xy - BOTTOM_LEFT.xy) / DX;
    let base_coord = vec2<i32>(floor(relative - 0.5));
    let v_p = vec3<f32>(p.velocity.xy, 0.0);
    let stress = fluid_stress(p_idx);

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            let current_coord = base_coord + vec2<i32>(i, j);
            if current_coord.x < 0 || current_coord.x >= i32(N) ||
                current_coord.y < 0 || current_coord.y >= i32(N) {
                continue;
            }
            let node_pos = vec3<f32>(
                BOTTOM_LEFT.x + f32(current_coord.x) * DX,
                BOTTOM_LEFT.y + f32(current_coord.y) * DX,
                0.0,
            );
            let xi_xp = node_pos - vec3<f32>(p.position.xy, 0.0);
            let r = xi_xp.xy / DX;
            let weight = quadratic_kernel(r.x) * quadratic_kernel(r.y);
            let affine_contribution = D_INV * p.B * xi_xp;
            let total_velocity = vec3<f32>((v_p + affine_contribution).xy, 0.0);
            let mass_contribution = weight * p.mass;
            let momentum_contribution = mass_contribution * total_velocity;
            let force_contribution = weight * p.rest_volume * D_INV * stress * xi_xp
                - weight * p.mass * vec3<f32>(0.0, 9.81, 0.0);

            let idx = get_grid_index(vec2<u32>(current_coord));
            atomicAdd(&grid[idx].mass, i32(round(mass_contribution * FIXED_POINT_SCALE)));
            atomicAdd(&grid[idx].momentum_x, i32(round(momentum_contribution.x * FIXED_POINT_SCALE)));
            atomicAdd(&grid[idx].momentum_y, i32(round(momentum_contribution.y * FIXED_POINT_SCALE)));
            atomicAdd(&grid[idx].force_x, i32(round(force_contribution.x * FIXED_POINT_SCALE)));
            atomicAdd(&grid[idx].force_y, i32(round(force_contribution.y * FIXED_POINT_SCALE)));
        }
    }
}
