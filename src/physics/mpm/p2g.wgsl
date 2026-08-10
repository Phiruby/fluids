struct Particle {
    deformation_gradient: mat3x3<f32>,
    B: mat3x3<f32>,
    position: vec3<f32>,
    velocity: vec3<f32>,
    rest_volume: f32,
    mass: f32,
}

struct GridParticle {
    /// all the atomic<u32> are treated as f32 in the bit level.
    /// my machine doesn't support atomic<f32> extension for vulkan, so doing this
    momentum_x: atomic<u32>,
    momentum_y: atomic<u32>,
    momentum_z: atomic<u32>,
    mass: atomic<u32>,
    force_x: atomic<u32>,
    force_y: atomic<u32>,
    force_z: atomic<u32>,
}

const DX: f32 = 0.01;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const N: u32 = 192u;
const D_INV = 4.0 / (DX * DX);
const GAMMA = 7.0;
const K = 1000.0;

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid: array<GridParticle>;

fn atomic_add_mass(idx: u32, value: f32) {
    var old_bits = atomicLoad(&grid[idx].mass);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&grid[idx].mass, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

fn atomic_add_momentum_x(idx: u32, value: f32) {
    var old_bits = atomicLoad(&grid[idx].momentum_x);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&grid[idx].momentum_x, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

fn atomic_add_momentum_y(idx: u32, value: f32) {
    var old_bits = atomicLoad(&grid[idx].momentum_y);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&grid[idx].momentum_y, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

fn atomic_add_force_x(idx: u32, value: f32) {
    var old_bits = atomicLoad(&grid[idx].force_x);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&grid[idx].force_x, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

fn atomic_add_force_y(idx: u32, value: f32) {
    var old_bits = atomicLoad(&grid[idx].force_y);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&grid[idx].force_y, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

fn fluid_stress(idx: u32) -> mat3x3<f32> {
    let f = particles[idx].deformation_gradient;
    let j = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let p = max(-0.1, K * (pow(j, -GAMMA) - 1.0));
    let s = j * p;

    let velocity_gradient = D_INV * particles[idx].B;
    var strain: mat3x3<f32> = velocity_gradient;
    // 2d for now. will add z once we transition to 3d
    let tr = strain[0][0] + strain[1][1];
    strain[0][1] = tr;
    strain[1][0] = tr;
    return mat3x3<f32>(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0);
    // return mat3x3<f32>(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0) + 0.9 * strain;
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
            atomic_add_mass(idx, mass_contribution);
            atomic_add_momentum_x(idx, momentum_contribution.x);
            atomic_add_momentum_y(idx, momentum_contribution.y);
            atomic_add_force_x(idx, force_contribution.x);
            atomic_add_force_y(idx, force_contribution.y);
        }
    }
}
