#include <constants.wgsl>
struct Particle {
    deformation_gradient: mat3x3<f32>,
    B: mat3x3<f32>,
    position: vec3<f32>,
    velocity: vec3<f32>,
    rest_density: f32,
    mass: f32,
}

struct ElasticParticle {
    elastic_deformation_gradient: mat3x3<f32>,
    B: mat3x3<f32>,
    position: vec3<f32>,
    velocity: vec3<f32>,
    plastic_deformation_determinant: f32,
    rest_volume: f32,
    mass: f32,
}

struct GridParticle {
    // all the atomic<u32> are treated as f32 in the bit level.
    /// my machine doesn't support atomic<f32> extension for vulkan, so doing this
    momentum_x: atomic<u32>,
    momentum_y: atomic<u32>,
    momentum_z: atomic<u32>,
    mass: atomic<u32>,
    force_x: atomic<u32>,
    force_y: atomic<u32>,
    force_z: atomic<u32>,
}

const D_INV = 4.0 / (DX * DX);
const EOS_STIFFNESS = 100000.0;
const EOS_POWER = 7.0;

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid: array<GridParticle>;
@group(0) @binding(2) var<storage, read> elastic_particles: array<ElasticParticle>;

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

fn atomic_add_momentum_z(idx: u32, value: f32) {
    var old_bits = atomicLoad(&grid[idx].momentum_z);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&grid[idx].momentum_z, old_bits, new_bits);
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

fn atomic_add_force_z(idx: u32, value: f32) {
    var old_bits = atomicLoad(&grid[idx].force_z);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&grid[idx].force_z, old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}


fn det2(a: f32, b: f32, c: f32, d: f32) -> f32 {
    return a * d - b * c;
}

fn inverse_mat3(m: mat3x3<f32>) -> mat3x3<f32> {
    let row0 = m[0];
    let row1 = m[1];
    let row2 = m[2];
    let m0 = vec3<f32>(
        det2(row1.y, row1.z, row2.y, row2.z),
        det2(row1.z, row1.x, row2.z, row2.x),
        det2(row1.x, row1.y, row2.x, row2.y)
    );
    let m1 = vec3<f32>(
        det2(row2.y, row2.z, row0.y, row0.z),
        det2(row2.z, row2.x, row0.z, row0.x),
        det2(row2.x, row2.y, row0.x, row0.y)
    );
    let m2 = vec3<f32>(
        det2(row0.y, row0.z, row1.y, row1.z),
        det2(row0.z, row0.x, row1.x, row1.z),
        det2(row0.x, row0.y, row1.x, row1.y)
    );
    let adj = transpose(mat3x3<f32>(m0, m1, m2));
    let det = dot(row0, m0);
    return (1.0 / det) * adj;
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

fn grid_mass(idx: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&grid[idx].mass));
}

fn scatter_mass_momentum(
    position: vec3<f32>,
    velocity: vec3<f32>,
    B: mat3x3<f32>,
    mass: f32,
) {
    let relative = (position - BOTTOM_LEFT) / DX;
    let base_coord = vec3<i32>(floor(relative - 0.5));

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            for (var k = 0; k < 3; k++) {
                let current_coord = base_coord + vec3<i32>(i, j, k);
                if current_coord.x < 0 || current_coord.x >= i32(N) ||
                    current_coord.y < 0 || current_coord.y >= i32(N) ||
                    current_coord.z < 0 || current_coord.z >= i32(N) {
                    continue;
                }
                let node_pos = BOTTOM_LEFT + vec3<f32>(current_coord) * DX;
                let xi_xp = node_pos - position;
                let r = xi_xp / DX;
                let weight = quadratic_kernel(r.x) * quadratic_kernel(r.y) * quadratic_kernel(r.z);
                let affine_contribution = D_INV * B * xi_xp;
                let total_velocity = velocity + affine_contribution;
                let mass_contribution = weight * mass;
                let momentum_contribution = mass_contribution * total_velocity;
                let idx = get_grid_index(vec3<u32>(current_coord));
                atomic_add_mass(idx, mass_contribution);
                atomic_add_momentum_x(idx, momentum_contribution.x);
                atomic_add_momentum_y(idx, momentum_contribution.y);
                atomic_add_momentum_z(idx, momentum_contribution.z);
            }
        }
    }
}

@compute
@workgroup_size(256)
fn particles_to_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&particles) { return; }
    let p = particles[p_idx];
    // for out-of-scene particles (e.g fluid particles waiting to be pumped in)
    if p.position.x > 50.0 { return; }
    scatter_mass_momentum(p.position, p.velocity, p.B, p.mass);
}

@compute
@workgroup_size(256)
fn elastic_particles_to_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&elastic_particles) { return; }
    let p = elastic_particles[p_idx];
    scatter_mass_momentum(p.position, p.velocity, p.B, p.mass);
}

fn push_to_grid(position: vec3<f32>, volume: f32, stress: mat3x3<f32>) {
    let relative = (position - BOTTOM_LEFT) / DX;
    let base_coord = vec3<i32>(floor(relative - 0.5));

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            for (var k = 0; k < 3; k++) {
                let current_coord = base_coord + vec3<i32>(i, j, k);
                if current_coord.x < 0 || current_coord.x >= i32(N) ||
                    current_coord.y < 0 || current_coord.y >= i32(N) ||
                    current_coord.z < 0 || current_coord.z >= i32(N) {
                    continue;
                }
                let node_pos = BOTTOM_LEFT + vec3<f32>(current_coord) * DX;
                let xi_xp = node_pos - position;
                let r = xi_xp / DX;
                let weight = quadratic_kernel(r.x) * quadratic_kernel(r.y) * quadratic_kernel(r.z);
                let force_contribution = -weight * volume * D_INV * stress * xi_xp;
                let idx = get_grid_index(vec3<u32>(current_coord));
                atomic_add_force_x(idx, force_contribution.x);
                atomic_add_force_y(idx, force_contribution.y);
                atomic_add_force_z(idx, force_contribution.z);
            }
        }
    }
}

@compute
@workgroup_size(256)
fn fluid_particles_to_grid_stress(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&particles) { return; }

    let p = particles[p_idx];
    if p.position.x > 50.0 { return; }
    let relative = (p.position - BOTTOM_LEFT) / DX;
    let base_coord = vec3<i32>(floor(relative - 0.5));

    var density = 0.0;
    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            for (var k = 0; k < 3; k++) {
                let current_coord = base_coord + vec3<i32>(i, j, k);
                if current_coord.x < 0 || current_coord.x >= i32(N) ||
                    current_coord.y < 0 || current_coord.y >= i32(N) ||
                    current_coord.z < 0 || current_coord.z >= i32(N) {
                    continue;
                }
                let node_pos = BOTTOM_LEFT + vec3<f32>(current_coord) * DX;
                let r = (node_pos - p.position) / DX;
                let weight = quadratic_kernel(r.x) * quadratic_kernel(r.y) * quadratic_kernel(r.z);
                let idx = get_grid_index(vec3<u32>(current_coord));
                density += grid_mass(idx) * weight;
            }
        }
    }
    if density < 1e-8 {
        return;
    }
    let volume = (p.mass / density) * DX * DX * DX;
    let pressure = max(-0.1, EOS_STIFFNESS * (pow(density / p.rest_density, EOS_POWER) - 1.0));
    var stress: mat3x3<f32> = mat3x3<f32>(
        -pressure, 0.0, 0.0,
        0.0, -pressure, 0.0,
        0.0, 0.0, -pressure,
    );
    // var velocity_gradient: mat3x3<f32> = D_INV * p.B;
    // velocity_gradient = velocity_gradient + transpose(velocity_gradient);
    // stress += 0.9 * velocity_gradient;

    push_to_grid(p.position, volume, stress);
}

const POISSON_RATIO = 0.2;
const YOUNG = 100000.0;
const MU = YOUNG / (2.0 * (1.0 + POISSON_RATIO));
const LAMBDA = YOUNG * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO));
const XI = 4.0;

@compute
@workgroup_size(256)
fn elastic_particles_to_grid_stress(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&elastic_particles) { return; }

    let p = elastic_particles[p_idx];
    let F = p.elastic_deformation_gradient;
    let J = max(determinant(F), 1e-6);
    let F_inv_T = transpose(inverse_mat3(F));
    let identity = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

    let exponential = clamp(XI * (1.0 - p.plastic_deformation_determinant), -10.0, 100.0);
    let mu = MU * exp(exponential);
    let lambda = LAMBDA * exp(exponential);
    // Neo-Hookean Kirchhoff stress τ = P F^T
    let tau = mu * (F - F_inv_T) * transpose(F) + lambda * log(J) * identity;

    push_to_grid(p.position, p.rest_volume, tau);
}
