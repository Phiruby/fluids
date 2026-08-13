#include <../svd.wgsl>
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

const D_INV = 4.0 / (DX * DX);
const DT = 0.0001;

const THETA_C = 0.2;
const THETA_S = 0.2;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> grid: array<GridParticle>;
@group(0) @binding(2) var<uniform> collider: ColliderUniform;
// TODO: binding elastic and fluid together will cause fluid and elastic compute shaders
// to run sequentially (wgpu automatically adds them)
@group(0) @binding(3) var<storage, read_write> elastic_particles: array<ElasticParticle>;

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
    return mat3x3<f32>(v1 * v2.x, v1 * v2.y, v1 * v2.z);
}

struct Updates {
    new_B: mat3x3<f32>,
    new_pos: vec3<f32>,
    new_v: vec3<f32>,
}

fn get_updates(position: vec3<f32>) -> Updates {
    var result: Updates;
    let relative = (position - BOTTOM_LEFT) / DX;
    let base_coord = vec3<i32>(floor(relative - 0.5));

    var new_v = vec3<f32>(0.0);
    var new_B = mat3x3<f32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

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
                let idx = get_grid_index(vec3<u32>(current_coord));

                if grid[idx].mass <= 1e-8 { continue; }
                let vi = vec3<f32>(
                    f32(grid[idx].momentum_x),
                    f32(grid[idx].momentum_y),
                    f32(grid[idx].momentum_z),
                ) / f32(grid[idx].mass);
                new_v += vi * weight;
                new_B += weight * outer(vi, xi_xp);
            }
        }
    }

    result.new_v = new_v;
    result.new_B = new_B;
    result.new_pos = position + new_v * DT;
    return result;
}

fn clamp_to_domain(p: vec3<f32>) -> vec3<f32> {
    let lo = BOTTOM_LEFT + vec3<f32>(DX);
    let hi = BOTTOM_LEFT + vec3<f32>(f32(N - 2) * DX);
    return clamp(p, lo, hi);
}

fn apply_wall_velocity(position: vec3<f32>, velocity: vec3<f32>) -> vec3<f32> {
    var v = velocity;
    let wall_min = collider.min;
    let wall_max = collider.max;
    // ideally we want x_n = x + v (and no DT multiple. this is so we predict 1 second ahead).
    // but this caused unusual behavior, so predict 5 steps ahead
    let x_n = position + 15.0 * DT * v;
    if x_n.x < wall_min.x { v.x += wall_min.x - x_n.x; }
    if x_n.x > wall_max.x { v.x += wall_max.x - x_n.x; }
    if x_n.y < wall_min.y { v.y += wall_min.y - x_n.y; }
    if x_n.y > wall_max.y { v.y += wall_max.y - x_n.y; }
    if x_n.z < wall_min.z { v.z += wall_min.z - x_n.z; }
    if x_n.z > wall_max.z { v.z += wall_max.z - x_n.z; }
    return v;
}

@compute
@workgroup_size(256)
fn fluid_grid_to_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&particles) { return; }
    var p = particles[p_idx];
    if p.position.x > 50.0 { return; }

    let updates = get_updates(p.position);
    p.velocity = updates.new_v;
    p.B = updates.new_B;
    let C = D_INV * p.B;
    var j = determinant(p.deformation_gradient);
    j *= 1.0 + DT * (C[0][0] + C[1][1] + C[2][2]);
    j = max(j, 0.1);
    let s = pow(j, 1.0 / 3.0);
    p.deformation_gradient = mat3x3<f32>(
        s, 0.0, 0.0,
        0.0, s, 0.0,
        0.0, 0.0, s,
    );
    p.position = clamp_to_domain(updates.new_pos);
    p.velocity = apply_wall_velocity(p.position, p.velocity);

    particles[p_idx] = p;
}

struct PlasticUpdate {
    new_elastic_gradient: mat3x3<f32>,
    new_plastic_determinant: f32,
}

// eq (79) - (86) https://alexey.stomakhin.com/research/siggraph2016_mpm.pdf
fn plastic_projection(
    new_elastic_gradient: mat3x3<f32>,
    plastic_deformation_determinant: f32,
) -> PlasticUpdate {
    var result: PlasticUpdate;
    let svd_decomp = svd3x3(new_elastic_gradient);

    var U = svd_decomp.U;
    var S = svd_decomp.S;
    if S[0] < 0.0 { S[0] = -S[0]; U[0] = -U[0]; }
    if S[1] < 0.0 { S[1] = -S[1]; U[1] = -U[1]; }
    if S[2] < 0.0 { S[2] = -S[2]; U[2] = -U[2]; }

    let clamped = vec3<f32>(
        clamp(S[0], 1.0 - THETA_C, 1.0 + THETA_S),
        clamp(S[1], 1.0 - THETA_C, 1.0 + THETA_S),
        clamp(S[2], 1.0 - THETA_C, 1.0 + THETA_S),
    );
    let det_scale = (clamped[0] / max(S[0], 1e-8))
        * (clamped[1] / max(S[1], 1e-8))
        * (clamped[2] / max(S[2], 1e-8));
    let new_determinant = plastic_deformation_determinant / det_scale;

    let clamped_S = mat3x3<f32>(
        clamped[0], 0.0, 0.0,
        0.0, clamped[1], 0.0,
        0.0, 0.0, clamped[2],
    );
    result.new_elastic_gradient = U * clamped_S * transpose(svd_decomp.V);
    result.new_plastic_determinant = new_determinant;
    return result;
}

@compute
@workgroup_size(256)
fn elastic_grid_to_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&elastic_particles) { return; }
    var p = elastic_particles[p_idx];

    let updates = get_updates(p.position);
    p.velocity = updates.new_v;
    p.B = updates.new_B;
    let C = D_INV * p.B;
    let identity = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    // applying eq 17 from mls-mpm to elastic component (ignoring plastic component)
    p.elastic_deformation_gradient = (identity + DT * C) * p.elastic_deformation_gradient;
    p.position = clamp_to_domain(updates.new_pos);

    let plastic_update = plastic_projection(
        p.elastic_deformation_gradient,
        p.plastic_deformation_determinant,
    );
    p.elastic_deformation_gradient = plastic_update.new_elastic_gradient;
    p.plastic_deformation_determinant = plastic_update.new_plastic_determinant;
    p.velocity = apply_wall_velocity(p.position, p.velocity);

    elastic_particles[p_idx] = p;
}
