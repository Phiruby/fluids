#include <../svd.wgsl>

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

const DX: f32 = 0.01;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const N: u32 = 192u;
const D_INV = 4.0 / (DX * DX);
const DT = 0.0001;

// for elastics
const THETA_C = 0.01;
const THETA_S = 0.01;

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

fn get_grid_index(coord: vec2<u32>) -> u32 {
    return coord.x * N + coord.y;
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
    let relative = (position.xy - BOTTOM_LEFT.xy) / DX;
    let base_coord = vec2<i32>(floor(relative - 0.5));

    var new_v = vec3<f32>(0.0);
    var new_B = mat3x3<f32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var new_pos = vec3<f32>(0.0);

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
            let xi_xp = node_pos - vec3<f32>(position.xy, 0.0);
            let r = xi_xp.xy / DX;
            let weight = quadratic_kernel(r.x) * quadratic_kernel(r.y);
            let idx = get_grid_index(vec2<u32>(current_coord));

            if grid[idx].mass <= 1e-8 { continue; }
            let vi = vec3<f32>(
                f32(grid[idx].momentum_x),
                f32(grid[idx].momentum_y),
                0.0,
            ) / f32(grid[idx].mass);
            new_v += vi * weight;
            new_B += weight * outer(vi, xi_xp);
            new_pos += (node_pos + vi * DT) * weight;
        }
    }

    result.new_v = new_v;
    result.new_B = new_B;
    result.new_pos = new_pos;
    return result;
}

@compute
@workgroup_size(256)
fn fluid_grid_to_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&particles) { return; }
    var p = particles[p_idx];

    let updates = get_updates(p.position);
    let new_v = updates.new_v;
    let new_B = updates.new_B;
    let new_pos = updates.new_pos;
    p.velocity = vec3<f32>(new_v.xy, 0.0);
    p.B = new_B;
    // eq (17)
    let C = D_INV * p.B;
    var j = p.deformation_gradient[0][0] * p.deformation_gradient[1][1]
        - p.deformation_gradient[0][1] * p.deformation_gradient[1][0];
    j *= 1.0 + DT * (C[0][0] + C[1][1]);
    j = max(j, 0.1);
    let s = sqrt(j);
    p.deformation_gradient = mat3x3<f32>(
        s, 0.0, 0.0,
        0.0, s, 0.0,
        0.0, 0.0, 1.0,
    );
    p.position = vec3<f32>(
        clamp(new_pos.x, BOTTOM_LEFT.x + DX, BOTTOM_LEFT.x + f32(N - 2) * DX),
        clamp(new_pos.y, BOTTOM_LEFT.y + DX, BOTTOM_LEFT.y + f32(N - 2) * DX),
        0.0,
    );

    let wall_min = collider.min.xy;
    let wall_max = collider.max.xy;
    // ideally we want x_n = x + v (and no DT multiple. this is so we predict 1 second ahead).
    // but this caused unusual behavior, so predict 5 steps ahead
    let x_n = p.position.xy + 15 * DT * p.velocity.xy;
    if x_n.x < wall_min.x {
        p.velocity.x += wall_min.x - x_n.x;
    }
    if x_n.x > wall_max.x {
        p.velocity.x += wall_max.x - x_n.x;
    }
    if x_n.y < wall_min.y {
        p.velocity.y += wall_min.y - x_n.y;
    }
    if x_n.y > wall_max.y {
        p.velocity.y += wall_max.y - x_n.y;
    }

    particles[p_idx] = p;
}

fn det2(m: mat3x3<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}
struct PlasticUpdate {
    new_elastic_gradient: mat3x3<f32>,
    new_plastic_determinant: f32,
}

// eq (79) - (86) https://alexey.stomakhin.com/research/siggraph2016_mpm.pdf
fn plastic_projection(new_elastic_gradient: mat3x3<f32>, plastic_deformation_determinant: f32, C: mat3x3<f32>) -> PlasticUpdate {
    var result: PlasticUpdate;
    let svd_decomp = svd3x3(new_elastic_gradient);

    let clamped = vec3<f32>(
        clamp(svd_decomp.S[0], 1.0 - THETA_C, 1.0 + THETA_S),
        clamp(svd_decomp.S[1], 1.0 - THETA_C, 1.0 + THETA_S),
        0.0,
    );
    // determinant update is not shown in the pdf. but from eq (86), u can derive it urself
    // this is how much the clamped elastic grad's det scaled by
    let det_scale = (clamped[0] / svd_decomp.S[0]) * (clamped[1] / svd_decomp.S[1]) * (clamped[2] / svd_decomp.S[2]);
    let new_determinant = plastic_deformation_determinant / det_scale;

    let clamped_S = mat3x3<f32>(clamped[0], 0.0, 0.0, 0.0, clamped[1], 0.0, 0.0, 0.0, clamped[2]);
    let new_gradient = svd_decomp.U * clamped_S * transpose(svd_decomp.V);
    result.new_elastic_gradient = new_gradient;
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
    let new_v = updates.new_v;
    let new_B = updates.new_B;
    let new_pos = updates.new_pos;
    p.velocity = vec3<f32>(new_v.xy, 0.0);
    p.B = new_B;
    // eq (17)
    let C = D_INV * p.B;
    let identity = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    // appyling eq 17 from mls-mpm to elastic component
    p.elastic_deformation_gradient = (identity + DT * C) * p.elastic_deformation_gradient;
    p.position = vec3<f32>(
        clamp(new_pos.x, BOTTOM_LEFT.x + DX, BOTTOM_LEFT.x + f32(N - 2) * DX),
        clamp(new_pos.y, BOTTOM_LEFT.y + DX, BOTTOM_LEFT.y + f32(N - 2) * DX),
        0.0,
    );
    let plastic_update = plastic_projection(p.elastic_deformation_gradient, p.plastic_deformation_determinant, C);
    p.elastic_deformation_gradient = plastic_update.new_elastic_gradient;
    p.plastic_deformation_determinant = plastic_update.new_plastic_determinant;

    let wall_min = collider.min.xy;
    let wall_max = collider.max.xy;
    // ideally we want x_n = x + v (and no DT multiple. this is so we predict 1 second ahead).
    // but this caused unusual behavior, so predict 5 steps ahead
    let x_n = p.position.xy + 15 * DT * p.velocity.xy;
    if x_n.x < wall_min.x {
        p.velocity.x += wall_min.x - x_n.x;
    }
    if x_n.x > wall_max.x {
        p.velocity.x += wall_max.x - x_n.x;
    }
    if x_n.y < wall_min.y {
        p.velocity.y += wall_min.y - x_n.y;
    }
    if x_n.y > wall_max.y {
        p.velocity.y += wall_max.y - x_n.y;
    }

    elastic_particles[p_idx] = p;
}
