struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>,
    density: f32,
    mass: f32,
    renderable: u32,
    nneighbors: u32,
    neighbors: array<u32, 27>, // NOTE: assuming max 27 neighbors
    L: mat3x3<f32>,
}

struct SimParams {
    ray_origin: vec3<f32>,
    mouse_pressed: u32,
    ray_end: vec3<f32>,
    grid_n: u32,
    ref_min: vec3<f32>,
    ref_max: vec3<f32>,
}

struct InteractionState {
    selected_face: i32,
}

const DT: f32 = 0.0004;
const VELOCITY_DAMPING: f32 = 0.995;
const PENALTY_ALPHA: f32 = 0.9;
const STRETCH_FORCE: f32 = 60.0;
const STRETCH_DAMPING: f32 = 8.0;
const MAX_STRETCH: f32 = 5.05;
const COLOR_DEFAULT: vec3<f32> = vec3<f32>(0.72, 0.45, 0.95);
const COLOR_HOVER: vec3<f32> = vec3<f32>(1.0, 0.85, 0.15);

@group(0) @binding(0) var<storage, read> reference_particles: array<Particle>;
@group(0) @binding(1) var<storage, read> current_particles: array<Particle>;
@group(0) @binding(2) var<storage, read> reference_gradients: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> reference_kernel_weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_particles: array<Particle>;
@group(0) @binding(5) var<uniform> params: SimParams;
@group(0) @binding(6) var<storage, read_write> interaction: InteractionState;

// For debugging (everything was NaN!)
fn is_finite(x: f32) -> bool {
    // NaN: x != x; Inf: x*0 != 0
    return (x == x) && (x * 0.0 == 0.0);
}

fn is_finite_vec3(v: vec3<f32>) -> bool {
    return is_finite(v.x) && is_finite(v.y) && is_finite(v.z);
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_particles = arrayLength(&reference_particles);
    let particle_index = global_id.x;
    if particle_index >= total_particles {
        return;
    }

    let n = params.grid_n;
    let hovered = find_hovered_face(n);
    var selected = interaction.selected_face;
    if params.mouse_pressed != 0u {
        if selected < 0 {
            selected = hovered;
        }
    } else {
        selected = -1;
    }
    if particle_index == 0u {
        interaction.selected_face = selected;
    }

    var particle = current_particles[particle_index];
    let pre_force_position = particle.position;
    var force = compute_force(particle_index);
    let penalty_force = compute_penalty_force(
        particle_index,
        initial_deformation_tensor_estimate(particle_index),
    );
    force += penalty_force;

    if selected >= 0 {
        let face = u32(selected);
        if particle_on_face(particle_index, face, n) {
            force += face_stretch_force(particle, face, stretch_target(face, n));
        }
    }

    if is_finite_vec3(force)
        && is_finite_vec3(particle.velocity)
        && is_finite_vec3(particle.position)
        && is_finite(particle.mass)
        && particle.mass != 0.0
    {
        particle.velocity += force * DT / particle.mass;
        particle.velocity *= VELOCITY_DAMPING;
        particle.position += particle.velocity * DT;
    }

    if selected >= 0 {
        let opposite = opposite_face(u32(selected));
        if particle_on_face(particle_index, opposite, n) {
            pin_particle_axis(&particle, pre_force_position, opposite);
        }
    }

    let hover_hit = hovered >= 0 && particle_on_face(particle_index, u32(hovered), n);
    let selected_hit = selected >= 0 && particle_on_face(particle_index, u32(selected), n);
    if hover_hit || selected_hit {
        particle.normal = COLOR_HOVER;
    } else {
        particle.normal = COLOR_DEFAULT;
    }

    dst_particles[particle_index] = particle;
}

fn neighbor_count(i: u32) -> u32 {
    return min(current_particles[i].nneighbors, 27u);
}

fn grid_ijk(idx: u32, n: u32) -> vec3<u32> {
    let n2 = n * n;
    return vec3<u32>(idx / n2, (idx / n) % n, idx % n);
}

fn particle_on_face(idx: u32, face: u32, n: u32) -> bool {
    let ijk = grid_ijk(idx, n);
    switch face {
        case 0u: { return ijk.x == 0u; }
        case 1u: { return ijk.y == 0u; }
        case 2u: { return ijk.z == 0u; }
        case 3u: { return ijk.x == n - 1u; }
        case 4u: { return ijk.y == n - 1u; }
        case 5u: { return ijk.z == n - 1u; }
        default: { return false; }
    }
}

fn face_axis(face: u32) -> u32 {
    return face % 3u;
}

fn opposite_face(face: u32) -> u32 {
    return (face + 3u) % 6u;
}

fn face_normal(face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 1u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 3u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 1.0, 0.0); }
        default: { return vec3<f32>(0.0, 0.0, 1.0); }
    }
}

fn vec_component(v: vec3<f32>, axis: u32) -> f32 {
    switch axis {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        default: { return v.z; }
    }
}

fn face_component(v: vec3<f32>, face: u32) -> f32 {
    return vec_component(v, face_axis(face));
}

fn face_vertex_index(face: u32, a: u32, b: u32, n: u32) -> u32 {
    let n2 = n * n;
    switch face {
        case 0u: { return a * n + b; }                         // i = 0
        case 1u: { return a * n2 + b; }                        // j = 0
        case 2u: { return a * n2 + b * n; }                    // k = 0
        case 3u: { return (n - 1u) * n2 + a * n + b; }         // i = n-1
        case 4u: { return a * n2 + (n - 1u) * n + b; }         // j = n-1
        default: { return a * n2 + b * n + (n - 1u); }         // k = n-1
    }
}

struct FaceBounds {
    center: vec3<f32>,
    min_p: vec3<f32>,
    max_p: vec3<f32>,
}

// Face bounds are computed as:
// face center +/- extent
fn face_bounds(face: u32, n: u32) -> FaceBounds {
    var sum = vec3<f32>(0.0);
    var min_p = vec3<f32>(1e20);
    var max_p = vec3<f32>(-1e20);
    for (var a = 0u; a < n; a++) {
        for (var b = 0u; b < n; b++) {
            let p = current_particles[face_vertex_index(face, a, b, n)].position;
            sum += p;
            min_p = min(min_p, p);
            max_p = max(max_p, p);
        }
    }
    let center = sum / f32(n * n);
    let half = max(max_p - min_p, vec3<f32>(1e-4)) * 0.5;
    return FaceBounds(center, center - half, center + half);
}

fn face_center(face: u32, n: u32) -> vec3<f32> {
    return face_bounds(face, n).center;
}

fn cube_center(n: u32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var face = 0u; face < 6u; face++) {
        sum += face_center(face, n);
    }
    return sum / 6.0;
}

fn point_in_aabb(p: vec3<f32>, min_p: vec3<f32>, max_p: vec3<f32>) -> bool {
    return all(p >= min_p) && all(p <= max_p);
}

// First check if the face plane intersects the ray. Then perform
// face bound checks
fn ray_intersect_face_t(origin: vec3<f32>, dir: vec3<f32>, face: u32, n: u32) -> f32 {
    let bounds = face_bounds(face, n);
    let axis = face_axis(face);
    let coord = vec_component(bounds.center, axis);
    let denom = vec_component(dir, axis);
    if abs(denom) < 1e-6 {
        return -1.0;
    }
    let t = (coord - vec_component(origin, axis)) / denom;
    if t <= 0.0 {
        return -1.0;
    }
    let hit = origin + dir * t;
    if !point_in_aabb(hit, bounds.min_p, bounds.max_p) {
        return -1.0;
    }
    return t;
}

fn find_hovered_face(n: u32) -> i32 {
    let origin = params.ray_origin;
    let delta = params.ray_end - params.ray_origin;
    let len = length(delta);
    if len < 1e-6 {
        return -1;
    }
    let dir = delta / len;
    var best_face: i32 = -1;
    var best_t = 1e20;
    for (var face = 0u; face < 6u; face++) {
        let t = ray_intersect_face_t(origin, dir, face, n);
        if t > 0.0 && t < best_t {
            best_t = t;
            best_face = i32(face);
        }
    }
    return best_face;
}

// Solves the optimization problem:
// min_{u, t} || ray_origin + t * ray_dir - u * axis ||^2
// and then projects ray_orgin + t * ray_dir onto axis after we
// solved for t
fn project_ray_to_axis(origin: vec3<f32>, end: vec3<f32>, axis: vec3<f32>) -> vec3<f32> {
    let e = vec3<f32>(1.0, 1.0, 1.0);
    let axis_n = normalize(axis);
    let sum_r0 = dot(e, origin);
    let dir = normalize(end - origin);
    let sum_r = dot(e, dir);
    let sum_v = dot(e, axis_n);
    let u = (sum_r0 + sum_r0 * sum_r) / (1.0 + sum_r * sum_v);
    let t = max(sum_r0 - u * sum_v, 0.0);
    let closest = origin + dir * t;
    return axis_n * dot(closest, axis_n);
}

fn clamp_stretch_target(face: u32, target_num: f32) -> f32 {
    let axis = face_axis(face);
    let mn = vec_component(params.ref_min, axis);
    let mx = vec_component(params.ref_max, axis);
    let extra = (mx - mn) * (MAX_STRETCH - 1.0);
    if face <= 2u {
        return clamp(target_num, mn - extra, mn);
    }
    return clamp(target_num, mx, mx + extra);
}

fn stretch_target(face: u32, n: u32) -> f32 {
    let normal = face_normal(face);
    let center = cube_center(n);
    let projected = project_ray_to_axis(params.ray_origin, params.ray_end, normal) - center;
    return clamp_stretch_target(face, face_component(projected, face));
}

fn face_stretch_force(particle: Particle, face: u32, target_axis: f32) -> vec3<f32> {
    let axis = face_axis(face);
    let current_axis = vec_component(particle.position, axis);
    let velocity_axis = vec_component(particle.velocity, axis);
    let force_scalar = STRETCH_FORCE * (target_axis - current_axis) - STRETCH_DAMPING * velocity_axis;
    var force = vec3<f32>(0.0);
    switch axis {
        case 0u: { force.x = force_scalar; }
        case 1u: { force.y = force_scalar; }
        default: { force.z = force_scalar; }
    }
    return force;
}

// NOTE: this function fully assisted by AI
fn pin_particle_axis(particle: ptr<function, Particle>, pinned_position: vec3<f32>, face: u32) {
    let axis = face_axis(face);
    switch axis {
        case 0u: {
            (*particle).position.x = pinned_position.x;
            (*particle).velocity.x = 0.0;
        }
        case 1u: {
            (*particle).position.y = pinned_position.y;
            (*particle).velocity.y = 0.0;
        }
        default: {
            (*particle).position.z = pinned_position.z;
            (*particle).velocity.z = 0.0;
        }
    }
}

fn initial_deformation_tensor_estimate(
    i: u32,
) -> mat3x3<f32> {
    let nparticles = arrayLength(&current_particles);
    var deformation = mat3x3<f32>(
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
    );
    let nneighbors = neighbor_count(i);
    for (var nbr: u32 = 0; nbr < nneighbors; nbr++) {
        let j = current_particles[i].neighbors[nbr];
        let x_ji = current_particles[j].position - current_particles[i].position;
        let grad_w_ij = reference_gradients[i * nparticles + j];
        let v_j = reference_particles[j].mass / reference_particles[j].density;
        // x_ji * grad^T  (same as CPU: x_ji * grad_w_ij.transpose())
        deformation += v_j * outer_prod(x_ji, grad_w_ij);
    }
    return current_particles[i].L * deformation;
}

fn compute_rotations(deformation: mat3x3<f32>) -> mat3x3<f32> {
    return polar_decomposition_rotation(deformation);
}

fn compute_grad_star(i: u32, j: u32, rotations: mat3x3<f32>) -> vec3<f32> {
    let nparticles = arrayLength(&current_particles);
    return rotations * current_particles[i].L * reference_gradients[i * nparticles + j];
}

fn deformation_tensor_star(i: u32, rotations: mat3x3<f32>) -> mat3x3<f32> {
    let identity = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    var nbsum = mat3x3<f32>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    for (var nbr: u32 = 0; nbr < neighbor_count(i); nbr++) {
        let j = current_particles[i].neighbors[nbr];
        let grad_star = compute_grad_star(i, j, rotations);
        let ref_volume = reference_particles[j].mass / reference_particles[j].density;
        let x_ji = current_particles[j].position - current_particles[i].position;
        let X_ji = reference_particles[j].position - reference_particles[i].position;
        // (x_ji - R X_ji) * grad*^T
        nbsum += ref_volume * outer_prod((x_ji - rotations * X_ji), grad_star);
    }
    return identity + nbsum;
}

struct StressAndRotation {
    stress_tensor: mat3x3<f32>,
    rotation: mat3x3<f32>,
}

fn stress_tensor_and_rotation(i: u32) -> StressAndRotation {
    let initial_deformation = initial_deformation_tensor_estimate(i);
    let rotation = compute_rotations(initial_deformation);
    let deformation = deformation_tensor_star(i, rotation);
    let k = 100000.0;
    let nu = 0.3;
    let mu = k / (2.0 * (1.0 + nu));
    let lambda = k * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let identity = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    let strain = 0.5 * (deformation + transpose(deformation)) - identity;
    let strain_trace = strain[0][0] + strain[1][1] + strain[2][2];
    return StressAndRotation(2.0 * mu * strain + lambda * strain_trace * identity, rotation);
}

fn compute_force(i: u32) -> vec3<f32> {
    let stress_and_rotation = stress_tensor_and_rotation(i);
    let stress_tensor = stress_and_rotation.stress_tensor;
    let rotation = stress_and_rotation.rotation;
    var force = vec3<f32>(0.0, 0.0, 0.0);
    let ref_volume_i = reference_particles[i].mass / reference_particles[i].density;
    for (var nbr: u32 = 0; nbr < neighbor_count(i); nbr++) {
        let j = current_particles[i].neighbors[nbr];
        let ref_volume_j = reference_particles[j].mass / reference_particles[j].density;
        // TODO: redundant computation (multiple threads compute this); maybe separate step?
        let stress_and_rotation_j = stress_tensor_and_rotation(j);
        let stress_j = stress_and_rotation_j.stress_tensor;
        let rotation_j = stress_and_rotation_j.rotation;
        force += ref_volume_i * ref_volume_j * (
            stress_tensor * compute_grad_star(i, j, rotation)
            - stress_j * compute_grad_star(j, i, rotation_j)
        );
    }
    return force;
}

fn compute_penalty_force(i: u32, deformation_i: mat3x3<f32>) -> vec3<f32> {
    let nparticles = arrayLength(&current_particles);
    let k = 100000.0;
    var force = vec3<f32>(0.0, 0.0, 0.0);
    for (var nbr: u32 = 0; nbr < neighbor_count(i); nbr++) {
        let j = current_particles[i].neighbors[nbr];
        if j == i {
            continue;
        }
        let ref_volume_j = reference_particles[j].mass / reference_particles[j].density;
        let ref_volume_i = reference_particles[i].mass / reference_particles[i].density;
        let x_ji = current_particles[j].position - current_particles[i].position;
        let X_ji = reference_particles[j].position - reference_particles[i].position;
        let X_ji_norm_sq = dot(X_ji, X_ji);
        let x_ji_len = length(x_ji);
        if X_ji_norm_sq < 1e-20 || x_ji_len < 1e-20 {
            continue;
        }
        let kernel_weight = reference_kernel_weights[i * nparticles + j] / X_ji_norm_sq;
        // TODO: again, computing deformation[j]; redundant work
        let deformation_j = initial_deformation_tensor_estimate(j);
        let e_ji = deformation_i * X_ji - x_ji;
        let e_ij = x_ji - deformation_j * X_ji;
        let X_ji_len = sqrt(X_ji_norm_sq);
        force += kernel_weight * ref_volume_i * ref_volume_j * (
            dot(e_ji, x_ji) / x_ji_len - dot(e_ij, X_ji) / X_ji_len
        ) * (x_ji / x_ji_len);
    }
    // Match CPU: -0.5 * alpha * k * sum
    return -0.5 * PENALTY_ALPHA * k * force;
}

fn adjugate3x3(m: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];

    let d00 = c1.y * c2.z - c1.z * c2.y;
    let d01 = c1.x * c2.z - c1.z * c2.x;
    let d02 = c1.x * c2.y - c1.y * c2.x;
    let d10 = c0.y * c2.z - c0.z * c2.y;
    let d11 = c0.x * c2.z - c0.z * c2.x;
    let d12 = c0.x * c2.y - c0.y * c2.x;
    let d20 = c0.y * c1.z - c0.z * c1.y;
    let d21 = c0.x * c1.z - c0.z * c1.x;
    let d22 = c0.x * c1.y - c0.y * c1.x;

    return mat3x3<f32>(
        vec3<f32>( d00, -d10,  d20),
        vec3<f32>(-d01,  d11, -d21),
        vec3<f32>( d02, -d12,  d22)
    );
}

fn frobenius_norm_sq(m: mat3x3<f32>) -> f32 {
    return dot(m[0], m[0]) + dot(m[1], m[1]) + dot(m[2], m[2]);
}

fn identity3() -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
}

// See link below (though we implement Higham's scaling factor, not the Halley's version)
// https://www.researchgate.net/publication/220656598_Optimizing_Halley's_Iteration_for_Computing_the_Matrix_Polar_Decomposition
fn polar_decomposition_rotation(deformation: mat3x3<f32>) -> mat3x3<f32> {
    var X = deformation;
    if frobenius_norm_sq(X) < 1e-20 {
        return identity3();
    }

    for (var i = 0; i < 4; i = i + 1) {
        let detX = determinant(X);
        let normX_sq = frobenius_norm_sq(X);
        if abs(detX) < 1e-12 || normX_sq < 1e-20 {
            return identity3();
        }
        let adjX = adjugate3x3(X);
        let normInvX_sq = frobenius_norm_sq(adjX) / (detX * detX);
        let gamma = sqrt(sqrt(normInvX_sq / normX_sq));
        X = 0.5 * ((gamma * X) + (1.0 / (gamma * detX)) * adjX);
    }

    return X;
}

// a * b^T  (column c = a * b[c]), matching nalgebra `a * b.transpose()`
fn outer_prod(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        a * b.x,
        a * b.y,
        a * b.z,
    );
}
