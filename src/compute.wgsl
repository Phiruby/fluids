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
const DT: f32 = 0.002;
const VELOCITY_DAMPING: f32 = 0.995;
const PENALTY_ALPHA: f32 = 0.9;

@group(0) @binding(0) var<storage, read> reference_particles: array<Particle>;
@group(0) @binding(1) var<storage, read> current_particles: array<Particle>;
@group(0) @binding(2) var<storage, read> reference_gradients: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> reference_kernel_weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_particles: array<Particle>;

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

    var particle = current_particles[particle_index];
    let force = compute_force(particle_index);
    let penalty_force = compute_penalty_force(
        particle_index,
        initial_deformation_tensor_estimate(particle_index),
    );

    if is_finite_vec3(force)
        && is_finite_vec3(penalty_force)
        && is_finite_vec3(particle.velocity)
        && is_finite_vec3(particle.position)
        && is_finite(particle.mass)
        && particle.mass != 0.0
    {
        particle.velocity += (force + penalty_force) * DT / particle.mass;
        particle.velocity *= VELOCITY_DAMPING;
        particle.position += particle.velocity * DT;
    }
    dst_particles[particle_index] = particle;
}


fn neighbor_count(i: u32) -> u32 {
    return min(current_particles[i].nneighbors, 27u);
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
