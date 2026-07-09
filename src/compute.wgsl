struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    mass: f32,
    density: f32,
    renderable: u32,
    nneighbors: u32,
    neighbors: array<u32, 27>, // NOTE: assuming max 27 neighbors
    L: mat3x3<f32>,
}


@group(0) @binding(0) var<storage, read> reference_particles: array<Particle>;
@group(0) @binding(1) var<storage, read> current_particles: array<Particle>;
@group(0) @binding(2) var<storage, read> reference_gradients: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> reference_kernel_weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_particles: array<Particle>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_particles = arrayLength(&reference_particles);
    let particle_index = global_id.x;
    if particle_index >= total_particles {
        return;
    }
    let x = reference_gradients[particle_index];
    // NOTE: I will assume the particles are indexed with i * n ^ 2 + j * n + k
    let force = compute_force(particle_index);
    let penalty_force = compute_penalty_force(particle_index, initial_deformation_tensor_estimate(particle_index));
    dst_particles[particle_index] = current_particles[particle_index];
    dst_particles[particle_index].velocity += (force + penalty_force) * 0.001 / current_particles[particle_index].mass;
    dst_particles[particle_index].position += dst_particles[particle_index].velocity * 0.001;
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
    for (var nbr: u32 = 0; nbr < current_particles[i].nneighbors; nbr++) {
        let j = current_particles[i].neighbors[nbr];
        let x_ji = current_particles[j].position - current_particles[i].position;
        let grad_w_ij = reference_gradients[i * nparticles + j];
        let v_j = reference_particles[j].mass / reference_particles[j].density;
        // NOTE: columnar matrix I think: so first arg is col 0, second col 1, ...
        let outer_prod = mat3x3<f32>(
            x_ji * grad_w_ij.x,
            x_ji * grad_w_ij.y,
            x_ji * grad_w_ij.z 
        );
        deformation += v_j * outer_prod;
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
    let nparticles = arrayLength(&current_particles);
    let identity = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    var nbsum = mat3x3 <f32>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    for (var nbr: u32 = 0; nbr < current_particles[i].nneighbors; nbr++) {
        let j = current_particles[i].neighbors[nbr];
        let grad_star = compute_grad_star(i, j, rotations);
        let ref_volume = reference_particles[j].mass / reference_particles[j].density;
        let x_ji = current_particles[j].position - current_particles[i].position;
        let X_ji = reference_particles[j].position - reference_particles[i].position;
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
    for (var nbr: u32 = 0; nbr < current_particles[i].nneighbors; nbr++) {
        let j = current_particles[i].neighbors[nbr];
        let ref_volume_j = reference_particles[j].mass / reference_particles[j].density;
        // TODO: redundant computation (multiple threads compute this); maybe separate step?
        let stress_j = stress_tensor_and_rotation(j).stress_tensor;
        force += ref_volume_i * ref_volume_j * stress_tensor * compute_grad_star(i, j, rotation) - stress_j * compute_grad_star(j, i, rotation);
    }
    return force;
}

fn compute_penalty_force(i: u32, deformation: mat3x3<f32>) -> vec3<f32> {
    let nparticles = arrayLength(&current_particles);
    let alpha = 0.0;
    let k = 100000.0;
    var force = vec3<f32>(0.0, 0.0, 0.0);
    for (var nbr: u32 = 0; nbr < current_particles[i].nneighbors; nbr++) {
        let j = current_particles[i].neighbors[nbr];
        let ref_volume_j = reference_particles[j].mass / reference_particles[j].density;
        let ref_volume_i = reference_particles[i].mass / reference_particles[i].density;
        let x_ji = current_particles[j].position - current_particles[i].position;
        let x_ij = -x_ji;
        let X_ji = reference_particles[j].position - reference_particles[i].position;
        let X_ij = -X_ji;
        let X_ij_norm_sq = length(X_ji) * length(X_ji);
        let kernel_weight = reference_kernel_weights[i * nparticles + j] / X_ij_norm_sq;
        // TODO: again, computing deformation[j]; redundant work
        let initial_deformation_j = initial_deformation_tensor_estimate(j);
        let rotation_j = compute_rotations(initial_deformation_j);
        let deformation_j = deformation_tensor_star(j, rotation_j);
        let e_ij = deformation_j * X_ij + x_ij;
        let e_ji = deformation * X_ji - x_ji;
        
        force += kernel_weight * ref_volume_i * ref_volume_j * (
            dot(e_ji, x_ji) / length(x_ji) - dot(e_ij, x_ij) / length(x_ij)) * x_ji / length(x_ji);
    }
    return 0.5 * alpha * k * force;
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

// See link below (though we implement Highham's scaling factor, not the Halley's version)
// https://www.researchgate.net/publication/220656598_Optimizing_Halley's_Iteration_for_Computing_the_Matrix_Polar_Decomposition
fn polar_decomposition_rotation(deformation: mat3x3<f32>) -> mat3x3<f32> {
    var X = deformation;
    
    for (var i = 0; i < 4; i = i + 1) {
        var detX = determinant(X);
        let adjX = adjugate3x3(X);
        
        let normX_sq    = frobenius_norm_sq(X);
        let normInvX_sq = frobenius_norm_sq(adjX) / (detX * detX);
        let gamma = sqrt(sqrt(normInvX_sq / normX_sq));
        X = 0.5 * ((gamma * X) + (1.0 / (gamma * detX)) * adjX);
    }

    return X;
}

fn outer_prod(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(a.x * b.x, a.x * b.y, a.x * b.z),
        vec3<f32>(a.y * b.x, a.y * b.y, a.y * b.z),
        vec3<f32>(a.z * b.x, a.z * b.y, a.z * b.z),
    );
}