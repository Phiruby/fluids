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
@group(0) @binding(3) var<storage, read_write> dst_particles: array<Particle>;

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
    dst_particles[particle_index] = current_particles[particle_index];
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
    let (_, R) = deformation.polar();
    return R;
}