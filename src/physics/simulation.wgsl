struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>,
    density: f32,
    rest_density: f32,
    mass: f32,
    cell_hash: u32,
    // 1 if not boundary, 0 if boundary particle
    boundary: u32}

// NOTE: this should be synced with spatial_hash.wgsl
const CELL_SIZE: f32 = 0.1;
const HASHTABLE_SIZE: u32 = 8192u;
const PI = 3.14159265358979323846;
const SIGMA = 8.0 / (PI * pow(CELL_SIZE, 3.0));
const DT: f32 = 0.002;
const K: f32 = 50.0;
const ALPHA: f32 = 1.0; // for cohesision

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> hash_offsets: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output_particles: array<Particle>;
@group(0) @binding(3) var<storage, read_write> debug: array<vec3<f32>>;

fn cell_index(position: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(
        i32(floor(position.x / CELL_SIZE)),
        i32(floor(position.y / CELL_SIZE)),
        i32(floor(position.z / CELL_SIZE)),
    );
}

fn hash_position(cell_id: vec3<i32>) -> u32 {
    let x = u32(cell_id.x);
    let y = u32(cell_id.y);
    let z = u32(cell_id.z);
    let kappa = 9u * (x % 3u) + 3u * (y % 3u) + (z % 3u);
    let beta = ((17u * x) ^ (31u * y) ^ (61u * z)) % HASHTABLE_SIZE;
    return 27u * beta + kappa;
}

fn neighbor_hashes(idx: u32) -> array<u32, 27> {
    let cur = cell_index(particles[idx].position);
    var neighbors: array<u32, 27>;
    var i = 0u;

    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                neighbors[i] = hash_position(cur + vec3<i32>(dx, dy, dz));
                i++;
            }
        }
    }
    return neighbors;
}

fn cohesion_kernel(i: u32, j: u32) -> f32 {
    let r = length(particles[i].position - particles[j].position);
    let scale = 32 / (PI * pow(CELL_SIZE, 9.0));
    if 2 * r > CELL_SIZE && r <= CELL_SIZE {
        return scale * pow(CELL_SIZE - r, 3.0) * pow(r, 3.0);
    }
    if r > 0 && 2 * r <= CELL_SIZE {
        return scale * 2 * pow(CELL_SIZE - r, 3.0) * pow(r, 3.0) - pow(CELL_SIZE, 6.0) / 64;
    }
    return 0.0;
}

// gradient w.r.t first input for cubic spline kernel (L2)
fn kernel_gradient(i: u32, j: u32) -> vec3<f32> {
    let r = particles[i].position - particles[j].position;
    let r_norm = length(r);
    if r_norm == 0.0 {
        return vec3<f32>(0.0);
    }
    let q = r_norm / CELL_SIZE;
    let dq_dx = r / (CELL_SIZE * r_norm);
    if 0.0 <= q && q <= 0.5 {
        return SIGMA * (18.0 * q * q - 12.0 * q) * dq_dx;
    } else if 0.5 < q && q <= 1.0 {
        return SIGMA * (-6.0 * (1.0 - q) * (1.0 - q)) * dq_dx;
    }
    return vec3<f32>(0.0);
}

fn surface_tension_force(idx: u32) -> vec3<f32> {
    let neighbors = neighbor_hashes(idx);
    var surface_tension_force: vec3<f32> = vec3<f32>(0.0);
    let m_i = particles[idx].mass;
    let rho0 = particles[idx].rest_density;
    for (var i: u32 = 0u; i < 27u; i++) {
        let offset = hash_offsets[neighbors[i]];
        let start = offset.x;
        let end = offset.y;
        for (var j: u32 = start; j < end; j++) {
            if distance(particles[j].position, particles[idx].position) <= CELL_SIZE {
                let is_fluid = f32(particles[j].boundary);
                var dir: vec3<f32> = particles[idx].position - particles[j].position;
                dir = normalize(dir);
                let cohesion_force = -ALPHA * m_i * particles[j].mass * cohesion_kernel(idx, j);
                let curvature_force = -ALPHA * m_i * (particles[idx].normal - particles[j].normal);
                let Kij = 2 * rho0 / (particles[idx].density + particles[j].density);
                surface_tension_force += Kij * (cohesion_force + curvature_force);
            }
        }
    }
    return surface_tension_force;
}

// Eqn (102) from https://arxiv.org/pdf/2009.06944
fn viscous_force(idx: u32) -> vec3<f32> {
    let neighbors = neighbor_hashes(idx);
    var sum: vec3<f32> = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 27u; i++) {
        let offset = hash_offsets[neighbors[i]];
        let start = offset.x;
        let end = offset.y;
        for (var j: u32 = start; j < end; j++) {
            if distance(particles[j].position, particles[idx].position) <= CELL_SIZE {
                let is_fluid = f32(particles[j].boundary);
                let v_j = particles[j].mass / particles[j].density;
                let x_ij = particles[idx].position - particles[j].position;
                let v_ij = particles[idx].velocity - particles[j].velocity;
                let grad = kernel_gradient(idx, j);
                sum += is_fluid * v_j * dot(v_ij, x_ij) / (dot(x_ij, x_ij) + 0.01 * pow(CELL_SIZE, 2.0)) * grad;
            }
        }
    }
    return 10.0 * sum;
}

// Eq (53) and the simple pressure formula in section 4.4. Output is nabla pressure / density (pressure acceleration)
fn pressure_gradient(idx: u32) -> vec3<f32> {
    let p_i = K * max(0.0, (particles[idx].density - particles[idx].rest_density));
    let neighbors = neighbor_hashes(idx);
    var sum: vec3<f32> = vec3<f32>(0.0);
    let ratio = p_i / pow(particles[idx].density, 2.0);

    for (var i: u32 = 0u; i < 27u; i++) {
        let offset = hash_offsets[neighbors[i]];
        let start = offset.x;
        let end = offset.y;
        for (var j: u32 = start; j < end; j++) {
            if distance(particles[j].position, particles[idx].position) <= CELL_SIZE {
                // pressure mirroring: fluid uses EOS; boundary mirrors p_i
                let is_fluid = f32(particles[j].boundary);
                let p_j = is_fluid * K * max(0.0, (particles[j].density - particles[j].rest_density))
                  + (1.0 - is_fluid) * p_i;
                let ratio_j = p_j / pow(particles[j].density, 2.0);
                var delta = particles[j].mass * (ratio + ratio_j) * kernel_gradient(idx, j);
                sum -= delta;
            }
        }
    }
    return sum;
}

fn height_color(y: f32) -> vec3<f32> {
    let t = clamp((y - 0.05) / 1.1, 0.0, 1.0);
    if t < 0.33 {
        let u = t / 0.33;
        return mix(vec3<f32>(0.05, 0.25, 0.95), vec3<f32>(0.1, 0.85, 0.85), u);
    } else if t < 0.66 {
        let u = (t - 0.33) / 0.33;
        return mix(vec3<f32>(0.1, 0.85, 0.85), vec3<f32>(0.95, 0.9, 0.15), u);
    }
    let u = (t - 0.66) / 0.34;
    return mix(vec3<f32>(0.95, 0.9, 0.15), vec3<f32>(0.95, 0.2, 0.1), u);
}

@compute
@workgroup_size(256)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    var particle: Particle = particles[gid.x];
    if particle.boundary == 0u {
        output_particles[gid.x] = particle;
        return;
    }

    let density = particle.density;
    // I am unsure why I am having to multiply by 200.
    // f_visc = mu * (nabla^2 v). But mu=nu * density. So scaling by nu makes things smoother (again, not sure why
    // given that the viscocity force needs to be multiplied by mu, not nu)
    let viscosity = 200.0 * viscous_force(gid.x) / density;
    let pressure_acceleration = pressure_gradient(gid.x);
    let surface_tension_force = surface_tension_force(gid.x) / particles[gid.x].mass;

    let dv = (viscosity + pressure_acceleration + vec3<f32>(0.0, -9.81, 0.0)) * DT;
    particle.velocity += dv;
    particle.position += particle.velocity * DT;
    particle.normal = height_color(particle.position.y);
    output_particles[gid.x] = particle;
}

