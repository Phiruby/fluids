
struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>,
    density: f32,
    rest_density: f32,
    mass: f32,
    cell_hash: u32,
}

// NOTE: this should be synced with spatial_hash.wgsl
const CELL_SIZE: f32 = 0.1;
const HASHTABLE_SIZE: u32 = 8192u;
const SIGMA = 8.0 / (3.14159 * pow(CELL_SIZE, 3.0));
const DT: f32 = 0.002;
const K: f32 = 0.5;

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> hash_offsets: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output_particles: array<Particle>;

fn cell_index(position: vec3<f32>) -> vec3<u32> {
    return vec3<u32>(
        u32(position.x / CELL_SIZE),
        u32(position.y / CELL_SIZE),
        u32(position.z / CELL_SIZE),
    );
}

fn hash_position(cell_id: vec3<u32>) -> u32 {
    let kappa = 9u * (cell_id.x % 3u) + 3u * (cell_id.y % 3u) + (cell_id.z % 3u);
    let beta = ((17u * cell_id.x) ^ (31u * cell_id.y) ^ (61u * cell_id.z)) % HASHTABLE_SIZE;
    return 27u * beta + kappa;
}

fn neighbor_hashes(idx: u32) -> array<u32, 27> {
    let cur = cell_index(particles[idx].position);
    var neighbors: array<u32, 27>;
    var i = 0u;

    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                neighbors[i] = hash_position(vec3<u32>(
                    u32(i32(cur.x) + dx),
                    u32(i32(cur.y) + dy),
                    u32(i32(cur.z) + dz),
                ));
                i++;
            }
        }
    }
    return neighbors;
}

// gradient w.r.t first input for cubic spline kernel
fn kernel_gradient(i: u32, j: u32) -> vec3<f32> {
  let r = particles[i].position - particles[j].position;
  let r_norm = dot(abs(r), vec3<f32>(1.0));
  let q = r_norm / CELL_SIZE;
  let partial_q_partial_x = vec3<f32>(
    r.x / (pow(CELL_SIZE, 2.0) * q),
    r.y / (pow(CELL_SIZE, 2.0) * q),
    r.z / (pow(CELL_SIZE, 2.0) * q),
  );

  if (0 <= q && q <= 0.5) {
    return SIGMA * (18.0 * pow(q, 2.0) * partial_q_partial_x - 12.0 * q * partial_q_partial_x);
  } else if (0.5 < q && q <= 1.0) {
    return SIGMA * -6.0 * pow(1.0 - q, 2.0) * partial_q_partial_x;
  }
  return vec3<f32>(0.0);
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
      if (distance(particles[j].position, particles[idx].position) <= CELL_SIZE) {
        let v_j = particles[j].mass / particles[j].density;
        let x_ij = particles[idx].position - particles[j].position;
        let v_ij = particles[idx].velocity - particles[j].velocity;
        // TODO: need gradient of kernel 
        let grad = kernel_gradient(idx, j);
        sum += v_j * dot(v_ij, x_ij) / (dot(x_ij, x_ij) + 0.01 * pow(CELL_SIZE, 2.0)) * grad;
      }
    }
  }
  return sum;
}

// TODO: recomputation of density; can simultaneously get pressure gradient and viscocities in one loop
// Eq (53) and the simple pressure formula in section 4.4. Output is nabla pressure / density (pressure acceleration)
fn pressure_gradient(idx: u32) -> vec3<f32> {
  let p_i = K * (particles[idx].density - particles[idx].rest_density);
  let neighbors = neighbor_hashes(idx);
  var sum: vec3<f32> = vec3<f32>(0.0);

  let ratio = p_i / pow(particles[idx].density, 2.0);
  for (var i: u32 = 0u; i < 27u; i++) {
    let offset = hash_offsets[neighbors[i]];
    let start = offset.x;
    let end = offset.y;
    for (var j: u32 = start; j < end; j++) {
      if (distance(particles[j].position, particles[idx].position) <= CELL_SIZE) {
        let p_j = K * (particles[j].density - particles[j].rest_density);
        let ratio_j = p_j / pow(particles[j].density, 2.0);
        sum -= particles[j].mass * (ratio + ratio_j) * kernel_gradient(idx, j);

      }
    }
  }
  return sum;
}

@compute
@workgroup_size(256)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let density = particles[gid.x].density;
  let viscosity = viscous_force(gid.x) / density;
  let pressure_acceleration = pressure_gradient(gid.x);
  let dv = (viscosity + pressure_acceleration + vec3<f32>(0.0, 0.0, -9.81)) * DT;
  var particle: Particle = particles[gid.x];
  particle.velocity += dv;
  particle.position += particle.velocity * DT;
  output_particles[gid.x] = particle;
}
