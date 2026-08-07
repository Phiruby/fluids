
struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>,
    density: f32,
    rest_density: f32,
    mass: f32,
    cell_hash: u32,
    boundary: u32,
}

// NOTE: this should be synced with spatial_hash.wgsl
const CELL_SIZE: f32 = 0.1;
const HASHTABLE_SIZE: u32 = 8192u;
const SIGMA = 8.0 / (3.14159 * pow(CELL_SIZE, 3.0));

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> hash_offsets: array<vec2<u32>>;

@group(0) @binding(2) var<storage, read_write> debug: array<vec3<f32>>;
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

// cubic spline kernel (L2 support radius = CELL_SIZE)
fn kernel_weight(i: u32, j: u32) -> f32 {
  let q = distance(particles[j].position, particles[i].position) / CELL_SIZE;
  if (0 <= q && q <= 0.5) {
      return SIGMA * (6.0 * pow(q, 3.0) - 6.0 * pow(q, 2.0) + 1.0);
  } else if (0.5 < q && q <= 1.0) {
    return SIGMA * 2.0 * pow(1 - q, 3.0);
  }
  return 0.0;
}

fn is_finite(x: f32) -> bool {
  // NaN: x != x; Inf: x*0 != 0
  return (x == x) && (x * 0.0 == 0.0);
}

// computes delta density;
// for boundary particles, the output will always be 0 (we will maintain same density for boundaries)
fn compute_density(idx: u32) -> f32 {
   let neighbors = neighbor_hashes(idx);
   var density: f32 = 0.0;
   let is_fluid = f32(particles[idx].boundary);
   for (var i: u32 = 0u; i < 27u; i++) {
    let offset = hash_offsets[neighbors[i]];
    let start = offset.x;
    let end = offset.y;
    for (var j: u32 = start; j < end; j++) {
      if (distance(particles[j].position, particles[idx].position) <= CELL_SIZE) {
        let kw = kernel_weight(idx, j);
        density += is_fluid * particles[j].mass * kw;

        if (!is_finite(particles[j].mass) || !is_finite(f32(is_fluid)) || !is_finite(kw)) {
          debug[idx] = vec3<f32>(f32(particles[j].density), particles[j].rest_density, f32(particles[j].boundary));
        }
      }
    }
  }
  return density;
}

// computes densities and nabla W_{ij}
@compute
@workgroup_size(256)
fn compute_quantities(@builtin(global_invocation_id) gid: vec3<u32>) {
  // NOTE: ok to write back to same input buffer since computed quantities here do not make use of density;
  // so no race conditions
  let rho = compute_density(gid.x);
  if (particles[gid.x].boundary == 0u) {
    particles[gid.x].density = particles[gid.x].rest_density;
  } else {
    particles[gid.x].density = rho;
  }
}
