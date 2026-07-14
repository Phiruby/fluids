
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

const NU: f32 = 0.3; // viscosity
// NOTE: this should be synced with spatial_hash.wgsl
const CELL_SIZE: f32 = 0.1;
const HASHTABLE_SIZE: u32 = 8192u;
const SIGMA = 8.0 / (3.14159 * pow(CELL_SIZE, 3.0));

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> hash_offsets: array<vec2<u32>>;

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

// cubic spline kernel
fn kernel_weight(i: u32, j: u32) -> f32 {
  // l1 norm of dist / cell size
  let q = dot(abs(particles[j].position - particles[i].position), vec3<f32>(1.0)) / CELL_SIZE;
  if (0 <= q && q <= 0.5) {
      return SIGMA * (6.0 * pow(q, 3.0) - 6.0 * pow(q, 2.0) + 1.0);
  } else if (0.5 < q && q <= 1.0) {
    return SIGMA * 2.0 * pow(1 - q, 3.0);
  }
  return 0.0;
}

fn compute_density(idx: u32) -> f32 {
   let neighbors = neighbor_hashes(idx);
   var density: f32 = 0.0;
   for (var i: u32 = 0u; i < 27u; i++) {
    let offset = hash_offsets[neighbors[i]];
    let start = offset.x;
    let end = offset.y;
    for (var j: u32 = start; j < end; j++) {
      if (distance(particles[j].position, particles[idx].position) <= CELL_SIZE) {
        let kw = kernel_weight(idx, j);
        density += particles[j].mass * kw;
      }
    }
  }
  return density;
}

// computes densities and nabla W_{ij}
@compute
@workgroup_size(256)
fn compute_quantities(@builtin(global_invocation_id) gid: vec3<u32>) {
  let density = compute_density(gid.x);
  // NOTE: ok to write back to same input buffer since computed quantities here do not make use of density;
  // so no race conditions
  particles[gid.x].density = density;
}
