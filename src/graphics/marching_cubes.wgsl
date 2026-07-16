// NOTE: particles and neighbors should be buffers from @simulation.wgsl

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>,
    density: f32,
    rest_density: f32,
    mass: f32,
    cell_hash: u32,
    boundary: u32}

// marching cubes will be performed over cube defined by this top left and bottom right
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(-1.5, -2.5, 0.0);
const N: u32 = 100;
const ISO_LEVEL: f32 = 1000.0;

const MARCHING_CUBES_CELL_SIZE: f32 = 0.1;

// the cell size is the size used in spatial hashing, separate from marching cubes
const CELL_SIZE: f32 = 0.1;
const HASHTABLE_SIZE: u32 = 8192u;
const SIGMA = 8.0 / (3.14159 * pow(CELL_SIZE, 3.0));
const DT: f32 = 0.002;
const K: f32 = 0.5;

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> hash_offsets: array<vec2<u32>>;

//not to be confused, but this is for the spatial hash (in @simulation.wgsl)
// whereas the function below is for the marching cubes bit
fn cell_index(position: vec3<f32>) -> vec3<u32> {
    return vec3<u32>(
        u32(position.x / CELL_SIZE),
        u32(position.y / CELL_SIZE),
        u32(position.z / CELL_SIZE),
    );
}

fn marching_cubes_cell_index(position: vec3<f32>) -> vec3<u32> {
    return vec3<u32>(
        u32((position.x - BOTTOM_LEFT.x) / MARCHING_CUBES_CELL_SIZE),
        u32((position.y - BOTTOM_LEFT.y) / MARCHING_CUBES_CELL_SIZE),
        u32((position.z - BOTTOM_LEFT.z) / MARCHING_CUBES_CELL_SIZE),
    );
}

fn neighbor_hashes(pos: vec3<f32>) -> array<u32, 27> {
    let cur = cell_index(pos);
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

fn hash_position(cell_id: vec3<u32>) -> u32 {
    let kappa = 9u * (cell_id.x % 3u) + 3u * (cell_id.y % 3u) + (cell_id.z % 3u);
    let beta = ((17u * cell_id.x) ^ (31u * cell_id.y) ^ (61u * cell_id.z)) % HASHTABLE_SIZE;
    return 27u * beta + kappa;
}

// cubic spline kernel; taken from quantities.wgsl
fn kernel_weight(pos_i: vec3<f32>, pos_j: vec3<f32>) -> f32 {
    // l1 norm of dist / cell size
    let q = dot(abs(pos_j - pos_i), vec3<f32>(1.0)) / CELL_SIZE;
    if 0 <= q && q <= 0.5 {
        return SIGMA * (6.0 * pow(q, 3.0) - 6.0 * pow(q, 2.0) + 1.0);
    } else if 0.5 < q && q <= 1.0 {
        return SIGMA * 2.0 * pow(1 - q, 3.0);
    }
    return 0.0;
}

fn evaluate_density(pos: vec3<f32>) -> f32 {
    let neighbors = neighbor_hashes(pos);
    let density = 0.0;
    for (var i: u32 = 0u; i < 27u; i++) {
        let offset = hash_offsets[neighbors[i]];
        let start = offset.x;
        let end = offset.y;
        for (var j: u32 = start; j < end; j++) {
            if distance(particles[j].position, pos) <= CELL_SIZE {
                let density = kernel_weight(particles[j].position, pos) * particles[j].mass;
            }
        }
    }
    return density;
}

@compute
@workgroup_size(256)
fn active_edges(@builtin(global_invocation_id) gid: vec3<u32>) {

    let i = gid.x;
    let grid_x = i % N;
    let grid_y = u32((i % (N * N)) / N);
    let grid_z = u32(i / (N * N));
    let start = vec3<f32>(
        f32(grid_x) * MARCHING_CUBES_CELL_SIZE,
        f32(grid_y) * MARCHING_CUBES_CELL_SIZE,
        f32(grid_z) * MARCHING_CUBES_CELL_SIZE
    );
    // TODO: out of grid edges need to be handled?
    let edge_x_end = start + vec3<f32>(MARCHING_CUBES_CELL_SIZE, 0, 0);
    let edge_y_end = start + vec3<f32>(0, MARCHING_CUBES_CELL_SIZE, 0);
    let edge_z_end = start + vec3<f32>(0, 0, MARCHING_CUBES_CELL_SIZE);

    let density_current = evaluate_density(start);
    let density_x_end = evaluate_density(edge_x_end);
    let density_y_end = evaluate_density(edge_y_end);
    let density_z_end = evaluate_density(edge_z_end);

    var nvertices: u32 = 0;
    // check if density equals desired iso level in between the points
    if (density_current - ISO_LEVEL) * (density_x_end - ISO_LEVEL) < 0.0 { nvertices++; }
    if (density_current - ISO_LEVEL) * (density_y_end - ISO_LEVEL) < 0.0 { nvertices++; }
    if (density_current - ISO_LEVEL) * (density_z_end - ISO_LEVEL) < 0.0 { nvertices++; }
}
