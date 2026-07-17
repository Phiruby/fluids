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

struct SortUniform {
    iteration: u32, // 0..7 (4 bits each * 8 iterations = 32 bits)
    num_particles: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32>;
@group(0) @binding(2) var<uniform> params: SortUniform;
@group(0) @binding(3) var<storage, read_write> sorted_particles: array<Particle>;
// idx i stores [start, end) for hash value i
@group(0) @binding(4) var<storage, read_write> hash_offsets: array<vec2<u32>, 27 * HASHTABLE_SIZE>;

const RADIX_BUCKETS: u32 = 16u;
const WG_SIZE: u32 = 256u;
const CELL_SIZE: f32 = 0.1;
const HASHTABLE_SIZE: u32 = 8192u;

// Use floor→i32 so negative world coords (this scene is largely in -x/-z)
// don't all collapse into cell 0 via u32(negative) clamping.
fn cell_index(position: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(
        i32(floor(position.x / CELL_SIZE)),
        i32(floor(position.y / CELL_SIZE)),
        i32(floor(position.z / CELL_SIZE)),
    );
}

fn hash_cell(cell_id: vec3<i32>) -> u32 {
    // u32(i32) wraps negatives; % keeps them in-range for the table.
    let x = u32(cell_id.x);
    let y = u32(cell_id.y);
    let z = u32(cell_id.z);
    let kappa = 9u * (x % 3u) + 3u * (y % 3u) + (z % 3u);
    let beta = ((17u * x) ^ (31u * y) ^ (61u * z)) % HASHTABLE_SIZE;
    return 27u * beta + kappa;
}

fn hash_position(position: vec3<f32>) -> u32 {
    return hash_cell(cell_index(position));
}

fn digit(key: u32) -> u32 {
    return (key >> (params.iteration * 4u)) & 0xFu;
}

@compute
@workgroup_size(256)
fn build_hash_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.num_particles { return; }
    particles[gid.x].cell_hash = hash_position(particles[gid.x].position);
}

var<workgroup> local_counts: array<atomic<u32>, 16>;

@compute
@workgroup_size(256)
fn histogram_shader(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if lid.x < RADIX_BUCKETS {
        atomicStore(&local_counts[lid.x], 0u);
    }
    workgroupBarrier();

    if gid.x < params.num_particles {
        atomicAdd(&local_counts[digit(particles[gid.x].cell_hash)], 1u);
    }
    workgroupBarrier();

    if lid.x < RADIX_BUCKETS {
        histogram[wid.x * RADIX_BUCKETS + lid.x] = atomicLoad(&local_counts[lid.x]);
    }
}

// scan over (bucket, workgroup) so all digit-d keys precede digit-(d+1).
// see fig 3a or fig 4 of https://arxiv.org/pdf/2206.01784
@compute
@workgroup_size(1)
fn prefix_sum_shader() {
    var offset = 0u;
    let num_wgs = (params.num_particles + WG_SIZE - 1u) / WG_SIZE;
    for (var b = 0u; b < RADIX_BUCKETS; b++) {
        for (var w = 0u; w < num_wgs; w++) {
            let idx = w * RADIX_BUCKETS + b;
            let n = histogram[idx];
            histogram[idx] = offset;
            offset += n;
        }
    }
}

var<workgroup> bucket_base: array<u32, 16>;
var<workgroup> local_digits: array<u32, 256>; // store digit associated with each worker's local thread

@compute
@workgroup_size(256)
fn scatter_shader(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if lid.x < RADIX_BUCKETS {
        bucket_base[lid.x] = histogram[wid.x * RADIX_BUCKETS + lid.x];
    }

    let valid = gid.x < params.num_particles;
    var d = 0u;
    if valid {
        d = digit(particles[gid.x].cell_hash);
    }
    local_digits[lid.x] = d;
    workgroupBarrier();

    if valid {
        var rank = 0u;
        // this is here to ensure stable sort
        for (var i = 0u; i < lid.x; i++) {
            rank += u32(local_digits[i] == d);
        }
        sorted_particles[bucket_base[d] + rank] = particles[gid.x];
    }
}

@compute
@workgroup_size(256)
fn clear_hash_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    hash_offsets[gid.x] = vec2<u32>(0u, 0u);
}

@compute
@workgroup_size(256)
fn build_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.num_particles { return; }
    let hash = sorted_particles[i].cell_hash;

    // beggining of hash
    if i == 0u || sorted_particles[i - 1u].cell_hash != hash {
        hash_offsets[hash].x = i;
    }

    // end of hash section
    if i == params.num_particles - 1 || sorted_particles[i + 1u].cell_hash != hash {
        hash_offsets[hash].y = i + 1u;
    }
}

