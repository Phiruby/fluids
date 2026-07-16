@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix_sums: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;

const n: u32 = 512; // elements per workgroup
const nlevels: u32 = 9; // log2(n)
var<workgroup> temp: array<u32, 532>;

// input: input array
// prefix_sums: contains the outputted prefixed sums (without workergroup offsets)
// sums: the cummulative sum for each workergroup
@compute
@workgroup_size(256)
fn prefix_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let thread_id = lid.x;
    let global_id = gid.x;

    if thread_id < (n >> 1) {
        temp[2u * thread_id] = input[2u * global_id];
        temp[2u * thread_id + 1u] = input[2u * global_id + 1u];
    }
    workgroupBarrier();

    var offset = 1u;
    for (var level = 1u; level <= nlevels; level++) {
        let level_idx = 1u << (nlevels - level);
        if thread_id < level_idx {
            let ai = offset * (2u * thread_id + 1u) - 1u;
            let bi = offset * (2u * thread_id + 2u) - 1u;
            temp[bi] += temp[ai];
        }
        workgroupBarrier();
        offset *= 2u;
    }

    if thread_id == 0u {
        sums[wid.x] = temp[n - 1u];
        temp[n - 1u] = 0u;
    }
    workgroupBarrier();

    for (var level = 1u; level <= nlevels; level++) {
        offset >>= 1u;
        if thread_id < (1u << (level - 1u)) {
            let ai = offset * (2u * thread_id + 1u) - 1u;
            let bi = offset * (2u * thread_id + 2u) - 1u;
            let temporary = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += temporary;
        }
        workgroupBarrier();
    }

    if thread_id < (n >> 1) {
        prefix_sums[2u * global_id] = temp[2u * thread_id];
        prefix_sums[2u * global_id + 1u] = temp[2u * thread_id + 1u];
    }
}

// input: input array
// prefix_sums: contains the prefix sums before the offsets are added
// sums: sums[i] contains the required offset for workgroup i
@compute
@workgroup_size(256)
fn aggregate_offsets(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if lid.x < (n >> 1) {
        prefix_sums[2u * gid.x] += sums[wid.x];
        prefix_sums[2u * gid.x + 1u] += sums[wid.x];
    }
}
