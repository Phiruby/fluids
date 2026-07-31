
struct Uniform {
    texture_width: f32,
    texture_height: f32,
}
@group(0) @binding(0) var input_depth: texture_2d<f32>;
@group(0) @binding(1) var output_depth: texture_storage_2d<f32>;
@group(0) @binding(2) var<uniform> ubo: Uniform;

var<workgroup> depth_cache: array<array<f32, 18>, 18>;
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_id: vec3<u32>,
    @builtin(local_invocation_id) local_index: vec3<u32>,
) {
}
