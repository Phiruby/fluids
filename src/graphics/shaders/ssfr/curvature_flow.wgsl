
struct Uniform {
    viewport_width: f32,
    viewport_height: f32,
    focal_x: f32,
    focal_y: f32,
    dt: f32,
    depth_threshold: f32,
}
@group(0) @binding(0) var input_depth: texture_2d<f32>;
@group(0) @binding(1) var output_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> ubo: Uniform;
@group(0) @binding(3) var normals: texture_storage_2d<rgba32float, write>;

const INVALID_DEPTH: f32 = 0.0;

fn is_valid_depth(z: f32) -> bool {
    // in fluid_depth.wgsl, depth is set 0 for non-fluid voxels. so we ignore the background stuff here
    return z > 1e-6;
}

fn finite_diff_x(z_center: f32, z_l: f32, z_r: f32) -> f32 {
    let valid_l = is_valid_depth(z_l) && abs(z_l - z_center) <= ubo.depth_threshold;
    let valid_r = is_valid_depth(z_r) && abs(z_r - z_center) <= ubo.depth_threshold;
    if valid_l && valid_r {
        return (z_r - z_l) * 0.5;
    } else if valid_r {
        return z_r - z_center;
    } else if valid_l {
        return z_center - z_l;
    }
    return 0.0;
}

fn finite_diff_y(z_center: f32, z_b: f32, z_t: f32) -> f32 {
    let valid_b = is_valid_depth(z_b) && abs(z_b - z_center) <= ubo.depth_threshold;
    let valid_t = is_valid_depth(z_t) && abs(z_t - z_center) <= ubo.depth_threshold;
    if valid_b && valid_t {
        return (z_t - z_b) * 0.5;
    } else if valid_t {
        return z_t - z_center;
    } else if valid_b {
        return z_center - z_b;
    }
    return 0.0;
}

// Eq (6) in https://wstahw.win.tue.nl/edu/2IV06/andrei/particle_rendering/provided/p91-van_der_laan.pdf
fn divergence_normal(x: u32, y: u32) -> f32 {
    let z_center = depth_cache[x][y];
    if !is_valid_depth(z_center) {
        return 0.0;
    }

    let z_l = depth_cache[x - 1u][y];
    let z_r = depth_cache[x + 1u][y];
    let z_b = depth_cache[x][y - 1u];
    let z_t = depth_cache[x][y + 1u];

    let z_bl = depth_cache[x - 1u][y - 1u];
    let z_br = depth_cache[x + 1u][y - 1u];
    let z_tl = depth_cache[x - 1u][y + 1u];
    let z_tr = depth_cache[x + 1u][y + 1u];

    // zero out depth discontinuities (e.g if point b/w fluid and background);
    // also zero out if it is a boundary voxel
    let thresh = ubo.depth_threshold;
    if !is_valid_depth(z_l) || !is_valid_depth(z_r) || !is_valid_depth(z_t) || !is_valid_depth(z_b)
        || abs(z_r - z_center) > thresh || abs(z_l - z_center) > thresh
        || abs(z_t - z_center) > thresh || abs(z_b - z_center) > thresh {
        return 0.0;
    }

    let pzpx = (z_r - z_l) * 0.5;
    let pzpy = (z_t - z_b) * 0.5;

    let p2zpx2 = z_r - 2.0 * z_center + z_l;
    let p2zpy2 = z_t - 2.0 * z_center + z_b;

    let p2zpxpy = 0.25 * (z_tr - z_tl - z_br + z_bl);

    let cx = 2.0 / (ubo.viewport_width * ubo.focal_x);
    let cy = 2.0 / (ubo.viewport_height * ubo.focal_y);

    let cx2 = cx * cx;
    let cy2 = cy * cy;
    let cx2cy2 = cx2 * cy2;
    let pdpx = 2.0 * (cy2 * pzpx * p2zpx2 + cx2 * pzpy * p2zpxpy + cx2cy2 * z_center * pzpx);
    let pdpy = 2.0 * (cx2 * pzpy * p2zpy2 + cy2 * pzpx * p2zpxpy + cx2cy2 * z_center * pzpy);
    let d = cy2 * pzpx * pzpx + cx2 * pzpy * pzpy + cx2cy2 * z_center * z_center;

    if d <= 1e-6 {
        return 0.0;
    }
    let ex = 0.5 * pzpx * pdpx - p2zpx2 * d;
    let ey = 0.5 * pzpy * pdpy - p2zpy2 * d;
    return (cy * ex + cx * ey) / pow(d, 1.5);
}

var<workgroup> depth_cache: array<array<f32, 18>, 18>;

fn load_depth_tile(
    workgroup_id: vec3<u32>,
    local_id: vec3<u32>,
) {
    let local_index = local_id.y * 16u + local_id.x;
    let tile_origin = vec2<i32>(workgroup_id.xy * 16u) - vec2<i32>(1, 1);
    let dims = vec2<i32>(textureDimensions(input_depth));

    let cache_x = local_index % 18u;
    let cache_y = local_index / 18u;
    let sample_pos = clamp(tile_origin + vec2<i32>(i32(cache_x), i32(cache_y)), vec2<i32>(0), dims - 1);
    depth_cache[cache_x][cache_y] = textureLoad(input_depth, sample_pos, 0).r;

    if local_index + 256u < 324u {
        let idx = local_index + 256u;
        let cache_x2 = idx % 18u;
        let cache_y2 = idx / 18u;
        let sample_pos2 = clamp(tile_origin + vec2<i32>(i32(cache_x2), i32(cache_y2)), vec2<i32>(0), dims - 1);
        depth_cache[cache_x2][cache_y2] = textureLoad(input_depth, sample_pos2, 0).r;
    }
}

@compute @workgroup_size(16, 16, 1)
fn curvature_flow(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    load_depth_tile(workgroup_id, local_id);
    workgroupBarrier();

    let dims = textureDimensions(input_depth);
    let pixel_x = i32(workgroup_id.x * 16u + local_id.x);
    let pixel_y = i32(workgroup_id.y * 16u + local_id.y);
    if pixel_x >= i32(dims.x) || pixel_y >= i32(dims.y) {
        return;
    }

    let z0 = depth_cache[local_id.x + 1u][local_id.y + 1u];
    if !is_valid_depth(z0) {
        textureStore(output_depth, vec2<i32>(pixel_x, pixel_y), vec4<f32>(INVALID_DEPTH, 0.0, 0.0, 1.0));
        return;
    }
    let h = divergence_normal(local_id.x + 1u, local_id.y + 1u);
    let z_new = max(z0 + h * ubo.dt, 0.0);
    textureStore(
        output_depth,
        vec2<i32>(pixel_x, pixel_y),
        vec4<f32>(z_new, 0.0, 0.0, 1.0),
    );
}

// Eq (4) in file:///home/kyber/Downloads/Screen_space_fluid_rendering_with_curvature_flow.pdf;
// important: the normal is in view space
fn get_normal(x: u32, y: u32) -> vec4<f32> {
    let z_center = depth_cache[x][y];
    if !is_valid_depth(z_center) {
        return vec4<f32>(0.0);
    }
    let z_l = depth_cache[x - 1u][y];
    let z_r = depth_cache[x + 1u][y];
    let z_b = depth_cache[x][y - 1u];
    let z_t = depth_cache[x][y + 1u];

    let pzpx = finite_diff_x(z_center, z_l, z_r);
    let pzpy = finite_diff_y(z_center, z_b, z_t);

    let cx = 2.0 / (ubo.viewport_width * ubo.focal_x);
    let cy = 2.0 / (ubo.viewport_height * ubo.focal_y);

    let d = cy * cy * pzpx * pzpx + cx * cx * pzpy * pzpy + cx * cx * cy * cy * z_center * z_center;
    if d <= 1e-12 {
        return vec4<f32>(0.0, 0.0, -1.0, 1.0);
    }
    let inv_len = inverseSqrt(d);
    let n = vec3<f32>(
        -cy * pzpx,
        -cx * pzpy,
        cx * cy * z_center,
    ) * inv_len;
    return vec4<f32>(n, 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn compute_normals(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    load_depth_tile(workgroup_id, local_id);
    workgroupBarrier();

    let dims = textureDimensions(input_depth);
    let pixel_x = i32(workgroup_id.x * 16u + local_id.x);
    let pixel_y = i32(workgroup_id.y * 16u + local_id.y);
    if pixel_x >= i32(dims.x) || pixel_y >= i32(dims.y) {
        return;
    }

    let normal = get_normal(local_id.x + 1u, local_id.y + 1u);
    textureStore(normals, vec2<i32>(pixel_x, pixel_y), normal);
}
