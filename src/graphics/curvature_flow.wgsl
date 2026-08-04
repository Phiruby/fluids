
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

// Eq (6) in https://wstahw.win.tue.nl/edu/2IV06/andrei/particle_rendering/provided/p91-van_der_laan.pdf
fn divergence_normal(x: u32, y: u32) -> f32 {
    let z_center = depth_cache[x][y];
    let z_l = depth_cache[x - 1u][y];
    let z_r = depth_cache[x + 1u][y];
    let z_b = depth_cache[x][y - 1u];
    let z_t = depth_cache[x][y + 1u];

    let z_bl = depth_cache[x - 1u][y - 1u];
    let z_br = depth_cache[x + 1u][y - 1u];
    let z_tl = depth_cache[x - 1u][y + 1u];
    let z_tr = depth_cache[x + 1u][y + 1u];

    // zero out depth discontinuities (e.g if point b/w fluid and background)
    let thresh = ubo.depth_threshold;
    if abs(z_r - z_center) > thresh || abs(z_l - z_center) > thresh ||
        abs(z_t - z_center) > thresh || abs(z_b - z_center) > thresh {
        return 0.0;
    }
    let pzpx = (z_r - z_l) * 0.5;
    let pzpy = (z_t - z_b) * 0.5;

    let p2zpx2 = z_r - 2.0 * z_center + z_l;
    let p2zpy2 = z_t - 2.0 * z_center + z_b;

    let p2zpxpy = 0.25 * (z_tr - z_tl - z_br + z_bl);
    let p2zpypx = p2zpxpy;

    let cx = 2.0 / (ubo.viewport_width * ubo.focal_x);
    let cy = 2.0 / (ubo.viewport_height * ubo.focal_y);

    let cx2 = cx * cx;
    let cy2 = cy * cy;
    let cx2cy2 = cx2 * cy2;
    let pdpx = 2.0 * (cy2 * pzpx * p2zpx2 + cx2 * pzpy * p2zpxpy + cx2cy2 * z_center * pzpx);
    let pdpy = 2.0 * (cx2 * pzpy * p2zpy2 + cy2 * pzpx * p2zpypx + cx2cy2 * z_center * pzpy);
    let d = cy2 * pzpx * pzpx + cx2 * pzpy * pzpy + cx2cy2 * z_center * z_center;

    if d <= 1e-6 {
        return 0.0;
    }
    let ex = 0.5 * pzpx * pdpx - p2zpx2 * d;
    let ey = 0.5 * pzpy * pdpy - p2zpy2 * d;
    return (cy * ex + cx * ey) / pow(d, 1.5);
}

var<workgroup> depth_cache: array<array<f32, 18>, 18>;

@compute @workgroup_size(16, 16, 1)
fn curvature_flow(
    @builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // each workgroup gets 16x16 pixel group;
    let local_index = local_id.y * 16u + local_id.x;
    let tile_origin = vec2<i32>(workgroup_id.xy * 16u) - vec2<i32>(1, 1);

    let cache_x = local_index % 18u;
    let cache_y = local_index / 18u;
    let sample_pos = tile_origin + vec2<i32>(i32(cache_x), i32(cache_y));
    depth_cache[cache_x][cache_y] = textureLoad(input_depth, sample_pos, 0).r;

    // only the first ~68 threads will load padded depth cache (i.e boundary values)
    if local_index + 256u < 324u {
        let idx = local_index + 256u;
        let cache_x = idx % 18u;
        let cache_y = idx / 18u;
        let sample_pos = tile_origin + vec2<i32>(i32(cache_x), i32(cache_y));
        depth_cache[cache_x][cache_y] = textureLoad(input_depth, sample_pos, 0).r;
    }
    workgroupBarrier();

    let div = divergence_normal(local_id.x + 1, local_id.y + 1);
    let h = div / 2.0;
    var dz: f32;
    dz = h * ubo.dt;

    let pixel_x = tile_origin.x + i32(local_id.x);
    let pixel_y = tile_origin.y + i32(local_id.y);
    textureStore(
        output_depth,
        vec2<i32>(i32(pixel_x), i32(pixel_y)),
        vec4<f32>(
            depth_cache[local_id.x + 1][local_id.y + 1] + dz,
            0.0,
            0.0,
            1.0,
        )
    );
}

fn get_normal(x: u32, y: u32) -> vec4<f32> {
    let z_center = depth_cache[x][y];
    let z_l = depth_cache[x - 1u][y];
    let z_r = depth_cache[x + 1u][y];
    let z_b = depth_cache[x][y - 1u];
    let z_t = depth_cache[x][y + 1u];

    let z_bl = depth_cache[x - 1u][y - 1u];
    let z_br = depth_cache[x + 1u][y - 1u];
    let z_tl = depth_cache[x - 1u][y + 1u];
    let z_tr = depth_cache[x + 1u][y + 1u];

    // zero out depth discontinuities (e.g if point b/w fluid and background)
    let thresh = ubo.depth_threshold;
    if abs(z_r - z_center) > thresh || abs(z_l - z_center) > thresh ||
        abs(z_t - z_center) > thresh || abs(z_b - z_center) > thresh {
        return vec4<f32>(0.0);
    }
    let pzpx = (z_r - z_l) * 0.5;
    let pzpy = (z_t - z_b) * 0.5;

    let cx = 2.0 / (ubo.viewport_width * ubo.focal_x);
    let cy = 2.0 / (ubo.viewport_height * ubo.focal_y);

    let cx2 = cx * cx;
    let cy2 = cy * cy;
    let d = cy2 * pzpx * pzpx + cx2 * pzpy * pzpy + cx2 * cy2 * z_center * z_center;

    return vec4<f32>(
        -cy * pzpx,
        -cx * pzpy,
        cx * cy * depth_cache[x][y],
        1.0
    ) / sqrt(d);
}

@compute @workgroup_size(16, 16, 1)
fn compute_normals(
    @builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // each workgroup gets 16x16 pixel group;
    let local_index = local_id.y * 16u + local_id.x;
    let tile_origin = vec2<i32>(workgroup_id.xy * 16u) - vec2<i32>(1, 1);

    let cache_x = local_index % 18u;
    let cache_y = local_index / 18u;
    let sample_pos = tile_origin + vec2<i32>(i32(cache_x), i32(cache_y));
    depth_cache[cache_x][cache_y] = textureLoad(input_depth, sample_pos, 0).r;

    // only the first ~68 threads will load padded depth cache (i.e boundary values)
    if local_index + 256u < 324u {
        let idx = local_index + 256u;
        let cache_x = idx % 18u;
        let cache_y = idx / 18u;
        let sample_pos = tile_origin + vec2<i32>(i32(cache_x), i32(cache_y));
        depth_cache[cache_x][cache_y] = textureLoad(input_depth, sample_pos, 0).r;
    }
    workgroupBarrier();
    let normal = get_normal(local_id.x + 1, local_id.y + 1);
    let pixel_x = tile_origin.x + i32(local_id.x);
    let pixel_y = tile_origin.y + i32(local_id.y);
    textureStore(
        output_depth,
        vec2<i32>(i32(pixel_x), i32(pixel_y)),
        normal,
    );
}

