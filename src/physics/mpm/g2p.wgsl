struct Particle {
    deformation_gradient: mat3x3<f32>,
    B: mat3x3<f32>,
    position: vec3<f32>,
    velocity: vec3<f32>,
    rest_volume: f32,
    mass: f32,
}

struct GridParticle {
    momentum_x: i32,
    momentum_y: i32,
    momentum_z: i32,
    mass: i32,
    force_x: i32,
    force_y: i32,
    force_z: i32,
}

const DX: f32 = 0.02;
const BOTTOM_LEFT: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const N: u32 = 64u;
const D_INV = 4.0 / (DX * DX);
const DT = 0.0001;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> grid: array<GridParticle>;

fn quadratic_kernel(r: f32) -> f32 {
    let ar = abs(r);
    if ar < 0.5 {
        return 0.75 - ar * ar;
    }
    if ar < 1.5 {
        return 0.5 * (1.5 - ar) * (1.5 - ar);
    }
    return 0.0;
}

fn get_grid_index(coord: vec2<u32>) -> u32 {
    return coord.x * N + coord.y;
}

fn outer(v1: vec3<f32>, v2: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(v1 * v2.x, v1 * v2.y, v1 * v2.z);
}

@compute
@workgroup_size(256)
fn grid_to_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    if p_idx >= arrayLength(&particles) { return; }
    var p = particles[p_idx];

    let relative = (p.position.xy - BOTTOM_LEFT.xy) / DX;
    let base_coord = vec2<i32>(floor(relative - 0.5));

    var new_v = vec3<f32>(0.0);
    var new_B = mat3x3<f32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var new_pos = vec3<f32>(0.0);

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            let current_coord = base_coord + vec2<i32>(i, j);
            if current_coord.x < 0 || current_coord.x >= i32(N) ||
                current_coord.y < 0 || current_coord.y >= i32(N) {
                continue;
            }
            let node_pos = vec3<f32>(
                BOTTOM_LEFT.x + f32(current_coord.x) * DX,
                BOTTOM_LEFT.y + f32(current_coord.y) * DX,
                0.0,
            );
            let xi_xp = node_pos - vec3<f32>(p.position.xy, 0.0);
            let r = xi_xp.xy / DX;
            let weight = quadratic_kernel(r.x) * quadratic_kernel(r.y);
            let idx = get_grid_index(vec2<u32>(current_coord));

            if grid[idx].mass <= 0 { continue; }
            let vi = vec3<f32>(
                f32(grid[idx].momentum_x),
                f32(grid[idx].momentum_y),
                0.0,
            ) / f32(grid[idx].mass);
            new_v += vi * weight;
            new_B += weight * outer(vi, xi_xp);
            new_pos += (node_pos + vi * DT) * weight;
        }
    }

    p.velocity = vec3<f32>(new_v.xy, 0.0);
    p.B = new_B;
    // eq (17)
    let C = D_INV * p.B;
    var j = p.deformation_gradient[0][0] * p.deformation_gradient[1][1]
        - p.deformation_gradient[0][1] * p.deformation_gradient[1][0];
    j *= 1.0 + DT * (C[0][0] + C[1][1]);
    j = max(j, 0.1);
    let s = sqrt(j);
    p.deformation_gradient = mat3x3<f32>(
        s, 0.0, 0.0,
        0.0, s, 0.0,
        0.0, 0.0, 1.0,
    );
    p.position = vec3<f32>(
        clamp(new_pos.x, DX, f32(N - 2) * DX),
        clamp(new_pos.y, DX, f32(N - 2) * DX),
        0.0,
    );
    particles[p_idx] = p;
}
