struct Particle {
    position: vec3<f32>,
    mass: f32,
    velocity: vec3<f32>,
    rest_volume: f32,
    reference_position: vec3<f32>,
    nneighbors: u32,
    color: vec3<f32>,
    renderable: u32,
    neighbors: array<u32, 27>,
    pinned: u32,
    L: mat3x3<f32>,
}

struct KernelParams {
    h: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> kernel: KernelParams;

const PI = 3.14159265358979323846;

fn kernel_gradient(r: vec3<f32>) -> vec3<f32> {
    let r_norm = length(r);
    if r_norm < 1e-12 {
        return vec3<f32>(0.0);
    }
    let q = r_norm / kernel.h;
    let sigma = 8.0 / (PI * pow(kernel.h, 3.0));
    let dq = r / (kernel.h * r_norm);
    if q <= 0.5 {
        return sigma * (18.0 * q * q - 12.0 * q) * dq;
    } else if q <= 1.0 {
        return sigma * (-6.0 * (1.0 - q) * (1.0 - q)) * dq;
    }
    return vec3<f32>(0.0);
}

fn identity3() -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
}

fn invert3(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = determinant(m);
    if abs(det) < 1e-10 {
        return identity3();
    }
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    let a00 = c1.y * c2.z - c1.z * c2.y;
    let a01 = c0.z * c2.y - c0.y * c2.z;
    let a02 = c0.y * c1.z - c0.z * c1.y;
    let a10 = c1.z * c2.x - c1.x * c2.z;
    let a11 = c0.x * c2.z - c0.z * c2.x;
    let a12 = c0.z * c1.x - c0.x * c1.z;
    let a20 = c1.x * c2.y - c1.y * c2.x;
    let a21 = c0.y * c2.x - c0.x * c2.y;
    let a22 = c0.x * c1.y - c0.y * c1.x;
    return (1.0 / det) * mat3x3<f32>(
        vec3<f32>(a00, a01, a02),
        vec3<f32>(a10, a11, a12),
        vec3<f32>(a20, a21, a22),
    );
}

// Eq (156) https://arxiv.org/pdf/2009.06944
@compute
@workgroup_size(64)
fn precompute_L(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&particles) {
        return;
    }
    var M = mat3x3<f32>(vec3(0.0), vec3(0.0), vec3(0.0));
    let Xi = particles[i].reference_position;
    let n = particles[i].nneighbors;
    for (var k = 0u; k < n; k++) {
        let j = particles[i].neighbors[k];
        let Xj = particles[j].reference_position;
        let X_ij = Xi - Xj;
        let X_ji = Xj - Xi;
        let grad = kernel_gradient(X_ij);
        let Vj = particles[j].rest_volume;
        // outer: X_ji * grad^T
        M += Vj * mat3x3<f32>(X_ji * grad.x, X_ji * grad.y, X_ji * grad.z);
    }
    particles[i].L = invert3(M);
}
