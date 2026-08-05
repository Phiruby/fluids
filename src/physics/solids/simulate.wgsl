// Explicit SPH elastic solids: eqs 153–160 (arxiv:2009.06944).
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

struct SimParams {
    kernel_h: f32,
    dt: f32,
    young: f32,
    poisson: f32,
    floor_y: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

@group(0) @binding(0) var<storage, read> src: array<Particle>;
@group(0) @binding(1) var<storage, read_write> dst: array<Particle>;
@group(0) @binding(2) var<uniform> params: SimParams;

const PI = 3.14159265358979323846;
const DAMPING = 0.998;
const GRAVITY = vec3<f32>(0.0, -9.81, 0.0);

fn kernel_weight(r_norm: f32) -> f32 {
    let q = r_norm / params.kernel_h;
    let sigma = 8.0 / (PI * pow(params.kernel_h, 3.0));
    if q <= 0.5 {
        return sigma * (6.0 * q * q * q - 6.0 * q * q + 1.0);
    } else if q <= 1.0 {
        return sigma * 2.0 * pow(1.0 - q, 3.0);
    }
    return 0.0;
}

fn kernel_gradient(r: vec3<f32>) -> vec3<f32> {
    let r_norm = length(r);
    if r_norm < 1e-12 {
        return vec3<f32>(0.0);
    }
    let q = r_norm / params.kernel_h;
    let sigma = 8.0 / (PI * pow(params.kernel_h, 3.0));
    let dq = r / (params.kernel_h * r_norm);
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

fn outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a * b.x, a * b.y, a * b.z);
}

fn frobenius_sq(m: mat3x3<f32>) -> f32 {
    return dot(m[0], m[0]) + dot(m[1], m[1]) + dot(m[2], m[2]);
}

fn adjugate3(m: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    return mat3x3<f32>(
        vec3<f32>(c1.y * c2.z - c1.z * c2.y, c0.z * c2.y - c0.y * c2.z, c0.y * c1.z - c0.z * c1.y),
        vec3<f32>(c1.z * c2.x - c1.x * c2.z, c0.x * c2.z - c0.z * c2.x, c0.z * c1.x - c0.x * c1.z),
        vec3<f32>(c1.x * c2.y - c1.y * c2.x, c0.y * c2.x - c0.x * c2.y, c0.x * c1.y - c0.y * c1.x),
    );
}

// implement polar decomposition; see link below (though we implement Highham's scaling factor, not the Halley's version)
// https://www.researchgate.net/publication/220656598_Optimizing_Halley's_Iteration_for_Computing_the_Matrix_Polar_Decomposition
fn extract_rotation(F: mat3x3<f32>) -> mat3x3<f32> {
    var X = F;
    if frobenius_sq(X) < 1e-20 {
        return identity3();
    }
    for (var it = 0; it < 4; it++) {
        let detX = determinant(X);
        let nX = frobenius_sq(X);
        if abs(detX) < 1e-12 || nX < 1e-20 {
            return identity3();
        }
        let adj = adjugate3(X);
        let nInv = frobenius_sq(adj) / (detX * detX);
        let gamma = sqrt(sqrt(nInv / nX));
        X = 0.5 * ((gamma * X) + (1.0 / (gamma * detX)) * adj);
    }
    return X;
}

fn ref_grad(i: u32, j: u32) -> vec3<f32> {
    return kernel_gradient(src[i].reference_position - src[j].reference_position);
}

// Eq 157 https://arxiv.org/pdf/2009.06944
fn deformation_gradient(i: u32) -> mat3x3<f32> {
    var J = mat3x3<f32>(vec3(0.0), vec3(0.0), vec3(0.0));
    let n = src[i].nneighbors;
    for (var k = 0u; k < n; k++) {
        let j = src[i].neighbors[k];
        let x_ji = src[j].position - src[i].position;
        J += src[j].rest_volume * outer(x_ji, ref_grad(i, j));
    }
    return src[i].L * J;
}

// Eq (159) https://arxiv.org/pdf/2009.06944
fn corotated_deformation(i: u32, R: mat3x3<f32>) -> mat3x3<f32> {
    var sum = mat3x3<f32>(vec3(0.0), vec3(0.0), vec3(0.0));
    let Li = src[i].L;
    let n = src[i].nneighbors;
    for (var k = 0u; k < n; k++) {
        let j = src[i].neighbors[k];
        let x_ji = src[j].position - src[i].position;
        let X_ji = src[j].reference_position - src[i].reference_position;
        let grad_star = R * Li * ref_grad(i, j);
        sum += src[j].rest_volume * outer(x_ji - R * X_ji, grad_star);
    }
    return identity3() + sum;
}

// Eq (150)
fn piola_stress(Jstar: mat3x3<f32>) -> mat3x3<f32> {
    let mu = params.young / (2.0 * (1.0 + params.poisson));
    let lambda = params.young * params.poisson
        / ((1.0 + params.poisson) * (1.0 - 2.0 * params.poisson));
    let eps = 0.5 * (Jstar + transpose(Jstar)) - identity3();
    let tr = eps[0][0] + eps[1][1] + eps[2][2];
    return 2.0 * mu * eps + lambda * tr * identity3();
}

fn grad_star(i: u32, j: u32, R: mat3x3<f32>) -> vec3<f32> {
    return R * src[i].L * ref_grad(i, j);
}

// Eq (160)
fn elastic_force(i: u32) -> vec3<f32> {
    let J_i = deformation_gradient(i);
    let R_i = extract_rotation(J_i);
    let P_i = piola_stress(corotated_deformation(i, R_i));
    var F = vec3<f32>(0.0);
    let Vi = src[i].rest_volume;
    let n = src[i].nneighbors;
    for (var k = 0u; k < n; k++) {
        let j = src[i].neighbors[k];
        let J_j = deformation_gradient(j);
        let R_j = extract_rotation(J_j);
        let P_j = piola_stress(corotated_deformation(j, R_j));
        let Vj = src[j].rest_volume;
        F += Vi * Vj * (P_i * grad_star(i, j, R_i) - P_j * grad_star(j, i, R_j));
    }
    return F;
}

@compute
@workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&src) {
        return;
    }
    var p = src[i];
    if p.pinned != 0u {
        p.position = p.reference_position;
        p.velocity = vec3<f32>(0.0);
        dst[i] = p;
        return;
    }
    let force = elastic_force(i) + p.mass * GRAVITY;
    p.velocity += force * (params.dt / p.mass);
    p.velocity *= DAMPING;
    p.position += p.velocity * params.dt;
    // if p.position.y < params.floor_y {
    //     p.position.y = params.floor_y;
    //     p.velocity.y = max(p.velocity.y, 0.0);
    // }
    dst[i] = p;
}
