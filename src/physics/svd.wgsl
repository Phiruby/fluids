// Struct to store the final Singular Value Decomposition
struct SVD3x3 {
    U: mat3x3<f32>, // Left singular vectors (columns)
    S: vec3<f32>,   // Singular values
    V: mat3x3<f32>, // Right singular vectors (columns)
};

// Constant parameters matching the McAdams et al. CPU/C++ implementation constants
const FOUR_GAMMA_SQUARED: f32 = 5.828427124; // sqrt(8) + 3
const C_STAR: f32 = 0.923879532;            // cos(pi / 8)
const S_STAR: f32 = 0.3826834323;           // sin(pi / 8)
const EPSILON: f32 = 1e-6;

// Helper to compute a quick, accurate inverse square root via Newton steps
fn accurate_rsqrt(x: f32) -> f32 {
    let xhalf = 0.5 * x;
    // Approximating the bit-level inverse square root behavior safely in WGSL via built-ins
    var y = inverseSqrt(x);
    // Perform a Newton-Raphson refinement step for high mathematical precision
    y = y * (1.5 - xhalf * y * y);
    return y;
}

fn accurate_sqrt(x: f32) -> f32 {
    return x * accurate_rsqrt(x);
}

// Utility function to convert a unit quaternion to a 3x3 rotation matrix
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.w; let x = q.x; let y = q.y; let z = q.z;
    let qxx = x * x; let qyy = y * y; let qzz = z * z;
    let qxz = x * z; let qxy = x * y; let qyz = y * z;
    let qwx = w * x; let qwy = w * y; let qwz = w * z;

    let m0 = vec3<f32>(1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy + qwz), 2.0 * (qxz - qwy));
    let m1 = vec3<f32>(2.0 * (qxy - qwz), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz + qwx));
    let m2 = vec3<f32>(2.0 * (qxz + qwy), 2.0 * (qyz - qwx), 1.0 - 2.0 * (qxx + qyy));

    return mat3x3<f32>(m0, m1, m2);
}

// Approximates the Givens Quaternion for symmetric matrix diagonalizations
fn approximate_givens_quaternion(a11: f32, a12: f32, a22: f32) -> vec2<f32> {
    let ch = 2.0 * (a11 - a22);
    let sh = a12;

    if FOUR_GAMMA_SQUARED * sh * sh < ch * ch {
        let w = accurate_rsqrt(ch * ch + sh * sh);
        return vec2<f32>(ch * w, sh * w);
    } else {
        return vec2<f32>(C_STAR, S_STAR);
    }
}

// Custom structure to bypass pointer-passing limitations in core WGSL functions
struct JacobiState {
    s11: f32,
    s21: f32,
    s22: f32,
    s31: f32,
    s32: f32,
    s33: f32,
    qV: vec4<f32>}

// Performs a Jacobi conjugation step and updates the cumulative V rotation quaternion
fn jacobi_conjugation(x_idx: u32, y_idx: u32, z_idx: u32, state_in: JacobiState) -> JacobiState {
    var state = state_in;
    let g = approximate_givens_quaternion(state.s11, state.s21, state.s22);

    let scale = g.x * g.x + g.y * g.y;
    let a = (g.x * g.x - g.y * g.y) / scale;
    let b = (2.0 * g.y * g.x) / scale;

    let _s11 = state.s11; let _s21 = state.s21; let _s22 = state.s22;
    let _s31 = state.s31; let _s32 = state.s32; let _s33 = state.s33;

    state.s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
    state.s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
    state.s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
    state.s31 = a * _s31 + b * _s32;
    state.s32 = -b * _s31 + a * _s32;
    state.s33 = _s33;

    var tmp = vec3<f32>(state.qV.x * g.y, state.qV.y * g.y, state.qV.z * g.y);
    let sh_w = g.y * state.qV.w;

    state.qV.x *= g.x; state.qV.y *= g.x; state.qV.z *= g.x; state.qV.w *= g.x;

    var q_arr = array<f32, 4>(state.qV.x, state.qV.y, state.qV.z, state.qV.w);
    q_arr[z_idx] += sh_w;
    q_arr[3] -= tmp[z_idx];
    q_arr[x_idx] += tmp[y_idx];
    q_arr[y_idx] -= tmp[x_idx];

    state.qV = vec4<f32>(q_arr[0], q_arr[1], q_arr[2], q_arr[3]);

    // Rearrange tracking values for the sequential layout loop pass
    let t_s11 = state.s22; let t_s21 = state.s32; let t_s22 = state.s33;
    let t_s31 = state.s21; let t_s32 = state.s31; let t_s33 = state.s11;

    state.s11 = t_s11; state.s21 = t_s21; state.s22 = t_s22;
    state.s31 = t_s31; state.s32 = t_s32; state.s33 = t_s33;

    return state;
}

// Evaluates the symmetric matrix to determine the diagonalization transformation V
fn jacobi_eigenanalysis(A_sym: mat3x3<f32>) -> vec4<f32> {
    var state: JacobiState;
    state.s11 = A_sym[0][0]; state.s21 = A_sym[0][1]; state.s31 = A_sym[0][2];
    state.s22 = A_sym[1][1]; state.s32 = A_sym[1][2]; state.s33 = A_sym[2][2];
    state.qV = vec4<f32>(0.0, 0.0, 0.0, 1.0); // Identity quaternion

    // 4 full sweep iterations over index pairings 
    for (var i = 0; i < 4; i++) {
        state = jacobi_conjugation(0, 1, 2, state); // p,q = 0,1
        state = jacobi_conjugation(1, 2, 0, state); // p,q = 1,2
        state = jacobi_conjugation(2, 0, 1, state); // p,q = 0,2
    }
    return state.qV;
}

fn qr_givens_quaternion(a1: f32, a2: f32) -> vec2<f32> {
    let rho = accurate_sqrt(a1 * a1 + a2 * a2);
    var sh = select(0.0, a2, rho > EPSILON);
    var ch = abs(a1) + max(rho, EPSILON);

    if a1 < 0.0 {
        let temp = sh; sh = ch; ch = temp;
    }
    let w = accurate_rsqrt(ch * ch + sh * sh);
    return vec2<f32>(ch * w, sh * w);
}

struct QRResult {
    Q: mat3x3<f32>,
    R: mat3x3<f32>,
};

// Explicit analytical QR factorization for 3x3 matrices
fn qr_decomposition(B: mat3x3<f32>) -> QRResult {
    var r: mat3x3<f32>;
    var b = B;

    // First Givens Rotation
    let g1 = qr_givens_quaternion(b[0][0], b[0][1]);
    let a1 = 1.0 - 2.0 * g1.y * g1.y;
    let b1 = 2.0 * g1.x * g1.y;

    r[0][0] = a1 * b[0][0] + b1 * b[0][1]; r[1][0] = a1 * b[1][0] + b1 * b[1][1]; r[2][0] = a1 * b[2][0] + b1 * b[2][1];
    let r21 = -b1 * b[0][0] + a1 * b[0][1]; let r22 = -b1 * b[1][0] + a1 * b[1][1]; let r23 = -b1 * b[2][0] + a1 * b[2][1];
    let r31 = b[0][2]; let r32 = b[1][2]; let r33 = b[2][2];

    // Second Givens Rotation
    let g2 = qr_givens_quaternion(r[0][0], r31);
    let a2 = 1.0 - 2.0 * g2.y * g2.y;
    let b2 = 2.0 * g2.x * g2.y;

    b[0][0] = a2 * r[0][0] + b2 * r31; b[1][0] = a2 * r[1][0] + b2 * r32; b[2][0] = a2 * r[2][0] + b2 * r33;
    let b21 = r21; let b22 = r22; let b23 = r23;
    let b31 = -b2 * r[0][0] + a2 * r31; let b32 = -b2 * r[1][0] + a2 * r32; let b33 = -b2 * r[2][0] + a2 * r33;

    // Third Givens Rotation
    let g3 = qr_givens_quaternion(b22, b32);
    let a3 = 1.0 - 2.0 * g3.y * g3.y;
    let b3 = 2.0 * g3.x * g3.y;

    r[0][0] = b[0][0]; r[1][0] = b[1][0]; r[2][0] = b[2][0];
    r[0][1] = a3 * b21 + b3 * b31; r[1][1] = a3 * b22 + b3 * b32; r[2][1] = a3 * b23 + b3 * b33;
    r[0][2] = -b3 * b21 + a3 * b31; r[1][2] = -b3 * b22 + a3 * b32; r[2][2] = -b3 * b23 + a3 * b33;

    // Construct cumulative Q rotation matrix tracking quaternions
    let sh12 = g1.y * g1.y; let sh22 = g2.y * g2.y; let sh32 = g3.y * g3.y;
    var Q: mat3x3<f32>;

    Q[0][0] = (-1.0 + 2.0 * sh12) * (-1.0 + 2.0 * sh22);
    Q[1][0] = 4.0 * g2.x * g3.x * (-1.0 + 2.0 * sh12) * g2.y * g3.y + 2.0 * g1.x * g1.y * (-1.0 + 2.0 * sh32);
    Q[2][0] = 4.0 * g1.x * g3.x * g1.y * g3.y - 2.0 * g2.x * (-1.0 + 2.0 * sh12) * g2.y * (-1.0 + 2.0 * sh32);

    Q[0][1] = 2.0 * g1.x * g1.y * (1.0 - 2.0 * sh22);
    Q[1][1] = -8.0 * g1.x * g2.x * g3.x * g1.y * g2.y * g3.y + (-1.0 + 2.0 * sh12) * (-1.0 + 2.0 * sh32);
    Q[2][1] = -2.0 * g3.x * g3.y + 4.0 * g1.y * (g3.x * g1.y * g3.y + g1.x * g2.x * g2.y * (-1.0 + 2.0 * sh32));

    Q[0][2] = 2.0 * g2.x * g2.y;
    Q[1][2] = 2.0 * g3.x * (1.0 - 2.0 * sh22) * g3.y;
    Q[2][2] = (-1.0 + 2.0 * sh22) * (-1.0 + 2.0 * sh32);

    var res: QRResult;
    res.Q = Q;
    res.R = r;
    return res;
}

/**
 * Computes SVD based on the McAdams et al. high performance implementation.
 * Eliminates execution branch paths where possible for optimized compute shader usage.
 */
fn svd3x3(A: mat3x3<f32>) -> SVD3x3 {
    // 1. Compute normal matrix symmetric system: ATA = A^T * A
    let ATA = transpose(A) * A;

    // 2. Perform symmetric eigenanalysis via quaternions to find V
    let qV = jacobi_eigenanalysis(ATA);
    var V = quat_to_mat3(qV);

    // 3. Compute B = A * V
    var B = A * V;

    // 4. Sort columns of B and V in descending magnitude order 
    let rho0 = dot(B[0], B[0]);
    let rho1 = dot(B[1], B[1]);
    let rho2 = dot(B[2], B[2]);

    if rho0 < rho1 {
        let tmp_b = B[0]; B[0] = -B[1]; B[1] = tmp_b;
        let tmp_v = V[0]; V[0] = -V[1]; V[1] = tmp_v;
    }
    if dot(B[0], B[0]) < rho2 {
        let tmp_b = B[0]; B[0] = -B[2]; B[2] = tmp_b;
        let tmp_v = V[0]; V[0] = -V[2]; V[2] = tmp_v;
    }
    if dot(B[1], B[1]) < dot(B[2], B[2]) {
        let tmp_b = B[1]; B[1] = -B[2]; B[2] = tmp_b;
        let tmp_v = V[1]; V[1] = -V[2]; V[2] = tmp_v;
    }

    // 5. Compute QR decomposition of the sorted B matrix to get final U and S elements
    let qr = qr_decomposition(B);

    var result: SVD3x3;
    result.U = qr.Q;
    result.S = vec3<f32>(qr.R[0][0], qr.R[1][1], qr.R[2][2]);
    result.V = V;
    return result;
}

