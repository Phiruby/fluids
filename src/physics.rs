mod spatialhash;
use crate::Vertex;
use nalgebra::{Matrix3, Vector3};
use std::ops::Add;

const DT: f32 = 0.01;
const VELOCITY_DAMPING: f32 = 0.995;
const STRETCH_RATE: f32 = 0.15;
const MAX_STRETCH: f32 = 1.05;
#[derive(Debug, Clone)]
#[allow(unused)]
struct PhysicsVertex {
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    renderable: bool,
    mass: f32,
}


impl From<PhysicsVertex> for Vertex {
    fn from(vertex: PhysicsVertex) -> Self {
        Vertex {
            // hackery to access vector from position: https://docs.rs/nalgebra/latest/nalgebra/base/type.Vector3.html
            position: vertex.position.data.0[0],
            color: [0.5, 0.0, 0.5],
        }
    }
}

fn neighbor_sum<F, T>(
    delta: F,
    neighbors: &[usize],
    zero: T,
) -> T
where
    F: Fn(usize) -> T,
    T: Add<Output = T>,
{
    let mut sum = zero;
    for &neighbor_idx in neighbors {
        sum = sum + delta(neighbor_idx);
    }
    sum
}

fn precompute_neighbors(positions: &[Vector3<f32>], kernel: &CubicKernel) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); positions.len()];
    for i in 0..positions.len() {
        for j in 0..positions.len() {
            if (positions[i] - positions[j]).norm() <= kernel.h {
                neighbors[i].push(j);
            }
        }
    }
    neighbors
}


#[derive(Debug, Clone)]
pub struct CubicKernel {
    pub h: f32,
}

impl CubicKernel {
    pub fn new(h: f32) -> Self {
        Self { h }
    }

    fn weight(&self, r_norm: f32) -> f32 {
        let q = r_norm / self.h;
        if q >= 1.0 || q < 0.0 {
            return 0.0;
        }

        // 3D normalization constant for Wendland C2
        let sigma = 21.0 / (2.0 * std::f32::consts::PI * self.h.powf(3.0));
        
        let term = 1.0 - q;
        let w = term.powf(4.0) * (4.0 * q + 1.0);
        
        w * sigma
    }

    fn weight_gradient(&self, r: Vector3<f32>) -> Vector3<f32> {
        let r_norm = r.norm();
        if r_norm == 0.0 {
            return Vector3::zeros();
        }

        let q = r_norm / self.h;
        if q >= 1.0 {
            return Vector3::zeros();
        }

        let sigma = 21.0 / (2.0 * std::f32::consts::PI * self.h.powf(3.0));

        // dW/dq = -20 * q * (1 - q)^3
        let dw_dq = -20.0 * q * (1.0 - q).powf(3.0);
        
        // dq/dx = r / (h * ||r||)
        let dq_dx = r / (self.h * r_norm);

        // Chain rule: dW/dx = dW/dq * dq/dx
        let gradient = (dw_dq * sigma) * dq_dx;
        gradient
    }

    /// Performs pairwise evaluation of Wendland C2 kernel between the two.
    /// Index [i][j] of output is W_{ij} = W(x_i - x_j).
    pub fn eval(
        &self,
        x_i: &[Vector3<f32>],
        x_j: &[Vector3<f32>],
        neighbors: &[Vec<usize>],
    ) -> Vec<Vec<f32>> {
        let mut w = vec![vec![0.0; x_j.len()]; x_i.len()];
        for i in 0..x_i.len() {
            for &j in &neighbors[i] {
                let r = x_i[i] - x_j[j];
                w[i][j] = self.weight(r.norm());
            }
        }
        w
    }

    /// Returns gradient w.r.t x_i's of W_{ij} = W(x_i - x_j);
    /// i.e this returns Matrix so that G[i][j] = nabla_i W_{ij}
    pub fn gradient(
        &self,
        x_i: &[Vector3<f32>],
        x_j: &[Vector3<f32>],
        neighbors: &[Vec<usize>],
    ) -> Vec<Vec<Vector3<f32>>> {
        let mut g = vec![vec![Vector3::new(0.0, 0.0, 0.0); x_j.len()]; x_i.len()];
        for i in 0..x_i.len() {
            for &j in &neighbors[i] {
                let r = x_i[i] - x_j[j];
                g[i][j] = self.weight_gradient(r);
            }
        }
        g
    }
}

#[derive(Debug)]
struct CubeShape {
    pub vertices: Vec<PhysicsVertex>,
    n: usize,
    t: f32,
}

/// reference_gradients: [i][j] = nabla_i W_{ij}
#[derive(Debug)]
pub struct Cube {
    reference: CubeShape,
    neighbors: Vec<Vec<usize>>,
    L: Vec<Matrix3<f32>>,
    reference_gradients: Vec<Vec<Vector3<f32>>>,
    reference_volumes: Vec<f32>,
    reference_kernel_weights: Vec<Vec<f32>>,
    pub current: CubeShape,
    kernel: CubicKernel,
}

impl Cube {
    pub fn new(n: usize, spacing: f32, kernel: &CubicKernel) -> Self {
        let reference = CubeShape::new_static(n, spacing);
        let reference_positions = reference
            .vertices
            .iter()
            .map(|v| v.position)
            .collect::<Vec<_>>();
        let neighbors = precompute_neighbors(&reference_positions, kernel);
        let reference_densities = compute_densities(&reference.vertices, kernel, &neighbors);
        
        let reference_volumes: Vec<f32> = reference_densities
            .iter()
            .zip(&reference.vertices)
            .map(|(d, v)| v.mass / d)
            .collect();

        let reference_gradients = kernel.gradient(
            &reference_positions,
            &reference_positions,
            &neighbors,
        );
        let reference_distances = pairwise_displacement(&reference.vertices, &reference.vertices);
        let mut L = Vec::with_capacity(reference.vertices.len());
        for i in 0..reference.vertices.len() {
            let mut L_i = Matrix3::zeros();
            for j in 0..reference.vertices.len() {
                // NOTE: this uses inverse of (154) instead of the provided formula (156) in https://arxiv.org/pdf/2009.06944
                L_i += reference_volumes[j] * reference_distances[j][i] * reference_gradients[i][j].transpose();
            }
            L.push(L_i.try_inverse().unwrap());
        }
        let reference_kernel_weights = kernel.eval(
            &reference_positions,
            &reference_positions,
            &neighbors,
        );
        Self {
            reference,
            neighbors,
            reference_volumes,
            reference_kernel_weights,
            current: CubeShape::new(n, spacing),
            L,
            reference_gradients,
            kernel: kernel.clone(),
        }
    }

    pub fn update(&mut self) {
        let forces = compute_forces(self, &self.kernel);
        self.current.vertices.iter_mut().zip(forces).for_each(|(v, f)| {
            v.velocity += f * DT / v.mass;
            v.velocity *= VELOCITY_DAMPING;
        });
        self.current.update();
        let stretch_amount = (self.current.t * STRETCH_RATE).min(MAX_STRETCH);
        apply_uniaxial_x_stretch(&self.reference, &mut self.current, stretch_amount);
    }

    pub fn to_renderable(&self) -> (Vec<Vertex>, Vec<u16>) {
        self.current.to_renderable()
    }

    pub fn to_particle_renderable(&self) -> Vec<Vertex> {
        let (min_x, max_x) = x_face_bounds(&self.reference);
        self.current
            .vertices
            .iter()
            .zip(self.reference.vertices.iter())
            .map(|(current, reference)| {
                let color = if (reference.position.x - min_x).abs() < 1e-6 {
                    [0.25, 0.8, 1.0]
                } else if (reference.position.x - max_x).abs() < 1e-6 {
                    [1.0, 0.35, 0.35]
                } else if current.renderable {
                    [1.0, 0.75, 0.55]
                } else {
                    [0.9, 0.9, 1.0]
                };

                Vertex {
                    position: [current.position.x, current.position.y, current.position.z],
                    color,
                }
            })
            .collect()
    }
}

impl CubeShape {
    fn build_vertices(n: usize, spacing: f32) -> Vec<PhysicsVertex> {
        let mut vertices = Vec::new();
        let spacing_x = 1.0 / n as f32;
        let spacing_y = 1.0 / n as f32;
        let spacing_z = spacing;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let renderable =
                        i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1;
                    vertices.push(PhysicsVertex {
                        position: Vector3::new(
                            i as f32 * spacing_x,
                            j as f32 * spacing_y,
                            k as f32 * spacing_z,
                        ),
                        velocity: Vector3::new(0.0, 0.0, 0.0),
                        renderable,
                        mass: 1.0,
                    });
                }
            }
        }
        vertices
    }

    fn new_static(n: usize, spacing: f32) -> Self {
        Self {
            vertices: Self::build_vertices(n, spacing),
            n,
            t: 0.0,
        }
    }

    pub fn new(n: usize, spacing: f32) -> Self {
        Self {
            vertices: Self::build_vertices(n, spacing),
            n,
            t: 0.0,
        }
    }

    fn grid_index(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.n * self.n + j * self.n + k
    }

    pub fn update(&mut self) {
        self.t += DT;
        for v in self.vertices.iter_mut() {
            v.position += v.velocity * DT;
        }
    }

    // a=bottom-left, b=bottom-right, c=top-right, d=top-left (viewed from outside)
    fn push_quad(indices: &mut Vec<u16>, a: u16, b: u16, c: u16, d: u16, flip: bool) {
        if flip {
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        } else {
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    pub fn to_renderable(&self) -> (Vec<Vertex>, Vec<u16>) {
        let n = self.n;
        let mut render_map = vec![None; self.vertices.len()];
        let mut vertices = Vec::new();
        for (grid_idx, v) in self.vertices.iter().enumerate() {
            if v.renderable {
                render_map[grid_idx] = Some(vertices.len() as u16);
                vertices.push(v.clone().into());
            }
        }
        let idx = |i: usize, j: usize, k: usize| render_map[self.grid_index(i, j, k)].unwrap();

        let mut indices = Vec::new();

        for j in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(0, j, k),
                    idx(0, j + 1, k),
                    idx(0, j + 1, k + 1),
                    idx(0, j, k + 1),
                    false,
                );
            }
        }
        for j in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(n - 1, j, k),
                    idx(n - 1, j + 1, k),
                    idx(n - 1, j + 1, k + 1),
                    idx(n - 1, j, k + 1),
                    true,
                );
            }
        }
        for i in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, 0, k),
                    idx(i + 1, 0, k),
                    idx(i + 1, 0, k + 1),
                    idx(i, 0, k + 1),
                    false,
                );
            }
        }
        for i in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, n - 1, k),
                    idx(i + 1, n - 1, k),
                    idx(i + 1, n - 1, k + 1),
                    idx(i, n - 1, k + 1),
                    true,
                );
            }
        }
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, j, 0),
                    idx(i + 1, j, 0),
                    idx(i + 1, j + 1, 0),
                    idx(i, j + 1, 0),
                    false,
                );
            }
        }
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, j, n - 1),
                    idx(i + 1, j, n - 1),
                    idx(i + 1, j + 1, n - 1),
                    idx(i, j + 1, n - 1),
                    true,
                );
            }
        }

        (vertices, indices)
    }

}

fn x_face_bounds(shape: &CubeShape) -> (f32, f32) {
    let min_x = shape
        .vertices
        .iter()
        .map(|v| v.position.x)
        .fold(f32::INFINITY, f32::min);
    let max_x = shape
        .vertices
        .iter()
        .map(|v| v.position.x)
        .fold(f32::NEG_INFINITY, f32::max);
    (min_x, max_x)
}

fn apply_uniaxial_x_stretch(reference: &CubeShape, current: &mut CubeShape, stretch_amount: f32) {
    let (min_x, max_x) = x_face_bounds(reference);

    for (reference_vertex, current_vertex) in reference.vertices.iter().zip(current.vertices.iter_mut()) {
        if (reference_vertex.position.x - min_x).abs() < 1e-6 {
            current_vertex.position.x = reference_vertex.position.x;
            current_vertex.velocity.x = 0.0;
        } else if (reference_vertex.position.x - max_x).abs() < 1e-6 {
            current_vertex.position.x = reference_vertex.position.x + stretch_amount;
            current_vertex.velocity.x = 0.0;
        }
    }
}

/// Eq (11) in https://arxiv.org/pdf/2009.06944
fn compute_densities(
    vertices: &[PhysicsVertex],
    kernel: &CubicKernel,
    neighbors: &[Vec<usize>],
) -> Vec<f32> {
    let mut densities = vec![0.0; vertices.len()];
    for i in 0..vertices.len() {
        let density = neighbor_sum(
            |j| {
                let r = vertices[i].position - vertices[j].position;
                vertices[j].mass * kernel_weight(kernel, r.norm())
            },
            &neighbors[i],
            0.0,
        );
        densities[i] = density;
    }
    densities
}

fn kernel_weight(kernel: &CubicKernel, r_norm: f32) -> f32 {
    kernel.weight(r_norm)
}

fn pairwise_displacement(x: &[PhysicsVertex], y: &[PhysicsVertex]) -> Vec<Vec<Vector3<f32>>> {
    let mut displacements = vec![vec![Vector3::new(0.0, 0.0, 0.0); y.len()]; x.len()];
    for i in 0..x.len() {
        for j in 0..y.len() {
            displacements[i][j] = x[i].position - y[j].position;
        }
    }
    displacements
}

fn strain_tensor(deformation: &Matrix3<f32>) -> Matrix3<f32> {
    0.5 * (deformation + deformation.transpose()) - Matrix3::identity()
}


fn compute_rotations(deformations: &[Matrix3<f32>]) -> Vec<Matrix3<f32>> {
    let mut rotations: Vec<Matrix3<f32>> = Vec::with_capacity(deformations.len());
    for i in 0..deformations.len() {
        // TODO: can be a lot faster if we don't use polar decomp;
        // also unsure from the paper whether this is right
        let (_, R_i) = deformations[i].polar();
        rotations.push(R_i);
    }
    rotations
}

fn initial_deformation_tensor_estimate(
    current_vertices: &[PhysicsVertex],
    reference_gradients: &[Vec<Vector3<f32>>],
    L: &[Matrix3<f32>],
    reference_volumes: &[f32],
    neighbors: &[Vec<usize>],
) -> Vec<Matrix3<f32>> {
    let mut deformations: Vec<Matrix3<f32>> = Vec::with_capacity(current_vertices.len());
    for i in 0..current_vertices.len() {
        let f_i = neighbor_sum(
            |j| {
                let x_ji = current_vertices[j].position - current_vertices[i].position;
                let grad_w_ij = reference_gradients[i][j];
                let v_j = reference_volumes[j];
                v_j * x_ji * grad_w_ij.transpose()
            },
            &neighbors[i],
            Matrix3::zeros(),
        );
        deformations.push(L[i] * f_i);
    }
    deformations
}

/// Naive impl of (159) in https://arxiv.org/pdf/2009.06944
fn deformation_tensor_star(
    current_vertices: &[PhysicsVertex],
    reference_vertices: &[PhysicsVertex],
    reference_gradients_star: &[Vec<Vector3<f32>>],
    rotations: &[Matrix3<f32>],
    reference_volumes: &[f32],
    neighbors: &[Vec<usize>],
) -> Vec<Matrix3<f32>> {
    let mut deformations_star = Vec::with_capacity(current_vertices.len());
    for i in 0..current_vertices.len() {
        let r_i = rotations[i];
        let f_i_star = neighbor_sum(
            |j| {
                let x_ji = current_vertices[j].position - current_vertices[i].position;
                let x_ji_ref = reference_vertices[j].position - reference_vertices[i].position;
                let grad_w_ij_star = reference_gradients_star[i][j];
                let v_j = reference_volumes[j];
                v_j * (x_ji - r_i * x_ji_ref) * grad_w_ij_star.transpose()
            },
            &neighbors[i],
            Matrix3::zeros(),
        );
        deformations_star.push(f_i_star + Matrix3::identity());
    }
    deformations_star
}

/// impl of (158) in https://arxiv.org/pdf/2009.06944
fn compute_grad_star(
    gradients: &[Vec<Vector3<f32>>],
    rotations: &[Matrix3<f32>],
    L: &[Matrix3<f32>],
) -> Vec<Vec<Vector3<f32>>> {
    let mut grad_star = Vec::with_capacity(gradients.len());
    for i in 0..gradients.len() {
        grad_star.push(Vec::with_capacity(gradients[i].len()));
        for j in 0..gradients[i].len() {
            grad_star[i].push(rotations[i] * L[i] * gradients[i][j]);
        }
    }
    grad_star
}

fn stress_tensor(
    deformations_star: &[Matrix3<f32>],
    mu: f32,
    lambda: f32
) -> Vec<Matrix3<f32>> {
    let mut stress_tensors = Vec::with_capacity(deformations_star.len());
    for i in 0..deformations_star.len() {
        let strain = strain_tensor(&deformations_star[i]);
        stress_tensors.push(2.0 * mu * strain + lambda * strain.trace() * Matrix3::identity());
    }
    stress_tensors
}

fn penalty_forces(
    reference_volumes: &[f32],
    k: f32,
    alpha: f32,
    deformations: &[Matrix3<f32>],
    reference_kernel_weights: &[Vec<f32>],
    current_vertices: &[PhysicsVertex],
    reference_vertices: &[PhysicsVertex],
    neighbors: &[Vec<usize>],
) -> Vec<Vector3<f32>> {
    let mut forces = Vec::with_capacity(current_vertices.len());
    for i in 0..current_vertices.len() {
        let force = -0.5 * alpha * k * neighbor_sum(
            |j| {
                if j == i { return Vector3::zeros(); }
                let x_ji_ref = reference_vertices[j].position - reference_vertices[i].position;
                let x_ji = current_vertices[j].position - current_vertices[i].position;
                let e_ji = deformations[i] * x_ji_ref - x_ji;
                let e_ij = x_ji - deformations[j] * x_ji_ref;
                reference_volumes[i] * reference_volumes[j] * (reference_kernel_weights[i][j] / x_ji_ref.norm_squared()) 
                    * (e_ji.dot(&x_ji) / x_ji.norm() - e_ij.dot(&x_ji_ref) / x_ji_ref.norm()) * x_ji / x_ji.norm()
                    
            },
            &neighbors[i],
            Vector3::zeros(),
        );
        forces.push(force);
    }
    forces
}

pub fn compute_forces(
    shape: &Cube,
    _kernel: &CubicKernel,
) -> Vec<Vector3<f32>> {
    let current = &shape.current;
    let initial_deformations = initial_deformation_tensor_estimate(
        &current.vertices,
        &shape.reference_gradients,
        &shape.L,
        &shape.reference_volumes,
        &shape.neighbors,
    );
    let rotations = compute_rotations(&initial_deformations);
    let grad_star = compute_grad_star(&shape.reference_gradients, &rotations, &shape.L);
    let deformations_star = deformation_tensor_star(
        &current.vertices,
        &shape.reference.vertices,
        &grad_star,
        &rotations,
        &shape.reference_volumes,
        &shape.neighbors,
    );
    let reference_volumes = &shape.reference_volumes;
    let k = 7000.0;
    let nu = 0.35;
    let mu = k / (2.0 * (1.0 + nu));
    let lambda = k * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let stress_tensors = stress_tensor(&deformations_star, mu, lambda);
    let mut forces = Vec::with_capacity(current.vertices.len());
    for i in 0..current.vertices.len() {
        let force = neighbor_sum(
            |j| {
                reference_volumes[i] * reference_volumes[j]
                    * (stress_tensors[i] * grad_star[i][j] - stress_tensors[j] * grad_star[j][i])
            },
            &shape.neighbors[i],
            Vector3::zeros(),
        );
        forces.push(force);
    }
    let penalty_forces = penalty_forces(
        &shape.reference_volumes,
        k,
        0.9, // alpha: what to choose?
        &initial_deformations,
        &shape.reference_kernel_weights,
        &current.vertices,
        &shape.reference.vertices,
        &shape.neighbors,
    );
    forces.iter_mut().zip(penalty_forces.iter()).for_each(|(f, pf)| {
        *f += pf;
    });
    forces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoketest_cube_new() {
        let cube = CubeShape::new(3, 1.0);
        assert_eq!(cube.vertices.len(), 27);
        assert_eq!(cube.vertices.iter().filter(|v| v.renderable).count(), 26);
    }

    #[test]
    fn test_renderable() {
        let cube = CubeShape::new(2, 1.0);
        let (vertices, indices) = cube.to_renderable();
        assert_eq!(vertices.len(), 8);
        assert_eq!(indices.len(), 36);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_renderable_n3() {
        let cube = CubeShape::new(3, 1.0);
        let (vertices, indices) = cube.to_renderable();
        assert_eq!(vertices.len(), 26);
        assert_eq!(indices.len(), 6 * 4 * 6);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_cubic_kernel() {
        let kernel = CubicKernel::new(1.0);
        let x_i = vec![Vector3::new(0.0, 0.0, 0.0)];
        let x_j = vec![Vector3::new(1.0, 0.0, 0.0)];
        let neighbors = precompute_neighbors(&x_j, &kernel);
        let w = kernel.eval(&x_i, &x_j, &neighbors);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].len(), 1);
        assert_eq!(w[0][0], 0.0);
    }

    #[test]
    fn test_stress_tensor_zero() {
        let deformation = Matrix3::identity();
        let stress = stress_tensor(
            &[deformation], 0.0, 0.0
        )[0];
        assert_eq!(stress, Matrix3::zeros());
    }

    #[test]
    fn no_deformation() {
        let m = 3;
        let kernel = CubicKernel::new(1.0 / (m - 1) as f32);
        let cube = Cube::new(m, 1.0 / m as f32, &kernel);
        let deformations = initial_deformation_tensor_estimate(
            &cube.reference.vertices,
            &cube.reference_gradients,
            &cube.L,
            &cube.reference_volumes,
            &cube.neighbors,
        );
        assert!(deformations[0].relative_eq(&Matrix3::identity(), 1e-6, 1e-6));
    }
    #[test]
    fn test_neighbor_sum() {
        let neighbors = vec![1, 3, 5];
        let sum = neighbor_sum(|idx| idx, &neighbors, 0);
        assert_eq!(sum, 9);
    }

    #[test]
    fn uniaxial_stretch_constrains_only_x_on_end_faces() {
        let reference = CubeShape::new_static(3, 1.0);
        let mut current = CubeShape::new(3, 1.0);

        for vertex in &mut current.vertices {
            vertex.position += Vector3::new(0.2, 0.3, 0.4);
            vertex.velocity = Vector3::new(1.0, 2.0, 3.0);
        }

        apply_uniaxial_x_stretch(&reference, &mut current, 0.25);
        let (min_x, max_x) = x_face_bounds(&reference);

        for (reference_vertex, current_vertex) in reference.vertices.iter().zip(current.vertices.iter()) {
            if (reference_vertex.position.x - min_x).abs() < 1e-6 {
                assert_eq!(current_vertex.position.x, reference_vertex.position.x);
                assert_eq!(current_vertex.velocity.x, 0.0);
            } else if (reference_vertex.position.x - max_x).abs() < 1e-6 {
                assert_eq!(current_vertex.position.x, reference_vertex.position.x + 0.25);
                assert_eq!(current_vertex.velocity.x, 0.0);
            }
        }
    }
}
