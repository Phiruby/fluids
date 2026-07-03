mod spatialhash;
pub use spatialhash::SpatialHash3D;
use crate::Vertex;
use nalgebra::{Matrix3, Vector3};
use std::ops::Add;
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
    hash: &SpatialHash3D<Vec<usize>>,
    position: Vector3<f32>,
    zero: T,
) -> T
where
    F: Fn(usize) -> T,
    T: Add<Output = T>,
{
    let mut sum = zero;
    for (_, cell_residents) in hash.neighbors(position) {
        for neighbor_idx in cell_residents {
            sum = sum + delta(*neighbor_idx);
        }
    }
    sum
}

#[derive(Debug, Clone)]
pub struct CubicKernel {
    pub h: f32,
}

impl CubicKernel {
    pub fn new(h: f32) -> Self {
        Self { h }
    }
    /// Performs pairwise evaluation of caussian kernel betwen the two.
    /// Index [i][j] of output is W_{ij} = W(x_i - x_j).
    /// Cubic kernel is impl of (4) in https://arxiv.org/pdf/2009.06944
    fn eval(
        &self, 
        x_i: &[Vector3<f32>], 
        x_j: &[Vector3<f32>],
    ) -> Vec<Vec<f32>> {
        let mut W = vec![vec![0.0; x_j.len()]; x_i.len()];
        let sigma = 8.0 / (std::f32::consts::PI * self.h.powf(3.0));
        for i in 0..x_i.len() {
            for j in 0..x_j.len() {
                let r = x_i[i] - x_j[j];
                let r_norm = r.norm();
                let q = r_norm / self.h;
                if 0.0 <= q && q <= 0.5 {
                    W[i][j] = 6.0 * (q.powf(3.0) - q.powf(2.0) + 1.0 / 6.0);
                } else if 0.5 < q && q <= 1.0 {
                    W[i][j] = 2.0 * (1.0 - q).powf(3.0);
                } else {
                    W[i][j] = 0.0;
                }
                W[i][j] *= sigma;
            }
        }
        W
    }

    /// Returns gradient w.r.t x_i's of W_{ij} = W(x_i - x_j);
    /// i.e this returns Matrix so that G[i][j] = nabla_i W_{ij}
    fn gradient(&self, x_i: &[Vector3<f32>], x_j: &[Vector3<f32>]) -> Vec<Vec<Vector3<f32>>> {
        let mut G = vec![vec![Vector3::new(0.0, 0.0, 0.0); x_j.len()]; x_i.len()];
        let sigma = 8.0 / (std::f32::consts::PI * self.h.powf(3.0));
        for i in 0..x_i.len() {
            for j in 0..x_j.len() {
                let r = x_i[i] - x_j[j];
                let r_norm = r.norm();
                if r_norm == 0.0 {
                    G[i][j] = Vector3::zeros();
                    continue;
                }
                let q = r_norm / self.h;
                let partial_q_partial_xi = |basis| r[basis] / (self.h * q);
                let pqpx = Vector3::new(
                    partial_q_partial_xi(0),
                    partial_q_partial_xi(1),
                    partial_q_partial_xi(2),
                );
                if 0.0 <= q && q <= 0.5 {
                    G[i][j] = 18.0 * q.powf(2.0) * pqpx - 12.0 * q * pqpx;
                } else if 0.5 < q && q <= 1.0 {
                    G[i][j] = -6.0 * (1.0 - q).powf(2.0) * pqpx;
                } else {
                    G[i][j] = Vector3::zeros();
                }
                G[i][j] *= sigma;
            }
        }
        G
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
    L: Vec<Matrix3<f32>>,
    reference_gradients: Vec<Vec<Vector3<f32>>>,
    reference_volumes: Vec<f32>,
    pub current: CubeShape,
    kernel: CubicKernel,
}

impl Cube {
    pub fn new(
        n: usize,
        spacing: f32,
        kernel: &CubicKernel,
        hash: &mut SpatialHash3D<Vec<usize>>,
    ) -> Self {
        let reference = CubeShape::new_static(n, spacing / 2.0);
        hash.clear(&mut |cell| cell.clear());
        for (idx, v) in reference.vertices.iter().enumerate() {
            hash.insert(v.position, idx);
        }
        let reference_densities = compute_densities(&reference.vertices, kernel, hash);
        // NOTE: inserting hash just for compute_densities and clearing
        hash.clear(&mut |cell| cell.clear());
        
        let reference_volumes: Vec<f32> = reference_densities
            .iter()
            .zip(&reference.vertices)
            .map(|(d, v)| v.mass / d)
            .collect();

        let reference_gradients = kernel.gradient(
            &reference.vertices
                .iter()
                .map(|v| v.position)
                .collect::<Vec<_>>(),
            &reference.vertices
                .iter()
                .map(|v| v.position)
                .collect::<Vec<_>>(),
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
        
        Self {
            reference,
            reference_volumes,
            current: CubeShape::new(n, spacing, hash),
            L,
            reference_gradients,
            kernel: kernel.clone(),
        }
    }

    pub fn update(&mut self, hash: &mut SpatialHash3D<Vec<usize>>) {
        hash.clear(&mut |cell| cell.clear());
        self.current.update_hash(hash);
        let forces = compute_forces(self, &self.kernel, hash);
        self.current.vertices.iter_mut().zip(forces).for_each(|(v, f)| {
            v.velocity += f * 0.01 / v.mass;
        });
        self.current.update();
    }

    pub fn to_renderable(&self) -> (Vec<Vertex>, Vec<u16>) {
        self.current.to_renderable()
    }
}

impl CubeShape {
    fn build_vertices(n: usize, spacing: f32) -> Vec<PhysicsVertex> {
        let mut vertices = Vec::new();
        let spacing_x = 0.1;
        let spacing_y = 0.1;
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

    pub fn new(n: usize, spacing: f32, hash: &mut SpatialHash3D<Vec<usize>>) -> Self {
        let mut this = Self {
            vertices: Self::build_vertices(n, spacing),
            n,
            t: 0.0,
        };
        this.update_hash(hash);
        this
    }

    pub fn update_hash(&mut self, hash: &mut SpatialHash3D<Vec<usize>>) {
        for (idx, v) in self.vertices.iter().enumerate() {
            hash.insert(v.position, idx);
        }
    }

    fn grid_index(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.n * self.n + j * self.n + k
    }

    pub fn update(&mut self) {
        self.t += 0.01;
        for v in self.vertices.iter_mut() {
            v.position += v.velocity * 0.01;
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

/// Eq (11) in https://arxiv.org/pdf/2009.06944
fn compute_densities(
    vertices: &[PhysicsVertex],
    kernel: &CubicKernel,
    hash: &SpatialHash3D<Vec<usize>>,
) -> Vec<f32> {
    let mut densities = vec![0.0; vertices.len()];
    for i in 0..vertices.len() {
        let pos_i = vertices[i].position;
        densities[i] = neighbor_sum(
            |j| {
                let r = pos_i - vertices[j].position;
                vertices[j].mass * kernel_weight(kernel, r.norm())
            },
            hash,
            pos_i,
            0.0,
        );
    }
    densities
}

fn kernel_weight(kernel: &CubicKernel, r_norm: f32) -> f32 {
    let sigma = 8.0 / (std::f32::consts::PI * kernel.h.powf(3.0));
    let q = r_norm / kernel.h;
    let w = if (0.0..=0.5).contains(&q) {
        6.0 * (q.powf(3.0) - q.powf(2.0) + 1.0 / 6.0)
    } else if (0.5..=1.0).contains(&q) {
        2.0 * (1.0 - q).powf(3.0)
    } else {
        0.0
    };
    w * sigma
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
    hash: &SpatialHash3D<Vec<usize>>,
) -> Vec<Matrix3<f32>> {
    let mut deformations: Vec<Matrix3<f32>> = Vec::with_capacity(current_vertices.len());
    for i in 0..current_vertices.len() {
        let pos_i = current_vertices[i].position;
        let f_i = neighbor_sum(
            |j| {
                let x_ji = current_vertices[j].position - pos_i;
                let grad_w_ij = reference_gradients[i][j];
                let v_j = reference_volumes[j];
                v_j * x_ji * grad_w_ij.transpose()
            },
            hash,
            pos_i,
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
    hash: &SpatialHash3D<Vec<usize>>,
) -> Vec<Matrix3<f32>> {
    let mut deformations_star = Vec::with_capacity(current_vertices.len());
    for i in 0..current_vertices.len() {
        let r_i = rotations[i];
        let pos_i = current_vertices[i].position;
        let ref_pos_i = reference_vertices[i].position;
        let f_i_star = neighbor_sum(
            |j| {
                let x_ji = current_vertices[j].position - pos_i;
                let x_ji_ref = reference_vertices[j].position - ref_pos_i;
                let grad_w_ij_star = reference_gradients_star[i][j];
                let v_j = reference_volumes[j];
                v_j * (x_ji - r_i * x_ji_ref) * grad_w_ij_star.transpose()
            },
            hash,
            pos_i,
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

pub fn compute_forces(
    shape: &Cube,
    _kernel: &CubicKernel,
    hash: &SpatialHash3D<Vec<usize>>,
) -> Vec<Vector3<f32>> {
    let current = &shape.current;
    let initial_deformations = initial_deformation_tensor_estimate(
        &current.vertices,
        &shape.reference_gradients,
        &shape.L,
        &shape.reference_volumes,
        hash,
    );
    let rotations = compute_rotations(&initial_deformations);
    let grad_star = compute_grad_star(&shape.reference_gradients, &rotations, &shape.L);
    let deformations_star = deformation_tensor_star(
        &current.vertices,
        &shape.reference.vertices,
        &grad_star,
        &rotations,
        &shape.reference_volumes,
        hash,
    );
    let reference_volumes = &shape.reference_volumes;
    let stress_tensors = stress_tensor(&deformations_star, 1.0, 0.0);
    let mut forces = Vec::with_capacity(current.vertices.len());
    for i in 0..current.vertices.len() {
        let pos_i = current.vertices[i].position;
        let force = neighbor_sum(
            |j| {
                reference_volumes[i] * reference_volumes[j]
                    * (stress_tensors[i] * grad_star[i][j] - stress_tensors[j] * grad_star[j][i])
            },
            hash,
            pos_i,
            Vector3::zeros(),
        );
        forces.push(force);
    }
    forces
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hash() -> SpatialHash3D<Vec<usize>> {
        SpatialHash3D::new((8, 8, 8), || Vec::new(), 1.0)
    }

    #[test]
    fn smoketest_cube_new() {
        let mut hash = test_hash();
        let cube = CubeShape::new(3, 1.0, &mut hash);
        assert_eq!(cube.vertices.len(), 27);
        assert_eq!(cube.vertices.iter().filter(|v| v.renderable).count(), 26);
    }

    #[test]
    fn test_renderable() {
        let mut hash = test_hash();
        let cube = CubeShape::new(2, 1.0, &mut hash);
        let (vertices, indices) = cube.to_renderable();
        assert_eq!(vertices.len(), 8);
        assert_eq!(indices.len(), 36);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_renderable_n3() {
        let mut hash = test_hash();
        let cube = CubeShape::new(3, 1.0, &mut hash);
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
        let W = kernel.eval(&x_i, &x_j);
        assert_eq!(W.len(), 1);
        assert_eq!(W[0].len(), 1);
        assert_eq!(W[0][0], 0.0);
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
        let mut hash = test_hash();
        let mut cube = Cube::new(3, 1.0, &CubicKernel::new(1.0), &mut hash);
        hash.clear(&mut |cell| cell.clear());
        cube.current.update_hash(&mut hash);
        let deformations = initial_deformation_tensor_estimate(
            &cube.reference.vertices,
            &cube.reference_gradients,
            &cube.L,
            &cube.reference_volumes,
            &hash,
        );
        assert!(deformations[0].relative_eq(&Matrix3::identity(), 1e-6, 1e-6));
    }
    #[test]
    fn test_neighbor_sum() {
        let hash = SpatialHash3D::new((3, 3, 3), || Vec::new(), 1.0);
        let sum = neighbor_sum(|idx| idx, &hash, Vector3::new(0.0, 0.0, 0.0), 0);
        assert_eq!(sum, 0);
    }
}
