use crate::physics::{
    CubicKernel, PhysicsVertex, compute_densities, pairwise_displacement, precompute_neighbors,
};
use nalgebra::{Matrix3, Vector3};

#[derive(Debug)]
pub struct CubeShape {
    pub vertices: Vec<PhysicsVertex>,
    n: usize,
}

#[derive(Debug)]
pub struct Cube {
    pub reference: CubeShape,
    pub neighbors: Vec<Vec<usize>>,
    pub L: Vec<Matrix3<f32>>,
    pub reference_gradients: Vec<Vec<Vector3<f32>>>,
    pub reference_volumes: Vec<f32>,
    pub reference_kernel_weights: Vec<Vec<f32>>,
    pub current: CubeShape,
    pub kernel: CubicKernel,
    pub selected_face_index: Option<usize>,
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

        let reference_gradients =
            kernel.gradient(&reference_positions, &reference_positions, &neighbors);
        let reference_distances = pairwise_displacement(&reference.vertices, &reference.vertices);
        let mut L = Vec::with_capacity(reference.vertices.len());
        for i in 0..reference.vertices.len() {
            let mut L_i = Matrix3::zeros();
            for j in 0..reference.vertices.len() {
                // NOTE: this uses inverse of (154) instead of the provided formula (156) in https://arxiv.org/pdf/2009.06944
                L_i += reference_volumes[j]
                    * reference_distances[j][i]
                    * reference_gradients[i][j].transpose();
            }
            L.push(L_i.try_inverse().unwrap());
        }
        let reference_kernel_weights =
            kernel.eval(&reference_positions, &reference_positions, &neighbors);
        Self {
            reference,
            neighbors,
            reference_volumes,
            reference_kernel_weights,
            current: CubeShape::new(n, spacing),
            L,
            reference_gradients,
            kernel: kernel.clone(),
            selected_face_index: None,
        }
    }

    /// indices into the full compute particle buffer (not a compacted surface list).
    pub fn surface_indices(&self) -> Vec<u16> {
        self.current.surface_indices()
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
        }
    }

    pub fn new(n: usize, spacing: f32) -> Self {
        Self {
            vertices: Self::build_vertices(n, spacing),
            n,
        }
    }

    fn grid_index(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.n * self.n + j * self.n + k
    }

    // a=bottom-left, b=bottom-right, c=top-right, d=top-left (viewed from outside)
    fn push_quad(indices: &mut Vec<u16>, a: u16, b: u16, c: u16, d: u16, flip: bool) {
        if flip {
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        } else {
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    /// Surface triangle indices into the *full* particle grid (same order as the compute buffer).
    /// Use this when the mesh vertex buffer is the shared Particle buffer.
    pub fn surface_indices(&self) -> Vec<u16> {
        let n = self.n;
        let idx = |i: usize, j: usize, k: usize| self.grid_index(i, j, k) as u16;
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
        indices
    }
}
