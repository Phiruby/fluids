mod fluids;
mod shapes;
use crate::graphics::Vertex;
use crate::{MouseState, Ray};
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::Bvh;
use bvh::ray::Ray as BvhRay;
use nalgebra::{Matrix3, Point3, Vector3};
use std::collections::HashSet;
use std::ops::Add;

pub use fluids::FluidSim;
pub use shapes::*;

#[derive(Debug, Clone)]
#[allow(unused)]
pub struct PhysicsVertex {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub renderable: bool,
    pub mass: f32,
}

impl From<PhysicsVertex> for Vertex {
    fn from(vertex: PhysicsVertex) -> Self {
        Vertex {
            // hackery to access vector from position: https://docs.rs/nalgebra/latest/nalgebra/base/type.Vector3.html
            position: vertex.position.data.0[0],
            color: [0.5, 0.0, 0.5],
            normal: [0.0, 0.0, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FluidParticle {
    pub position: [f32; 3],
    _pad0: u32,
    pub velocity: [f32; 3],
    _pad1: u32,
    pub normal: [f32; 3],
    pub density: f32,
    pub rest_density: f32,
    pub mass: f32,
    pub cell_hash: u32,
    pub boundary: u32,
}

impl Default for FluidParticle {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            _pad0: 0,
            velocity: [0.0; 3],
            _pad1: 0,
            normal: [0.0; 3],
            density: 0.0,
            rest_density: 0.0,
            mass: 0.0,
            cell_hash: 0,
            boundary: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ElasticSolidParticle {
    pub position: [f32; 3],
    _pad0: u32,
    pub velocity: [f32; 3],
    _pad1: u32,
    pub normal: [f32; 3],
    pub density: f32,
    pub mass: f32,
    pub renderable: u32,
    pub nneighbors: u32,
    pub neighbors: [u32; 27],
    _pad_l: [u32; 2],
    pub L: [f32; 12],
}

impl Default for ElasticSolidParticle {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            _pad0: 0,
            velocity: [0.0; 3],
            _pad1: 0,
            normal: [0.0; 3],
            density: 0.0,
            mass: 0.0,
            renderable: 0,
            nneighbors: 0,
            neighbors: [0; 27],
            _pad_l: [0; 2],
            L: [0.0; 12],
        }
    }
}

impl ElasticSolidParticle {
    pub fn new(
        position: Vector3<f32>,
        velocity: Vector3<f32>,
        mass: f32,
        density: f32,
        renderable: u32,
        nneighbors: u32,
        mut neighbors: Vec<usize>,
        L: Matrix3<f32>,
    ) -> Self {
        assert!(neighbors.len() <= 27, "Maximum 27 neighbors");
        assert!(mass > 0.0, "Mass must be positive");
        assert!(density > 0.0, "Density must be positive");
        neighbors.resize(27, 0);
        let neighbors = neighbors.into_iter().map(|n| n as u32).collect::<Vec<_>>();
        let neighbors_array = neighbors.try_into().expect("Maximum 27 neighbors");
        Self {
            position: position.data.0[0],
            _pad0: 0,
            velocity: velocity.data.0[0],
            _pad1: 0,
            normal: [0.72, 0.45, 0.95],
            density,
            mass,
            renderable,
            nneighbors,
            neighbors: neighbors_array,
            _pad_l: [0; 2],
            L: mat3_to_gpu(L),
        }
    }
}

pub fn neighbor_sum<F, T>(delta: F, neighbors: &[usize], zero: T) -> T
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

pub fn precompute_neighbors(positions: &[Vector3<f32>], kernel: &CubicKernel) -> Vec<Vec<usize>> {
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

    pub fn weight(&self, r_norm: f32) -> f32 {
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
/// Eq (11) in https://arxiv.org/pdf/2009.06944
pub fn compute_densities(
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

pub fn pairwise_displacement(x: &[PhysicsVertex], y: &[PhysicsVertex]) -> Vec<Vec<Vector3<f32>>> {
    let mut displacements = vec![vec![Vector3::new(0.0, 0.0, 0.0); y.len()]; x.len()];
    for i in 0..x.len() {
        for j in 0..y.len() {
            displacements[i][j] = x[i].position - y[j].position;
        }
    }
    displacements
}

fn mat3_to_gpu(m: Matrix3<f32>) -> [f32; 12] {
    let cols = m.data.0;
    [
        cols[0][0], cols[0][1], cols[0][2], 0.0, cols[1][0], cols[1][1], cols[1][2], 0.0,
        cols[2][0], cols[2][1], cols[2][2], 0.0,
    ]
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
    fn test_neighbor_sum() {
        let neighbors = vec![1, 3, 5];
        let sum = neighbor_sum(|idx| idx, &neighbors, 0);
        assert_eq!(sum, 9);
    }
}
