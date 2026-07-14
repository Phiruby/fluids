mod boundaries;
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

pub use boundaries::ContainerBoundary;
pub use fluids::FluidSim;
pub use shapes::*;

pub const FLUID_COLOR: [f32; 3] = [0.2, 0.5, 0.95];

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

impl FluidParticle {
    /// Per-instance layout so the render pipeline can read particles straight
    /// from the simulation buffer (position, `normal` as color, boundary flag).
    pub fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::offset_of!(FluidParticle, normal) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::offset_of!(FluidParticle, boundary) as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }

    /// Far-away static particle for padding the buffer to a workgroup multiple.
    /// `mass > 0` keeps its self-density nonzero (no division by zero).
    pub fn inert() -> Self {
        Self {
            position: [1000.0, 1000.0, 1000.0],
            mass: 1.0,
            boundary: 0,
            ..Default::default()
        }
    }
}

/// n x n x n cube of fluid particles
pub fn fluid_cube(n: usize, spacing: f32, origin: Vector3<f32>) -> Vec<FluidParticle> {
    let mut particles = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                particles.push(FluidParticle {
                    position: [
                        origin.x + i as f32 * spacing,
                        origin.y + j as f32 * spacing,
                        origin.z + k as f32 * spacing,
                    ],
                    normal: FLUID_COLOR,
                    mass: 1.0,
                    boundary: 1,
                    ..Default::default()
                });
            }
        }
    }
    particles
}

/// set each particle's rest_density (and density) from its rest configuration;
/// NOTE: fluid particles (`boundary == 1`) get their own initial density. boundary
/// particles instead take the fluid's rest density
pub fn assign_rest_densities(particles: &mut [FluidParticle], kernel: &CubicKernel) {
    let vertices: Vec<PhysicsVertex> = particles
        .iter()
        .map(|p| PhysicsVertex {
            position: Vector3::new(p.position[0], p.position[1], p.position[2]),
            velocity: Vector3::zeros(),
            renderable: false,
            mass: p.mass,
        })
        .collect();
    let positions: Vec<Vector3<f32>> = vertices.iter().map(|v| v.position).collect();
    let neighbors = precompute_neighbors(&positions, kernel);
    let densities = compute_densities(&vertices, kernel, &neighbors);

    let fluid_rest = particles
        .iter()
        .zip(&densities)
        .filter(|(p, _)| p.boundary == 1)
        .map(|(_, &d)| d)
        .fold(0.0_f32, f32::max);

    for (particle, &density) in particles.iter_mut().zip(&densities) {
        particle.density = density;
        particle.rest_density = if particle.boundary == 1 {
            density
        } else {
            fluid_rest
        };
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
        let sigma = 8.0 / (std::f32::consts::PI * self.h.powi(3));
        if q <= 0.5 {
            sigma * (6.0 * q.powi(3) - 6.0 * q.powi(2) + 1.0)
        } else if q <= 1.0 {
            sigma * 2.0 * (1.0 - q).powi(3)
        } else {
            0.0
        }
    }

    fn weight_gradient(&self, r: Vector3<f32>) -> Vector3<f32> {
        let r_norm = r.norm();
        if r_norm == 0.0 {
            return Vector3::zeros();
        }
        let q = r_norm / self.h;
        let sigma = 8.0 / (std::f32::consts::PI * self.h.powi(3));
        let dw_dq = if q <= 0.5 {
            sigma * (18.0 * q.powi(2) - 12.0 * q)
        } else if q <= 1.0 {
            sigma * -6.0 * (1.0 - q).powi(2)
        } else {
            0.0
        };
        // dq/dx = r / (h * ||r||)
        (dw_dq / (self.h * r_norm)) * r
    }

    /// Performs pairwise evaluation of the cubic spline kernel between the two.
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

pub fn kernel_weight(kernel: &CubicKernel, r_norm: f32) -> f32 {
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
