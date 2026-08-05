mod boundaries;
mod fluids;
mod solids;
use nalgebra::Vector3;
use std::ops::Add;

pub use boundaries::ContainerBoundary;
pub use fluids::FluidSim;
pub(crate) use fluids::PrefixSum;
pub use solids::{ElasticSolidSim, elastic_cube};

pub const FLUID_COLOR: [f32; 3] = [0.2, 0.5, 0.95];
pub const SOLID_COLOR: [f32; 3] = [0.85, 0.45, 0.25];

#[derive(Debug, Clone)]
#[allow(unused)]
pub struct PhysicsVertex {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub renderable: bool,
    pub mass: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
pub struct ElasticSolidParticle {
    pub position: [f32; 3],
    pub mass: f32,
    pub velocity: [f32; 3],
    pub rest_volume: f32,
    pub reference_position: [f32; 3],
    pub nneighbors: u32,
    pub color: [f32; 3],
    pub renderable: u32,
    pub neighbors: [u32; 27],
    pub pinned: u32,
    /// mat3x3, with 16 bytes padding
    pub L: [f32; 12],
}

impl Default for ElasticSolidParticle {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            mass: 1.0,
            velocity: [0.0; 3],
            rest_volume: 0.0,
            reference_position: [0.0; 3],
            nneighbors: 0,
            color: SOLID_COLOR,
            renderable: 1,
            neighbors: [0; 27],
            pinned: 0,
            L: [0.0; 12],
        }
    }
}

const _: () = assert!(std::mem::size_of::<ElasticSolidParticle>() == 224);

impl ElasticSolidParticle {
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
                    offset: std::mem::offset_of!(Self, color) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::offset_of!(Self, renderable) as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

pub struct ElasticSolid {
    pub particles: Vec<ElasticSolidParticle>,
    pub kernel_h: f32,
}

impl ElasticSolid {
    pub fn new(particles: Vec<ElasticSolidParticle>, kernel_h: f32) -> Self {
        Self {
            particles,
            kernel_h,
        }
    }

    /// make n x n x n cube
    pub fn cube(
        n: usize,
        spacing: f32,
        origin: Vector3<f32>,
        kernel: &CubicKernel,
        pin_top: bool,
    ) -> Self {
        let mut positions = Vec::with_capacity(n * n * n);
        let mut pinned_flags = Vec::with_capacity(n * n * n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    positions.push(Vector3::new(
                        origin.x + i as f32 * spacing,
                        origin.y + j as f32 * spacing,
                        origin.z + k as f32 * spacing,
                    ));
                    pinned_flags.push(pin_top && j == n - 1);
                }
            }
        }
        let neighbors = precompute_neighbors(&positions, kernel);
        let vertices: Vec<PhysicsVertex> = positions
            .iter()
            .map(|&p| PhysicsVertex {
                position: p,
                velocity: Vector3::zeros(),
                renderable: true,
                mass: 1.0,
            })
            .collect();
        let densities = compute_densities(&vertices, kernel, &neighbors);

        let mut particles = Vec::with_capacity(positions.len());
        for (i, &pos) in positions.iter().enumerate() {
            let mut nbrs = neighbors[i].clone();
            let nneighbors = nbrs.len().min(27) as u32;
            // TODO: truncation can be done to choose closest neighbors
            nbrs.truncate(27);
            nbrs.resize(27, 0);
            let density = densities[i].max(1e-8);
            particles.push(ElasticSolidParticle {
                position: pos.into(),
                mass: 1.0,
                velocity: [0.0; 3],
                rest_volume: 1.0 / density,
                reference_position: pos.into(),
                nneighbors,
                color: SOLID_COLOR,
                renderable: 1,
                neighbors: nbrs.into_iter().map(|j| j as u32).collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                pinned: pinned_flags[i] as u32,
                L: [0.0; 12],
            });
        }
        Self::new(particles, kernel.h)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
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

    pub fn inert() -> Self {
        Self {
            position: [1000.0, 1000.0, 1000.0],
            mass: 1.0,
            density: 1.0,
            rest_density: 1.0,
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

pub fn lattice_rest_density(spacing: f32, mass: f32, kernel: &CubicKernel) -> f32 {
    let reach = (kernel.h / spacing).ceil() as i32;
    let mut density = 0.0;
    for i in -reach..=reach {
        for j in -reach..=reach {
            for k in -reach..=reach {
                let r = Vector3::new(i as f32, j as f32, k as f32) * spacing;
                density += mass * kernel.weight(r.norm());
            }
        }
    }
    density
}

/// all particles given the same rest density (and initial density as well):
/// helped stabilize sim since boundary fluid particles naturally are less densed,
/// which was causing all sorts of problems
pub fn assign_rest_densities(
    particles: &mut [FluidParticle],
    boundary_particles: &mut [FluidParticle],
    spacing: f32,
    kernel: &CubicKernel,
) {
    let mass = particles.first().map(|p| p.mass).unwrap_or(1.0);
    let rest_density = lattice_rest_density(spacing, mass, kernel);

    for p in particles.iter_mut() {
        p.rest_density = rest_density;
        p.density = rest_density;
    }
    for p in boundary_particles.iter_mut() {
        p.rest_density = rest_density;
        p.density = rest_density;
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
                vertices[j].mass * kernel.weight(r.norm())
            },
            &neighbors[i],
            0.0,
        );
        densities[i] = density;
    }
    densities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_kernel() {
        let kernel = CubicKernel::new(1.0);
        let x_i = vec![Vector3::new(0.0, 0.0, 0.0)];
        let x_j = vec![Vector3::new(1.0, 0.0, 0.0)];
        let neighbors = precompute_neighbors(&x_j, &kernel);
        let w = kernel.weight((x_i[0] - x_j[0]).norm());
        assert_eq!(neighbors.len(), 1);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn test_neighbor_sum() {
        let neighbors = vec![1, 3, 5];
        let sum = neighbor_sum(|idx| idx, &neighbors, 0);
        assert_eq!(sum, 9);
    }
}
