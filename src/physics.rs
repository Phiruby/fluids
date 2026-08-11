mod mpm;
mod pipelines;
mod sph;
use nalgebra::Vector3;
use std::ops::Add;

pub use mpm::{MpmFluidSim, mpm_collider_bounds, mpm_grid_bounds};
pub(crate) use pipelines::PrefixSum;
pub use sph::boundaries::ContainerBoundary;
pub use sph::fluids::FluidSim;

pub const FLUID_COLOR: [f32; 3] = [0.2, 0.5, 0.95];

pub trait GpuParticle {
    /// this should return fields as used in wgsl
    fn wgsl_fields() -> &'static str;
}

/// for particles that can be drawn with the instanced cube pipeline
pub trait RenderParticle: GpuParticle {
    fn instance_layout() -> wgpu::VertexBufferLayout<'static>;
    fn instance_wgsl_fields() -> &'static str;
    fn vs_main_body() -> &'static str;
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

impl GpuParticle for FluidParticle {
    fn wgsl_fields() -> &'static str {
        r#"
        position: vec3<f32>,
        velocity: vec3<f32>,
        normal: vec3<f32>,
        density: f32,
        rest_density: f32,
        mass: f32,
        cell_hash: u32,
        boundary: u32,
        "#
    }
}

impl RenderParticle for FluidParticle {
    fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
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

    fn instance_wgsl_fields() -> &'static str {
        r#"
        @location(2) instance_position: vec3<f32>,
        @location(3) color: vec3<f32>,
        @location(4) boundary: u32,
        "#
    }

    fn vs_main_body() -> &'static str {
        r#"
        if input.boundary == 0u { return culled(); }
        return particle_out(input, input.color, 1.0);
        "#
    }
}

impl FluidParticle {
    /// Just to pad the buffers
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

/// column major 3x3 (with 16 byte padding)
pub type WgslMat3 = [[f32; 4]; 3];

fn wgsl_mat3_identity() -> WgslMat3 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MpmFluidParticle {
    pub deformation_gradient: WgslMat3,
    pub b: WgslMat3,
    pub position: [f32; 3],
    _pad0: f32,
    pub velocity: [f32; 3],
    /// computed similar to sph (sum w_i * m_i)
    pub rest_density: f32,
    pub mass: f32,
    _pad1: [f32; 3],
}

impl Default for MpmFluidParticle {
    fn default() -> Self {
        Self {
            deformation_gradient: wgsl_mat3_identity(),
            b: [[0.0; 4]; 3],
            position: [0.0; 3],
            _pad0: 0.0,
            velocity: [0.0; 3],
            rest_density: 0.0,
            mass: 0.0,
            _pad1: [0.0; 3],
        }
    }
}

impl GpuParticle for MpmFluidParticle {
    fn wgsl_fields() -> &'static str {
        r#"
        deformation_gradient: mat3x3<f32>,
        B: mat3x3<f32>,
        position: vec3<f32>,
        velocity: vec3<f32>,
        rest_density: f32,
        mass: f32,
        "#
    }
}

impl RenderParticle for MpmFluidParticle {
    fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: std::mem::offset_of!(MpmFluidParticle, position) as wgpu::BufferAddress,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }

    fn instance_wgsl_fields() -> &'static str {
        r#"
        @location(2) instance_position: vec3<f32>,
        "#
    }

    fn vs_main_body() -> &'static str {
        r#"
        var out = particle_out(input, vec3<f32>(1.0, 1.0, 1.0), 1.0);
        out.world_normal = vec3<f32>(0.0);
        return out;
        "#
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MpmElasticParticle {
    pub elastic_deformation_gradient: WgslMat3,
    pub b: WgslMat3,
    pub position: [f32; 3],
    _pad0: f32,
    pub velocity: [f32; 3],
    pub plastic_deformation_determinant: f32,
    pub rest_volume: f32,
    pub mass: f32,
    _pad1: [f32; 2],
}

impl Default for MpmElasticParticle {
    fn default() -> Self {
        Self {
            elastic_deformation_gradient: wgsl_mat3_identity(),
            b: [[0.0; 4]; 3],
            position: [0.0; 3],
            _pad0: 0.0,
            velocity: [0.0; 3],
            plastic_deformation_determinant: 1.0,
            rest_volume: 0.0,
            mass: 0.0,
            _pad1: [0.0; 2],
        }
    }
}

impl GpuParticle for MpmElasticParticle {
    fn wgsl_fields() -> &'static str {
        r#"
        elastic_deformation_gradient: mat3x3<f32>,
        B: mat3x3<f32>,
        position: vec3<f32>,
        velocity: vec3<f32>,
        plastic_deformation_determinant: f32,
        rest_volume: f32,
        mass: f32,
        "#
    }
}

impl RenderParticle for MpmElasticParticle {
    fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: std::mem::offset_of!(MpmElasticParticle, position) as wgpu::BufferAddress,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }

    fn instance_wgsl_fields() -> &'static str {
        r#"
        @location(2) instance_position: vec3<f32>,
        "#
    }

    fn vs_main_body() -> &'static str {
        r#"
        var out = particle_out(input, vec3<f32>(0.85, 0.55, 0.25), 1.0);
        out.world_normal = vec3<f32>(0.0);
        return out;
        "#
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MpmGridNode {
    pub momentum_x: f32,
    pub momentum_y: f32,
    pub momentum_z: f32,
    pub mass: f32,
    pub force_x: f32,
    pub force_y: f32,
    pub force_z: f32,
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

fn mpm_quadratic_kernel(r: f32) -> f32 {
    let ar = r.abs();
    if ar < 0.5 {
        0.75 - ar * ar
    } else if ar < 1.5 {
        let t = 1.5 - ar;
        0.5 * t * t
    } else {
        0.0
    }
}

/// we want to compute rest density of particles in the grid. to do this, we
/// want the particle's neighborhood to be filled. so we construct n x n x n grid, compute density with
/// rho = \sum_{i} w_i * m_i (where we sum over grid nodes). we take rho of the center particle in the grid
/// to be the representative rest density (since its neighborhood is full)
pub fn mpm_lattice_rest_density(spacing: f32, mass: f32, dx: f32) -> f32 {
    let n = 7usize;
    let origin = 0.0;
    let mut grid: std::collections::HashMap<(i32, i32, i32), f32> =
        std::collections::HashMap::new();
    let mut positions = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                positions.push([
                    origin + i as f32 * spacing,
                    origin + j as f32 * spacing,
                    origin + k as f32 * spacing,
                ]);
            }
        }
    }
    for p in &positions {
        let base_x = ((p[0] / dx) - 0.5).floor() as i32;
        let base_y = ((p[1] / dx) - 0.5).floor() as i32;
        let base_z = ((p[2] / dx) - 0.5).floor() as i32;
        for di in 0..3 {
            for dj in 0..3 {
                for dk in 0..3 {
                    let cx = base_x + di;
                    let cy = base_y + dj;
                    let cz = base_z + dk;
                    let rx = (cx as f32 * dx - p[0]) / dx;
                    let ry = (cy as f32 * dx - p[1]) / dx;
                    let rz = (cz as f32 * dx - p[2]) / dx;
                    let w = mpm_quadratic_kernel(rx)
                        * mpm_quadratic_kernel(ry)
                        * mpm_quadratic_kernel(rz);
                    *grid.entry((cx, cy, cz)).or_insert(0.0) += w * mass;
                }
            }
        }
    }
    let p = positions[n * n * n / 2];
    let base_x = ((p[0] / dx) - 0.5).floor() as i32;
    let base_y = ((p[1] / dx) - 0.5).floor() as i32;
    let base_z = ((p[2] / dx) - 0.5).floor() as i32;
    let mut density = 0.0;
    for di in 0..3 {
        for dj in 0..3 {
            for dk in 0..3 {
                let cx = base_x + di;
                let cy = base_y + dj;
                let cz = base_z + dk;
                let rx = (cx as f32 * dx - p[0]) / dx;
                let ry = (cy as f32 * dx - p[1]) / dx;
                let rz = (cz as f32 * dx - p[2]) / dx;
                let w = mpm_quadratic_kernel(rx)
                    * mpm_quadratic_kernel(ry)
                    * mpm_quadratic_kernel(rz);
                density += grid.get(&(cx, cy, cz)).copied().unwrap_or(0.0) * w;
            }
        }
    }
    density
}

pub fn mpm_lattice_rest_volume(spacing: f32, mass: f32, dx: f32) -> f32 {
    let density = mpm_lattice_rest_density(spacing, mass, dx);
    (mass / density) * dx * dx * dx
}

pub fn mpm_fluid_block(
    nx: usize,
    ny: usize,
    nz: usize,
    spacing: f32,
    origin: Vector3<f32>,
) -> Vec<MpmFluidParticle> {
    let rest_volume = spacing * spacing * spacing;
    let density = 1000.0;
    let mass = density * rest_volume;
    let rest_density = mpm_lattice_rest_density(spacing, mass, mpm::MPM_DX);
    let mut particles = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                particles.push(MpmFluidParticle {
                    deformation_gradient: wgsl_mat3_identity(),
                    b: [[0.0; 4]; 3],
                    position: [
                        origin.x + i as f32 * spacing,
                        origin.y + j as f32 * spacing,
                        origin.z + k as f32 * spacing,
                    ],
                    rest_density,
                    mass,
                    ..Default::default()
                });
            }
        }
    }
    particles
}

pub fn mpm_elastic_block(
    nx: usize,
    ny: usize,
    nz: usize,
    spacing: f32,
    origin: Vector3<f32>,
) -> Vec<MpmElasticParticle> {
    let material_density = 400.0;
    let mass = material_density * spacing * spacing * spacing;
    let rest_volume = mpm_lattice_rest_volume(spacing, mass, mpm::MPM_DX);
    let mut particles = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                particles.push(MpmElasticParticle {
                    elastic_deformation_gradient: wgsl_mat3_identity(),
                    b: [[0.0; 4]; 3],
                    position: [
                        origin.x + i as f32 * spacing,
                        origin.y + j as f32 * spacing,
                        origin.z + k as f32 * spacing,
                    ],
                    plastic_deformation_determinant: 1.0,
                    rest_volume,
                    mass,
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
