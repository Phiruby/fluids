use crate::graphics::{create_pipeline, PipelineInfo, PipelineState};
use crate::Vertex;
use cgmath::Point3;
use wgpu::util::DeviceExt;
use bytemuck;
use nalgebra::Vector3;
use nalgebra::Matrix3;
use crate::physics::Cube;
use std::ops::Add;
use crate::physics::CubicKernel;
use crate::physics::PhysicsVertex;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct Particle {
    position: [f32; 3],
    _pad: u32,
    velocity: [f32; 3],
    _pad1: u32,
    mass: f32,
    density: f32,
    renderable: u32,
    nneighbors: u32,
    neighbors: [u32; 27],
    _pad3: [u32; 3],
    L: [f32; 9],
    _pad4: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ReferenceGradient {
    gradient: [f32; 3],
    _pad: [u32; 3],
}

impl ReferenceGradient {
    fn new(
        gradient: [f32; 3],
    ) -> Self {
        Self { gradient, _pad: [0; 3] }
    }
}

impl Particle {
    fn new(
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
        neighbors.resize(27, 0);
        let neighbors = neighbors.into_iter().map(|n| n as u32).collect::<Vec<_>>();
        let neighbors_array = neighbors.try_into().expect("Maximum 27 neighbors");
        Self { 
            position: position.data.0[0], 
            _pad: 0, 
            velocity: velocity.data.0[0], 
            _pad1: 0, 
            mass, 
            density, 
            renderable, 
            nneighbors, 
            neighbors: neighbors_array, 
            _pad3: [0; 3], 
            L: L.data.0.into_iter().flatten().collect::<Vec<f32>>().try_into().expect("Maximum 9 elements"), 
            _pad4: [0; 3],
        }
    }
}

pub struct PhysicsPipeline {
    pipeline: wgpu::ComputePipeline,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl PhysicsPipeline {
    pub fn new(
        device: &wgpu::Device,
        cube: Cube
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../compute.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Physics pipeline"),
            module: &shader,
            entry_point: None,
            layout: None, // TODO: do we need global bind for compute?
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let reference_densities = compute_densities(&cube.reference.vertices, &cube.kernel, &cube.neighbors);

        let reference_particles = cube.reference
            .vertices
            .iter()
            .zip(cube.neighbors.iter())
            .zip(reference_densities.iter())
            .zip(cube.L.iter())
            .map(|(((vertex, neighbors), density), L)| Particle::new(
                vertex.position,
                vertex.velocity,
                vertex.mass,
                *density,
                0,
                neighbors.len() as u32,
                neighbors.to_owned(),
                *L,
            ))
            .collect::<Vec<_>>();

        let current_particles = cube.current
            .vertices
            .iter()
            .zip(cube.neighbors.iter())
            .zip(reference_densities.iter())
            .zip(cube.L.iter())
            .map(|(((vertex, neighbors), density), L)| Particle::new(
                vertex.position,
                vertex.velocity, vertex.mass, *density, 0, neighbors.len() as u32, neighbors.to_owned(), *L))
            .collect::<Vec<_>>();
        
        let reference_gradients: Vec<ReferenceGradient> = cube.reference_gradients
            .into_iter()
            .flatten()
            .map(|gradient| ReferenceGradient::new(gradient.data.0[0]))
            .collect();
        
        let reference_kernel_weights: Vec<f32> = cube.reference_kernel_weights
            .into_iter()
            .flatten()
            .collect();

        let out = vec![Particle::default(); current_particles.len()];

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input buffer"),
            contents: bytemuck::cast_slice(&reference_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Output buffer"),
            contents: bytemuck::cast_slice(&out),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let reference_gradients_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reference gradients buffer"),
            contents: bytemuck::cast_slice(&reference_gradients),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let reference_kernel_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reference kernel weights buffer"),
            contents: bytemuck::cast_slice(&reference_kernel_weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let current_particles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Current particles buffer"),
            contents: bytemuck::cast_slice(&current_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current_particles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reference_gradients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: reference_kernel_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        Self { pipeline, input_buffer, output_buffer, bind_group }
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        start: Point3<f32>,
        end: Point3<f32>,
        color: [f32; 3],
    ) {
       todo!();
    }

    pub fn compute(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(0, 1, 1);
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

fn kernel_weight(kernel: &CubicKernel, r_norm: f32) -> f32 {
    kernel.weight(r_norm)
}
