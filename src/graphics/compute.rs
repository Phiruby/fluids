use crate::graphics::{create_pipeline, PipelineInfo, PipelineState};
use crate::Vertex;
use cgmath::Point3;
use wgpu::util::DeviceExt;
use bytemuck;
use nalgebra::Vector3;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
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
        position: [f32; 3],
        velocity: [f32; 3],
        mass: f32,
        density: f32,
        renderable: u32,
        nneighbors: u32,
        neighbors: [u32; 27],
    ) -> Self {
        Self { 
            position, 
            _pad: 0, 
            velocity, 
            _pad1: 0, 
            mass, 
            density, 
            renderable, 
            nneighbors, 
            neighbors, 
            _pad3: [0; 3], 
            L: [0.0; 9], 
            _pad4: [0; 3],
        }
    }
}

// struct Cube {
//     reference: CubeShape,
//     neighbors: Vec<Vec<usize>>,
//     L: Vec<Matrix3<f32>>,
//     reference_gradients: Vec<Vec<Vector3<f32>>>,
//     reference_volumes: Vec<f32>,
//     reference_kernel_weights: Vec<Vec<f32>>,
//     pub current: CubeShape,
//     pub cube_faces: [CubeFace; 6],
// }

pub struct PhysicsPipeline {
    pipeline: wgpu::ComputePipeline,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl PhysicsPipeline {
    pub fn new(
        device: &wgpu::Device,
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

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input buffer"),
            contents: bytemuck::cast_slice(&[Particle::new([0.0; 3], [0.0; 3], 1.0, 1.0, 0, 0, [0; 27])]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Output buffer"),
            contents: bytemuck::cast_slice(&[Particle::new([0.0; 3], [0.0; 3], 1.0, 1.0, 0, 0, [0; 27])]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let reference_gradients_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reference gradients buffer"),
            contents: bytemuck::cast_slice(&[ReferenceGradient::new([0.0; 3])]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let current_particles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Current particles buffer"),
            contents: bytemuck::cast_slice(&[Particle::new([0.0; 3], [0.0; 3], 1.0, 1.0, 0, 0, [0; 27])]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        dbg!(&pipeline.get_bind_group_layout(0));

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
