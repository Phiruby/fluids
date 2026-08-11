use crate::graphics::BufferDebug;
use crate::physics::FluidParticle;
use crate::physics::pipelines::{PrefixSum, SpatialHashGPU, pipeline};
use wgpu::util::DeviceExt;
const WG_SIZE: u32 = 256;
const RADIX_BUCKETS: u32 = 16;
const RADIX_PASSES: usize = 8;
const HASHTABLE_SIZE: u64 = 8192;

pub struct ComputeDensities {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    debug_buffer: wgpu::Buffer,
    debugger: BufferDebug,
}

impl ComputeDensities {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("quantities.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute densities"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density_computation"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let compute_densities = pipeline(device, &pipeline_layout, &module, "compute_quantities");
        let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug buffer"),
            size: 1024 * 1024 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            pipeline: compute_densities,
            layout,
            debug_buffer,
            debugger: BufferDebug::new(device, 1024 * 1024 * 4),
        }
    }
    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        hash_offsets: &wgpu::Buffer,
        num_particles: u32,
    ) {
        // TODO: does this need to be done every compute call?
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hash_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.debug_buffer.as_entire_binding(),
                },
            ],
        });
        let ndispatches = num_particles / 256;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(ndispatches, 1, 1);
        }
    }
}

struct Simulation {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    debug_buffer: wgpu::Buffer,
    debugger: BufferDebug,
}
impl Simulation {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("simulation.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("run simulation"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // debug buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simulation"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let simulate = pipeline(device, &pipeline_layout, &module, "simulate");
        let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug buffer"),
            size: 1024 * 1024 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            pipeline: simulate,
            layout,
            debug_buffer,
            debugger: BufferDebug::new(device, 1024 * 1024 * 4),
        }
    }

    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input_particles: &wgpu::Buffer,
        output_particles: &wgpu::Buffer,
        hash_offsets: &wgpu::Buffer,
        num_particles: u32,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hash_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.debug_buffer.as_entire_binding(),
                },
            ],
        });
        let ndispatches = num_particles / 256;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(ndispatches, 1, 1);
        }
    }
}

/// Run SPH step with ping-ponging buffers
pub struct FluidSim {
    spatial_hash: SpatialHashGPU,
    compute_densities: ComputeDensities,
    simulation: Simulation,
    particles: [wgpu::Buffer; 2],
    num_particles: u32,
    current: usize,
    debugger: BufferDebug,
}

impl FluidSim {
    pub fn new(device: &wgpu::Device, particles: &[FluidParticle]) -> Self {
        let num_particles = particles.len() as u32;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let buffers = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("fluid particles {i}")),
                contents: bytemuck::cast_slice(particles),
                usage,
            })
        });
        Self {
            spatial_hash: SpatialHashGPU::new::<FluidParticle>(device, num_particles),
            compute_densities: ComputeDensities::new(device),
            simulation: Simulation::new(device),
            particles: buffers,
            num_particles,
            current: 0,
            debugger: BufferDebug::new(
                device,
                num_particles as u64 * std::mem::size_of::<FluidParticle>() as u64,
            ),
        }
    }

    /// (sort, densities, integrate) and swaps the buffers
    pub fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.num_particles == 0 {
            return;
        }
        let src = self.current;
        let dst = 1 - self.current;
        let hash_offsets = &self.spatial_hash.hash_offsets_buf;

        self.spatial_hash.sort(
            device,
            queue,
            encoder,
            &self.particles[src],
            self.num_particles,
        );
        self.compute_densities.compute(
            device,
            queue,
            encoder,
            &self.particles[src],
            hash_offsets,
            self.num_particles,
        );
        self.simulation.run(
            device,
            queue,
            encoder,
            &self.particles[src],
            &self.particles[dst],
            hash_offsets,
            self.num_particles,
        );
        self.current = dst;
    }

    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particles[self.current]
    }

    pub fn hash_offsets_buffer(&self) -> &wgpu::Buffer {
        &self.spatial_hash.hash_offsets_buf
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}
