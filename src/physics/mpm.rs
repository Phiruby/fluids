use crate::physics::{MpmFluidParticle, MpmGridNode};
use wgpu::util::DeviceExt;

/// keep this in sync with @p2g.wgsl
const N: u64 = 32;
const WG_SIZE: u32 = 256;

/// interpolate data from particles to grid nodes
struct P2G {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    grid: wgpu::Buffer,
}

impl P2G {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("mpm/p2g.wgsl"));
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
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("p2g"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("particles_to_grid"),
            cache: Default::default(),
            compilation_options: Default::default(),
        });
        let grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm grid"),
            size: size_of::<MpmGridNode>() as u64 * N * N * N,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            grid,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        num_particles: u32,
    ) {
        encoder.clear_buffer(&self.grid, 0, None);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p2g"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_particles.div_ceil(WG_SIZE), 1, 1);
    }
}

/// compute forces on eulerian grid and update velocity
struct GridSimulation {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl GridSimulation {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("mpm/grid_simulation.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("run simulation"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simulation"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("grid_sim"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("update_momentum"),
            cache: Default::default(),
            compilation_options: Default::default(),
        });
        Self { pipeline, layout }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        grid: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_sim"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid.as_entire_binding(),
            }],
        });
        // TODO: we should make subpass to filter nodes with 0 mass
        let n_nodes = (N * N * N) as u32;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_nodes.div_ceil(WG_SIZE), 1, 1);
    }
}

/// transfer from grid to particle
struct G2P {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl G2P {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("mpm/g2p.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("g2p"),
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
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("g2p"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("g2p"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("grid_to_particles"),
            cache: Default::default(),
            compilation_options: Default::default(),
        });
        Self { pipeline, layout }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        grid: &wgpu::Buffer,
        num_particles: u32,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g2p"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_particles.div_ceil(WG_SIZE), 1, 1);
    }
}

pub struct MpmFluidSim {
    p2g: P2G,
    grid_simulation: GridSimulation,
    g2p: G2P,
    particles: wgpu::Buffer,
    num_particles: u32,
}

impl MpmFluidSim {
    pub fn new(device: &wgpu::Device, particles: &[MpmFluidParticle]) -> Self {
        let num_particles = particles.len() as u32;
        let particles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mpm particles"),
            contents: bytemuck::cast_slice(particles),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            p2g: P2G::new(device),
            grid_simulation: GridSimulation::new(device),
            g2p: G2P::new(device),
            particles,
            num_particles,
        }
    }

    pub fn step(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.num_particles == 0 {
            return;
        }
        self.p2g
            .compute(device, encoder, &self.particles, self.num_particles);
        self.grid_simulation
            .compute(device, encoder, &self.p2g.grid);
        self.g2p.compute(
            device,
            encoder,
            &self.particles,
            &self.p2g.grid,
            self.num_particles,
        );
    }

    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particles
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}
