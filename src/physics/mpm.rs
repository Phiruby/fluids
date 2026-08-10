mod objects;

use crate::physics::{MpmFluidParticle, MpmGridNode};
use wgpu::util::DeviceExt;

pub use objects::mpm_collider_bounds;

pub const MPM_N: u32 = 192;
pub const MPM_DX: f32 = 0.01;
pub const MPM_BOTTOM_LEFT: [f32; 3] = [0.0, 0.0, 0.0];

const N: u64 = MPM_N as u64;
const WG_SIZE: u32 = 256;

pub fn mpm_grid_bounds() -> ([f32; 3], [f32; 3]) {
    let min = MPM_BOTTOM_LEFT;
    let extent = MPM_N as f32 * MPM_DX;
    let max = [min[0] + extent, min[1] + extent, min[2] + MPM_DX];
    (min, max)
}

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
            size: size_of::<MpmGridNode>() as u64 * N * N,
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
    collider_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColliderUniform {
    min: [f32; 3],
    _pad0: f32,
    max: [f32; 3],
    _pad1: f32,
}

impl GridSimulation {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("mpm/grid_simulation.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("run simulation"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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
            label: Some("grid_sim"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("update_momentum"),
            cache: Default::default(),
            compilation_options: Default::default(),
        });
        let (min, max) = objects::mpm_collider_bounds();
        let collider_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mpm collider"),
            contents: bytemuck::bytes_of(&ColliderUniform {
                min,
                _pad0: 0.0,
                max,
                _pad1: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            pipeline,
            layout,
            collider_buffer,
        }
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
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.collider_buffer.as_entire_binding(),
                },
            ],
        });
        let n_nodes = (N * N) as u32;
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
        collider: &wgpu::Buffer,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: collider.as_entire_binding(),
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
        const SUBSTEPS: u32 = 8;
        for _ in 0..SUBSTEPS {
            self.substep(device, encoder);
        }
    }

    fn substep(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.p2g
            .compute(device, encoder, &self.particles, self.num_particles);
        self.grid_simulation
            .compute(device, encoder, &self.p2g.grid);
        self.g2p.compute(
            device,
            encoder,
            &self.particles,
            &self.p2g.grid,
            &self.grid_simulation.collider_buffer,
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

#[cfg(test)]
mod mpm_step_tests {
    use super::*;
    use crate::graphics::BufferDebug;
    use crate::physics::mpm_fluid_block;
    use nalgebra::Vector3;

    fn gpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device")
    }

    #[test]
    fn particles_move_after_step() {
        let (device, queue) = gpu();
        let particles = mpm_fluid_block(8, 8, 0.02, Vector3::new(0.2, 0.6, 0.0));
        let before = particles[0].position;
        let sim = MpmFluidSim::new(&device, &particles);
        let dbg = BufferDebug::new(&device, sim.particles.size());
        let grid_dbg = BufferDebug::new(&device, sim.p2g.grid.size());

        let mut enc = device.create_command_encoder(&Default::default());
        sim.step(&device, &mut enc);
        queue.submit(Some(enc.finish()));

        let after =
            dbg.read::<MpmFluidParticle>(&device, &queue, &sim.particles, particles.len() as u64);
        let grid = grid_dbg.read::<MpmGridNode>(&device, &queue, &sim.p2g.grid, (N * N) as u64);
        let nonzero_mass = grid.iter().filter(|n| n.mass > 0.0).count();

        assert!(nonzero_mass > 0, "P2G deposited no mass");
        assert!(
            after[0].position != before || after[0].velocity != [0.0; 3],
            "particle did not move or gain velocity"
        );
        assert_eq!(after[0].position[2], 0.0, "2D sim should keep z=0");
    }
}
