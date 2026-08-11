mod objects;

use crate::load_wgsl;
use crate::physics::{MpmElasticParticle, MpmFluidParticle, MpmGridNode};

use wgpu::util::DeviceExt;

pub use objects::mpm_collider_bounds;

pub const MPM_N: u32 = 32;
pub const MPM_DX: f32 = 0.02;
pub const MPM_BOTTOM_LEFT: [f32; 3] = [0.0, 0.0, 0.0];

const N: u64 = MPM_N as u64;
const WG_SIZE: u32 = 256;

pub fn mpm_grid_bounds() -> ([f32; 3], [f32; 3]) {
    let min = MPM_BOTTOM_LEFT;
    let extent = MPM_N as f32 * MPM_DX;
    let max = [min[0] + extent, min[1] + extent, min[2] + extent];
    (min, max)
}

/// interpolate data from particles to grid nodes (two-pass: mass/momentum, then density stress).
/// two-pass idea from https://nialltl.neocities.org/articles/mpm_guide
struct P2G {
    fluid_mass_pipeline: wgpu::ComputePipeline,
    elastic_mass_pipeline: wgpu::ComputePipeline,
    fluid_stress_pipeline: wgpu::ComputePipeline,
    elastic_stress_pipeline: wgpu::ComputePipeline,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
            label: Some("simulation"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let fluid_mass_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("p2g_mass"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("particles_to_grid"),
            cache: Default::default(),
            compilation_options: Default::default(),
        });
        let elastic_mass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("p2g_elastic_mass"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("elastic_particles_to_grid"),
                cache: Default::default(),
                compilation_options: Default::default(),
            });
        let fluid_stress_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("p2g_stress"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("fluid_particles_to_grid_stress"),
                cache: Default::default(),
                compilation_options: Default::default(),
            });
        let elastic_stress_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("p2g_elastic_stress"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("elastic_particles_to_grid_stress"),
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
            fluid_mass_pipeline,
            elastic_mass_pipeline,
            fluid_stress_pipeline,
            elastic_stress_pipeline,
            layout,
            grid,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        elastic_particles: &wgpu::Buffer,
        num_particles: u32,
        num_elastic: u32,
    ) {
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: elastic_particles.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            if num_particles > 0 {
                pass.set_pipeline(&self.fluid_mass_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_particles.div_ceil(WG_SIZE), 1, 1);
            }
            if num_elastic > 0 {
                pass.set_pipeline(&self.elastic_mass_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_elastic.div_ceil(WG_SIZE), 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            if num_particles > 0 {
                pass.set_pipeline(&self.fluid_stress_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_particles.div_ceil(WG_SIZE), 1, 1);
            }
            if num_elastic > 0 {
                pass.set_pipeline(&self.elastic_stress_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_elastic.div_ceil(WG_SIZE), 1, 1);
            }
        }
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
        let n_nodes = (N * N * N) as u32;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_nodes.div_ceil(WG_SIZE), 1, 1);
    }
}

/// transfer from grid to particle
struct G2P {
    fluid_pipeline: wgpu::ComputePipeline,
    elastic_pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl G2P {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = load_wgsl!(device, "src/physics/mpm/g2p.wgsl");
        // let module = device.create_shader_module(wgpu::include_wgsl!("mpm/g2p.wgsl"));
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
            label: Some("g2p"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let fluid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("g2p"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("fluid_grid_to_particles"),
            cache: Default::default(),
            compilation_options: Default::default(),
        });
        let elastic_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("g2p_elastic"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("elastic_grid_to_particles"),
            cache: Default::default(),
            compilation_options: Default::default(),
        });
        Self {
            fluid_pipeline,
            elastic_pipeline,
            layout,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        elastic_particles: &wgpu::Buffer,
        grid: &wgpu::Buffer,
        collider: &wgpu::Buffer,
        num_particles: u32,
        num_elastic: u32,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: elastic_particles.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        if num_particles > 0 {
            pass.set_pipeline(&self.fluid_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_particles.div_ceil(WG_SIZE), 1, 1);
        }
        if num_elastic > 0 {
            pass.set_pipeline(&self.elastic_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_elastic.div_ceil(WG_SIZE), 1, 1);
        }
    }
}

fn particle_storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    particles: &[T],
) -> wgpu::Buffer {
    if particles.is_empty() {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_of::<T>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    } else {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(particles),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }
}

pub struct MpmFluidSim {
    p2g: P2G,
    grid_simulation: GridSimulation,
    g2p: G2P,
    particles: wgpu::Buffer,
    elastic_particles: wgpu::Buffer,
    num_particles: u32,
    num_elastic: u32,
}

impl MpmFluidSim {
    pub fn new(
        device: &wgpu::Device,
        particles: &[MpmFluidParticle],
        elastic_particles: &[MpmElasticParticle],
    ) -> Self {
        let num_particles = particles.len() as u32;
        let num_elastic = elastic_particles.len() as u32;
        let particles = particle_storage_buffer(device, "mpm particles", particles);
        let elastic_particles =
            particle_storage_buffer(device, "mpm elastic particles", elastic_particles);
        Self {
            p2g: P2G::new(device),
            grid_simulation: GridSimulation::new(device),
            g2p: G2P::new(device),
            particles,
            elastic_particles,
            num_particles,
            num_elastic,
        }
    }

    pub fn step(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.num_particles == 0 && self.num_elastic == 0 {
            return;
        }
        const SUBSTEPS: u32 = 8;
        for _ in 0..SUBSTEPS {
            self.substep(device, encoder);
        }
    }

    fn substep(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.p2g.grid, 0, None);
        self.p2g.compute(
            device,
            encoder,
            &self.particles,
            &self.elastic_particles,
            self.num_particles,
            self.num_elastic,
        );
        self.grid_simulation
            .compute(device, encoder, &self.p2g.grid);
        self.g2p.compute(
            device,
            encoder,
            &self.particles,
            &self.elastic_particles,
            &self.p2g.grid,
            &self.grid_simulation.collider_buffer,
            self.num_particles,
            self.num_elastic,
        );
    }

    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particles
    }

    pub fn elastic_particle_buffer(&self) -> &wgpu::Buffer {
        &self.elastic_particles
    }

    pub fn grid_buffer(&self) -> &wgpu::Buffer {
        &self.p2g.grid
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }

    pub fn num_elastic_particles(&self) -> u32 {
        self.num_elastic
    }

    pub fn num_grid_nodes(&self) -> u32 {
        (N * N * N) as u32
    }
}
