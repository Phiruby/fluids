use crate::physics::{CubicKernel, ElasticSolid, ElasticSolidParticle};
use wgpu::util::DeviceExt;

const WG: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    kernel_h: f32,
    dt: f32,
    young: f32,
    poisson: f32,
    floor_y: f32,
    _pad: [f32; 3],
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        module,
        entry_point: Some(entry),
        layout: Some(layout),
        compilation_options: Default::default(),
        cache: Default::default(),
    })
}

/// pipeline for elastic solid: [1] precompute L_i, [2] SPH simulation step; only [2] done each step
pub struct ElasticSolidSim {
    precompute: wgpu::ComputePipeline,
    simulate: wgpu::ComputePipeline,
    precompute_layout: wgpu::BindGroupLayout,
    simulate_layout: wgpu::BindGroupLayout,
    particles: [wgpu::Buffer; 2],
    params_buf: wgpu::Buffer,
    kernel_h_buf: wgpu::Buffer,
    num_particles: u32,
    current: usize,
}

impl ElasticSolidSim {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, solid: &ElasticSolid) -> Self {
        let num_particles = solid.particles.len() as u32;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let particles = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("elastic particles {i}")),
                contents: bytemuck::cast_slice(&solid.particles),
                usage,
            })
        });

        let params = SimParams {
            kernel_h: solid.kernel_h,
            dt: 0.0004,
            young: 100_000.0,
            poisson: 0.3,
            floor_y: 0.05,
            _pad: [0.0; 3],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("elastic sim params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // uniform buffer must be 16-byte aligned
        let kernel_h_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("elastic kernel h"),
            contents: bytemuck::bytes_of(&[solid.kernel_h, 0.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let precompute_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("elastic precompute"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                ],
            });
        let simulate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("elastic simulate"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        });

        let pre_mod = device.create_shader_module(wgpu::include_wgsl!("solids/precompute.wgsl"));
        let sim_mod = device.create_shader_module(wgpu::include_wgsl!("solids/simulate.wgsl"));
        let pre_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("elastic precompute"),
            bind_group_layouts: &[Some(&precompute_layout)],
            immediate_size: 0,
        });
        let sim_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("elastic simulate"),
            bind_group_layouts: &[Some(&simulate_layout)],
            immediate_size: 0,
        });

        let sim = Self {
            precompute: pipeline(device, &pre_pl, &pre_mod, "precompute_L"),
            simulate: pipeline(device, &sim_pl, &sim_mod, "simulate"),
            precompute_layout,
            simulate_layout,
            particles,
            params_buf,
            kernel_h_buf,
            num_particles,
            current: 0,
        };
        // precompute L_i once at construction
        let mut enc = device.create_command_encoder(&Default::default());
        sim.precompute_l(device, &mut enc);
        queue.submit(Some(enc.finish()));
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(
            &sim.particles[0],
            0,
            &sim.particles[1],
            0,
            (num_particles as u64) * std::mem::size_of::<ElasticSolidParticle>() as u64,
        );
        queue.submit(Some(enc.finish()));
        sim
    }

    fn precompute_l(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.precompute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.particles[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.kernel_h_buf.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.precompute);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(self.num_particles.div_ceil(WG), 1, 1);
    }

    pub fn step(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.num_particles == 0 {
            return;
        }
        let src = self.current;
        let dst = 1 - self.current;
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.simulate_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.particles[src].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.particles[dst].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.simulate);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.num_particles.div_ceil(WG), 1, 1);
        }
        self.current = dst;
    }

    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particles[self.current]
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}

pub fn elastic_cube(
    n: usize,
    spacing: f32,
    origin: nalgebra::Vector3<f32>,
    pin_top: bool,
) -> ElasticSolid {
    let kernel = CubicKernel::new(spacing * 2.1);
    ElasticSolid::cube(n, spacing, origin, &kernel, pin_top)
}
