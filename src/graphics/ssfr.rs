use crate::physics::{FluidParticle, PrefixSum};

/// After SPH sim, the particles buffer contains a mix of
/// boundary and fluid particles. this performs compaction
/// to get a buffer with just fluid particles
struct FluidCompaction {
    build_fluid_mask_pipeline: wgpu::ComputePipeline,
    fluid_compaction_pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    prefix_sum: PrefixSum,
    fluid_mask_buffer: wgpu::Buffer,
    offsets_buffer: wgpu::Buffer,
    compact_fluids_buffer: wgpu::Buffer,
    nparticles: u32,
}

impl FluidCompaction {
    pub fn new(device: &wgpu::Device, nparticles: u32) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("fluid_compaction.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid compaction"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Compaction Pipeline Layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build_fluid_mask_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("build_fluid_mask"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("build_fluid_mask"),
                cache: Default::default(),
                compilation_options: Default::default(),
            });
        let fluid_compaction_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fluid_compaction"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("fluid_compaction"),
                cache: Default::default(),
                compilation_options: Default::default(),
            });

        let scan_len = nparticles.div_ceil(512).max(1) * 512;
        let mask_bytes = (nparticles as u64) * 4;
        let offsets_bytes = (scan_len as u64) * 4;
        let compact_bytes = (nparticles as u64) * std::mem::size_of::<FluidParticle>() as u64;

        let fluid_mask_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Mask Buffer"),
            size: mask_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Offsets Buffer"),
            size: offsets_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let compact_fluids_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compact Fluids Buffer"),
            size: compact_bytes.max(std::mem::size_of::<FluidParticle>() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prefix_sum = PrefixSum::new(device, scan_len);

        Self {
            build_fluid_mask_pipeline,
            fluid_compaction_pipeline,
            layout,
            prefix_sum,
            fluid_mask_buffer,
            compact_fluids_buffer,
            offsets_buffer,
            nparticles,
        }
    }

    pub fn compact(
        &self,
        device: &wgpu::Device,
        particles_buffer: &wgpu::Buffer,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Mask Bind Group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.fluid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.compact_fluids_buffer.as_entire_binding(),
                },
            ],
        });

        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Compaction Bind Group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.compact_fluids_buffer.as_entire_binding(),
                },
            ],
        });
        let ndispatches = self.nparticles.div_ceil(256);
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

            compute_pass.set_pipeline(&self.build_fluid_mask_pipeline);
            compute_pass.set_bind_group(0, &bg1, &[]);
            compute_pass.dispatch_workgroups(ndispatches, 1, 1);
        }

        self.prefix_sum.get_prefix_sum(
            device,
            command_encoder,
            &self.fluid_mask_buffer,
            &self.offsets_buffer,
            self.nparticles,
        );

        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&self.fluid_compaction_pipeline);
            compute_pass.set_bind_group(0, &bg2, &[]);
            compute_pass.dispatch_workgroups(ndispatches, 1, 1);
        }
    }

    pub fn get_compact_fluids_buffer(&self) -> &wgpu::Buffer {
        &self.compact_fluids_buffer
    }
}

struct FluidDepth {
    pipeline: wgpu::RenderPipeline,
}

impl FluidDepth {
    pub fn new(device: &wgpu::Device) -> Self {
        // let module = device.create_shader_module(wgpu::include_wgsl!("ssfr.wgsl"));
        let module = device.create_shader_module(wgpu::include_wgsl!("fluid_depth.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid_depth"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
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
            label: Some("Fluid depth calculation"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let fluid_depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth fluid pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                buffers: &[], // no buffers needed since we use sph buffer
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
        });

        Self {
            pipeline: fluid_depth_pipeline,
        }
    }
}

/// Perform screen-space fluid rendering
/// see https://wstahw.win.tue.nl/edu/2IV06/andrei/particle_rendering/provided/p91-van_der_laan.pdf
pub struct ScreenSpaceRendering {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::BufferDebug;
    use wgpu::util::DeviceExt;

    fn gpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device")
    }

    fn fluid_at(x: f32) -> FluidParticle {
        let mut p = FluidParticle::default();
        p.position = [x, 0.0, 0.0];
        p.boundary = 1;
        p.mass = 1.0;
        p
    }

    fn boundary_at(x: f32) -> FluidParticle {
        let mut p = FluidParticle::default();
        p.position = [x, 0.0, 0.0];
        p.boundary = 0;
        p.mass = 1.0;
        p
    }

    #[test]
    fn test_fluid_compaction() {
        let particles = vec![
            fluid_at(0.0),
            boundary_at(1.0),
            fluid_at(2.0),
            boundary_at(3.0),
            fluid_at(4.0),
            boundary_at(5.0),
        ];
        let expected = vec![fluid_at(0.0), fluid_at(2.0), fluid_at(4.0)];

        let (device, q) = gpu();
        let dbg = BufferDebug::new(
            &device,
            (expected.len() * std::mem::size_of::<FluidParticle>()) as u64,
        );

        let mut enc = device.create_command_encoder(&Default::default());
        let pipeline = FluidCompaction::new(&device, particles.len() as u32);

        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Particles Buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        pipeline.compact(&device, &in_buf, &mut enc);
        q.submit(Some(enc.finish()));

        let got = dbg.read::<FluidParticle>(
            &device,
            &q,
            pipeline.get_compact_fluids_buffer(),
            expected.len() as u64,
        );

        assert_eq!(got, expected);
    }
}
