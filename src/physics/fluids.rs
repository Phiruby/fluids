use crate::graphics::BufferDebug;
use wgpu::util::DeviceExt;

const WG_SIZE: u32 = 256;
const RADIX_BUCKETS: u32 = 16;
const RADIX_PASSES: usize = 8;
const HASHTABLE_SIZE: u64 = 8192;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    _pad0: u32,
    pub velocity: [f32; 3],
    _pad1: u32,
    pub normal: [f32; 3],
    pub density: f32,
    pub mass: f32,
    pub cell_hash: u32,
    _pad2: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SortUniform {
    iteration: u32,
    num_particles: u32,
    _pad: [u32; 2],
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

pub struct SpatialHashGPU {
    hash: wgpu::ComputePipeline,
    histogram: wgpu::ComputePipeline,
    prefix: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    reset_hash_offsets: wgpu::ComputePipeline,
    hash_offsets: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// One uniform buffer per radix pass (avoids overwrite-before-submit).
    uniforms: [wgpu::Buffer; RADIX_PASSES],
    histogram_buf: wgpu::Buffer,
    hash_offsets_buf: wgpu::Buffer,
    sorted: wgpu::Buffer,
    max_particles: u32,
    buffer_debug: BufferDebug,
}

impl SpatialHashGPU {
    pub fn new(device: &wgpu::Device, max_particles: u32) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("spatial_hash.wgsl"));
        // Explicit layout: auto layout from `build_hash_table` only sees 2 bindings.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial hash"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("spatial hash"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let hash = pipeline(device, &pipeline_layout, &module, "build_hash_table");
        let histogram = pipeline(device, &pipeline_layout, &module, "histogram_shader");
        let prefix = pipeline(device, &pipeline_layout, &module, "prefix_sum_shader");
        let scatter = pipeline(device, &pipeline_layout, &module, "scatter_shader");
        let reset_hash_offsets = pipeline(device, &pipeline_layout, &module, "clear_hash_offsets");
        let hash_offsets = pipeline(device, &pipeline_layout, &module, "build_offsets");

        let max_wgs = (max_particles + WG_SIZE - 1) / WG_SIZE;
        let uniforms = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("sort uniform {i}")),
                size: std::mem::size_of::<SortUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let histogram_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix histogram"),
            size: (max_wgs * RADIX_BUCKETS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hash_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash offsets"),
            size: 27 * HASHTABLE_SIZE * 8 * 2 as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sorted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorted particles"),
            size: max_particles as u64 * std::mem::size_of::<Particle>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            hash,
            histogram,
            prefix,
            scatter,
            reset_hash_offsets,
            hash_offsets,
            layout,
            uniforms,
            histogram_buf,
            hash_offsets_buf,
            sorted,
            max_particles,
            buffer_debug: BufferDebug::new(
                device,
                max_particles as u64 * std::mem::size_of::<Particle>() as u64,
            ),
        }
    }

    fn bind(
        &self,
        device: &wgpu::Device,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.histogram_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.hash_offsets_buf.as_entire_binding(),
                },
            ],
        })
    }

    pub fn sort(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        num_particles: u32,
    ) {
        assert!(num_particles <= self.max_particles);
        let wgs = (num_particles + WG_SIZE - 1) / WG_SIZE;

        for (i, buf) in self.uniforms.iter().enumerate() {
            queue.write_buffer(
                buf,
                0,
                bytemuck::bytes_of(&SortUniform {
                    iteration: i as u32,
                    num_particles,
                    _pad: [0; 2],
                }),
            );
        }

        let hash_bg = self.bind(device, particles, &self.sorted, &self.uniforms[0]);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.hash);
            pass.set_bind_group(0, &hash_bg, &[]);
            pass.dispatch_workgroups(wgs, 1, 1);
        }

        // pass i even: particles sorted via a; odd via b. after 8 passes, data is in `particles` var
        for i in 0..RADIX_PASSES {
            let (src, dst) = if i % 2 == 0 {
                (particles, &self.sorted)
            } else {
                (&self.sorted as &wgpu::Buffer, particles)
            };
            let bg = self.bind(device, src, dst, &self.uniforms[i]);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_bind_group(0, &bg, &[]);
            pass.set_pipeline(&self.histogram);
            pass.dispatch_workgroups(wgs, 1, 1);
            pass.set_pipeline(&self.prefix);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.scatter);
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        // calculate offsets for sorted hash
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        let (src, dst) = (particles, &self.sorted);
        let bg = self.bind(device, src, dst, &self.uniforms[7]);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_pipeline(&self.hash_offsets);
        let wgs = 27 * HASHTABLE_SIZE / (WG_SIZE as u64);
        pass.dispatch_workgroups(wgs as u32, 1, 1);
    }
}

pub struct FluidSim {
    spatial_hash: SpatialHashGPU,
}

impl FluidSim {
    pub fn new(device: &wgpu::Device, max_particles: u32) -> Self {
        Self {
            spatial_hash: SpatialHashGPU::new(device, max_particles),
        }
    }

    pub fn sort_particles(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        num_particles: u32,
    ) {
        self.spatial_hash
            .sort(device, queue, encoder, particles, num_particles);
    }
}
