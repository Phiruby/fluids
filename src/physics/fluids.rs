use crate::graphics::BufferDebug;
use crate::physics::FluidParticle;
use wgpu::util::DeviceExt;
const WG_SIZE: u32 = 256;
const RADIX_BUCKETS: u32 = 16;
const RADIX_PASSES: usize = 8;
const HASHTABLE_SIZE: u64 = 8192;

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

/// At the moment, limited by 512^2 elements in the array:
/// each workergroup takes a chunk of 512 elements. These output cummulative sum
/// in their chunk to the `summation_buffer`. We again compute a prefix sum on the summation_buffer
/// (which has a limit of 512, since a single worker group can handle at most one layer of recursion
/// for now).
pub struct PrefixSum {
    prefix_sum_pipeline: wgpu::ComputePipeline,
    aggregate_offsets_pipeline: wgpu::ComputePipeline,
    summation_buffer: wgpu::Buffer,
    offsets_buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
}

impl PrefixSum {
    pub fn new(device: &wgpu::Device, max_particles: u32) -> Self {
        assert!(max_particles <= 512 * 512, "Max particles must be <= 512^2");
        let module = device.create_shader_module(wgpu::include_wgsl!("prefix_sum.wgsl"));
        // Explicit layout: auto layout from `build_hash_table` only sees 2 bindings.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix sum"),
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
            label: Some("prefix sum"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let workergroup_prefixsum = pipeline(device, &pipeline_layout, &module, "prefix_sum");
        let aggregate_offsets = pipeline(device, &pipeline_layout, &module, "aggregate_offsets");

        let max_wgs = (max_particles + WG_SIZE - 1) / WG_SIZE;
        let summation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("workgroup summation"),
            size: (max_particles * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("summation buffer prefix sum"),
            size: (max_particles * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            summation_buffer,
            prefix_sum_pipeline: workergroup_prefixsum,
            aggregate_offsets_pipeline: aggregate_offsets,
            layout,
            offsets_buffer,
        }
    }

    pub fn get_prefix_sum(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        prefix_sum_output_buffer: &wgpu::Buffer,
        n_input_elements: u32,
    ) {
        // NOTE: we dont need to bind every frame; huge slow down
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prefix_sum_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.summation_buffer.as_entire_binding(),
                },
            ],
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.summation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.offsets_buffer.as_entire_binding(),
                },
            ],
        });

        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prefix_sum_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.offsets_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.prefix_sum_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let nworkers = n_input_elements.div_ceil(512);
        pass.dispatch_workgroups(nworkers, 1, 1);
        // get prefix sum of the summation buffer; the offset buffer will contain
        // offsets for each workgroup after this finishes
        pass.set_pipeline(&self.prefix_sum_pipeline);
        pass.set_bind_group(0, &bg2, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        // finally, add offsets to the prefix sum
        pass.set_pipeline(&self.aggregate_offsets_pipeline);
        pass.set_bind_group(0, &bg3, &[]);
        pass.dispatch_workgroups(nworkers, 1, 1);
    }
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
    pub(in crate::physics) hash_offsets_buf: wgpu::Buffer,
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
            size: max_particles as u64 * std::mem::size_of::<FluidParticle>() as u64,
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
                max_particles as u64 * std::mem::size_of::<FluidParticle>() as u64,
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

        {
            let clear_bg = self.bind(device, particles, &self.sorted, &self.uniforms[0]);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.reset_hash_offsets);
            pass.set_bind_group(0, &clear_bg, &[]);
            let clear_wgs = (27 * HASHTABLE_SIZE / WG_SIZE as u64) as u32;
            pass.dispatch_workgroups(clear_wgs, 1, 1);
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
        let o = self
            .debugger
            .read::<[f32; 3]>(device, queue, &self.debug_buffer, 1);
        dbg!(o);
        let ndispatches = num_particles / 256;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(ndispatches, 1, 1);
        }
    }
}

// Performs sph discretized calculations of navier stokes and outputs new particles
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
        // TODO: does this need to be done every compute call?
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
        // let o = self.debugger.read::<[f32; 3]>(
        //     device,
        //     queue,
        //     &self.debug_buffer,
        //     1,
        // );
        // dbg!(o);
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
            spatial_hash: SpatialHashGPU::new(device, num_particles),
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
        let o =
            self.debugger
                .read::<FluidParticle>(device, queue, &self.particles[self.current], 1);
        dbg!(o);
        self.current = dst;
    }

    /// get the buffer w/ most recent particle state
    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particles[self.current]
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}

#[cfg(test)]
mod prefix_sum_test {
    use super::*;

    fn gpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device")
    }

    #[test]
    fn prefix_sum_exclusive() {
        let (device, queue) = gpu();
        let n = 1236u32;
        let input: Vec<u32> = (1..=n).collect();
        let expected: Vec<u32> = {
            let mut e = vec![0u32; n as usize];
            for i in 1..n as usize {
                e[i] = e[i - 1] + input[i - 1];
            }
            e
        };

        let prefix = PrefixSum::new(&device, n);
        let dbg = BufferDebug::new(&device, (n as u64) * 4);
        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&Default::default());
        prefix.get_prefix_sum(&device, &mut enc, &in_buf, &out_buf, n);
        queue.submit(Some(enc.finish()));

        let got = dbg.read::<u32>(&device, &queue, &out_buf, n as u64);
        dbg!(&got[..16], &expected[..16]);
        dbg!(got.len(), expected.len());
        assert_eq!(got, expected);
    }
}
