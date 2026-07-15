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

struct ActiveEdges ();

impl ActiveEdges {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("marchine_cubes.wgsl"));
        // Explicit layout: auto layout from `build_hash_table` only sees 2 bindings.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("marching cubes - active edges"),
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
}

/// Perform marching cubes; steps inspired by https://www.willusher.io/graphics/2024/04/22/webgpu-marching-cubes/#conclusion-webgpu-performance-vs-vulkan
/// but modified to output vertices AND indices (to avoid duplicate vertices). Steps:
/// 1. Determine active edges;
/// 2. Count number of unique vertices based on above; compute offsets per thread
/// 3. Create the vertices
/// 4. For each voxel, use edge index to find the vertex index in the buffer (can use offsets buffer from step 2)
pub struct MarchingCubes {
    level_set_threshold: f32
}

impl MarchingCubes {
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
        let o =
            self.debugger
                .read::<FluidParticle>(device, queue, &self.particles[self.current], 1);
        dbg!(o);
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

    /// get the buffer w/ most recent particle state
    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particles[self.current]
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}
