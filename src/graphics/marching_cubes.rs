use crate::physics::PrefixSum;
use wgpu::util::DeviceExt;

const WG_SIZE: u32 = 256;
const GRID_N: u32 = 40;
const EDGE_N: u32 = GRID_N + 1;
const NUM_VOXELS: u32 = GRID_N * GRID_N * GRID_N;
const NUM_EDGE_CELLS: u32 = EDGE_N * EDGE_N * EDGE_N;
const NUM_EDGES: u32 = 3 * NUM_EDGE_CELLS;
const MAX_INDICES_PER_VOXEL: u32 = 15;
const DEFAULT_ISO_LEVEL: f32 = 2000.0; // ~0.25 × typical rest density (~8e3)
const ISO_LEVEL_MIN: f32 = 100.0;
const ISO_LEVEL_MAX: f32 = 10000.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct McParams {
    iso_level: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchingCubesVertex {
    pub position: [f32; 3],
    _pad0: f32,
    pub normal: [f32; 3],
    _pad1: f32,
}

impl MarchingCubesVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
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

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn mc_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
            uniform_entry(6),
        ],
    })
}

/// Main goal here is to write out to index buffer. for that, we need to:
/// 1. compute number of indices needed per voxel
/// 2. determine offsets for each voxel to write to in the index buffer (via prefix sum)
/// 3. write the indices
struct CompactIndices {
    count: wgpu::ComputePipeline,
    write: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    prefix_sum: PrefixSum,
    counts: wgpu::Buffer,
    offsets: wgpu::Buffer,
    indices: wgpu::Buffer,
    indirect: wgpu::Buffer,
    scan_count: u32,
}

impl CompactIndices {
    fn new(device: &wgpu::Device, module: &wgpu::ShaderModule) -> Self {
        let layout = mc_bind_group_layout(device, "mc compact indices");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc compact indices"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let count = pipeline(device, &pipeline_layout, module, "count_indices");
        let write = pipeline(device, &pipeline_layout, module, "write_indices");

        let scan_count = (NUM_VOXELS + 1).next_multiple_of(512);
        let prefix_sum = PrefixSum::new(device, scan_count);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        Self {
            count,
            write,
            layout,
            prefix_sum,
            counts: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc index counts"),
                size: scan_count as u64 * 4,
                usage,
                mapped_at_creation: false,
            }),
            offsets: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc index offsets"),
                size: scan_count as u64 * 4,
                usage,
                mapped_at_creation: false,
            }),
            indices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc compact indices"),
                size: NUM_VOXELS as u64 * MAX_INDICES_PER_VOXEL as u64 * 4,
                usage: usage | wgpu::BufferUsages::INDEX,
                mapped_at_creation: false,
            }),
            indirect: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc draw indirect"),
                contents: bytemuck::cast_slice(&[0u32, 1, 0, 0, 0]),
                usage: wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
            scan_count,
        }
    }

    fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        hash_offsets: &wgpu::Buffer,
        edge_offsets: &wgpu::Buffer,
        vertices: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) {
        let bg_counts = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: edge_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.count);
            pass.set_bind_group(0, &bg_counts, &[]);
            pass.dispatch_workgroups(NUM_VOXELS.div_ceil(WG_SIZE), 1, 1);
        }

        self.prefix_sum.get_prefix_sum(
            device,
            encoder,
            &self.counts,
            &self.offsets,
            self.scan_count,
        );

        let bg_write = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: edge_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.write);
            pass.set_bind_group(0, &bg_write, &[]);
            pass.dispatch_workgroups(NUM_VOXELS.div_ceil(WG_SIZE), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.offsets,
            NUM_VOXELS as u64 * 4,
            &self.indirect,
            0,
            4,
        );
    }

    fn index_buffer(&self) -> &wgpu::Buffer {
        &self.indices
    }

    fn indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect
    }
}

pub struct MarchingCubes {
    active_edges: wgpu::ComputePipeline,
    add_vertices: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    prefix_sum: PrefixSum,
    edge_mask: wgpu::Buffer,
    edge_offsets: wgpu::Buffer,
    vertices: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    compact_indices: CompactIndices,
    edge_scan_count: u32,
    iso_level: f32,
}

impl MarchingCubes {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("marching_cubes.wgsl"));
        let layout = mc_bind_group_layout(device, "marching cubes");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("marching cubes"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let active_edges = pipeline(device, &pipeline_layout, &module, "active_edges");
        let add_vertices = pipeline(device, &pipeline_layout, &module, "add_vertices");
        let compact_indices = CompactIndices::new(device, &module);

        let edge_scan_count = (NUM_EDGES + 1).next_multiple_of(512);
        let prefix_sum = PrefixSum::new(device, edge_scan_count);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let iso_level = DEFAULT_ISO_LEVEL;
        let params = McParams {
            iso_level,
            _pad: [0.0; 3],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            active_edges,
            add_vertices,
            layout,
            prefix_sum,
            edge_mask: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc edge mask"),
                size: edge_scan_count as u64 * 4,
                usage,
                mapped_at_creation: false,
            }),
            edge_offsets: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc edge offsets"),
                size: edge_scan_count as u64 * 4,
                usage,
                mapped_at_creation: false,
            }),
            vertices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc vertices"),
                size: NUM_EDGES as u64 * std::mem::size_of::<MarchingCubesVertex>() as u64,
                usage: usage | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            }),
            params_buf,
            compact_indices,
            edge_scan_count,
            iso_level,
        }
    }

    pub fn iso_level(&self) -> f32 {
        self.iso_level
    }

    pub fn set_iso_level(&mut self, queue: &wgpu::Queue, iso_level: f32) {
        self.iso_level = iso_level.clamp(ISO_LEVEL_MIN, ISO_LEVEL_MAX);
        let params = McParams {
            iso_level: self.iso_level,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Adjust iso level by a delta (e.g. from mouse wheel). Returns the new value.
    pub fn nudge_iso_level(&mut self, queue: &wgpu::Queue, delta: f32) -> f32 {
        self.set_iso_level(queue, self.iso_level + delta);
        self.iso_level
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        particles: &wgpu::Buffer,
        hash_offsets: &wgpu::Buffer,
        edges: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("marching cubes"),
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
                    resource: edges.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.compact_indices.counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.compact_indices.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.params_buf.as_entire_binding(),
                },
            ],
        })
    }

    pub fn extract(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        hash_offsets: &wgpu::Buffer,
    ) {
        let bg_mask = self.bind_group(device, particles, hash_offsets, &self.edge_mask);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.active_edges);
            pass.set_bind_group(0, &bg_mask, &[]);
            pass.dispatch_workgroups(NUM_EDGE_CELLS.div_ceil(WG_SIZE), 1, 1);
        }

        self.prefix_sum.get_prefix_sum(
            device,
            encoder,
            &self.edge_mask,
            &self.edge_offsets,
            self.edge_scan_count,
        );

        let bg_offsets = self.bind_group(device, particles, hash_offsets, &self.edge_offsets);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.add_vertices);
            pass.set_bind_group(0, &bg_offsets, &[]);
            pass.dispatch_workgroups(NUM_EDGES.div_ceil(WG_SIZE), 1, 1);
        }

        self.compact_indices.run(
            device,
            encoder,
            particles,
            hash_offsets,
            &self.edge_offsets,
            &self.vertices,
            &self.params_buf,
        );
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertices
    }

    pub fn index_buffer(&self) -> &wgpu::Buffer {
        self.compact_indices.index_buffer()
    }

    pub fn indirect_buffer(&self) -> &wgpu::Buffer {
        self.compact_indices.indirect_buffer()
    }

    pub fn grid_n(&self) -> u32 {
        GRID_N
    }

    pub fn num_voxels(&self) -> u32 {
        NUM_VOXELS
    }

    pub fn num_edges(&self) -> u32 {
        NUM_EDGES
    }
}
