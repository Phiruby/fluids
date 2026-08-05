use crate::physics::{FluidParticle, PrefixSum};
use crate::texture::Texture;
use wgpu::util::DeviceExt;
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
    layout: wgpu::BindGroupLayout,
    pub depth_texture: Texture,
}

impl FluidDepth {
    pub fn new(
        device: &wgpu::Device,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        extend_width: u32,
        extend_height: u32,
    ) -> Self {
        // TODO: we can use create_pipeline from crate::graphics
        let module = device.create_shader_module(wgpu::include_wgsl!("fluid_depth.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid_depth"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid depth calculation"),
            bind_group_layouts: &[Some(uniform_bind_group_layout), Some(&layout)],
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
                targets: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let depth_texture =
            Texture::create_depth_texture(device, extend_width, extend_height, "fluid depth ssfr");
        Self {
            pipeline: fluid_depth_pipeline,
            layout,
            depth_texture,
        }
    }

    fn get_render_pass_descriptor(&self) -> wgpu::RenderPassDescriptor<'_> {
        wgpu::RenderPassDescriptor {
            label: Some("Fluid depth calculation render pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    // TODO: verify 1.0 is enough
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }
    }

    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fluid_particles: &wgpu::Buffer,
        nparticles: u32,
        uniforms: &wgpu::BindGroup,
    ) {
        let render_pass_desc = self.get_render_pass_descriptor();
        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, uniforms, &[]);
        let particles_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fluid_particles.as_entire_binding(),
            }],
        });
        render_pass.set_bind_group(1, &particles_binding, &[]);
        // TODO: should be instanced with rectangles?
        render_pass.draw(0..4, 0..nparticles);
    }
}

struct FluidThickness {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    thickness_texture: Texture,
}

impl FluidThickness {
    pub fn new(
        device: &wgpu::Device,
        uniform_bind_group_layout: wgpu::BindGroupLayout,
        extent_width: u32,
        extent_height: u32,
    ) -> Self {
        // TODO: can pull out commonalities with Fluid Depth pipeline
        let module = device.create_shader_module(wgpu::include_wgsl!("fluid_depth.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid_thickness"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid thickness calculation"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout), Some(&layout)],
            immediate_size: 0,
        });
        let fluid_thickness_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fluid thickness pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vertex_main"),
                    buffers: &[], // no buffers needed since we use sph buffer
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("frag_thickness"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R16Float,
                        //::ADDITIVE is not being found for some reason so explicit definition
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::RED,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let thickness_texture = Texture::from_descriptor(
            device,
            &wgpu::TextureDescriptor {
                label: Some("thickness"),
                size: wgpu::Extent3d {
                    width: extent_width,
                    height: extent_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        Self {
            pipeline: fluid_thickness_pipeline,
            layout,
            thickness_texture,
        }
    }

    pub fn thickness_texture(&self) -> &Texture {
        &self.thickness_texture
    }
    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fluid_particles: &wgpu::Buffer,
        nparticles: u32,
        uniforms: &wgpu::BindGroup,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.thickness_texture.view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, uniforms, &[]);
        let particles_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fluid_particles.as_entire_binding(),
            }],
        });
        render_pass.set_bind_group(1, &particles_binding, &[]);
        render_pass.draw(0..4, 0..nparticles);
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CurvatureFlowUniform {
    viewport_width: f32,
    viewport_height: f32,
    focal_x: f32,
    focal_y: f32,
    dt: f32,
    depth_threshold: f32,
}

struct CurvatureFlow {
    curvature_flow_pipeline: wgpu::ComputePipeline,
    normals_pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    niterations: u32,
    alternate_depth1: Texture,
    alternate_depth2: Texture,
    normals: Texture,
    extent: wgpu::Extent3d,
    uniform_buffer: wgpu::Buffer,
    uniform: CurvatureFlowUniform,
    depth_staging: wgpu::Buffer,
    padded_bytes_per_row: u32,
}

impl CurvatureFlow {
    pub fn new(
        device: &wgpu::Device,
        extent_width: u32,
        extent_height: u32,
        focal_x: f32,
        focal_y: f32,
    ) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("curvature_flow.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("curvature flow"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    // TODO: verify that texture format is fine
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let uniform = CurvatureFlowUniform {
            viewport_width: extent_width as f32,
            viewport_height: extent_height as f32,
            focal_x,
            focal_y,
            dt: 0.001,
            depth_threshold: 0.1,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("curvature flow uniform buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("curvature flow"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let curvature_flow_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("curvature flow"),
                module: &module,
                entry_point: Some("curvature_flow"),
                layout: Some(&pipeline_layout),
                compilation_options: Default::default(),
                cache: Default::default(),
            });
        let normals_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("curvature flow"),
            module: &module,
            entry_point: Some("compute_normals"),
            layout: Some(&pipeline_layout),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let ping_pong_depth_desc = wgpu::TextureDescriptor {
            label: Some("curvature ping-pong depth"),
            size: wgpu::Extent3d {
                width: extent_width,
                height: extent_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let normal_map_desc = wgpu::TextureDescriptor {
            label: Some("curvature normal map"),
            size: wgpu::Extent3d {
                width: extent_width,
                height: extent_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let alternate_depth1 = Texture::from_descriptor(device, &ping_pong_depth_desc);
        let alternate_depth2 = Texture::from_descriptor(device, &ping_pong_depth_desc);
        let normals = Texture::from_descriptor(device, &normal_map_desc);

        let bytes_per_pixel = 4; // 32-bit float = 4 bytes
        let unpadded_bytes_per_row = extent_width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);

        let depth_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Depth Copy Staging Buffer"),
            size: (padded_bytes_per_row * extent_height) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            curvature_flow_pipeline,
            normals_pipeline,
            layout,
            niterations: 1,
            alternate_depth1,
            alternate_depth2,
            normals,
            extent: wgpu::Extent3d {
                width: extent_width,
                height: extent_height,
                depth_or_array_layers: 1,
            },
            uniform_buffer,
            uniform,
            depth_staging,
            padded_bytes_per_row,
        }
    }

    pub fn smooth(
        &self,
        device: &wgpu::Device,
        fluid_depth: &Texture,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &fluid_depth.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.depth_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.extent.height),
                },
            },
            self.extent,
        );

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.depth_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.extent.height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.alternate_depth1.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            self.extent,
        );
        let input = &self.alternate_depth1.view;
        let output = &self.alternate_depth2.view;
        let bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.normals.view),
                },
            ],
        });
        let bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(output),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.normals.view),
                },
            ],
        });
        let (h, w) = (fluid_depth.texture.height(), fluid_depth.texture.width());
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.curvature_flow_pipeline);
        for i in 0..self.niterations {
            let bg = if i % 2 == 0 {
                &bind_group1
            } else {
                &bind_group2
            };
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(w / 16, h / 16, 1);
        }
        pass.set_pipeline(&self.normals_pipeline);
        pass.set_bind_group(0, &bind_group2, &[]);
        pass.dispatch_workgroups(w / 16, h / 16, 1);
    }

    fn normal_map(&self) -> &wgpu::TextureView {
        &self.normals.view
    }
}

/// temporary pipeline to visualize depth after curvature flow
struct DepthViz {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl DepthViz {
    fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("visualize_depth.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("depth viz"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth viz"),
            bind_group_layouts: &[Some(&layout), Some(uniform_bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth viz"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout }
    }

    fn draw(
        &self,
        device: &wgpu::Device,
        normal_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        thickness_view: &wgpu::TextureView,
        render_pass: &mut wgpu::RenderPass,
        uniforms: &wgpu::BindGroup,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth viz"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(thickness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        });
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &bg, &[]);
        render_pass.set_bind_group(1, uniforms, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Perform screen-space fluid rendering
/// see https://wstahw.win.tue.nl/edu/2IV06/andrei/particle_rendering/provided/p91-van_der_laan.pdf
pub struct ScreenSpaceRendering {
    compaction: FluidCompaction,
    depth: FluidDepth,
    thickness: FluidThickness,
    smoothing: CurvatureFlow,
    depth_viz: DepthViz,
    n_fluid: u32,
}

impl ScreenSpaceRendering {
    pub fn new(
        device: &wgpu::Device,
        uniform_bind_group_layout: wgpu::BindGroupLayout,
        nparticles: u32,
        n_fluid: u32,
        surface_format: wgpu::TextureFormat,
        extent_width: u32,
        extent_height: u32,
        focal_x: f32,
        focal_y: f32,
    ) -> Self {
        Self {
            compaction: FluidCompaction::new(device, nparticles),
            depth: FluidDepth::new(
                device,
                &uniform_bind_group_layout,
                surface_format,
                extent_width,
                extent_height,
            ),
            depth_viz: DepthViz::new(device, surface_format, &uniform_bind_group_layout),
            thickness: FluidThickness::new(
                device,
                uniform_bind_group_layout,
                extent_width,
                extent_height,
            ),
            smoothing: CurvatureFlow::new(device, extent_width, extent_height, focal_x, focal_y),
            n_fluid,
        }
    }

    /// performs fluid compaction (i.e. filter boundary particles out of the SPH buffer).
    /// make sure to use same encoder. this should be called before using draw()
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        particles: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
        uniforms: &wgpu::BindGroup,
    ) {
        self.compaction.compact(device, particles, encoder);
        self.thickness.draw(
            device,
            encoder,
            self.compaction.get_compact_fluids_buffer(),
            self.n_fluid,
            uniforms,
        );
        self.depth.draw(
            device,
            encoder,
            self.compaction.get_compact_fluids_buffer(),
            self.n_fluid,
            uniforms,
        );
        let depth_texture = &self.depth.depth_texture;
        self.smoothing.smooth(device, depth_texture, encoder);
    }

    pub fn draw(
        &self,
        device: &wgpu::Device,
        render_pass: &mut wgpu::RenderPass,
        uniforms: &wgpu::BindGroup,
    ) {
        self.depth_viz.draw(
            device,
            self.smoothing.normal_map(),
            &self.depth.depth_texture.view,
            &self.thickness.thickness_texture().view,
            render_pass,
            uniforms,
        );
    }
}

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
