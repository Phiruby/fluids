use crate::graphics::{PipelineInfo, PipelineState, create_pipeline};
use crate::physics::RenderParticle;
use crate::texture::Texture;
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleMeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl ParticleMeshVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
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
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl LineVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
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
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

fn particle_cube(radius: f32) -> ([ParticleMeshVertex; 8], [u16; 36]) {
    let vertices = [
        ParticleMeshVertex {
            position: [-radius, -radius, -radius],
            normal: [-0.57735026, -0.57735026, -0.57735026],
        },
        ParticleMeshVertex {
            position: [radius, -radius, -radius],
            normal: [0.57735026, -0.57735026, -0.57735026],
        },
        ParticleMeshVertex {
            position: [radius, radius, -radius],
            normal: [0.57735026, 0.57735026, -0.57735026],
        },
        ParticleMeshVertex {
            position: [-radius, radius, -radius],
            normal: [-0.57735026, 0.57735026, -0.57735026],
        },
        ParticleMeshVertex {
            position: [-radius, -radius, radius],
            normal: [-0.57735026, -0.57735026, 0.57735026],
        },
        ParticleMeshVertex {
            position: [radius, -radius, radius],
            normal: [0.57735026, -0.57735026, 0.57735026],
        },
        ParticleMeshVertex {
            position: [radius, radius, radius],
            normal: [0.57735026, 0.57735026, 0.57735026],
        },
        ParticleMeshVertex {
            position: [-radius, radius, radius],
            normal: [-0.57735026, 0.57735026, 0.57735026],
        },
    ];
    let indices = [
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

/// draw particles as instanced cubes;
//  generic P provide instance layout + WGSL vertex fields (works for generic structs)
pub struct RenderPipeline<P: RenderParticle> {
    pipeline: PipelineState,
    mesh_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    _marker: PhantomData<P>,
}

impl<P: RenderParticle> RenderPipeline<P> {
    pub fn new(
        device: &wgpu::Device,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        particle_radius: f32,
    ) -> Self {
        let shader_src = include_str!("particle_render.wgsl")
            .replace("${INSTANCE_FIELDS}", P::instance_wgsl_fields())
            .replace("${VS_MAIN_BODY}", P::vs_main_body());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle render"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_src)),
        });
        let vertex_buffers = [ParticleMeshVertex::layout(), P::instance_layout()];
        let pipeline = create_pipeline(
            device,
            &PipelineInfo {
                shader: &shader,
                vertex_entry_point: "vs_main",
                fragment_entry_point: "fs_main",
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                global_bind_group_layout,
                vertex_buffers: &vertex_buffers,
                fragment_targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
            },
        );
        let (mesh, indices) = particle_cube(particle_radius);
        let mesh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle mesh buffer"),
            contents: bytemuck::cast_slice(&mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            pipeline,
            mesh_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            _marker: PhantomData,
        }
    }

    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass,
        global_bind_group: &wgpu::BindGroup,
        particle_buffer: &wgpu::Buffer,
        num_instances: u32,
    ) {
        self.pipeline.bind(render_pass, global_bind_group);
        render_pass.set_vertex_buffer(0, self.mesh_buffer.slice(..));
        render_pass.set_vertex_buffer(1, particle_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..num_instances);
    }
}

fn box_wireframe_vertices(min: [f32; 3], max: [f32; 3], color: [f32; 3]) -> [LineVertex; 24] {
    let corners = [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [max[0], max[1], max[2]],
        [min[0], max[1], max[2]],
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let mut verts = [LineVertex {
        position: [0.0; 3],
        color,
    }; 24];
    for (i, &(a, b)) in edges.iter().enumerate() {
        verts[i * 2] = LineVertex {
            position: corners[a],
            color,
        };
        verts[i * 2 + 1] = LineVertex {
            position: corners[b],
            color,
        };
    }
    verts
}

pub struct WireframeBox {
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
}

impl WireframeBox {
    pub fn new(device: &wgpu::Device, min: [f32; 3], max: [f32; 3], color: [f32; 3]) -> Self {
        let verts = box_wireframe_vertices(min, max, color);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wireframe box"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            vertex_buffer,
            num_vertices: verts.len() as u32,
        }
    }
}

pub struct WireframeBoxPipeline {
    pipeline: PipelineState,
}

pub struct MarchingCubesRenderPipeline {
    pipeline: PipelineState,
}

impl MarchingCubesRenderPipeline {
    pub fn new(
        device: &wgpu::Device,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
        let pipeline = create_pipeline(
            device,
            &PipelineInfo {
                shader: &shader,
                vertex_entry_point: "vs_mc_main",
                fragment_entry_point: "fs_main",
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                global_bind_group_layout,
                vertex_buffers: &[crate::graphics::marching_cubes::MarchingCubesVertex::layout()],
                fragment_targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
            },
        );
        Self { pipeline }
    }

    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass,
        global_bind_group: &wgpu::BindGroup,
        vertices: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        indirect: &wgpu::Buffer,
    ) {
        self.pipeline.bind(render_pass, global_bind_group);
        render_pass.set_vertex_buffer(0, vertices.slice(..));
        render_pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed_indirect(indirect, 0);
    }
}

impl WireframeBoxPipeline {
    pub fn new(
        device: &wgpu::Device,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
        let pipeline = create_pipeline(
            device,
            &PipelineInfo {
                shader: &shader,
                vertex_entry_point: "vs_line_main",
                fragment_entry_point: "fs_main",
                primitive_topology: wgpu::PrimitiveTopology::LineList,
                global_bind_group_layout,
                vertex_buffers: &[LineVertex::layout()],
                fragment_targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
            },
        );
        Self { pipeline }
    }

    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass,
        global_bind_group: &wgpu::BindGroup,
        wireframe: &WireframeBox,
    ) {
        self.pipeline.bind(render_pass, global_bind_group);
        render_pass.set_vertex_buffer(0, wireframe.vertex_buffer.slice(..));
        render_pass.draw(0..wireframe.num_vertices, 0..1);
    }
}

pub struct RigidBodyPipeline {
    pipeline: wgpu::RenderPipeline,
    bodies_layout: wgpu::BindGroupLayout,
    mesh_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl RigidBodyPipeline {
    pub fn new(
        device: &wgpu::Device,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("rigid_render.wgsl"));
        let bodies_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rigid bodies"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rigid render"),
            bind_group_layouts: &[Some(global_bind_group_layout), Some(&bodies_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rigid render"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[ParticleMeshVertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let (mesh, indices) = particle_cube(1.0);
        let mesh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid mesh"),
            contents: bytemuck::cast_slice(&mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid index"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            pipeline,
            bodies_layout,
            mesh_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
        }
    }

    pub fn draw(
        &self,
        device: &wgpu::Device,
        render_pass: &mut wgpu::RenderPass,
        global_bind_group: &wgpu::BindGroup,
        rigid_buffer: &wgpu::Buffer,
        num_instances: u32,
    ) {
        if num_instances == 0 {
            return;
        }
        let bodies = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rigid bodies"),
            layout: &self.bodies_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: rigid_buffer.as_entire_binding(),
            }],
        });
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, global_bind_group, &[]);
        render_pass.set_bind_group(1, &bodies, &[]);
        render_pass.set_vertex_buffer(0, self.mesh_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..num_instances);
    }
}
