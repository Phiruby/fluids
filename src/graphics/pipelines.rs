use crate::graphics::{create_pipeline, PipelineInfo, PipelineState};
use crate::Vertex;
use cgmath::Point3;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleMeshVertex {
    position: [f32; 3],
}

impl ParticleMeshVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
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
        ParticleMeshVertex { position: [-radius, -radius, -radius] },
        ParticleMeshVertex { position: [ radius, -radius, -radius] },
        ParticleMeshVertex { position: [ radius,  radius, -radius] },
        ParticleMeshVertex { position: [-radius,  radius, -radius] },
        ParticleMeshVertex { position: [-radius, -radius,  radius] },
        ParticleMeshVertex { position: [ radius, -radius,  radius] },
        ParticleMeshVertex { position: [ radius,  radius,  radius] },
        ParticleMeshVertex { position: [-radius,  radius,  radius] },
    ];
    let indices = [
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7,
        0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

fn color_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::REPLACE),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

pub struct RenderPipeline {
    pipeline: PipelineState,
    mesh_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    num_instances: u32,
}

impl RenderPipeline {
    pub fn new(
        device: &wgpu::Device,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        particle_radius: f32,
        instances: &[Vertex],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
        let pipeline = create_pipeline(
            device,
            &PipelineInfo {
                shader: &shader,
                vertex_entry_point: "vs_main",
                fragment_entry_point: "fs_main",
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                global_bind_group_layout,
                vertex_buffers: &[ParticleMeshVertex::layout(), Vertex::instance_layout()],
                fragment_targets: &[Some(color_target(surface_format))],
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
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle instance buffer"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            pipeline,
            mesh_buffer,
            instance_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            num_instances: instances.len() as u32,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, instances: &[Vertex]) {
        self.num_instances = instances.len() as u32;
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
    }

    pub fn draw(&self, render_pass: &mut wgpu::RenderPass, global_bind_group: &wgpu::BindGroup) {
        self.pipeline.bind(render_pass, global_bind_group);
        render_pass.set_vertex_buffer(0, self.mesh_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
    }
}

pub struct RayPipeline {
    pipeline: PipelineState,
    vertex_buffer: wgpu::Buffer,
}

impl RayPipeline {
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
                fragment_targets: &[Some(color_target(surface_format))],
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
            },
        );
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ray buffer"),
            contents: bytemuck::cast_slice(&[LineVertex {
                position: [0.0; 3],
                color: [0.2, 1.0, 0.2],
            }; 2]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self { pipeline, vertex_buffer }
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        start: Point3<f32>,
        end: Point3<f32>,
        color: [f32; 3],
    ) {
        let verts = [
            LineVertex { position: start.into(), color },
            LineVertex { position: end.into(), color },
        ];
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&verts));
    }

    pub fn draw(&self, render_pass: &mut wgpu::RenderPass, global_bind_group: &wgpu::BindGroup) {
        self.pipeline.bind(render_pass, global_bind_group);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..2, 0..1);
    }
}
