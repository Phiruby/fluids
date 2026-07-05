use crate::graphics::{PipelineState, PipelineInfo};
use crate::graphics;


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleMeshVertex {
    position: [f32; 3],
}

impl ParticleMeshVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleMeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

pub struct RenderPipeline {
    pipeline: PipelineState,
    index_buffer: wgpu::Buffer,
}

impl RenderPipeline {
    pub fn new(
        device: &wgpu::Device,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("src/shader.wgsl"));
        let pipeline = graphics::create_pipeline(
            device,
            &PipelineInfo {
                shader: &shader,
                vertex_entry_point: "vs_main",
                fragment_entry_point: "fs_main",
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                global_bind_group_layout: &global_bind_group_layout,
                vertex_buffers: &[ParticleMeshVertex::layout(), Vertex::instance_layout()],
                fragment_targets: &[Some(wgpu::ColorTargetState { 
                    format: surface_format, 
                    blend: Some(wgpu::BlendState::REPLACE), 
                    write_mask: wgpu::ColorWrites::ALL 
                })],
            }
        );
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        Self { pipeline, index_buffer }
    }

    pub fn update(
        &self,
        render_pass: &mut wgpu::RenderPass,
        global_bind_group: &wgpu::BindGroup,
        num_indices: u32,
        num_instances: u32,
    ) {
        self.pipeline.update(render_pass, global_bind_group);
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
    }
}