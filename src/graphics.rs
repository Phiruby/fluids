use crate::texture::Texture;
use wgpu;

pub mod debug_plot;
pub mod marching_cubes;
mod pipelines;
mod ssfr;
use flume::bounded;
pub use debug_plot::{LivePlot, PlotWindow};
pub use marching_cubes::MarchingCubes;
pub use pipelines::{
    MarchingCubesRenderPipeline, RenderPipeline, WireframeBox, WireframeBoxPipeline,
};
pub use ssfr::ScreenSpaceRendering;
use std::fmt::Debug;

pub(in crate::graphics) struct PipelineInfo<'a> {
    shader: &'a wgpu::ShaderModule,
    vertex_entry_point: &'a str,
    fragment_entry_point: &'a str,
    primitive_topology: wgpu::PrimitiveTopology,
    global_bind_group_layout: &'a wgpu::BindGroupLayout,
    vertex_buffers: &'a [wgpu::VertexBufferLayout<'a>],
    fragment_targets: &'a [Option<wgpu::ColorTargetState>],
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
}

pub(in crate::graphics) struct PipelineState {
    pipeline: wgpu::RenderPipeline,
}

impl PipelineState {
    pub(in crate::graphics) fn bind(
        &self,
        render_pass: &mut wgpu::RenderPass,
        global_bind_group: &wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, global_bind_group, &[]);
    }
}

pub(in crate::graphics) fn create_pipeline(
    device: &wgpu::Device,
    pipeline_info: &PipelineInfo,
) -> PipelineState {
    let shader = pipeline_info.shader;
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[Some(pipeline_info.global_bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(pipeline_info.vertex_entry_point),
            buffers: pipeline_info.vertex_buffers,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(pipeline_info.fragment_entry_point),
            targets: pipeline_info.fragment_targets,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: pipeline_info.primitive_topology,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: Texture::DEPTH_FORMAT,
            depth_write_enabled: Some(pipeline_info.depth_write_enabled),
            depth_compare: Some(pipeline_info.depth_compare),
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
    PipelineState { pipeline }
}

pub struct BufferDebug {
    staging_buffer: wgpu::Buffer,
    plot: Option<LivePlot>,
    sample_every: u32,
    frame: u32,
}

impl BufferDebug {
    pub fn new(device: &wgpu::Device, buffer_size: u64) -> Self {
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            staging_buffer,
            plot: None,
            sample_every: 1,
            frame: 0,
        }
    }

    /// Same as [`Self::new`], plus a live plot handle (window attached later on the event loop).
    pub fn with_plot(device: &wgpu::Device, buffer_size: u64, title: &str) -> Self {
        let mut dbg = Self::new(device, buffer_size);
        dbg.plot = Some(LivePlot::new(title));
        dbg.sample_every = 2;
        dbg
    }

    pub fn sample_every(&mut self, n: u32) {
        self.sample_every = n.max(1);
    }

    pub fn plot(&self) -> Option<&LivePlot> {
        self.plot.as_ref()
    }

    /// Copy `buffer[..count]` to CPU. Submits its own copy — not for in-flight encoders.
    /// `buffer` must have `COPY_SRC` usage.
    pub fn read<T: bytemuck::Pod + bytemuck::Zeroable>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u64,
    ) -> Vec<T> {
        let size = (count * std::mem::size_of::<T>() as u64).min(self.staging_buffer.size());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BufferDebug copy"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &self.staging_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = bounded(1);
        self.staging_buffer
            .slice(..size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll while waiting for buffer map");
        rx.recv()
            .expect("map_async callback channel closed")
            .expect("buffer map failed");

        let values = {
            let data = self.staging_buffer.slice(..size).get_mapped_range();
            bytemuck::cast_slice::<u8, T>(&data).to_vec()
        };
        self.staging_buffer.unmap();
        values
    }

    /// Read buffer, reduce to named metrics, push to the live plot (if enabled).
    pub fn sample<T, F>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u64,
        t: f64,
        reduce: F,
    ) where
        T: bytemuck::Pod + bytemuck::Zeroable,
        F: FnOnce(&[T]) -> Vec<(String, f64)>,
    {
        self.frame = self.frame.wrapping_add(1);
        let Some(plot) = self.plot.clone() else {
            return;
        };
        if !self.frame.is_multiple_of(self.sample_every) {
            return;
        }
        let data = self.read::<T>(device, queue, buffer, count);
        let values = reduce(&data);
        if values.is_empty() {
            log::warn!("BufferDebug::sample produced no metrics at t={t}");
        }
        plot.sample(t, values);
    }

    pub fn log<T: bytemuck::Pod + bytemuck::Zeroable + Debug>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) {
        let n = buffer.size() / std::mem::size_of::<T>() as u64;
        dbg!(self.read::<T>(device, queue, buffer, n));
    }
}
