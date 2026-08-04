mod camera;
mod graphics;
mod physics;
mod texture;
use crate::camera::CameraController;
use crate::graphics::{
    MarchingCubes, MarchingCubesRenderPipeline, RenderPipeline, ScreenSpaceRendering,
    WireframeBoxPipeline,
};
use crate::physics::{
    ContainerBoundary, CubicKernel, FluidParticle, FluidSim, assign_rest_densities, fluid_cube,
};
use crate::texture::Texture;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use bytemuck;
use cgmath::SquareMatrix;
use nalgebra::Vector3;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

#[derive(Clone, Copy, PartialEq, Eq)]
enum FluidRenderMode {
    Particles,
    MarchingCubes,
    ScreenSpaceFluidRendering,
}
// marching cubes slow and bad :(
const FLUID_RENDER_MODE: FluidRenderMode = FluidRenderMode::ScreenSpaceFluidRendering;

const FLUID_GRID_N: usize = 30;
const FLUID_SPACING: f32 = 0.03;
const FLUID_PARTICLE_RADIUS: f32 = 0.014;
const FLUID_ORIGIN: [f32; 3] = [-0.95, 0.12, -0.7];
const FLUID_CELL_SIZE: f32 = 0.1;
const FLUID_CONTAINER_LAYERS: usize = 2;
const FLUID_WORKGROUP: usize = 256;
const ISO_LEVEL_STEP: f32 = 100.0;

const CONTAINER_MIN: [f32; 3] = [-1.20, 0.05, -1.0];
const CONTAINER_MAX: [f32; 3] = [0.1, 1.85, 0.8];

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

pub(crate) struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}
impl Camera {
    fn build_view_projection_matrix(&self) -> (cgmath::Matrix4<f32>, cgmath::Matrix4<f32>) {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let projection =
            cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        return (OPENGL_TO_WGPU_MATRIX * projection, view);
    }
}
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
    container_pipeline: WireframeBoxPipeline,
    fluid_pipeline: RenderPipeline,
    marching_cubes: MarchingCubes,
    screen_space_rendering: ScreenSpaceRendering,
    mc_pipeline: MarchingCubesRenderPipeline,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    #[allow(unused)]
    camera_bind_group: wgpu::BindGroup,
    #[allow(unused)]
    camera_buffer: wgpu::Buffer,
    depth_texture: Texture,
    fluid_sim: FluidSim,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    particle_radius: f32,
    screen_height: f32,
    _pad: [f32; 2],
}
impl CameraUniform {
    fn new() -> Self {
        Self {
            view: cgmath::Matrix4::identity().into(),
            projection: cgmath::Matrix4::identity().into(),
            particle_radius: 0.1,
            screen_height: 100.0,
            _pad: [0.0; 2],
        }
    }

    fn update_view_projection(&mut self, camera: &Camera) {
        let (projection, view) = camera.build_view_projection_matrix();
        self.projection = projection.into();
        self.view = view.into();
    }
}
impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let device_limits = adapter.limits();

        #[cfg(not(target_arch = "wasm32"))]
        let mut required_limits = wgpu::Limits::default();
        #[cfg(target_arch = "wasm32")]
        let mut required_limits = wgpu::Limits::downlevel_webgl2_defaults();

        required_limits.max_storage_buffer_binding_size =
            device_limits.max_storage_buffer_binding_size;
        required_limits.max_buffer_size = device_limits.max_buffer_size;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;
        let surface_capabilities = surface.get_capabilities(&adapter);

        let surface_format = surface_capabilities
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_capabilities.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        let camera = Camera {
            eye: (-0.4, 1.4, 3.2).into(),
            target: (-0.55, 0.55, -0.1).into(),
            up: (0.0, 1.0, 0.0).into(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.particle_radius = FLUID_PARTICLE_RADIUS;
        camera_uniform.screen_height = config.height as f32;
        camera_uniform.update_view_projection(&camera);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let depth_texture =
            Texture::create_depth_texture(&device, config.width, config.height, "Depth texture");
        let container_pipeline = WireframeBoxPipeline::new(
            &device,
            &camera_bind_group_layout,
            config.format,
            CONTAINER_MIN,
            CONTAINER_MAX,
        );

        let mut fluid_particles =
            fluid_cube(FLUID_GRID_N, FLUID_SPACING, Vector3::from(FLUID_ORIGIN));
        let n_fluid = fluid_particles.len() as u32;
        let container = ContainerBoundary {
            top_left: nalgebra::Point3::new(CONTAINER_MIN[0], CONTAINER_MIN[1], CONTAINER_MIN[2]),
            bottom_right: nalgebra::Point3::new(
                CONTAINER_MAX[0],
                CONTAINER_MAX[1],
                CONTAINER_MAX[2],
            ),
            nlayers: FLUID_CONTAINER_LAYERS,
            spacing: FLUID_SPACING / 2.0,
        };
        let mut boundary_points = container.create_points(FLUID_CONTAINER_LAYERS);
        assign_rest_densities(
            &mut fluid_particles,
            &mut boundary_points,
            FLUID_SPACING,
            &CubicKernel::new(FLUID_CELL_SIZE),
        );
        fluid_particles.append(&mut boundary_points);
        let padded_len = fluid_particles.len().next_multiple_of(FLUID_WORKGROUP);
        fluid_particles.resize(padded_len, FluidParticle::inert());

        let fluid_sim = FluidSim::new(&device, &fluid_particles);
        let fluid_pipeline = RenderPipeline::new(
            &device,
            &camera_bind_group_layout,
            config.format,
            FLUID_PARTICLE_RADIUS,
        );
        let marching_cubes = MarchingCubes::new(&device);
        let mc_pipeline =
            MarchingCubesRenderPipeline::new(&device, &camera_bind_group_layout, config.format);
        // TODO: needs to be updated on screen resize
        let screen_space_rendering = ScreenSpaceRendering::new(
            &device,
            camera_bind_group_layout,
            fluid_particles.len() as u32,
            n_fluid,
            config.format,
            size.width,
            size.height,
        );
        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            container_pipeline,
            fluid_pipeline,
            marching_cubes,
            mc_pipeline,
            camera,
            camera_controller: CameraController::new(),
            camera_bind_group,
            camera_buffer,
            camera_uniform,
            depth_texture,
            fluid_sim,
            screen_space_rendering,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.camera.aspect = width as f32 / height as f32;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            let height = self.config.height;
            let width = self.config.width;
            self.depth_texture =
                Texture::create_depth_texture(&self.device, width, height, "Depth texture");
            self.camera_uniform.screen_height = height as f32;
        }
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.surface.configure(&self.device, &self.config);
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return Ok(());
            }

            wgpu::CurrentSurfaceTexture::Lost => {
                anyhow::bail!("Device lost");
            }

            _ => anyhow::bail!("Unknown error"),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command encoder for rendering"),
            });

        self.fluid_sim.step(&self.device, &self.queue, &mut encoder);

        if FLUID_RENDER_MODE == FluidRenderMode::MarchingCubes {
            self.marching_cubes.extract(
                &self.device,
                &mut encoder,
                self.fluid_sim.particle_buffer(),
                self.fluid_sim.hash_offsets_buffer(),
            );
        }

        if FLUID_RENDER_MODE == FluidRenderMode::ScreenSpaceFluidRendering {
            // perform compaction (filter out boundary particles) b4 creating render pass
            self.screen_space_rendering.prepare(
                &self.device,
                self.fluid_sim.particle_buffer(),
                &mut encoder,
                &self.camera_bind_group,
            );
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            match FLUID_RENDER_MODE {
                FluidRenderMode::Particles => {
                    self.fluid_pipeline.draw(
                        &mut render_pass,
                        &self.camera_bind_group,
                        self.fluid_sim.particle_buffer(),
                        self.fluid_sim.num_particles(),
                    );
                }
                FluidRenderMode::MarchingCubes => {
                    self.mc_pipeline.draw(
                        &mut render_pass,
                        &self.camera_bind_group,
                        self.marching_cubes.vertex_buffer(),
                        self.marching_cubes.index_buffer(),
                        self.marching_cubes.indirect_buffer(),
                    );
                }
                FluidRenderMode::ScreenSpaceFluidRendering => self.screen_space_rendering.draw(
                    &self.device,
                    &mut render_pass,
                    &self.camera_bind_group,
                ),
            }

            self.container_pipeline
                .draw(&mut render_pass, &self.camera_bind_group);
        }

        self.queue.submit([encoder.finish()]);
        output.present();
        Ok(())
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if code == KeyCode::Escape && is_pressed {
            event_loop.exit();
            return;
        }
        if FLUID_RENDER_MODE == FluidRenderMode::MarchingCubes && is_pressed {
            let delta = match code {
                KeyCode::BracketRight | KeyCode::Equal => ISO_LEVEL_STEP,
                KeyCode::BracketLeft | KeyCode::Minus => -ISO_LEVEL_STEP,
                _ => 0.0,
            };
            if delta != 0.0 {
                let iso = self.marching_cubes.nudge_iso_level(&self.queue, delta);
                self.window
                    .set_title(&format!("fluids — iso level {iso:.1}"));
                return;
            }
        }
        self.camera_controller.process_key(code, is_pressed);
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        if FLUID_RENDER_MODE != FluidRenderMode::MarchingCubes {
            return;
        }
        let steps = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => (pos.y as f32 / 40.0),
        };
        let iso = self
            .marching_cubes
            .nudge_iso_level(&self.queue, steps * ISO_LEVEL_STEP);
        self.window
            .set_title(&format!("fluids — iso level {iso:.1}"));
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        self.camera_controller.process_mouse_button(button, state);
    }

    fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        self.camera_controller
            .process_cursor_moved(position, &mut self.camera);
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_projection(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            // If we are not on web we can use pollster to
            // await the
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Run the future asynchronously and use the
            // proxy to send the results to the event loop
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(
                                State::new(window)
                                    .await
                                    .expect("Unable to create canvas!!!")
                            )
                            .is_ok()
                    )
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("{e:?}");
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                state.handle_mouse_button(button, button_state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.handle_cursor_moved(position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                state.handle_mouse_wheel(delta);
            }
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = App::new();
        event_loop.run_app(&mut app)?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let app = App::new(&event_loop);
        event_loop.spawn_app(app);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
