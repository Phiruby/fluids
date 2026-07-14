mod camera;
mod graphics;
mod physics;
mod texture;
use crate::camera::CameraController;
use crate::graphics::compute::PhysicsPipeline;
use crate::graphics::{MeshRenderPipeline, RayPipeline, RenderPipeline};
use crate::physics::{
    ContainerBoundary, Cube, CubicKernel, FluidParticle, FluidSim, assign_rest_densities,
    fluid_cube,
};
use crate::texture::Texture;
use cgmath::SquareMatrix;
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
use cgmath;
use nalgebra::Vector3;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    Particles,
    Mesh,
}

const RENDER_MODE: RenderMode = RenderMode::Mesh;

const FLUID_GRID_N: usize = 8;
const FLUID_SPACING: f32 = 0.05;
const FLUID_PARTICLE_RADIUS: f32 = 0.02;
const FLUID_ORIGIN: [f32; 3] = [-0.6, 0.2, -0.2];
const FLUID_CELL_SIZE: f32 = 0.1;
const FLUID_CONTAINER_LAYERS: usize = 5;
const FLUID_WORKGROUP: usize = 256;

pub(crate) struct Ray {
    start: cgmath::Point3<f32>,
    end: cgmath::Point3<f32>,
}

impl Ray {
    pub fn project_to(&self, axis: Vector3<f32>) -> Vector3<f32> {
        let e = Vector3::new(1.0, 1.0, 1.0);
        let axis = axis.normalize();
        let start = Vector3::new(self.start.x, self.start.y, self.start.z);
        let end = Vector3::new(self.end.x, self.end.y, self.end.z);
        let sum_r0 = e.dot(&start);
        let dir = (end - start).normalize();
        let sum_r = e.dot(&dir);
        let sum_v = e.dot(&axis);
        let u = (sum_r0 + sum_r0 * sum_r) / (1.0 + sum_r * sum_v);
        let t = (sum_r0 - u * sum_v) / dir.norm();
        let closest_to_axis = start + dir * t.max(0.0);
        // project this to the axis
        let projected = axis * closest_to_axis.dot(&axis);
        projected
    }
}
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

pub(crate) struct MouseState {
    position: winit::dpi::PhysicalPosition<f64>,
    pub is_pressed: bool,
}
impl MouseState {
    fn new() -> Self {
        Self {
            position: winit::dpi::PhysicalPosition::new(0.0, 0.0),
            is_pressed: false,
        }
    }
    fn process_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left && state == ElementState::Pressed {
            self.is_pressed = true;
        } else if button == MouseButton::Left && state == ElementState::Released {
            self.is_pressed = false;
        }
    }
    fn process_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        self.position = position;
    }
    fn world_ray(&self, camera: &Camera, viewport_width: f32, viewport_height: f32) -> Ray {
        let viewport_width = viewport_width.max(1.0);
        let viewport_height = viewport_height.max(1.0);
        let ndc_x = (self.position.x as f32 / viewport_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (self.position.y as f32 / viewport_height) * 2.0;
        let inverse_view_projection = camera
            .build_view_projection_matrix()
            .invert()
            .expect("camera view projection matrix must be invertible");
        let near_clip = cgmath::Vector4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_clip = cgmath::Vector4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near_world = inverse_view_projection * near_clip;
        let far_world = inverse_view_projection * far_clip;
        let near_world = near_world.truncate() / near_world.w;
        let far_world = far_world.truncate() / far_world.w;
        Ray {
            start: cgmath::Point3::new(near_world.x, near_world.y, near_world.z),
            end: cgmath::Point3::new(far_world.x, far_world.y, far_world.z),
        }
    }
}

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
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let projection =
            cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        return OPENGL_TO_WGPU_MATRIX * projection * view;
    }
}
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
    mesh_pipeline: MeshRenderPipeline,
    ray_pipeline: RayPipeline,
    fluid_pipeline: RenderPipeline,
    physics_pipeline: PhysicsPipeline,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    #[allow(unused)]
    camera_bind_group: wgpu::BindGroup,
    #[allow(unused)]
    camera_buffer: wgpu::Buffer,
    depth_texture: Texture,
    #[allow(unused)]
    object: Cube,
    mouse_state: MouseState,
    fluid_sim: FluidSim,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
}
impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_projection: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_projection(&mut self, camera: &Camera) {
        self.view_projection = camera.build_view_projection_matrix().into();
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
            eye: (0.0, 1.0, 4.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: (0.0, 1.0, 0.0).into(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
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
                    visibility: wgpu::ShaderStages::VERTEX,
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

        let m = 10;
        let kernel = CubicKernel::new(1.0 / (m - 1) as f32);
        let cube = Cube::new(m, 1.0 / m as f32, &kernel);
        let mesh_pipeline = MeshRenderPipeline::new(
            &device,
            &camera_bind_group_layout,
            config.format,
            &cube.surface_indices(),
        );
        let ray_pipeline = RayPipeline::new(&device, &camera_bind_group_layout, config.format);
        let physics_pipeline = PhysicsPipeline::new(&device, Cube::new(m, 1.0 / m as f32, &kernel));
        let depth_texture = Texture::create_depth_texture(&device, &config, "Depth texture");

        // Construct: fluid cube init bounded by a cube-boundary
        let mut fluid_particles = fluid_cube(FLUID_GRID_N, FLUID_SPACING, Vector3::from(FLUID_ORIGIN));
        let container = ContainerBoundary {
            top_left: nalgebra::Point3::new(-0.75, 0.05, -0.4),
            bottom_right: nalgebra::Point3::new(-0.1, 0.7, 0.3),
            nlayers: FLUID_CONTAINER_LAYERS,
            spacing: FLUID_SPACING / 2.0,
        };
        let mut boundary_points = container.create_points(FLUID_CONTAINER_LAYERS);
        fluid_particles.append(&mut boundary_points);
        assign_rest_densities(&mut fluid_particles, &CubicKernel::new(FLUID_CELL_SIZE));
        let padded_len = fluid_particles.len().next_multiple_of(FLUID_WORKGROUP);
        fluid_particles.resize(padded_len, FluidParticle::inert());

        let fluid_sim = FluidSim::new(&device, &fluid_particles);
        let fluid_pipeline = RenderPipeline::new(
            &device,
            &camera_bind_group_layout,
            config.format,
            FLUID_PARTICLE_RADIUS,
        );
        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            mesh_pipeline,
            ray_pipeline,
            fluid_pipeline,
            physics_pipeline,
            camera,
            camera_controller: CameraController::new(),
            camera_bind_group,
            camera_buffer,
            camera_uniform,
            depth_texture,
            object: cube,
            mouse_state: MouseState::new(),
            fluid_sim,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.camera.aspect = width as f32 / height as f32;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.depth_texture =
                Texture::create_depth_texture(&self.device, &self.config, "Depth texture");
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

        self.physics_pipeline
            .compute(&self.device, &self.queue, &mut encoder);

        self.fluid_sim
            .step(&self.device, &self.queue, &mut encoder);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
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

            self.mesh_pipeline.draw(
                &mut render_pass,
                &self.camera_bind_group,
                &self.physics_pipeline.get_output_buffer(),
            );

            self.fluid_pipeline.draw(
                &mut render_pass,
                &self.camera_bind_group,
                self.fluid_sim.particle_buffer(),
                self.fluid_sim.num_particles(),
            );

            self.ray_pipeline
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
        self.camera_controller.process_key(code, is_pressed);
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        self.camera_controller.process_mouse_button(button, state);
        self.mouse_state.process_mouse_button(button, state);
    }

    fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        self.camera_controller
            .process_cursor_moved(position, &mut self.camera);
        self.mouse_state.process_cursor_moved(position);
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_projection(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        let ray = self.mouse_state.world_ray(
            &self.camera,
            self.config.width as f32,
            self.config.height as f32,
        );
        self.ray_pipeline
            .update(&self.queue, ray.start, ray.end, [0.2, 1.0, 0.2]);
        self.physics_pipeline
            .update(&self.queue, ray.start, ray.end, self.mouse_state.is_pressed);
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

