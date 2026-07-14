use crate::graphics::BufferDebug;
use crate::physics::Cube;
use crate::physics::{ElasticSolidParticle, PhysicsVertex};
use bytemuck;
use cgmath::Point3;
use wgpu::util::DeviceExt;
use crate::physics::{compute_densities, kernel_weight};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ReferenceGradient {
    gradient: [f32; 3],
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<ReferenceGradient>() == 16);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    ray_origin: [f32; 3],
    mouse_pressed: u32,
    ray_end: [f32; 3],
    grid_n: u32,
    ref_min: [f32; 3],
    _pad0: u32,
    ref_max: [f32; 3],
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InteractionState {
    selected_face: i32,
}

impl ReferenceGradient {
    fn new(gradient: [f32; 3]) -> Self {
        Self { gradient, _pad: 0 }
    }
}

pub struct PhysicsPipeline {
    pipeline: wgpu::ComputePipeline,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    alt_bind_group: wgpu::BindGroup,
    current_particles_buffer: wgpu::Buffer,
    reference_gradients_buffer: wgpu::Buffer,
    reference_kernel_weights_buffer: wgpu::Buffer,
    reference_particles_buffer: wgpu::Buffer,
    sim_params_buffer: wgpu::Buffer,
    interaction_buffer: wgpu::Buffer,
    sim_params: SimParams,
    particle_count: u32,
    use_alt: bool,
    debugger: BufferDebug,
}

impl PhysicsPipeline {
    pub fn new(device: &wgpu::Device, cube: Cube) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../compute.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Physics pipeline"),
            module: &shader,
            entry_point: None,
            layout: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let reference_densities =
            compute_densities(&cube.reference.vertices, &cube.kernel, &cube.neighbors);

        let reference_particles = cube
            .reference
            .vertices
            .iter()
            .zip(cube.neighbors.iter())
            .zip(reference_densities.iter())
            .zip(cube.L.iter())
            .map(|(((vertex, neighbors), density), L)| {
                ElasticSolidParticle::new(
                    vertex.position,
                    vertex.velocity,
                    vertex.mass,
                    *density,
                    vertex.renderable as u32,
                    neighbors.len() as u32,
                    neighbors.to_owned(),
                    *L,
                )
            })
            .collect::<Vec<_>>();

        let current_particles = cube
            .current
            .vertices
            .iter()
            .zip(cube.neighbors.iter())
            .zip(reference_densities.iter())
            .zip(cube.L.iter())
            .map(|(((vertex, neighbors), density), L)| {
                ElasticSolidParticle::new(
                    vertex.position,
                    vertex.velocity,
                    vertex.mass,
                    *density,
                    vertex.renderable as u32,
                    neighbors.len() as u32,
                    neighbors.to_owned(),
                    *L,
                )
            })
            .collect::<Vec<_>>();

        let reference_gradients: Vec<ReferenceGradient> = cube
            .reference_gradients
            .into_iter()
            .flatten()
            .map(|gradient| ReferenceGradient::new(gradient.data.0[0]))
            .collect();

        let reference_kernel_weights: Vec<f32> = cube
            .reference_kernel_weights
            .into_iter()
            .flatten()
            .collect();

        let out = current_particles.clone();
        let particle_count = current_particles.len() as u32;
        let grid_n = (particle_count as f32).cbrt().round() as u32;

        let (ref_min, ref_max) = reference_bounds(&cube.reference.vertices);

        let sim_params = SimParams {
            ray_origin: [0.0; 3],
            mouse_pressed: 0,
            ray_end: [0.0, 0.0, 1.0],
            grid_n,
            ref_min,
            _pad0: 0,
            ref_max,
            _pad1: 0,
        };

        let particle_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::VERTEX;

        let reference_particles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input buffer"),
                contents: bytemuck::cast_slice(&reference_particles),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Output buffer"),
            contents: bytemuck::cast_slice(&out),
            usage: particle_usage,
        });
        let reference_gradients_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reference gradients buffer"),
                contents: bytemuck::cast_slice(&reference_gradients),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let reference_kernel_weights_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reference kernel weights buffer"),
                contents: bytemuck::cast_slice(&reference_kernel_weights),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let current_particles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Current particles buffer"),
                contents: bytemuck::cast_slice(&current_particles),
                usage: particle_usage,
            });
        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim params buffer"),
            contents: bytemuck::cast_slice(&[sim_params]),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let interaction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Interaction state buffer"),
            contents: bytemuck::cast_slice(&[InteractionState { selected_face: -1 }]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reference_particles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current_particles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reference_gradients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: reference_kernel_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: interaction_buffer.as_entire_binding(),
                },
            ],
        });

        let alt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reference_particles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reference_gradients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: reference_kernel_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: current_particles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: interaction_buffer.as_entire_binding(),
                },
            ],
        });

        let buffer_size = reference_particles_buffer.size();
        Self {
            pipeline,
            output_buffer,
            bind_group,
            alt_bind_group,
            current_particles_buffer,
            reference_gradients_buffer,
            reference_kernel_weights_buffer,
            reference_particles_buffer,
            sim_params_buffer,
            interaction_buffer,
            sim_params,
            particle_count,
            use_alt: false,
            debugger: BufferDebug::new(device, buffer_size),
        }
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        start: Point3<f32>,
        end: Point3<f32>,
        mouse_pressed: bool,
    ) {
        self.sim_params.ray_origin = [start.x, start.y, start.z];
        self.sim_params.ray_end = [end.x, end.y, end.z];
        self.sim_params.mouse_pressed = u32::from(mouse_pressed);
        queue.write_buffer(
            &self.sim_params_buffer,
            0,
            bytemuck::cast_slice(&[self.sim_params]),
        );
    }

    pub fn compute(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // self.debugger
        //     .log::<Particle>(_device, _queue, &self.output_buffer);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        if self.use_alt {
            pass.set_bind_group(0, &self.alt_bind_group, &[]);
        } else {
            pass.set_bind_group(0, &self.bind_group, &[]);
        }
        self.use_alt = !self.use_alt;
        let workgroups = self.particle_count.div_ceil(256);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn get_output_buffer(&self) -> &wgpu::Buffer {
        if self.use_alt {
            &self.output_buffer
        } else {
            &self.current_particles_buffer
        }
    }
}

fn reference_bounds(vertices: &[PhysicsVertex]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices {
        let p = [v.position.x, v.position.y, v.position.z];
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    (min, max)
}
