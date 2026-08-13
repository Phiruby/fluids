use crate::graphics::SCENE_CLEAR;
use crate::physics::{InflowSource, MpmWorld, RigidBody, SphWorld, MpmFluidParticle, MpmElasticParticle, MpmFluidSim, MpmObject};
use crate::{FLUID_RENDER_MODE, FluidRenderMode, MPM_MATERIAL, MpmMaterial, State};

/// parked far from the domain so P2G/G2P skip them (x > 50 condition in shaders), but y < 0 so emit
/// does not treat them as pipe reservoir particles.
const PARK_HIDDEN: [f32; 3] = [1000.0, -1000.0, 1000.0];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SimBackend {
    Sph,
    Mpm,
}

pub trait SimulationBackend {
    type World: Default + 'static;
    fn into_particles(
        world: Self::World,
    ) -> (
        Vec<crate::physics::FluidParticle>,
        Vec<crate::physics::MpmFluidParticle>,
        Vec<crate::physics::MpmElasticParticle>,
        Vec<RigidBody>,
        Vec<InflowSource>,
    );

    fn finalize(world: Self::World, events: Vec<MpmSceneEvent>) -> (Self::World, MpmTimeline) {
        let _ = events;
        (world, MpmTimeline::default())
    }

    fn simulate(state: &mut State<Self>, encoder: &mut wgpu::CommandEncoder)
    where
        Self: Sized,
    {
        panic!("Not implemented");
    }
    fn draw(state: &mut State<Self>, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView)
    where
        Self: Sized,
    {
        panic!("Not implemented");
    }
}

pub enum MpmSceneEvent {
    Add(Box<dyn MpmObject>),
    Wait(f32),
}

struct DeferredSpawn {
    at_time: f32,
    rigids: Vec<(u32, RigidBody)>,
    particles: Vec<(u32, MpmFluidParticle)>,
}

#[derive(Default)]
pub struct MpmTimeline {
    sim_time: f32,
    pending: Vec<DeferredSpawn>,
}


impl MpmTimeline {
    pub fn tick(&mut self, dt: f32, queue: &wgpu::Queue, sim: &MpmFluidSim) {
        if self.pending.is_empty() {
            return;
        }
        self.sim_time += dt;
        let mut i = 0;
        while i < self.pending.len() {
            if self.sim_time + 1e-8 >= self.pending[i].at_time {
                let spawn = self.pending.remove(i);
                spawn.upload(queue, sim);
            } else {
                i += 1;
            }
        }
    }
}

impl DeferredSpawn {
    fn upload(&self, queue: &wgpu::Queue, sim: &MpmFluidSim) {
        if !self.rigids.is_empty() {
            let start = self.rigids[0].0;
            let bodies: Vec<RigidBody> = self.rigids.iter().map(|(_, b)| *b).collect();
            sim.write_rigid_bodies(queue, start, &bodies);
        }
        if !self.particles.is_empty() {
            let start = self.particles[0].0;
            let particles: Vec<MpmFluidParticle> = self.particles.iter().map(|(_, p)| *p).collect();
            sim.write_fluid_particles(queue, start, &particles);
        }
    }
}


pub enum Mpm {}
pub enum Sph {}

impl SimulationBackend for Mpm {
    type World = MpmWorld;

    fn finalize(world: MpmWorld, events: Vec<MpmSceneEvent>) -> (MpmWorld, MpmTimeline) {
        bake_mpm(world, events)
    }

    fn into_particles(
        world: Self::World,
    ) -> (
        Vec<crate::physics::FluidParticle>,
        Vec<crate::physics::MpmFluidParticle>,
        Vec<crate::physics::MpmElasticParticle>,
        Vec<RigidBody>,
        Vec<InflowSource>,
    ) {
        (
            Vec::new(),
            world.fluid_particles,
            world.elastic_particles,
            world.rigid_bodies,
            world.inflow_sources,
        )
    }

    fn simulate(state: &mut State<Mpm>, encoder: &mut wgpu::CommandEncoder) {
        state.mpm_fluid_sim.step(&state.device, encoder);
    }
    fn draw(state: &mut State<Mpm>, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        if FLUID_RENDER_MODE == FluidRenderMode::ScreenSpaceFluidRendering
            && MPM_MATERIAL == MpmMaterial::Fluid
        {
            {
                let mut pass = state.mpm_screen_space.begin_opaque_pass(encoder);
                state.sky_pipeline.draw(&mut pass);
                state.rigid_pipeline.draw(
                    &state.device,
                    &mut pass,
                    &state.camera_bind_group,
                    state.mpm_fluid_sim.rigid_buffer(),
                    state.mpm_fluid_sim.num_rigid_bodies(),
                );
            }
            state.mpm_screen_space.prepare(
                &state.device,
                state.mpm_fluid_sim.particle_buffer(),
                encoder,
                &state.camera_bind_group,
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
                        load: wgpu::LoadOp::Clear(SCENE_CLEAR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &state.depth_texture.view,
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

            state.sky_pipeline.draw(&mut render_pass);

            match FLUID_RENDER_MODE {
                FluidRenderMode::Particles => match MPM_MATERIAL {
                    MpmMaterial::Fluid => {
                        state.mpm_pipeline.draw(
                            &mut render_pass,
                            &state.camera_bind_group,
                            state.mpm_fluid_sim.particle_buffer(),
                            state.mpm_fluid_sim.num_particles(),
                        );
                    }
                    MpmMaterial::Elastic => {
                        state.mpm_elastic_pipeline.draw(
                            &mut render_pass,
                            &state.camera_bind_group,
                            state.mpm_fluid_sim.elastic_particle_buffer(),
                            state.mpm_fluid_sim.num_elastic_particles(),
                        );
                    }
                },
                FluidRenderMode::ScreenSpaceFluidRendering => {
                    assert!(
                        MPM_MATERIAL == MpmMaterial::Fluid,
                        "SSFR is only for MPM fluid particles"
                    );
                    state.mpm_screen_space.draw(
                        &state.device,
                        &mut render_pass,
                        &state.camera_bind_group,
                    );
                }
                _ => panic!("Mpm does not support this render mode"),
            }

            state.wireframe_pipeline.draw(
                &mut render_pass,
                &state.camera_bind_group,
                &state.mpm_wireframe,
            );
            state.wireframe_pipeline.draw(
                &mut render_pass,
                &state.camera_bind_group,
                &state.mpm_collider_wireframe,
            );
            if FLUID_RENDER_MODE != FluidRenderMode::ScreenSpaceFluidRendering {
                state.rigid_pipeline.draw(
                    &state.device,
                    &mut render_pass,
                    &state.camera_bind_group,
                    state.mpm_fluid_sim.rigid_buffer(),
                    state.mpm_fluid_sim.num_rigid_bodies(),
                );
            }
        }
    }
}

impl SimulationBackend for Sph {
    type World = SphWorld;

    fn into_particles(
        world: Self::World,
    ) -> (
        Vec<crate::physics::FluidParticle>,
        Vec<crate::physics::MpmFluidParticle>,
        Vec<crate::physics::MpmElasticParticle>,
        Vec<RigidBody>,
        Vec<InflowSource>,
    ) {
        (
            world.fluid_particles,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    fn simulate(state: &mut State<Sph>, encoder: &mut wgpu::CommandEncoder) {
        state.fluid_sim.step(&state.device, &state.queue, encoder);
    }
    fn draw(state: &mut State<Sph>, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        if FLUID_RENDER_MODE == FluidRenderMode::MarchingCubes {
            state.marching_cubes.extract(
                &state.device,
                encoder,
                state.fluid_sim.particle_buffer(),
                state.fluid_sim.hash_offsets_buffer(),
            );
        }

        if FLUID_RENDER_MODE == FluidRenderMode::ScreenSpaceFluidRendering {
            {
                let mut pass = state.sph_screen_space.begin_opaque_pass(encoder);
                state.sky_pipeline.draw(&mut pass);
            }
            state.sph_screen_space.prepare(
                &state.device,
                state.fluid_sim.particle_buffer(),
                encoder,
                &state.camera_bind_group,
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
                        load: wgpu::LoadOp::Clear(SCENE_CLEAR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &state.depth_texture.view,
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

            state.sky_pipeline.draw(&mut render_pass);

            match FLUID_RENDER_MODE {
                FluidRenderMode::Particles => {
                    state.fluid_pipeline.draw(
                        &mut render_pass,
                        &state.camera_bind_group,
                        state.fluid_sim.particle_buffer(),
                        state.fluid_sim.num_particles(),
                    );
                }
                FluidRenderMode::MarchingCubes => {
                    state.mc_pipeline.draw(
                        &mut render_pass,
                        &state.camera_bind_group,
                        state.marching_cubes.vertex_buffer(),
                        state.marching_cubes.index_buffer(),
                        state.marching_cubes.indirect_buffer(),
                    );
                }
                FluidRenderMode::ScreenSpaceFluidRendering => {
                    state.sph_screen_space.draw(
                        &state.device,
                        &mut render_pass,
                        &state.camera_bind_group,
                    );
                }
            }

            state.wireframe_pipeline.draw(
                &mut render_pass,
                &state.camera_bind_group,
                &state.sph_wireframe,
            );
        }
    }
}


pub(crate) fn bake_mpm(mut world: MpmWorld, events: Vec<MpmSceneEvent>) -> (MpmWorld, MpmTimeline) {
    let mut at_time = 0.0;
    let mut pending = Vec::new();
    for event in events {
        match event {
            MpmSceneEvent::Wait(seconds) => {
                at_time += seconds.max(0.0);
            }
            MpmSceneEvent::Add(object) => {
                let rigid_start = world.rigid_bodies.len();
                let particle_start = world.fluid_particles.len();
                object.insert(&mut world);
                if at_time > 0.0 {
                    pending.push(hide_inserted(
                        &mut world,
                        at_time,
                        rigid_start,
                        particle_start,
                    ));
                }
            }
        }
    }
    (
        world,
        MpmTimeline {
            sim_time: 0.0,
            pending,
        },
    )
}

fn hide_inserted(
    world: &mut MpmWorld,
    at_time: f32,
    rigid_start: usize,
    particle_start: usize,
) -> DeferredSpawn {
    let mut rigids = Vec::new();
    let live = world.rigid_bodies[rigid_start];
    rigids.push((rigid_start as u32, live));
    let mut hidden = live;
    hidden.position = PARK_HIDDEN;
    hidden.pinned = 1;
    hidden.inv_mass = 0.0;
    hidden.v_cm = [0.0; 3];
    hidden.omega = [0.0; 3];
    hidden.operand = 1;
    world.rigid_bodies[rigid_start] = hidden;

    let mut particles = Vec::new();
    for i in particle_start..world.fluid_particles.len() {
        let live = world.fluid_particles[i];
        particles.push((i as u32, live));
        world.fluid_particles[i].position = PARK_HIDDEN;
        world.fluid_particles[i].velocity = [0.0; 3];
    }
    DeferredSpawn {
        at_time,
        rigids,
        particles,
    }
}
