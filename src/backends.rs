use crate::physics::{MpmWorld, SphWorld};
use crate::{FluidRenderMode, MpmMaterial, State, FLUID_RENDER_MODE, MPM_MATERIAL};

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
    );

    fn simulate(state: &mut State<Self>, encoder: &mut wgpu::CommandEncoder)
    where
        Self: Sized,
    {
        panic!("Not implemented");
    }
    fn draw(
        state: &mut State<Self>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) where
        Self: Sized,
    {
        panic!("Not implemented");
    }
}

pub enum Mpm {}
pub enum Sph {}

impl SimulationBackend for Mpm {
    type World = MpmWorld;
    fn into_particles(
        world: Self::World,
    ) -> (
        Vec<crate::physics::FluidParticle>,
        Vec<crate::physics::MpmFluidParticle>,
        Vec<crate::physics::MpmElasticParticle>,
    ) {
        (Vec::new(), world.fluid_particles, world.elastic_particles)
    }

    fn simulate(state: &mut State<Mpm>, encoder: &mut wgpu::CommandEncoder) {
        state.mpm_fluid_sim.step(&state.device, encoder);
    }
    fn draw(state: &mut State<Mpm>, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        if FLUID_RENDER_MODE == FluidRenderMode::ScreenSpaceFluidRendering
            && MPM_MATERIAL == MpmMaterial::Fluid
        {
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
    ) {
        (world.fluid_particles, Vec::new(), Vec::new())
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
            // perform compaction (filter out boundary particles) b4 creating render pass
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
