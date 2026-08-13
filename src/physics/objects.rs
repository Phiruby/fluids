use crate::physics::{
    FLUID_COLOR, FluidParticle, MpmElasticParticle, MpmFluidParticle, mpm_fluid_block,
};
mod sdf;
use nalgebra::Vector3;
pub use sdf::{InflowSource, RigidBody, RigidBox, RigidPipe};

pub struct MpmWorld {
    pub elastic_particles: Vec<MpmElasticParticle>,
    pub fluid_particles: Vec<MpmFluidParticle>,
    pub rigid_bodies: Vec<RigidBody>,
    pub inflow_sources: Vec<InflowSource>,
}

impl Default for MpmWorld {
    fn default() -> Self {
        Self {
            elastic_particles: Vec::new(),
            fluid_particles: Vec::new(),
            rigid_bodies: Vec::new(),
            inflow_sources: Vec::new(),
        }
    }
}

pub struct SphWorld {
    pub fluid_particles: Vec<FluidParticle>,
}
impl Default for SphWorld {
    fn default() -> Self {
        Self {
            fluid_particles: Vec::new(),
        }
    }
}

/// inject particles into respective sim backend
pub trait MpmObject {
    fn insert(&self, world: &mut MpmWorld);
}
pub trait SphObject {
    fn insert(&self, world: &mut SphWorld);
}

pub struct BoundingBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

pub struct FluidCube {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub spacing: f32,
    pub origin: [f32; 3],
}

impl FluidCube {
    pub fn new(n: usize, spacing: f32, origin: [f32; 3]) -> Self {
        Self::block(n, n, n, spacing, origin)
    }

    pub fn block(nx: usize, ny: usize, nz: usize, spacing: f32, origin: [f32; 3]) -> Self {
        Self {
            nx,
            ny,
            nz,
            spacing,
            origin,
        }
    }
}

impl MpmObject for FluidCube {
    fn insert(&self, world: &mut MpmWorld) {
        world.fluid_particles.extend(mpm_fluid_block(
            self.nx,
            self.ny,
            self.nz,
            self.spacing,
            Vector3::from(self.origin),
        ));
    }
}

impl SphObject for FluidCube {
    fn insert(&self, world: &mut SphWorld) {
        let mut particles = Vec::with_capacity(self.nx * self.ny * self.nz);
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    particles.push(FluidParticle {
                        position: [
                            self.origin[0] + i as f32 * self.spacing,
                            self.origin[1] + j as f32 * self.spacing,
                            self.origin[2] + k as f32 * self.spacing,
                        ],
                        normal: FLUID_COLOR,
                        mass: 1.0,
                        boundary: 1,
                        ..Default::default()
                    });
                }
            }
        }
        world.fluid_particles.extend(particles);
    }
}
