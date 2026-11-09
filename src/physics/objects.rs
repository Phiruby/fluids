use crate::physics::{FluidParticle, MpmElasticParticle, MpmFluidParticle, mpm_fluid_block};
mod primitives;
mod sdf;
pub use primitives::{ElasticBox, FluidCube, RigidBox, RigidPipe};
pub use sdf::{InflowSource, RigidBody};

#[derive(Default)]
pub struct MpmWorld {
    pub elastic_particles: Vec<MpmElasticParticle>,
    pub fluid_particles: Vec<MpmFluidParticle>,
    pub rigid_bodies: Vec<RigidBody>,
    pub inflow_sources: Vec<InflowSource>,
}

#[derive(Default)]
pub struct SphWorld {
    pub fluid_particles: Vec<FluidParticle>,
}

/// inject particles into respective sim backend
pub trait MpmObject {
    fn insert(&self, world: &mut MpmWorld);
}
pub trait SphObject {
    fn insert(&self, world: &mut SphWorld);
}
