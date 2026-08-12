use crate::physics::{FluidParticle, MpmElasticParticle, MpmFluidParticle};

pub(crate) trait MpmObject {
    fn insert(world: &mut MpmWorld);
}

pub(crate) trait SphObject {
    fn insert(world: &mut SphWorld);
}

pub struct BoundingBox {
    min: [f32; 3],
    max: [f32; 3],
}
