use fluids::{FluidCube, Mpm, Scene, Sph};

const MPM_BLOCK_N: usize = 32;
const MPM_SPACING: f32 = 0.01;
const MPM_ORIGIN: [f32; 3] = [0.16, 0.28, 0.16];

const SPH_BLOCK_N: usize = 30;
const SPH_SPACING: f32 = 0.03;
const SPH_FLUID_ORIGIN: [f32; 3] = [-0.95, 0.50, -0.7];
fn main() {

    // Mpm version
    // let mut scene = Scene::<Mpm>::new();
    // scene.add(FluidCube::new(MPM_BLOCK_N, MPM_SPACING, MPM_ORIGIN));
    // scene.run().unwrap();

    // Sph version
    let mut scene = Scene::<Sph>::new();
    scene.add(FluidCube::new(SPH_BLOCK_N, SPH_SPACING, SPH_FLUID_ORIGIN));
    scene.run().unwrap();
}
