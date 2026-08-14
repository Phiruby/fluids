use fluids::FluidCube;
use fluids::{Mpm, MultiBoxWaterCube, PumpAtCube, Scene, Sph};
fn main() {
    // let mut scene = Scene::<Mpm>::new();
    // // scene.environment(MultiBoxWaterCube::default());
    // // scene.environment(PumpAtCube::default());
    // scene.run().unwrap();

    let mut scene = Scene::<Sph>::new();
    let fluid = FluidCube::new(30, 0.03, [-0.95, 0.50, -0.7]);
    scene.add(fluid);
    scene.run().unwrap();
}
