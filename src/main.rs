use fluids::{Mpm, MultiBoxWaterCube, PumpAtCube, Scene};

fn main() {
    let mut scene = Scene::<Mpm>::new();
    // scene.environment(MultiBoxWaterCube::default());
    scene.environment(PumpAtCube::default());
    scene.run().unwrap();
}
