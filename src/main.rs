use fluids::{Mpm, MultiBoxWaterCube, Scene, PumpAtCube};

fn main() {
    let mut scene = Scene::<Mpm>::new();
    // scene.environment(MultiBoxWaterCube::default());
    scene.environment(PumpAtCube::default());
    scene.run().unwrap();
}
