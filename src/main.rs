use fluids::ElasticBox;
use fluids::FluidCube;
use fluids::{Mpm, MultiBoxWaterCube, PumpAtCube, Scene, Sph};
fn main() {
    let export_grid = std::env::args().any(|arg| arg == "--export-grid");
    let mut scene = Scene::<Mpm>::new();
    scene.export_grid(export_grid);
    scene.environment(MultiBoxWaterCube::default());
    // scene.environment(PumpAtCube::default());
    scene.run().unwrap();

    let mut scene = Scene::<Sph>::new();
    let fluid = FluidCube::new(30, 0.03, [-0.95, 0.50, -0.7]);
    scene.add(fluid);
    scene.run().unwrap();
}
