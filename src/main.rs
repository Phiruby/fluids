use fluids::FluidCube;
use fluids::{Mpm, MultiBoxWaterCube, PumpAtCube, Scene, Sph};
use fluids::ElasticBox;
fn main() {
    // let mut scene = Scene::<Mpm>::new();
    // scene.environment(MultiBoxWaterCube::default());
    // // scene.environment(PumpAtCube::default());
    // scene.run().unwrap();

    // let mut scene = Scene::<Sph>::new();
    // let fluid = FluidCube::new(30, 0.03, [-0.95, 0.50, -0.7]);
    // scene.add(fluid);
    // scene.run().unwrap();
    let mut scene = Scene::<Mpm>::new();
    let elastic_box = ElasticBox {
        center: [0.0, 0.2, 0.0],
        nparticles: [30, 30, 30],
        spacing: 0.01,
        poisson_ratio: 0.3,
        youngs_modulus: 1000.0,
    };
    scene.add(elastic_box);
    scene.run().unwrap();
}
