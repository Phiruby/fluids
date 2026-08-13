use fluids::{FluidCube, Mpm, RigidBox, Scene};

const MPM_BLOCK_N: usize = 32;
const MPM_SPACING: f32 = 0.01;
const MPM_ORIGIN: [f32; 3] = [0.24, 0.18, 0.24];
const MPM_BOX_CENTER: [f32; 3] = [0.32, 0.09, 0.32];
const MPM_BOX_HALF: [f32; 3] = [0.12, 0.05, 0.12];

fn main() {
    let mut scene = Scene::<Mpm>::new();
    scene.add(RigidBox::new(MPM_BOX_CENTER, MPM_BOX_HALF));
    scene.add(FluidCube::new(MPM_BLOCK_N, MPM_SPACING, MPM_ORIGIN));
    scene.run().unwrap();
}
