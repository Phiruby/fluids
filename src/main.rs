use fluids::{Mpm, RigidBox, RigidPipe, Scene};

const MPM_BLOCK_N: usize = 32;
const MPM_SPACING: f32 = 0.01;
const MPM_ORIGIN: [f32; 3] = [0.24, 0.18, 0.24];
const MPM_BOX_CENTER: [f32; 3] = [0.32, 0.09, 0.32];
const MPM_BOX_HALF: [f32; 3] = [0.12, 0.05, 0.12];
const MPM_PIPE_CENTER: [f32; 3] = [0.32, 0.42, 0.32];
const MPM_PIPE_OUTER: f32 = 0.045;
const MPM_PIPE_INNER: f32 = 0.028;
const MPM_PIPE_HEIGHT: f32 = 0.16;

fn main() {
    let mut scene = Scene::<Mpm>::new();
    // scene.add(RigidBox::new(MPM_BOX_CENTER, MPM_BOX_HALF));
    // scene.add(FluidCube::new(MPM_BLOCK_N, MPM_SPACING, MPM_ORIGIN));
    scene.add(
        RigidPipe::new(
            MPM_PIPE_CENTER,
            MPM_PIPE_OUTER,
            MPM_PIPE_INNER,
            MPM_PIPE_HEIGHT,
        )
        .pinned()
        .with_flow([0.0, -10.0, 0.0]),
    );
    scene.run().unwrap();
}
