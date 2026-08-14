use crate::Scene;
use crate::backends::Mpm;
use crate::physics::{
    FluidCube, MpmFluidParticle, MpmFluidSim, MpmObject, MpmWorld, RigidBody, RigidBox, RigidPipe,
    mpm_collider_bounds, mpm_grid_bounds,
};
use nalgebra::{UnitQuaternion, Vector3};

const POOL_NX: usize = 64;
const POOL_NY: usize = 32;
const POOL_NZ: usize = 64;
const POOL_SPACING: f32 = 0.01;

pub trait MpmEnvironment {
    fn build(&self, scene: &mut Scene<Mpm>);
}

fn pool_particle_count() -> usize {
    POOL_NX * POOL_NY * POOL_NZ
}

fn domain_floor() -> f32 {
    mpm_collider_bounds().0[1]
}

fn domain_mid() -> [f32; 3] {
    let (min, max) = mpm_grid_bounds();
    [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ]
}

fn quat_align_y(dir: [f32; 3]) -> [f32; 4] {
    let to = Vector3::new(dir[0], dir[1], dir[2]);
    let to = to.try_normalize(1e-8).unwrap_or_else(Vector3::y);
    let q = UnitQuaternion::rotation_between(&Vector3::y(), &to)
        .unwrap_or_else(UnitQuaternion::identity);
    let v = q.into_inner().coords;
    [v.x, v.y, v.z, v.w]
}

/// pipe that pumps water horizontally at a rigid cube.
#[derive(Clone)]
pub struct PumpAtCube {
    pub cube_center: [f32; 3],
    pub cube_size: f32,
    pub pump_flow: [f32; 3],
}

impl PumpAtCube {
    pub fn new(cube_center: [f32; 3], cube_size: f32, pump_flow: [f32; 3]) -> Self {
        Self {
            cube_center,
            cube_size,
            pump_flow,
        }
    }
}

impl Default for PumpAtCube {
    fn default() -> Self {
        let mid = domain_mid();
        let y = domain_floor() + 0.08;
        Self {
            cube_center: [mid[0] + 0.22, y, mid[2]],
            cube_size: 0.08,
            pump_flow: [12.0, 0.0, 0.0],
        }
    }
}

impl PumpAtCube {
    fn spawn(&self, world: &mut MpmWorld) {
        RigidBox::new(
            self.cube_center,
            [self.cube_size, self.cube_size, self.cube_size],
        )
        .with_color([0.86, 0.42, 0.22])
        .insert(world);

        let flow = Vector3::from(self.pump_flow);
        let flow_len = flow.norm().max(1e-8);
        let dir = flow / flow_len;
        let pipe_height = 0.18;
        let standoff = self.cube_size + pipe_height * 0.5 + 0.05;
        let pipe_center = [
            self.cube_center[0] - dir.x * standoff,
            self.cube_center[1] - dir.y * standoff,
            self.cube_center[2] - dir.z * standoff,
        ];
        RigidPipe::new(pipe_center, 0.05, 0.032, pipe_height)
            .pinned()
            .with_orientation(quat_align_y(self.pump_flow))
            .with_flow([0.0, flow_len, 0.0])
            .with_color([0.58, 0.62, 0.68])
            .with_particle_count(pool_particle_count())
            .insert(world);
    }
}

impl MpmObject for PumpAtCube {
    fn insert(&self, world: &mut MpmWorld) {
        self.spawn(world);
    }
}

impl MpmEnvironment for PumpAtCube {
    fn build(&self, scene: &mut Scene<Mpm>) {
        scene.add(self.clone());
    }
}

/// fluid pool that settles, then density-varied boxes drop in;
// one would expect boxes denser than the fluid to sink (and what is observed!)
pub struct MultiBoxWaterCube {
    pub settle_time: f32,
}

impl MultiBoxWaterCube {
    pub fn new(settle_time: f32) -> Self {
        Self { settle_time }
    }
}

impl Default for MultiBoxWaterCube {
    fn default() -> Self {
        Self { settle_time: 1.00 }
    }
}

impl MpmEnvironment for MultiBoxWaterCube {
    fn build(&self, scene: &mut Scene<Mpm>) {
        let mid = domain_mid();
        let floor = domain_floor();
        let origin = [
            mid[0] - 0.5 * POOL_NX as f32 * POOL_SPACING,
            floor + 0.008,
            mid[2] - 0.5 * POOL_NZ as f32 * POOL_SPACING,
        ];

        scene.add(FluidCube::block(
            POOL_NX,
            POOL_NY,
            POOL_NZ,
            POOL_SPACING,
            origin,
        ));
        scene.wait(self.settle_time);

        let drop_y = floor + 0.55;
        let half = [0.045, 0.045, 0.045];
        scene.add(
            RigidBox::new([mid[0] - 0.12, drop_y, mid[2] - 0.06], half)
                .with_density(100.0)
                .with_color([0.86, 0.68, 0.32]),
        );
        scene.wait(0.12);
        scene.add(
            RigidBox::new([mid[0], drop_y + 0.02, mid[2] + 0.08], half)
                .with_density(200.0)
                .with_color([0.72, 0.26, 0.20]),
        );
        scene.wait(0.12);
        scene.add(
            RigidBox::new([mid[0] + 0.14, drop_y + 0.04, mid[2] - 0.04], half)
                .with_density(4000.0)
                .with_color([0.28, 0.34, 0.42]),
        );
    }
}
