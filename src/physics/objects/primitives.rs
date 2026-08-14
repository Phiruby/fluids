use super::{MpmObject, MpmWorld, SphObject, SphWorld};
use crate::physics::objects::sdf::AnalyticalSdf;
use crate::physics::objects::sdf::InflowSource;
use crate::physics::{FLUID_COLOR, FluidParticle, MpmFluidParticle, mpm_fluid_block};
use crate::physics::{MPM_DX, mpm_lattice_rest_density};
use nalgebra::Vector3;
pub struct RigidBox {
    pub center: [f32; 3],
    pub half_extents: [f32; 3],
    pub mass: f32,
    pub pin: bool,
    pub color: [f32; 3],
}

impl RigidBox {
    pub fn new(center: [f32; 3], half_extents: [f32; 3]) -> Self {
        let vol = 8.0 * half_extents[0] * half_extents[1] * half_extents[2];
        Self {
            center,
            half_extents,
            mass: 100.0 * vol,
            pin: false,
            color: [0.78, 0.58, 0.42],
        }
    }

    pub fn pinned(mut self) -> Self {
        self.pin = true;
        self
    }

    pub fn with_density(mut self, density: f32) -> Self {
        let vol = 8.0 * self.half_extents[0] * self.half_extents[1] * self.half_extents[2];
        self.mass = density * vol;
        self
    }

    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = mass;
        self
    }

    pub fn with_color(mut self, color: [f32; 3]) -> Self {
        self.color = color;
        self
    }
}

impl MpmObject for RigidBox {
    fn insert(&self, world: &mut MpmWorld) {
        let i = AnalyticalSdf::Box {
            half_extents: self.half_extents,
        }
        .push_into(world, self.center, [0.0, 0.0, 0.0, 1.0], self.mass);
        world.rigid_bodies[i as usize].color = self.color;
        if self.pin {
            world.rigid_bodies[i as usize] = world.rigid_bodies[i as usize].pinned();
        }
    }
}

const PIPE_FLUID_N: usize = 32;

pub struct RigidPipe {
    pub center: [f32; 3],
    pub outer_radius: f32,
    pub inner_radius: f32,
    pub height: f32,
    pub orientation: [f32; 4],
    pub flow: [f32; 3],
    pub pin: bool,
    pub particle_spacing: f32,
    pub color: [f32; 3],
    pub particle_count: usize,
}

impl RigidPipe {
    pub fn new(center: [f32; 3], outer_radius: f32, inner_radius: f32, height: f32) -> Self {
        Self {
            center,
            outer_radius,
            inner_radius,
            height,
            orientation: [0.0, 0.0, 0.0, 1.0],
            flow: [0.0, 0.0, 0.0],
            pin: false,
            particle_spacing: 0.01,
            color: [0.62, 0.66, 0.70],
            particle_count: PIPE_FLUID_N * PIPE_FLUID_N * PIPE_FLUID_N,
        }
    }

    pub fn pinned(mut self) -> Self {
        self.pin = true;
        self
    }

    pub fn with_flow(mut self, flow: [f32; 3]) -> Self {
        self.flow = flow;
        self
    }

    pub fn with_orientation(mut self, orientation: [f32; 4]) -> Self {
        self.orientation = orientation;
        self
    }

    pub fn with_color(mut self, color: [f32; 3]) -> Self {
        self.color = color;
        self
    }

    pub fn with_particle_count(mut self, count: usize) -> Self {
        self.particle_count = count;
        self
    }
}

fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let s = q[3];
    let uv_cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let t = [uv_cross[0] * 2.0, uv_cross[1] * 2.0, uv_cross[2] * 2.0];
    let ut_cross = [
        u[1] * t[2] - u[2] * t[1],
        u[2] * t[0] - u[0] * t[2],
        u[0] * t[1] - u[1] * t[0],
    ];
    [
        v[0] + s * t[0] + ut_cross[0],
        v[1] + s * t[1] + ut_cross[1],
        v[2] + s * t[2] + ut_cross[2],
    ]
}

fn fluid_in_cylinder(
    center: [f32; 3],
    orientation: [f32; 4],
    radius: f32,
    half_h: f32,
    spacing: f32,
    flow: [f32; 3],
) -> Vec<MpmFluidParticle> {
    let rest_volume = spacing * spacing * spacing;
    let mass = 1000.0 * rest_volume;
    let rest_density = mpm_lattice_rest_density(spacing, mass, MPM_DX);
    let velocity = quat_rotate(orientation, flow);
    let flow_len = (flow[0] * flow[0] + flow[1] * flow[1] + flow[2] * flow[2]).sqrt();
    let n_r = ((radius * 2.0) / spacing).ceil() as i32;
    let n_y = ((half_h * 2.0) / spacing).ceil() as i32;
    let mut particles = Vec::new();
    for iy in -n_y..=n_y {
        for ix in -n_r..=n_r {
            for iz in -n_r..=n_r {
                let local = [
                    ix as f32 * spacing,
                    iy as f32 * spacing,
                    iz as f32 * spacing,
                ];
                if local[0] * local[0] + local[2] * local[2] > radius * radius * 0.85 {
                    continue;
                }
                if local[1].abs() > half_h * 0.9 {
                    continue;
                }
                let w = quat_rotate(orientation, local);
                particles.push(MpmFluidParticle {
                    position: [center[0] + w[0], center[1] + w[1], center[2] + w[2]],
                    velocity,
                    rest_density,
                    mass,
                    ..Default::default()
                });
            }
        }
    }
    particles
}

impl MpmObject for RigidPipe {
    fn insert(&self, world: &mut MpmWorld) {
        let vol = std::f32::consts::PI
            * (self.outer_radius * self.outer_radius - self.inner_radius * self.inner_radius)
            * self.height;
        let i = AnalyticalSdf::Cylinder {
            radius: self.outer_radius,
            height: self.height,
        }
        .subtract(AnalyticalSdf::Cylinder {
            radius: self.inner_radius,
            height: self.height * 1.2,
        })
        .push_into(
            world,
            self.center,
            self.orientation,
            (2000.0 * vol).max(0.1),
        );
        if self.pin {
            world.rigid_bodies[i as usize] = world.rigid_bodies[i as usize].pinned();
        }
        world.rigid_bodies[i as usize].color = self.color;
        if self.flow[0] * self.flow[0] + self.flow[1] * self.flow[1] + self.flow[2] * self.flow[2]
            > 1e-16
        {
            world.inflow_sources.push(InflowSource {
                body: i,
                flow: self.flow,
                ..Default::default()
            });
        }
        let mut particles = fluid_in_cylinder(
            self.center,
            self.orientation,
            self.inner_radius,
            self.height * 0.5,
            self.particle_spacing,
            self.flow,
        );
        let rest_volume = self.particle_spacing.powi(3);
        let mass = 1000.0 * rest_volume;
        let rest_density = mpm_lattice_rest_density(self.particle_spacing, mass, MPM_DX);
        let target = self.particle_count.max(particles.len());
        // fill with particles waiting to be pumped
        particles.resize(
            target,
            MpmFluidParticle {
                position: [1000.0, 1000.0, 1000.0],
                rest_density,
                mass,
                ..Default::default()
            },
        );
        world.fluid_particles.extend(particles);
    }
}

pub struct FluidCube {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub spacing: f32,
    pub origin: [f32; 3],
}

impl FluidCube {
    pub fn new(n: usize, spacing: f32, origin: [f32; 3]) -> Self {
        Self::block(n, n, n, spacing, origin)
    }

    pub fn block(nx: usize, ny: usize, nz: usize, spacing: f32, origin: [f32; 3]) -> Self {
        Self {
            nx,
            ny,
            nz,
            spacing,
            origin,
        }
    }
}

impl MpmObject for FluidCube {
    fn insert(&self, world: &mut MpmWorld) {
        world.fluid_particles.extend(mpm_fluid_block(
            self.nx,
            self.ny,
            self.nz,
            self.spacing,
            Vector3::from(self.origin),
        ));
    }
}

impl SphObject for FluidCube {
    fn insert(&self, world: &mut SphWorld) {
        let mut particles = Vec::with_capacity(self.nx * self.ny * self.nz);
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    particles.push(FluidParticle {
                        position: [
                            self.origin[0] + i as f32 * self.spacing,
                            self.origin[1] + j as f32 * self.spacing,
                            self.origin[2] + k as f32 * self.spacing,
                        ],
                        normal: FLUID_COLOR,
                        mass: 1.0,
                        boundary: 1,
                        ..Default::default()
                    });
                }
            }
        }
        world.fluid_particles.extend(particles);
    }
}
