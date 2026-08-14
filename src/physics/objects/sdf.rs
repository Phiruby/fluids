use super::{MpmObject, MpmWorld};
use crate::physics::{MPM_DX, MpmFluidParticle, mpm_lattice_rest_density};

/// store a list of analytical sdfs supported.
/// all shapes' sdf will be constructed w.r.t origin;
/// it is up to you to translate points to these coordinates.
/// analytical forms from https://iquilezles.org/articles/distfunctions/
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SdfType(pub u32);

impl SdfType {
    pub const SPHERE: Self = SdfType(0);
    pub const CYLINDER: Self = SdfType(1);
    pub const BOX: Self = SdfType(2);
    pub const DIFFERENCE: Self = SdfType(3);
}

pub enum AnalyticalSdf {
    Sphere(f32),
    Cylinder {
        radius: f32,
        height: f32,
    },
    Box {
        half_extents: [f32; 3],
    },
    Difference {
        a: Box<AnalyticalSdf>,
        b: Box<AnalyticalSdf>,
    },
}

impl AnalyticalSdf {
    pub fn subtract(self, other: Self) -> Self {
        Self::Difference {
            a: Box::new(self),
            b: Box::new(other),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RigidBody {
    pub object_type: SdfType,
    pub invert: u32,
    pub pinned: u32,
    /// if 1, implies the body is part of operand (such as ::DIFFERENCE)
    pub operand: u32,
    pub position: [f32; 3],
    pub scale: f32,
    pub orientation: [f32; 4],
    /// params for your shape. expected values should be referred by the specific sdf implementation
    pub params: [f32; 4],
    pub v_cm: [f32; 3],
    pub mass: f32,
    pub omega: [f32; 3],
    pub inv_mass: f32,
    pub inv_inertia: [f32; 3],
    pub force: [f32; 3],
    pub torque: [f32; 3],
    pub color: [f32; 3],
}

/// for particle flow (e.g pipe pushing water)
#[repr(C)]
#[derive(Default, Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InflowSource {
    pub body: u32,
    pub _pad: [u32; 3],
    pub flow: [f32; 3],
    pub _pad1: f32,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            object_type: SdfType::SPHERE,
            invert: 0,
            pinned: 0,
            operand: 0,
            position: [0.0; 3],
            scale: 1.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            params: [0.0; 4],
            v_cm: [0.0; 3],
            mass: 1.0,
            omega: [0.0; 3],
            inv_mass: 1.0,
            inv_inertia: [1.0; 3],
            force: [0.0; 3],
            torque: [0.0; 3],
            color: [0.78, 0.58, 0.42],
        }
    }
}

impl RigidBody {
    pub fn pinned(mut self) -> Self {
        self.pinned = 1;
        self.inv_mass = 0.0;
        self
    }

    pub fn with_color(mut self, color: [f32; 3]) -> Self {
        self.color = color;
        self
    }
}

fn box_inv_inertia(mass: f32, h: [f32; 3]) -> [f32; 3] {
    let ixx = mass / 3.0 * (h[1] * h[1] + h[2] * h[2]);
    let iyy = mass / 3.0 * (h[0] * h[0] + h[2] * h[2]);
    let izz = mass / 3.0 * (h[0] * h[0] + h[1] * h[1]);
    [
        1.0 / ixx.max(1e-8),
        1.0 / iyy.max(1e-8),
        1.0 / izz.max(1e-8),
    ]
}

fn primitive_to_gpu(
    sdf: &AnalyticalSdf,
    position: [f32; 3],
    orientation: [f32; 4],
    mass: f32,
) -> RigidBody {
    let inv_mass = if mass > 1e-8 { 1.0 / mass } else { 0.0 };
    match sdf {
        AnalyticalSdf::Sphere(radius) => {
            let i = 0.4 * mass * radius * radius;
            RigidBody {
                object_type: SdfType::SPHERE,
                position,
                scale: *radius,
                orientation,
                params: [*radius, *radius, *radius, 0.0],
                mass,
                inv_mass,
                inv_inertia: [1.0 / i.max(1e-8); 3],
                ..Default::default()
            }
        }
        AnalyticalSdf::Cylinder { radius, height } => {
            let h = height * 0.5;
            let iyy = 0.5 * mass * radius * radius;
            let ixx = mass / 12.0 * (3.0 * radius * radius + height * height);
            RigidBody {
                object_type: SdfType::CYLINDER,
                position,
                orientation,
                params: [*radius, h, *radius, 0.0],
                mass,
                inv_mass,
                inv_inertia: [
                    1.0 / ixx.max(1e-8),
                    1.0 / iyy.max(1e-8),
                    1.0 / ixx.max(1e-8),
                ],
                ..Default::default()
            }
        }
        AnalyticalSdf::Box { half_extents } => RigidBody {
            object_type: SdfType::BOX,
            position,
            orientation,
            params: [half_extents[0], half_extents[1], half_extents[2], 0.0],
            mass,
            inv_mass,
            inv_inertia: box_inv_inertia(mass, *half_extents),
            ..Default::default()
        },
        AnalyticalSdf::Difference { .. } => panic!("child must be a primitive"),
    }
}

/// push the sdf under the assumption that the sdf is an operand (e.g part of ::difference of SdfType)
fn push_operand(sdf: &AnalyticalSdf, world: &mut MpmWorld) -> u32 {
    let mut rb = primitive_to_gpu(sdf, [0.0; 3], [0.0, 0.0, 0.0, 1.0], 0.0);
    rb.operand = 1;
    rb.mass = 0.0;
    rb.inv_mass = 0.0;
    world.rigid_bodies.push(rb);
    (world.rigid_bodies.len() - 1) as u32
}

impl AnalyticalSdf {
    pub fn push_into(
        self,
        world: &mut MpmWorld,
        position: [f32; 3],
        orientation: [f32; 4],
        mass: f32,
    ) -> u32 {
        let rb = match self {
            AnalyticalSdf::Difference { a, b } => {
                let mut rb = primitive_to_gpu(&a, position, orientation, mass);
                let ia = push_operand(&a, world);
                let ib = push_operand(&b, world);
                rb.object_type = SdfType::DIFFERENCE;
                rb.params = [ia as f32, ib as f32, 0.0, 0.0];
                rb
            }
            other => primitive_to_gpu(&other, position, orientation, mass),
        };
        world.rigid_bodies.push(rb);
        (world.rigid_bodies.len() - 1) as u32
    }
}
