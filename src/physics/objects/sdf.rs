use super::{MpmObject, MpmWorld};

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
}

pub enum AnalyticalSdf {
    Sphere(f32),
    Cylinder { radius: f32, height: f32 },
    Box { half_extents: [f32; 3] },
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RigidBody {
    pub object_type: SdfType,
    pub invert: u32,
    _pad0: [u32; 2],
    pub position: [f32; 3],
    pub scale: f32,
    pub orientation: [f32; 4],
    pub params: [f32; 4],
    pub v_cm: [f32; 3],
    pub mass: f32,
    pub omega: [f32; 3],
    pub inv_mass: f32,
    pub inv_inertia: [f32; 3],
    _pad1: f32,
    pub force: [f32; 3],
    _pad2: f32,
    pub torque: [f32; 3],
    _pad3: f32,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            object_type: SdfType::SPHERE,
            invert: 0,
            _pad0: [0; 2],
            position: [0.0; 3],
            scale: 1.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            params: [0.0; 4],
            v_cm: [0.0; 3],
            mass: 1.0,
            omega: [0.0; 3],
            inv_mass: 1.0,
            inv_inertia: [1.0; 3],
            _pad1: 0.0,
            force: [0.0; 3],
            _pad2: 0.0,
            torque: [0.0; 3],
            _pad3: 0.0,
        }
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

impl AnalyticalSdf {
    pub fn to_rigid_body(self, position: [f32; 3], orientation: [f32; 4], mass: f32) -> RigidBody {
        let inv_mass = if mass > 1e-8 { 1.0 / mass } else { 0.0 };
        match self {
            AnalyticalSdf::Sphere(radius) => {
                let i = 0.4 * mass * radius * radius;
                RigidBody {
                    object_type: SdfType::SPHERE,
                    position,
                    scale: radius,
                    orientation,
                    params: [radius, radius, radius, 0.0],
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
                    params: [radius, h, radius, 0.0],
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
                inv_inertia: box_inv_inertia(mass, half_extents),
                ..Default::default()
            },
        }
    }
}

pub struct RigidBox {
    pub center: [f32; 3],
    pub half_extents: [f32; 3],
    pub mass: f32,
}

impl RigidBox {
    pub fn new(center: [f32; 3], half_extents: [f32; 3]) -> Self {
        let vol = 8.0 * half_extents[0] * half_extents[1] * half_extents[2];
        Self {
            center,
            half_extents,
            mass: 2000.0 * vol,
        }
    }
}

impl MpmObject for RigidBox {
    fn insert(&self, world: &mut MpmWorld) {
        world.rigid_bodies.push(
            AnalyticalSdf::Box {
                half_extents: self.half_extents,
            }
            .to_rigid_body(self.center, [0.0, 0.0, 0.0, 1.0], self.mass),
        );
    }
}
