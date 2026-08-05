use crate::physics::FluidParticle;
use nalgebra::Point3;

/// A cube boundary for the fluid simulation
pub struct ContainerBoundary {
    pub top_left: Point3<f32>,
    pub bottom_right: Point3<f32>,
    pub nlayers: usize,
    pub spacing: f32,
}

impl ContainerBoundary {
    pub fn create_points(&self, nlayers: usize) -> Vec<FluidParticle> {
        let mut points = Vec::new();

        let min = self.top_left.coords.inf(&self.bottom_right.coords);
        let max = self.top_left.coords.sup(&self.bottom_right.coords);

        for layer in 0..nlayers {
            let d = layer as f32 * self.spacing;

            let xmin = min.x - d;
            let xmax = max.x + d;
            let ymin = min.y - d;
            let ymax = max.y + d;
            let zmin = min.z - d;
            let zmax = max.z + d;

            let mut x = xmin;
            while x <= xmax {
                let mut y = ymin;
                while y <= ymax {
                    // front
                    points.push(FluidParticle {
                        position: [x, y, zmin],
                        velocity: [0.0, 0.0, 0.0],
                        normal: [0.0, 0.0, 0.0],
                        density: 0.0,
                        rest_density: 0.0,
                        mass: 1.0,
                        boundary: 0,
                        ..Default::default()
                    });
                    // back
                    if zmax != zmin {
                        points.push(FluidParticle {
                            position: [x, y, zmax],
                            velocity: [0.0, 0.0, 0.0],
                            normal: [0.0, 0.0, 0.0],
                            density: 0.0,
                            rest_density: 0.0,
                            mass: 1.0,
                            boundary: 0,
                            ..Default::default()
                        });
                    }
                    y += self.spacing;
                }
                x += self.spacing;
            }

            let mut x = xmin;
            while x <= xmax {
                let mut z = zmin + self.spacing;
                while z < zmax {
                    // bottom
                    points.push(FluidParticle {
                        position: [x, ymin, z],
                        velocity: [0.0, 0.0, 0.0],
                        normal: [0.0, 0.0, 0.0],
                        density: 0.0,
                        rest_density: 0.0,
                        mass: 1.0,
                        boundary: 0,
                        ..Default::default()
                    });
                    // top
                    if ymax != ymin {
                        points.push(FluidParticle {
                            position: [x, ymax, z],
                            velocity: [0.0, 0.0, 0.0],
                            normal: [0.0, 0.0, 0.0],
                            density: 0.0,
                            rest_density: 0.0,
                            mass: 1.0,
                            boundary: 0,
                            ..Default::default()
                        });
                    }
                    z += self.spacing;
                }
                x += self.spacing;
            }

            let mut y = ymin + self.spacing;
            while y < ymax {
                let mut z = zmin + self.spacing;
                while z < zmax {
                    // left
                    points.push(FluidParticle {
                        position: [xmin, y, z],
                        velocity: [0.0, 0.0, 0.0],
                        normal: [0.0, 0.0, 0.0],
                        density: 0.0,
                        rest_density: 0.0,
                        mass: 1.0,
                        boundary: 0,
                        ..Default::default()
                    });
                    // right
                    if xmax != xmin {
                        points.push(FluidParticle {
                            position: [xmax, y, z],
                            velocity: [0.0, 0.0, 0.0],
                            normal: [0.0, 0.0, 0.0],
                            density: 0.0,
                            rest_density: 0.0,
                            mass: 1.0,
                            boundary: 0,
                            ..Default::default()
                        });
                    }
                    z += self.spacing;
                }
                y += self.spacing;
            }
        }

        points
    }
}