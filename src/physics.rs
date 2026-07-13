mod fluids;
use crate::Vertex;
use nalgebra::{Matrix3, Point3, Vector3};
use std::collections::HashSet;
use std::ops::Add;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::Bvh;
use bvh::ray::Ray as BvhRay;
use crate::{MouseState, Ray};

pub use fluids::FluidSim;

const DT: f32 = 0.002;
const VELOCITY_DAMPING: f32 = 0.995;
const STRETCH_FORCE: f32 = 60.0;
const STRETCH_DAMPING: f32 = 8.0;
const MAX_STRETCH: f32 = 5.05;
#[derive(Debug, Clone)]
#[allow(unused)]
pub struct PhysicsVertex {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub renderable: bool,
    pub mass: f32,
}


impl From<PhysicsVertex> for Vertex {
    fn from(vertex: PhysicsVertex) -> Self {
        Vertex {
            // hackery to access vector from position: https://docs.rs/nalgebra/latest/nalgebra/base/type.Vector3.html
            position: vertex.position.data.0[0],
            color: [0.5, 0.0, 0.5],
            normal: [0.0, 0.0, 1.0],
        }
    }
}

fn neighbor_sum<F, T>(
    delta: F,
    neighbors: &[usize],
    zero: T,
) -> T
where
    F: Fn(usize) -> T,
    T: Add<Output = T>,
{
    let mut sum = zero;
    for &neighbor_idx in neighbors {
        sum = sum + delta(neighbor_idx);
    }
    sum
}

fn precompute_neighbors(positions: &[Vector3<f32>], kernel: &CubicKernel) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); positions.len()];
    for i in 0..positions.len() {
        for j in 0..positions.len() {
            if (positions[i] - positions[j]).norm() <= kernel.h {
                neighbors[i].push(j);
            }
        }
    }
    neighbors
}


#[derive(Debug, Clone)]
pub struct CubicKernel {
    pub h: f32,
}

impl CubicKernel {
    pub fn new(h: f32) -> Self {
        Self { h }
    }

    pub fn weight(&self, r_norm: f32) -> f32 {
        let q = r_norm / self.h;
        if q >= 1.0 || q < 0.0 {
            return 0.0;
        }

        // 3D normalization constant for Wendland C2
        let sigma = 21.0 / (2.0 * std::f32::consts::PI * self.h.powf(3.0));
        
        let term = 1.0 - q;
        let w = term.powf(4.0) * (4.0 * q + 1.0);
        
        w * sigma
    }

    fn weight_gradient(&self, r: Vector3<f32>) -> Vector3<f32> {
        let r_norm = r.norm();
        if r_norm == 0.0 {
            return Vector3::zeros();
        }

        let q = r_norm / self.h;
        if q >= 1.0 {
            return Vector3::zeros();
        }

        let sigma = 21.0 / (2.0 * std::f32::consts::PI * self.h.powf(3.0));

        // dW/dq = -20 * q * (1 - q)^3
        let dw_dq = -20.0 * q * (1.0 - q).powf(3.0);
        
        // dq/dx = r / (h * ||r||)
        let dq_dx = r / (self.h * r_norm);

        // Chain rule: dW/dx = dW/dq * dq/dx
        let gradient = (dw_dq * sigma) * dq_dx;
        gradient
    }

    /// Performs pairwise evaluation of Wendland C2 kernel between the two.
    /// Index [i][j] of output is W_{ij} = W(x_i - x_j).
    pub fn eval(
        &self,
        x_i: &[Vector3<f32>],
        x_j: &[Vector3<f32>],
        neighbors: &[Vec<usize>],
    ) -> Vec<Vec<f32>> {
        let mut w = vec![vec![0.0; x_j.len()]; x_i.len()];
        for i in 0..x_i.len() {
            for &j in &neighbors[i] {
                let r = x_i[i] - x_j[j];
                w[i][j] = self.weight(r.norm());
            }
        }
        w
    }

    /// Returns gradient w.r.t x_i's of W_{ij} = W(x_i - x_j);
    /// i.e this returns Matrix so that G[i][j] = nabla_i W_{ij}
    pub fn gradient(
        &self,
        x_i: &[Vector3<f32>],
        x_j: &[Vector3<f32>],
        neighbors: &[Vec<usize>],
    ) -> Vec<Vec<Vector3<f32>>> {
        let mut g = vec![vec![Vector3::new(0.0, 0.0, 0.0); x_j.len()]; x_i.len()];
        for i in 0..x_i.len() {
            for &j in &neighbors[i] {
                let r = x_i[i] - x_j[j];
                g[i][j] = self.weight_gradient(r);
            }
        }
        g
    }
}


/// Represents a face of a cube
#[derive(Debug, Clone)]
pub struct CubeFace {
    pub face_index: usize,
    pub center: Point3<f32>,
    pub width: f32,
    pub height: f32,
    pub depth: f32,
    pub node_index: usize,
    pub vertex_indices: Vec<usize>,
}

impl Bounded<f32, 3> for CubeFace {
    fn aabb(&self) -> Aabb<f32, 3> {
        let half = nalgebra::Vector3::new(self.width / 2.0, self.height / 2.0, self.depth / 2.0);
        let min = self.center - half;
        let max = self.center + half;
        Aabb::with_bounds(min, max)
    }
}

impl BHShape<f32, 3> for CubeFace {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

fn grid_index(n: usize, i: usize, j: usize, k: usize) -> usize {
    i * n * n + j * n + k
}

fn build_face(
    n: usize,
    layer: usize,
    axis: usize,
    face_index: usize,
    vertices: &[PhysicsVertex],
) -> CubeFace {
    let mut vertex_indices = Vec::with_capacity(n * n);
    let mut pos_sum = Vector3::zeros();
    let mut min = Vector3::repeat(f32::INFINITY);
    let mut max = Vector3::repeat(f32::NEG_INFINITY);
    for a in 0..n {
        for b in 0..n {
            let idx = match axis {
                0 => grid_index(n, layer, a, b),
                1 => grid_index(n, a, layer, b),
                _ => grid_index(n, a, b, layer),
            };
            vertex_indices.push(idx);
            let p = vertices[idx].position;
            pos_sum += p;
            min = min.inf(&p);
            max = max.sup(&p);
        }
    }
    let extent = max - min;
    let eps = 1e-4;
    CubeFace {
        face_index,
        center: Point3::from(pos_sum / (n * n) as f32),
        width: extent.x.max(eps),
        height: extent.y.max(eps),
        depth: extent.z.max(eps),
        node_index: 0,
        vertex_indices,
    }
}

fn cubeface_component(position: Vector3<f32>, face_index: usize) -> f32 {
    match face_index {
        0 => position.x,
        1 => position.y,
        2 => position.z,
        3 => position.x,
        4 => position.y,
        5 => position.z,
        _ => panic!("invalid face index"),
    }
}

fn build_cube_faces(vertices: &[PhysicsVertex], n: usize) -> [CubeFace; 6] {
    [
        build_face(n, 0, 0, 0, vertices),
        build_face(n, 0, 1, 1, vertices),
        build_face(n, 0, 2, 2, vertices),
        build_face(n, n - 1, 0, 3, vertices),
        build_face(n, n - 1, 1, 4, vertices),
        build_face(n, n - 1, 2, 5, vertices),
    ]
}

fn update_cube_faces(faces: &mut [CubeFace; 6], vertices: &[PhysicsVertex]) {
    for face in faces.iter_mut() {
        let mut pos_sum = Vector3::zeros();
        let mut min = Vector3::repeat(f32::INFINITY);
        let mut max = Vector3::repeat(f32::NEG_INFINITY);
        for &idx in &face.vertex_indices {
            let p = vertices[idx].position;
            pos_sum += p;
            min = min.inf(&p);
            max = max.sup(&p);
        }
        let count = face.vertex_indices.len() as f32;
        face.center = Point3::from(pos_sum / count);
        let extent = max - min;
        let eps = 1e-4;
        face.width = extent.x.max(eps);
        face.height = extent.y.max(eps);
        face.depth = extent.z.max(eps);
    }
}

fn vec_component(v: Vector3<f32>, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn ray_intersect_face(
    origin: Point3<f32>,
    dir: Vector3<f32>,
    face: &CubeFace,
    vertices: &[PhysicsVertex],
) -> Option<f32> {
    let axis = match face.face_index {
        0 | 3 => 0,
        1 | 4 => 1,
        _ => 2,
    };
    let coord = vec_component(face.center.coords, axis);
    let denom = vec_component(dir, axis);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (coord - vec_component(origin.coords, axis)) / denom;
    if t <= 0.0 {
        return None;
    }
    // TODO: bound check per face?
    Some(t)
}
#[derive(Debug)]
pub struct CubeShape {
    pub vertices: Vec<PhysicsVertex>,
    n: usize,
    t: f32,
}

#[derive(Debug)]
pub struct Cube {
    pub reference: CubeShape,
    pub neighbors: Vec<Vec<usize>>,
    pub L: Vec<Matrix3<f32>>,
    pub reference_gradients: Vec<Vec<Vector3<f32>>>,
    pub reference_volumes: Vec<f32>,
    pub reference_kernel_weights: Vec<Vec<f32>>,
    pub current: CubeShape,
    pub cube_faces: [CubeFace; 6],
    pub face_bvh: Bvh<f32, 3>,
    pub kernel: CubicKernel,
    pub selected_face_index: Option<usize>,
}

impl Cube {
    pub fn new(n: usize, spacing: f32, kernel: &CubicKernel) -> Self {
        let reference = CubeShape::new_static(n, spacing);
        let reference_positions = reference
            .vertices
            .iter()
            .map(|v| v.position)
            .collect::<Vec<_>>();
        let neighbors = precompute_neighbors(&reference_positions, kernel);
        let reference_densities = compute_densities(&reference.vertices, kernel, &neighbors);
        
        let reference_volumes: Vec<f32> = reference_densities
            .iter()
            .zip(&reference.vertices)
            .map(|(d, v)| v.mass / d)
            .collect();

        let reference_gradients = kernel.gradient(
            &reference_positions,
            &reference_positions,
            &neighbors,
        );
        let mut cube_faces = build_cube_faces(&reference.vertices, n);
        let face_bvh = Bvh::build(&mut cube_faces);
        let reference_distances = pairwise_displacement(&reference.vertices, &reference.vertices);
        let mut L = Vec::with_capacity(reference.vertices.len());
        for i in 0..reference.vertices.len() {
            let mut L_i = Matrix3::zeros();
            for j in 0..reference.vertices.len() {
                // NOTE: this uses inverse of (154) instead of the provided formula (156) in https://arxiv.org/pdf/2009.06944
                L_i += reference_volumes[j] * reference_distances[j][i] * reference_gradients[i][j].transpose();
            }
            L.push(L_i.try_inverse().unwrap());
        }
        let reference_kernel_weights = kernel.eval(
            &reference_positions,
            &reference_positions,
            &neighbors,
        );
        Self {
            reference,
            neighbors,
            reference_volumes,
            reference_kernel_weights,
            current: CubeShape::new(n, spacing),
            L,
            reference_gradients,
            kernel: kernel.clone(),
            cube_faces,
            face_bvh,
            selected_face_index: None,
        }
    }

    pub fn update(&mut self, mouse_state: &MouseState, hovered_face: Option<usize>, ray: &Ray) {
        let mut forces = compute_forces(self, &self.kernel);

        if let Some(hovered_face) = hovered_face && mouse_state.is_pressed && let None = self.selected_face_index {
            self.selected_face_index = Some(hovered_face);
        } else if !mouse_state.is_pressed {
            self.selected_face_index = None;
        }

        let stretch_state = self.selected_face_index.map(|hovered_face| {
            let normal = face_normal(hovered_face);
            let cube_center = self.cube_faces.iter().fold(Vector3::zeros(), |acc, face| acc + face.center.coords) / 6.0;
            let ray_end_projected = ray.project_to(normal) - cube_center;
            let target_axis = cubeface_component(
                Vector3::new(ray_end_projected.x, ray_end_projected.y, ray_end_projected.z),
                hovered_face,
            );
            let target_axis = clamp_stretch_target(hovered_face, &self.reference, target_axis);
            let hovered_indices = self.cube_faces[hovered_face].vertex_indices.clone();
            let opposite = opposite_face(hovered_face);
            let opposite_indices = self.cube_faces[opposite].vertex_indices.clone();
            dbg!(&cube_center, &ray_end_projected, &target_axis, &hovered_face);
            (hovered_face, target_axis, hovered_indices, opposite, opposite_indices)
        });

        if let Some((hovered_face, target_axis, hovered_indices, _, _)) = &stretch_state {
            apply_face_stretch_forces(
                &self.current,
                *hovered_face,
                hovered_indices,
                *target_axis,
                &mut forces,
            );
        }

        self.current.vertices.iter_mut().zip(forces).for_each(|(v, f)| {
            v.velocity += f * DT / v.mass;
            v.velocity *= VELOCITY_DAMPING;
        });
        self.current.update();

        if let Some((_, _, _, opposite, opposite_indices)) = stretch_state {
            pin_face_vertices(&self.reference, &mut self.current, opposite, &opposite_indices);
        }

        update_cube_faces(&mut self.cube_faces, &self.current.vertices);
        // TODO: do we have to rebuild every frame? 
        self.face_bvh = Bvh::build(&mut self.cube_faces);
    }

    pub fn hovered_face_from_ray(
        &self,
        start: (f32, f32, f32),
        end: (f32, f32, f32),
    ) -> Option<usize> {
        let origin = Point3::new(start.0, start.1, start.2);
        let direction = Vector3::new(end.0 - start.0, end.1 - start.1, end.2 - start.2);
        self.hovered_face(origin, direction)
    }

    fn hovered_face(&self, origin: Point3<f32>, direction: Vector3<f32>) -> Option<usize> {
        let len = direction.norm();
        if len < 1e-6 {
            return None;
        }
        let dir = direction / len;
        let ray = BvhRay::new(origin, dir);
        let mut best: Option<(usize, f32)> = None;
        for face in self.face_bvh.traverse(&ray, &self.cube_faces) {
            if let Some(t) = ray_intersect_face(origin, dir, face, &self.current.vertices) {
                if best.map_or(true, |(_, bt)| t < bt) {
                    best = Some((face.face_index, t));
                }
            }
        }
        Some(best?.0)
    }

    pub fn to_renderable(&self) -> (Vec<Vertex>, Vec<u16>) {
        self.to_renderable_with_hover(None)
    }

    pub fn to_renderable_with_hover(&self, hovered_face: Option<usize>) -> (Vec<Vertex>, Vec<u16>) {
        let highlight: HashSet<_> = hovered_face
            // TODO: copying ~ n^2 indicies
            .map(|f| self.cube_faces[f].vertex_indices.iter().copied().collect())
            .unwrap_or_default();
        self.current.to_renderable_with_highlight(&highlight)
    }

    pub fn to_particle_renderable(&self) -> Vec<Vertex> {
        // TODO: verify 0 is x direction face
        let (min_x, max_x) = face_bounds(&self.reference, 0);
        self.current
            .vertices
            .iter()
            .zip(self.reference.vertices.iter())
            .map(|(current, reference)| {
                let color = if (reference.position.x - min_x).abs() < 1e-6 {
                    [0.25, 0.8, 1.0]
                } else if (reference.position.x - max_x).abs() < 1e-6 {
                    [1.0, 0.35, 0.35]
                } else if current.renderable {
                    [1.0, 0.75, 0.55]
                } else {
                    [0.9, 0.9, 1.0]
                };

                Vertex {
                    position: [current.position.x, current.position.y, current.position.z],
                    color,
                    normal: [0.0, 0.0, 1.0],
                }
            })
            .collect()
    }

    /// indices into the full compute particle buffer (not a compacted surface list).
    pub fn surface_indices(&self) -> Vec<u16> {
        self.current.surface_indices()
    }
}

impl CubeShape {
    fn build_vertices(n: usize, spacing: f32) -> Vec<PhysicsVertex> {
        let mut vertices = Vec::new();
        let spacing_x = 1.0 / n as f32;
        let spacing_y = 1.0 / n as f32;
        let spacing_z = spacing;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let renderable =
                        i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1;
                    vertices.push(PhysicsVertex {
                        position: Vector3::new(
                            i as f32 * spacing_x,
                            j as f32 * spacing_y,
                            k as f32 * spacing_z,
                        ),
                        velocity: Vector3::new(0.0, 0.0, 0.0),
                        renderable,
                        mass: 1.0,
                    });
                }
            }
        }
        vertices
    }

    fn new_static(n: usize, spacing: f32) -> Self {
        Self {
            vertices: Self::build_vertices(n, spacing),
            n,
            t: 0.0,
        }
    }

    pub fn new(n: usize, spacing: f32) -> Self {
        Self {
            vertices: Self::build_vertices(n, spacing),
            n,
            t: 0.0,
        }
    }

    fn grid_index(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.n * self.n + j * self.n + k
    }

    pub fn update(&mut self) {
        self.t += DT;
        for v in self.vertices.iter_mut() {
            v.position += v.velocity * DT;
        }
    }

    // a=bottom-left, b=bottom-right, c=top-right, d=top-left (viewed from outside)
    fn push_quad(indices: &mut Vec<u16>, a: u16, b: u16, c: u16, d: u16, flip: bool) {
        if flip {
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        } else {
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    /// Surface triangle indices into the *full* particle grid (same order as the compute buffer).
    /// Use this when the mesh vertex buffer is the shared Particle buffer.
    pub fn surface_indices(&self) -> Vec<u16> {
        let n = self.n;
        let idx = |i: usize, j: usize, k: usize| self.grid_index(i, j, k) as u16;
        let mut indices = Vec::new();

        for j in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(0, j, k),
                    idx(0, j + 1, k),
                    idx(0, j + 1, k + 1),
                    idx(0, j, k + 1),
                    false,
                );
            }
        }
        for j in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(n - 1, j, k),
                    idx(n - 1, j + 1, k),
                    idx(n - 1, j + 1, k + 1),
                    idx(n - 1, j, k + 1),
                    true,
                );
            }
        }
        for i in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, 0, k),
                    idx(i + 1, 0, k),
                    idx(i + 1, 0, k + 1),
                    idx(i, 0, k + 1),
                    false,
                );
            }
        }
        for i in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, n - 1, k),
                    idx(i + 1, n - 1, k),
                    idx(i + 1, n - 1, k + 1),
                    idx(i, n - 1, k + 1),
                    true,
                );
            }
        }
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, j, 0),
                    idx(i + 1, j, 0),
                    idx(i + 1, j + 1, 0),
                    idx(i, j + 1, 0),
                    false,
                );
            }
        }
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, j, n - 1),
                    idx(i + 1, j, n - 1),
                    idx(i + 1, j + 1, n - 1),
                    idx(i, j + 1, n - 1),
                    true,
                );
            }
        }
        indices
    }

    pub fn to_renderable(&self) -> (Vec<Vertex>, Vec<u16>) {
        self.to_renderable_with_highlight(&HashSet::new())
    }

    fn to_renderable_with_highlight(&self, highlight: &HashSet<usize>) -> (Vec<Vertex>, Vec<u16>) {
        let n = self.n;
        let mut render_map = vec![None; self.vertices.len()];
        let mut vertices = Vec::new();
        for (grid_idx, v) in self.vertices.iter().enumerate() {
            if v.renderable {
                render_map[grid_idx] = Some(vertices.len() as u16);
                let mut vertex: Vertex = v.clone().into();
                if highlight.contains(&grid_idx) {
                    vertex.set_color([1.0, 0.85, 0.15]);
                }
                vertices.push(vertex);
            }
        }
        let idx = |i: usize, j: usize, k: usize| render_map[self.grid_index(i, j, k)].unwrap();

        let mut indices = Vec::new();

        for j in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(0, j, k),
                    idx(0, j + 1, k),
                    idx(0, j + 1, k + 1),
                    idx(0, j, k + 1),
                    false,
                );
            }
        }
        for j in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(n - 1, j, k),
                    idx(n - 1, j + 1, k),
                    idx(n - 1, j + 1, k + 1),
                    idx(n - 1, j, k + 1),
                    true,
                );
            }
        }
        for i in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, 0, k),
                    idx(i + 1, 0, k),
                    idx(i + 1, 0, k + 1),
                    idx(i, 0, k + 1),
                    false,
                );
            }
        }
        for i in 0..n - 1 {
            for k in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, n - 1, k),
                    idx(i + 1, n - 1, k),
                    idx(i + 1, n - 1, k + 1),
                    idx(i, n - 1, k + 1),
                    true,
                );
            }
        }
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, j, 0),
                    idx(i + 1, j, 0),
                    idx(i + 1, j + 1, 0),
                    idx(i, j + 1, 0),
                    false,
                );
            }
        }
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                Self::push_quad(
                    &mut indices,
                    idx(i, j, n - 1),
                    idx(i + 1, j, n - 1),
                    idx(i + 1, j + 1, n - 1),
                    idx(i, j + 1, n - 1),
                    true,
                );
            }
        }

        let mut normal_sums = vec![Vector3::zeros(); vertices.len()];
        for tri in indices.chunks_exact(3) {
            let ia = tri[0] as usize;
            let ib = tri[1] as usize;
            let ic = tri[2] as usize;
            let pa = Vector3::new(
                vertices[ia].position[0],
                vertices[ia].position[1],
                vertices[ia].position[2],
            );
            let pb = Vector3::new(
                vertices[ib].position[0],
                vertices[ib].position[1],
                vertices[ib].position[2],
            );
            let pc = Vector3::new(
                vertices[ic].position[0],
                vertices[ic].position[1],
                vertices[ic].position[2],
            );
            let face_normal = (pb - pa).cross(&(pc - pa));
            if face_normal.norm_squared() > 1e-12 {
                normal_sums[ia] += face_normal;
                normal_sums[ib] += face_normal;
                normal_sums[ic] += face_normal;
            }
        }

        for (vertex, normal_sum) in vertices.iter_mut().zip(normal_sums.into_iter()) {
            let n = if normal_sum.norm_squared() > 1e-12 {
                normal_sum.normalize()
            } else {
                Vector3::new(0.0, 0.0, 1.0)
            };
            vertex.normal = [n.x, n.y, n.z];
        }

        (vertices, indices)
    }

}

fn face_bounds(shape: &CubeShape, face_index: usize) -> (f32, f32) {
    let component = |v: &PhysicsVertex| cubeface_component(v.position, face_index);
    let min = shape
        .vertices
        .iter()
        .map(component)
        .fold(f32::INFINITY, f32::min);
    let max = shape
        .vertices
        .iter()
        .map(component)
        .fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

fn opposite_face(face_index: usize) -> usize {
    match face_index {
        0 => 3,
        1 => 4,
        2 => 5,
        3 => 0,
        4 => 1,
        5 => 2,
        _ => panic!("invalid face index"),
    }
}

fn face_axis(face_index: usize) -> usize {
    match face_index {
        0 | 3 => 0,
        1 | 4 => 1,
        2 | 5 => 2,
        _ => panic!("invalid face index"),
    }
}

fn face_normal(face_index: usize) -> Vector3<f32> {
    match face_index {
        0 => Vector3::new(-1.0, 0.0, 0.0),
        1 => Vector3::new(0.0, -1.0, 0.0),
        2 => Vector3::new(0.0, 0.0, -1.0),
        3 => Vector3::new(1.0, 0.0, 0.0),
        4 => Vector3::new(0.0, 1.0, 0.0),
        5 => Vector3::new(0.0, 0.0, 1.0),
        _ => panic!("invalid face index"),
    }
}

fn clamp_stretch_target(face_index: usize, reference: &CubeShape, target: f32) -> f32 {
    let (min, max) = face_bounds(reference, face_index);
    let extra = (max - min) * (MAX_STRETCH - 1.0);
    if face_index <= 2 {
        target.clamp(min - extra, min)
    } else {
        target.clamp(max, max + extra)
    }
}

fn apply_face_stretch_forces(
    current: &CubeShape,
    face_index: usize,
    vertex_indices: &[usize],
    target_axis: f32,
    forces: &mut [Vector3<f32>],
) {
    let axis = face_axis(face_index);
    for &idx in vertex_indices {
        let vertex = &current.vertices[idx];
        let (current_axis, velocity_axis) = match axis {
            0 | 3 => (vertex.position.x, vertex.velocity.x),
            1 | 4 => (vertex.position.y, vertex.velocity.y),
            _ => (vertex.position.z, vertex.velocity.z),
        };
        let force_scalar =
            STRETCH_FORCE * (target_axis - current_axis) - STRETCH_DAMPING * velocity_axis;
        match axis {
            0 | 3 => forces[idx].x += force_scalar,
            1 | 4 => forces[idx].y += force_scalar,
            _ => forces[idx].z += force_scalar,
        }
    }
}

fn pin_face_vertices(
    reference: &CubeShape,
    current: &mut CubeShape,
    face_index: usize,
    vertex_indices: &[usize],
) {
    let axis = face_axis(face_index);
    for &idx in vertex_indices {
        let ref_v = &reference.vertices[idx];
        let cur_v = &mut current.vertices[idx];
        match axis {
            0 => {
                cur_v.position.x = ref_v.position.x;
                cur_v.velocity.x = 0.0;
            }
            1 => {
                cur_v.position.y = ref_v.position.y;
                cur_v.velocity.y = 0.0;
            }
            _ => {
                cur_v.position.z = ref_v.position.z;
                cur_v.velocity.z = 0.0;
            }
        }
    }
}

/// Eq (11) in https://arxiv.org/pdf/2009.06944
fn compute_densities(
    vertices: &[PhysicsVertex],
    kernel: &CubicKernel,
    neighbors: &[Vec<usize>],
) -> Vec<f32> {
    let mut densities = vec![0.0; vertices.len()];
    for i in 0..vertices.len() {
        let density = neighbor_sum(
            |j| {
                let r = vertices[i].position - vertices[j].position;
                vertices[j].mass * kernel_weight(kernel, r.norm())
            },
            &neighbors[i],
            0.0,
        );
        densities[i] = density;
    }
    densities
}

fn kernel_weight(kernel: &CubicKernel, r_norm: f32) -> f32 {
    kernel.weight(r_norm)
}

fn pairwise_displacement(x: &[PhysicsVertex], y: &[PhysicsVertex]) -> Vec<Vec<Vector3<f32>>> {
    let mut displacements = vec![vec![Vector3::new(0.0, 0.0, 0.0); y.len()]; x.len()];
    for i in 0..x.len() {
        for j in 0..y.len() {
            displacements[i][j] = x[i].position - y[j].position;
        }
    }
    displacements
}

fn strain_tensor(deformation: &Matrix3<f32>) -> Matrix3<f32> {
    0.5 * (deformation + deformation.transpose()) - Matrix3::identity()
}


fn compute_rotations(deformations: &[Matrix3<f32>]) -> Vec<Matrix3<f32>> {
    let mut rotations: Vec<Matrix3<f32>> = Vec::with_capacity(deformations.len());
    for i in 0..deformations.len() {
        // TODO: can be a lot faster if we don't use polar decomp;
        // also unsure from the paper whether this is right
        let (_, R_i) = deformations[i].polar();
        rotations.push(R_i);
    }
    rotations
}

fn initial_deformation_tensor_estimate(
    current_vertices: &[PhysicsVertex],
    reference_gradients: &[Vec<Vector3<f32>>],
    L: &[Matrix3<f32>],
    reference_volumes: &[f32],
    neighbors: &[Vec<usize>],
) -> Vec<Matrix3<f32>> {
    let mut deformations: Vec<Matrix3<f32>> = Vec::with_capacity(current_vertices.len());
    for i in 0..current_vertices.len() {
        let f_i = neighbor_sum(
            |j| {
                let x_ji = current_vertices[j].position - current_vertices[i].position;
                let grad_w_ij = reference_gradients[i][j];
                let v_j = reference_volumes[j];
                v_j * x_ji * grad_w_ij.transpose()
            },
            &neighbors[i],
            Matrix3::zeros(),
        );
        deformations.push(L[i] * f_i);
    }
    deformations
}

/// Naive impl of (159) in https://arxiv.org/pdf/2009.06944
fn deformation_tensor_star(
    current_vertices: &[PhysicsVertex],
    reference_vertices: &[PhysicsVertex],
    reference_gradients_star: &[Vec<Vector3<f32>>],
    rotations: &[Matrix3<f32>],
    reference_volumes: &[f32],
    neighbors: &[Vec<usize>],
) -> Vec<Matrix3<f32>> {
    let mut deformations_star = Vec::with_capacity(current_vertices.len());
    for i in 0..current_vertices.len() {
        let r_i = rotations[i];
        let f_i_star = neighbor_sum(
            |j| {
                let x_ji = current_vertices[j].position - current_vertices[i].position;
                let x_ji_ref = reference_vertices[j].position - reference_vertices[i].position;
                let grad_w_ij_star = reference_gradients_star[i][j];
                let v_j = reference_volumes[j];
                v_j * (x_ji - r_i * x_ji_ref) * grad_w_ij_star.transpose()
            },
            &neighbors[i],
            Matrix3::zeros(),
        );
        deformations_star.push(f_i_star + Matrix3::identity());
    }
    deformations_star
}

/// impl of (158) in https://arxiv.org/pdf/2009.06944
fn compute_grad_star(
    gradients: &[Vec<Vector3<f32>>],
    rotations: &[Matrix3<f32>],
    L: &[Matrix3<f32>],
) -> Vec<Vec<Vector3<f32>>> {
    let mut grad_star = Vec::with_capacity(gradients.len());
    for i in 0..gradients.len() {
        grad_star.push(Vec::with_capacity(gradients[i].len()));
        for j in 0..gradients[i].len() {
            grad_star[i].push(rotations[i] * L[i] * gradients[i][j]);
        }
    }
    grad_star
}

fn stress_tensor(
    deformations_star: &[Matrix3<f32>],
    mu: f32,
    lambda: f32
) -> Vec<Matrix3<f32>> {
    let mut stress_tensors = Vec::with_capacity(deformations_star.len());
    for i in 0..deformations_star.len() {
        let strain = strain_tensor(&deformations_star[i]);
        stress_tensors.push(2.0 * mu * strain + lambda * strain.trace() * Matrix3::identity());
    }
    stress_tensors
}

fn penalty_forces(
    reference_volumes: &[f32],
    k: f32,
    alpha: f32,
    deformations: &[Matrix3<f32>],
    reference_kernel_weights: &[Vec<f32>],
    current_vertices: &[PhysicsVertex],
    reference_vertices: &[PhysicsVertex],
    neighbors: &[Vec<usize>],
) -> Vec<Vector3<f32>> {
    let mut forces = Vec::with_capacity(current_vertices.len());
    for i in 0..current_vertices.len() {
        let force = -0.5 * alpha * k * neighbor_sum(
            |j| {
                if j == i { return Vector3::zeros(); }
                let x_ji_ref = reference_vertices[j].position - reference_vertices[i].position;
                let x_ji = current_vertices[j].position - current_vertices[i].position;
                let e_ji = deformations[i] * x_ji_ref - x_ji;
                let e_ij = x_ji - deformations[j] * x_ji_ref;
                reference_volumes[i] * reference_volumes[j] * (reference_kernel_weights[i][j] / x_ji_ref.norm_squared()) 
                    * (e_ji.dot(&x_ji) / x_ji.norm() - e_ij.dot(&x_ji_ref) / x_ji_ref.norm()) * x_ji / x_ji.norm()
                    
            },
            &neighbors[i],
            Vector3::zeros(),
        );
        forces.push(force);
    }
    forces
}

pub fn compute_forces(
    shape: &Cube,
    _kernel: &CubicKernel,
) -> Vec<Vector3<f32>> {
    let current = &shape.current;
    let initial_deformations = initial_deformation_tensor_estimate(
        &current.vertices,
        &shape.reference_gradients,
        &shape.L,
        &shape.reference_volumes,
        &shape.neighbors,
    );
    let rotations = compute_rotations(&initial_deformations);
    let grad_star = compute_grad_star(&shape.reference_gradients, &rotations, &shape.L);
    let deformations_star = deformation_tensor_star(
        &current.vertices,
        &shape.reference.vertices,
        &grad_star,
        &rotations,
        &shape.reference_volumes,
        &shape.neighbors,
    );
    let reference_volumes = &shape.reference_volumes;
    // approximate values for rubber
    let k = 100000.0;
    let nu = 0.30;
    let mu = k / (2.0 * (1.0 + nu));
    let lambda = k * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let stress_tensors = stress_tensor(&deformations_star, mu, lambda);
    let mut forces = Vec::with_capacity(current.vertices.len());
    for i in 0..current.vertices.len() {
        let force = neighbor_sum(
            |j| {
                reference_volumes[i] * reference_volumes[j]
                    * (stress_tensors[i] * grad_star[i][j] - stress_tensors[j] * grad_star[j][i])
            },
            &shape.neighbors[i],
            Vector3::zeros(),
        );
        forces.push(force);
    }
    let penalty_forces = penalty_forces(
        &shape.reference_volumes,
        k,
        0.9, // alpha: what to choose?
        &initial_deformations,
        &shape.reference_kernel_weights,
        &current.vertices,
        &shape.reference.vertices,
        &shape.neighbors,
    );
    forces.iter_mut().zip(penalty_forces.iter()).for_each(|(f, pf)| {
        *f += pf;
    });
    forces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoketest_cube_new() {
        let cube = CubeShape::new(3, 1.0);
        assert_eq!(cube.vertices.len(), 27);
        assert_eq!(cube.vertices.iter().filter(|v| v.renderable).count(), 26);
    }

    #[test]
    fn test_renderable() {
        let cube = CubeShape::new(2, 1.0);
        let (vertices, indices) = cube.to_renderable();
        assert_eq!(vertices.len(), 8);
        assert_eq!(indices.len(), 36);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_renderable_n3() {
        let cube = CubeShape::new(3, 1.0);
        let (vertices, indices) = cube.to_renderable();
        assert_eq!(vertices.len(), 26);
        assert_eq!(indices.len(), 6 * 4 * 6);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_cubic_kernel() {
        let kernel = CubicKernel::new(1.0);
        let x_i = vec![Vector3::new(0.0, 0.0, 0.0)];
        let x_j = vec![Vector3::new(1.0, 0.0, 0.0)];
        let neighbors = precompute_neighbors(&x_j, &kernel);
        let w = kernel.eval(&x_i, &x_j, &neighbors);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].len(), 1);
        assert_eq!(w[0][0], 0.0);
    }

    #[test]
    fn test_stress_tensor_zero() {
        let deformation = Matrix3::identity();
        let stress = stress_tensor(
            &[deformation], 0.0, 0.0
        )[0];
        assert_eq!(stress, Matrix3::zeros());
    }

    #[test]
    fn no_deformation() {
        let m = 3;
        let kernel = CubicKernel::new(1.0 / (m - 1) as f32);
        let cube = Cube::new(m, 1.0 / m as f32, &kernel);
        let deformations = initial_deformation_tensor_estimate(
            &cube.reference.vertices,
            &cube.reference_gradients,
            &cube.L,
            &cube.reference_volumes,
            &cube.neighbors,
        );
        assert!(deformations[0].relative_eq(&Matrix3::identity(), 1e-6, 1e-6));
    }
    #[test]
    fn test_neighbor_sum() {
        let neighbors = vec![1, 3, 5];
        let sum = neighbor_sum(|idx| idx, &neighbors, 0);
        assert_eq!(sum, 9);
    }

    #[test]
    fn pin_face_restores_axis_position_and_velocity() {
        let reference = CubeShape::new_static(3, 1.0);
        let mut current = CubeShape::new(3, 1.0);
        let faces = build_cube_faces(&reference.vertices, 3);

        for vertex in &mut current.vertices {
            vertex.position += Vector3::new(0.2, 0.3, 0.4);
            vertex.velocity = Vector3::new(1.0, 2.0, 3.0);
        }

        pin_face_vertices(&reference, &mut current, 0, &faces[0].vertex_indices);

        for &idx in &faces[0].vertex_indices {
            let reference_vertex = &reference.vertices[idx];
            let current_vertex = &current.vertices[idx];
            assert_eq!(current_vertex.position.x, reference_vertex.position.x);
            assert_eq!(current_vertex.velocity.x, 0.0);
        }
    }
}
