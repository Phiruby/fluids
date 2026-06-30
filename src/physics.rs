use crate::Vertex;

#[derive(Debug, Clone)]
#[allow(unused)]
struct PhysicsVertex {
    position: [f32; 3],
    velocity: [f32; 3],
    renderable: bool,
}

impl From<PhysicsVertex> for Vertex {
    fn from(vertex: PhysicsVertex) -> Self {
        Vertex {
            position: vertex.position,
            color: [0.5, 0.0, 0.5],
        }
    }
}

#[allow(unused)]
struct GaussianKernel {
    h: f32,
}

#[derive(Debug)]
pub struct Cube {
    pub vertices: Vec<PhysicsVertex>,
    n: usize,
    t: f32,
}

impl Cube {
    pub fn new(n: usize, spacing: f32) -> Self {
        let mut vertices = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let renderable = i == 0
                        || j == 0
                        || k == 0
                        || i == n - 1
                        || j == n - 1
                        || k == n - 1;
                    vertices.push(PhysicsVertex {
                        position: [i as f32 * spacing, j as f32 * spacing, k as f32 * spacing],
                        velocity: [0.0, 0.0, 0.0],
                        renderable,
                    });
                }
            }
        }
        Self { vertices, n, t: 0.0 }
    }

    fn grid_index(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.n * self.n + j * self.n + k
    }

    pub fn update(&mut self) {
        self.t += 0.001;
        for v in self.vertices.iter_mut() {
            v.position[0] += 0.1 * self.t;
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

    pub fn to_renderable(&self) -> (Vec<Vertex>, Vec<u16>) {
        let n = self.n;
        let mut render_map = vec![None; self.vertices.len()];
        let mut vertices = Vec::new();
        for (grid_idx, v) in self.vertices.iter().enumerate() {
            if v.renderable {
                render_map[grid_idx] = Some(vertices.len() as u16);
                vertices.push(v.clone().into());
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

        (vertices, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoketest_cube_new() {
        let cube = Cube::new(3, 1.0);
        assert_eq!(cube.vertices.len(), 27);
        assert_eq!(cube.vertices.iter().filter(|v| v.renderable).count(), 26);
    }

    #[test]
    fn test_renderable() {
        let cube = Cube::new(2, 1.0);
        let (vertices, indices) = cube.to_renderable();
        assert_eq!(vertices.len(), 8);
        assert_eq!(indices.len(), 36);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_renderable_n3() {
        let cube = Cube::new(3, 1.0);
        let (vertices, indices) = cube.to_renderable();
        assert_eq!(vertices.len(), 26);
        assert_eq!(indices.len(), 6 * 4 * 6);
        assert_eq!(indices.len() % 3, 0);
    }
}
