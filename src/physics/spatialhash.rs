use std::ops::{Deref, DerefMut};

use cgmath::Vector3;
use spatial_hash_3d::{BoxIdxIterator, BoxIterator, BoxIteratorMut, SpatialHashGrid};

#[derive(Debug)]
pub struct SpatialHash3D<T> {
    cube_sidelength: f32,
    bottom_left: Vector3<f32>,
    grid: SpatialHashGrid<T>,
}

impl<T> Deref for SpatialHash3D<T> {
    type Target = SpatialHashGrid<T>;
    fn deref(&self) -> &Self::Target {
        &self.grid
    }
}

impl<T> DerefMut for SpatialHash3D<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.grid
    }
}

impl<T> SpatialHash3D<T> {
    pub fn new<V>(partitions: (usize, usize, usize), filler: V, cube_sidelength: f32) -> Self
    where
        V: FnMut() -> T,
    {
        Self {
            cube_sidelength,
            bottom_left: Vector3::new(0.0, 0.0, 0.0),
            grid: SpatialHashGrid::new(partitions.0, partitions.1, partitions.2, filler),
        }
    }
    pub fn set_bottom_left(mut self, bottom_left: Vector3<f32>) -> Self {
        self.bottom_left = bottom_left;
        self
    }

    pub fn iter_cubes_mut(
        &mut self,
        min: Vector3<f32>,
        max: Vector3<f32>,
    ) -> BoxIteratorMut<'_, T> {
        let min = min - self.bottom_left;
        let max = max - self.bottom_left;
        let min = Vector3::new(
            (min / self.cube_sidelength).x as u32,
            (min / self.cube_sidelength).y as u32,
            (min / self.cube_sidelength).z as u32,
        );
        let max = Vector3::new(
            (max / self.cube_sidelength).x as u32,
            (max / self.cube_sidelength).y as u32,
            (max / self.cube_sidelength).z as u32,
        );
        self.grid.iter_cubes_mut(min, max)
    }

    pub fn neighbors(&self, pos: Vector3<f32>) -> BoxIterator<'_, T> {
        let off = self.cube_sidelength * Vector3::new(1.0, 1.0, 1.0);
        let min = (pos - self.bottom_left) - off;
        let max = (pos - self.bottom_left) + off;
        self.iter_cubes(min, max)
    }

    pub fn iter_cubes(&self, min: Vector3<f32>, max: Vector3<f32>) -> BoxIterator<'_, T> {
        let min = min - self.bottom_left;
        let max = max - self.bottom_left;
        let min = Vector3::new(
            (min / self.cube_sidelength).x as u32,
            (min / self.cube_sidelength).y as u32,
            (min / self.cube_sidelength).z as u32,
        );
        let max = Vector3::new(
            (max / self.cube_sidelength).x as u32,
            (max / self.cube_sidelength).y as u32,
            (max / self.cube_sidelength).z as u32,
        );
        self.grid.iter_cubes(min, max)
    }

    pub fn pos_to_index(&self, pos: Vector3<f32>) -> Option<usize> {
        let pos = pos - self.bottom_left;
        if pos.x < 0.0 || pos.y < 0.0 || pos.z < 0.0 {
            return None;
        }
        let p = Vector3::new(
            (pos / self.cube_sidelength).x as i32,
            (pos / self.cube_sidelength).y as i32,
            (pos / self.cube_sidelength).z as i32,
        );
        let p = Vector3::new(p.x as u32, p.y as u32, p.z as u32);
        self.grid.pos_to_index(p)
    }

    pub fn clear(&mut self, clear_fn: &mut impl FnMut(&mut T)) {
        let dims = self.grid.size();
        let dims = Vector3::new(dims.x as f32, dims.y as f32, dims.z as f32);
        let top_left = self.bottom_left + dims * self.cube_sidelength;
        let top_left = Vector3::new(
            (top_left.x / self.cube_sidelength) as u32,
            (top_left.y / self.cube_sidelength) as u32,
            (top_left.z / self.cube_sidelength) as u32,
        );

        let bottom_left = self.bottom_left - Vector3::new(1.0, 1.0, 1.0);
        let bottom_left = Vector3::new(
            (bottom_left.x / self.cube_sidelength) as u32,
            (bottom_left.y / self.cube_sidelength) as u32,
            (bottom_left.z / self.cube_sidelength) as u32,
        );

        self.grid
            .iter_cubes_mut(bottom_left, top_left)
            .for_each(|(_, _, v)| clear_fn(v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::Vector3;

    fn make_grid() -> SpatialHash3D<Vec<i32>> {
        SpatialHash3D::new((4, 4, 4), || Vec::new(), 1.0)
    }

    #[test]
    fn test_pos_to_index_basic() {
        let grid = make_grid();

        // داخل grid
        let idx = grid.pos_to_index(Vector3::new(1.2, 2.3, 3.4));
        assert!(idx.is_some());

        // edge case: exactly on boundary
        let idx = grid.pos_to_index(Vector3::new(0.0, 0.0, 0.0));
        assert!(idx.is_some());
    }

    #[test]
    fn test_pos_to_index_out_of_bounds() {
        let grid = make_grid();

        // outside grid (negative)
        let idx = grid.pos_to_index(Vector3::new(-10.0, 0.0, 0.0));
        assert!(idx.is_none());

        // outside grid (too large)
        let idx = grid.pos_to_index(Vector3::new(100.0, 100.0, 100.0));
        assert!(idx.is_none());
    }

    #[test]
    fn test_iter_cubes_mut_write() {
        let mut grid = make_grid();

        let min = Vector3::new(0.0, 0.0, 0.0);
        let max = Vector3::new(1.0, 1.0, 1.0);

        for (_, _, cell) in grid.iter_cubes_mut(min, max) {
            cell.push(42);
        }

        let mut count = 0;
        for (_, cell) in grid.iter_cubes(min, max) {
            if cell.contains(&42) {
                count += 1;
            }
        }

        assert!(count > 0);
    }

    #[test]
    fn test_bottom_left_offset() {
        let grid = SpatialHash3D::new((4, 4, 4), || Vec::<i32>::new(), 1.0)
            .set_bottom_left(Vector3::new(10.0, 10.0, 10.0));

        // This should map to (0,0,0) in grid space
        let idx = grid.pos_to_index(Vector3::new(10.1, 10.1, 10.1));
        dbg!(idx);
        assert!(idx.is_some());

        // This is before bottom_left → should fail
        let idx = grid.pos_to_index(Vector3::new(9.9, 10.0, 10.0));
        assert!(idx.is_none());
    }

    #[test]
    fn test_clear() {
        let mut grid = make_grid();

        // Fill some cells
        for (_, _, cell) in
            grid.iter_cubes_mut(Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 3.0, 3.0))
        {
            cell.push(123);
        }

        // Clear all
        grid.clear(&mut |cell| cell.clear());

        // Verify all empty
        for (_, cell) in grid.iter_cubes(Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 3.0, 3.0)) {
            assert!(cell.is_empty());
        }
    }

    #[test]
    fn test_float_truncation_behavior() {
        let grid = make_grid();

        // These should land in same cube due to truncation
        let a = grid.pos_to_index(Vector3::new(1.1, 1.1, 1.1));
        let b = grid.pos_to_index(Vector3::new(1.9, 1.9, 1.9));

        assert_eq!(a, b);
    }
}
