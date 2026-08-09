use super::{MPM_BOTTOM_LEFT, MPM_DX, MPM_N};

pub struct StationaryBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}


const WALL_MARGIN: f32 = 3.0 * MPM_DX;

fn grid_node_max_xy() -> [f32; 2] {
    let e = (MPM_N - 1) as f32 * MPM_DX;
    [MPM_BOTTOM_LEFT[0] + e, MPM_BOTTOM_LEFT[1] + e]
}

pub fn mpm_collider() -> StationaryBox {
    let gmax = grid_node_max_xy();
    StationaryBox {
        min: [
            MPM_BOTTOM_LEFT[0] + WALL_MARGIN,
            MPM_BOTTOM_LEFT[1] + WALL_MARGIN,
            -1.0,
        ],
        max: [gmax[0] - WALL_MARGIN, gmax[1] - WALL_MARGIN, 1.0],
    }
}

pub fn mpm_collider_bounds() -> ([f32; 3], [f32; 3]) {
    let c = mpm_collider();
    (c.min, c.max)
}
