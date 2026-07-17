// NOTE: particles and hash_offsets should be the buffers from the SPH simulation.
// Keep `N` in sync with `GRID_N` in marching_cubes.rs (PrefixSum limit: 3*N^3 <= 512^2).

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>,
    density: f32,
    rest_density: f32,
    mass: f32,
    cell_hash: u32,
    boundary: u32,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

struct McParams {
    iso_level: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const BOTTOM_LEFT: vec3<f32> = vec3<f32>(-1.25, 0.0, -1.05);
const N: u32 = 40;
const EDGE_N: u32 = N + 1u;
const MARCHING_CUBES_CELL_SIZE: f32 = 0.05;

const CELL_SIZE: f32 = 0.1;
const HASHTABLE_SIZE: u32 = 8192u;
const SIGMA: f32 = 8.0 / (3.14159 * pow(CELL_SIZE, 3.0));

const EDGE_TABLE: array<u32, 256> = array<u32, 256>(
0x0u, 0x109u, 0x203u, 0x30au, 0x406u, 0x50fu, 0x605u, 0x70cu,
    0x80cu, 0x905u, 0xa0fu, 0xb06u, 0xc0au, 0xd03u, 0xe09u, 0xf00u,
    0x190u, 0x99u, 0x393u, 0x29au, 0x596u, 0x49fu, 0x795u, 0x69cu,
    0x99cu, 0x895u, 0xb9fu, 0xa96u, 0xd9au, 0xc93u, 0xf99u, 0xe90u,
    0x230u, 0x339u, 0x33u, 0x13au, 0x636u, 0x73fu, 0x435u, 0x53cu,
    0xa3cu, 0xb35u, 0x83fu, 0x936u, 0xe3au, 0xf33u, 0xc39u, 0xd30u,
    0x3a0u, 0x2a9u, 0x1a3u, 0xaau, 0x7a6u, 0x6afu, 0x5a5u, 0x4acu,
    0xbacu, 0xaa5u, 0x9afu, 0x8a6u, 0xfaau, 0xea3u, 0xda9u, 0xca0u,
    0x460u, 0x569u, 0x663u, 0x76au, 0x66u, 0x16fu, 0x265u, 0x36cu,
    0xc6cu, 0xd65u, 0xe6fu, 0xf66u, 0x86au, 0x963u, 0xa69u, 0xb60u,
    0x5f0u, 0x4f9u, 0x7f3u, 0x6fau, 0x1f6u, 0xffu, 0x3f5u, 0x2fcu,
    0xdfcu, 0xcf5u, 0xfffu, 0xef6u, 0x9fau, 0x8f3u, 0xbf9u, 0xaf0u,
    0x650u, 0x759u, 0x453u, 0x55au, 0x256u, 0x35fu, 0x55u, 0x15cu,
    0xe5cu, 0xf55u, 0xc5fu, 0xd56u, 0xa5au, 0xb53u, 0x859u, 0x950u,
    0x7c0u, 0x6c9u, 0x5c3u, 0x4cau, 0x3c6u, 0x2cfu, 0x1c5u, 0xccu,
    0xfccu, 0xec5u, 0xdcfu, 0xcc6u, 0xbcau, 0xac3u, 0x9c9u, 0x8c0u,
    0x8c0u, 0x9c9u, 0xac3u, 0xbcau, 0xcc6u, 0xdcfu, 0xec5u, 0xfccu,
    0xccu, 0x1c5u, 0x2cfu, 0x3c6u, 0x4cau, 0x5c3u, 0x6c9u, 0x7c0u,
    0x950u, 0x859u, 0xb53u, 0xa5au, 0xd56u, 0xc5fu, 0xf55u, 0xe5cu,
    0x15cu, 0x55u, 0x35fu, 0x256u, 0x55au, 0x453u, 0x759u, 0x650u,
    0xaf0u, 0xbf9u, 0x8f3u, 0x9fau, 0xef6u, 0xfffu, 0xcf5u, 0xdfcu,
    0x2fcu, 0x3f5u, 0xffu, 0x1f6u, 0x6fau, 0x7f3u, 0x4f9u, 0x5f0u,
    0xb60u, 0xa69u, 0x963u, 0x86au, 0xf66u, 0xe6fu, 0xd65u, 0xc6cu,
    0x36cu, 0x265u, 0x16fu, 0x66u, 0x76au, 0x663u, 0x569u, 0x460u,
    0xca0u, 0xda9u, 0xea3u, 0xfaau, 0x8a6u, 0x9afu, 0xaa5u, 0xbacu,
    0x4acu, 0x5a5u, 0x6afu, 0x7a6u, 0xaau, 0x1a3u, 0x2a9u, 0x3a0u,
    0xd30u, 0xc39u, 0xf33u, 0xe3au, 0x936u, 0x83fu, 0xb35u, 0xa3cu,
    0x53cu, 0x435u, 0x73fu, 0x636u, 0x13au, 0x33u, 0x339u, 0x230u,
    0xe90u, 0xf99u, 0xc93u, 0xd9au, 0xa96u, 0xb9fu, 0x895u, 0x99cu,
    0x69cu, 0x795u, 0x49fu, 0x596u, 0x29au, 0x393u, 0x99u, 0x190u,
    0xf00u, 0xe09u, 0xd03u, 0xc0au, 0xb06u, 0xa0fu, 0x905u, 0x80cu,
    0x70cu, 0x605u, 0x50fu, 0x406u, 0x30au, 0x203u, 0x109u, 0x0u
);

const TRI_TABLE: array<array<u32, 16>, 256> = array<array<u32, 16>, 256>(
    array(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 1u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 8u, 3u, 9u, 8u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 3u, 1u, 2u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 2u, 10u, 0u, 2u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 8u, 3u, 2u, 10u, 8u, 10u, 9u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 11u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 11u, 2u, 8u, 11u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 9u, 0u, 2u, 3u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 11u, 2u, 1u, 9u, 11u, 9u, 8u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 10u, 1u, 11u, 10u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 10u, 1u, 0u, 8u, 10u, 8u, 11u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 9u, 0u, 3u, 11u, 9u, 11u, 10u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 8u, 10u, 10u, 8u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 7u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 3u, 0u, 7u, 3u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 1u, 9u, 8u, 4u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 1u, 9u, 4u, 7u, 1u, 7u, 3u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 10u, 8u, 4u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 4u, 7u, 3u, 0u, 4u, 1u, 2u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 2u, 10u, 9u, 0u, 2u, 8u, 4u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 10u, 9u, 2u, 9u, 7u, 2u, 7u, 3u, 7u, 9u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 4u, 7u, 3u, 11u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(11u, 4u, 7u, 11u, 2u, 4u, 2u, 0u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 0u, 1u, 8u, 4u, 7u, 2u, 3u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 7u, 11u, 9u, 4u, 11u, 9u, 11u, 2u, 9u, 2u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 10u, 1u, 3u, 11u, 10u, 7u, 8u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 11u, 10u, 1u, 4u, 11u, 1u, 0u, 4u, 7u, 11u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 7u, 8u, 9u, 0u, 11u, 9u, 11u, 10u, 11u, 0u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 7u, 11u, 4u, 11u, 9u, 9u, 11u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 5u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 5u, 4u, 0u, 8u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 5u, 4u, 1u, 5u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 5u, 4u, 8u, 3u, 5u, 3u, 1u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 10u, 9u, 5u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 0u, 8u, 1u, 2u, 10u, 4u, 9u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 2u, 10u, 5u, 4u, 2u, 4u, 0u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 10u, 5u, 3u, 2u, 5u, 3u, 5u, 4u, 3u, 4u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 5u, 4u, 2u, 3u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 11u, 2u, 0u, 8u, 11u, 4u, 9u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 5u, 4u, 0u, 1u, 5u, 2u, 3u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 1u, 5u, 2u, 5u, 8u, 2u, 8u, 11u, 4u, 8u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 3u, 11u, 10u, 1u, 3u, 9u, 5u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 9u, 5u, 0u, 8u, 1u, 8u, 10u, 1u, 8u, 11u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 4u, 0u, 5u, 0u, 11u, 5u, 11u, 10u, 11u, 0u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 4u, 8u, 5u, 8u, 10u, 10u, 8u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 7u, 8u, 5u, 7u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 3u, 0u, 9u, 5u, 3u, 5u, 7u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 7u, 8u, 0u, 1u, 7u, 1u, 5u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 5u, 3u, 3u, 5u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 7u, 8u, 9u, 5u, 7u, 10u, 1u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 1u, 2u, 9u, 5u, 0u, 5u, 3u, 0u, 5u, 7u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 0u, 2u, 8u, 2u, 5u, 8u, 5u, 7u, 10u, 5u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 10u, 5u, 2u, 5u, 3u, 3u, 5u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 9u, 5u, 7u, 8u, 9u, 3u, 11u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 5u, 7u, 9u, 7u, 2u, 9u, 2u, 0u, 2u, 7u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 3u, 11u, 0u, 1u, 8u, 1u, 7u, 8u, 1u, 5u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(11u, 2u, 1u, 11u, 1u, 7u, 7u, 1u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 5u, 8u, 8u, 5u, 7u, 10u, 1u, 3u, 10u, 3u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 7u, 0u, 5u, 0u, 9u, 7u, 11u, 0u, 1u, 0u, 10u, 11u, 10u, 0u, 0xffffffffu),
    array(11u, 10u, 0u, 11u, 0u, 3u, 10u, 5u, 0u, 8u, 0u, 7u, 5u, 7u, 0u, 0xffffffffu),
    array(11u, 10u, 5u, 7u, 11u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 6u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 3u, 5u, 10u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 0u, 1u, 5u, 10u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 8u, 3u, 1u, 9u, 8u, 5u, 10u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 6u, 5u, 2u, 6u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 6u, 5u, 1u, 2u, 6u, 3u, 0u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 6u, 5u, 9u, 0u, 6u, 0u, 2u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 9u, 8u, 5u, 8u, 2u, 5u, 2u, 6u, 3u, 2u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 3u, 11u, 10u, 6u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(11u, 0u, 8u, 11u, 2u, 0u, 10u, 6u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 1u, 9u, 2u, 3u, 11u, 5u, 10u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 10u, 6u, 1u, 9u, 2u, 9u, 11u, 2u, 9u, 8u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 3u, 11u, 6u, 5u, 3u, 5u, 1u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 11u, 0u, 11u, 5u, 0u, 5u, 1u, 5u, 11u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 11u, 6u, 0u, 3u, 6u, 0u, 6u, 5u, 0u, 5u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 5u, 9u, 6u, 9u, 11u, 11u, 9u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 10u, 6u, 4u, 7u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 3u, 0u, 4u, 7u, 3u, 6u, 5u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 9u, 0u, 5u, 10u, 6u, 8u, 4u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 6u, 5u, 1u, 9u, 7u, 1u, 7u, 3u, 7u, 9u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 1u, 2u, 6u, 5u, 1u, 4u, 7u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 5u, 5u, 2u, 6u, 3u, 0u, 4u, 3u, 4u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 4u, 7u, 9u, 0u, 5u, 0u, 6u, 5u, 0u, 2u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 3u, 9u, 7u, 9u, 4u, 3u, 2u, 9u, 5u, 9u, 6u, 2u, 6u, 9u, 0xffffffffu),
    array(3u, 11u, 2u, 7u, 8u, 4u, 10u, 6u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 10u, 6u, 4u, 7u, 2u, 4u, 2u, 0u, 2u, 7u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 1u, 9u, 4u, 7u, 8u, 2u, 3u, 11u, 5u, 10u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 2u, 1u, 9u, 11u, 2u, 9u, 4u, 11u, 7u, 11u, 4u, 5u, 10u, 6u, 0xffffffffu),
    array(8u, 4u, 7u, 3u, 11u, 5u, 3u, 5u, 1u, 5u, 11u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 1u, 11u, 5u, 11u, 6u, 1u, 0u, 11u, 7u, 11u, 4u, 0u, 4u, 11u, 0xffffffffu),
    array(0u, 5u, 9u, 0u, 6u, 5u, 0u, 3u, 6u, 11u, 6u, 3u, 8u, 4u, 7u, 0xffffffffu),
    array(6u, 5u, 9u, 6u, 9u, 11u, 4u, 7u, 9u, 7u, 11u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 4u, 9u, 6u, 4u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 10u, 6u, 4u, 9u, 10u, 0u, 8u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 0u, 1u, 10u, 6u, 0u, 6u, 4u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 3u, 1u, 8u, 1u, 6u, 8u, 6u, 4u, 6u, 1u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 4u, 9u, 1u, 2u, 4u, 2u, 6u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 0u, 8u, 1u, 2u, 9u, 2u, 4u, 9u, 2u, 6u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 2u, 4u, 4u, 2u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 3u, 2u, 8u, 2u, 4u, 4u, 2u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 4u, 9u, 10u, 6u, 4u, 11u, 2u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 2u, 2u, 8u, 11u, 4u, 9u, 10u, 4u, 10u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 11u, 2u, 0u, 1u, 6u, 0u, 6u, 4u, 6u, 1u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 4u, 1u, 6u, 1u, 10u, 4u, 8u, 1u, 2u, 1u, 11u, 8u, 11u, 1u, 0xffffffffu),
    array(9u, 6u, 4u, 9u, 3u, 6u, 9u, 1u, 3u, 11u, 6u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 11u, 1u, 8u, 1u, 0u, 11u, 6u, 1u, 9u, 1u, 4u, 6u, 4u, 1u, 0xffffffffu),
    array(3u, 11u, 6u, 3u, 6u, 0u, 0u, 6u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 4u, 8u, 11u, 6u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 10u, 6u, 7u, 8u, 10u, 8u, 9u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 7u, 3u, 0u, 10u, 7u, 0u, 9u, 10u, 6u, 7u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 6u, 7u, 1u, 10u, 7u, 1u, 7u, 8u, 1u, 8u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 6u, 7u, 10u, 7u, 1u, 1u, 7u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 6u, 1u, 6u, 8u, 1u, 8u, 9u, 8u, 6u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 6u, 9u, 2u, 9u, 1u, 6u, 7u, 9u, 0u, 9u, 3u, 7u, 3u, 9u, 0xffffffffu),
    array(7u, 8u, 0u, 7u, 0u, 6u, 6u, 0u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 3u, 2u, 6u, 7u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 3u, 11u, 10u, 6u, 8u, 10u, 8u, 9u, 8u, 6u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 0u, 7u, 2u, 7u, 11u, 0u, 9u, 7u, 6u, 7u, 10u, 9u, 10u, 7u, 0xffffffffu),
    array(1u, 8u, 0u, 1u, 7u, 8u, 1u, 10u, 7u, 6u, 7u, 10u, 2u, 3u, 11u, 0xffffffffu),
    array(11u, 2u, 1u, 11u, 1u, 7u, 10u, 6u, 1u, 6u, 7u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 9u, 6u, 8u, 6u, 7u, 9u, 1u, 6u, 11u, 6u, 3u, 1u, 3u, 6u, 0xffffffffu),
    array(0u, 9u, 1u, 11u, 6u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 8u, 0u, 7u, 0u, 6u, 3u, 11u, 0u, 11u, 6u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 11u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 6u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 0u, 8u, 11u, 7u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 1u, 9u, 11u, 7u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 1u, 9u, 8u, 3u, 1u, 11u, 7u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 1u, 2u, 6u, 11u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 10u, 3u, 0u, 8u, 6u, 11u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 9u, 0u, 2u, 10u, 9u, 6u, 11u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 11u, 7u, 2u, 10u, 3u, 10u, 8u, 3u, 10u, 9u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 2u, 3u, 6u, 2u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 0u, 8u, 7u, 6u, 0u, 6u, 2u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 7u, 6u, 2u, 3u, 7u, 0u, 1u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 6u, 2u, 1u, 8u, 6u, 1u, 9u, 8u, 8u, 7u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 7u, 6u, 10u, 1u, 7u, 1u, 3u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 7u, 6u, 1u, 7u, 10u, 1u, 8u, 7u, 1u, 0u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 3u, 7u, 0u, 7u, 10u, 0u, 10u, 9u, 6u, 10u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 6u, 10u, 7u, 10u, 8u, 8u, 10u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 8u, 4u, 11u, 8u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 6u, 11u, 3u, 0u, 6u, 0u, 4u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 6u, 11u, 8u, 4u, 6u, 9u, 0u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 4u, 6u, 9u, 6u, 3u, 9u, 3u, 1u, 11u, 3u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 8u, 4u, 6u, 11u, 8u, 2u, 10u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 10u, 3u, 0u, 11u, 0u, 6u, 11u, 0u, 4u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 11u, 8u, 4u, 6u, 11u, 0u, 2u, 9u, 2u, 10u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 9u, 3u, 10u, 3u, 2u, 9u, 4u, 3u, 11u, 3u, 6u, 4u, 6u, 3u, 0xffffffffu),
    array(8u, 2u, 3u, 8u, 4u, 2u, 4u, 6u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 4u, 2u, 4u, 6u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 9u, 0u, 2u, 3u, 4u, 2u, 4u, 6u, 4u, 3u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 9u, 4u, 1u, 4u, 2u, 2u, 4u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 1u, 3u, 8u, 6u, 1u, 8u, 4u, 6u, 6u, 10u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 1u, 0u, 10u, 0u, 6u, 6u, 0u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 6u, 3u, 4u, 3u, 8u, 6u, 10u, 3u, 0u, 3u, 9u, 10u, 9u, 3u, 0xffffffffu),
    array(10u, 9u, 4u, 6u, 10u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 9u, 5u, 7u, 6u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 3u, 4u, 9u, 5u, 11u, 7u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 0u, 1u, 5u, 4u, 0u, 7u, 6u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(11u, 7u, 6u, 8u, 3u, 4u, 3u, 5u, 4u, 3u, 1u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 5u, 4u, 10u, 1u, 2u, 7u, 6u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 11u, 7u, 1u, 2u, 10u, 0u, 8u, 3u, 4u, 9u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 6u, 11u, 5u, 4u, 10u, 4u, 2u, 10u, 4u, 0u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 4u, 8u, 3u, 5u, 4u, 3u, 2u, 5u, 10u, 5u, 2u, 11u, 7u, 6u, 0xffffffffu),
    array(7u, 2u, 3u, 7u, 6u, 2u, 5u, 4u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 5u, 4u, 0u, 8u, 6u, 0u, 6u, 2u, 6u, 8u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 6u, 2u, 3u, 7u, 6u, 1u, 5u, 0u, 5u, 4u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 2u, 8u, 6u, 8u, 7u, 2u, 1u, 8u, 4u, 8u, 5u, 1u, 5u, 8u, 0xffffffffu),
    array(9u, 5u, 4u, 10u, 1u, 6u, 1u, 7u, 6u, 1u, 3u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 6u, 10u, 1u, 7u, 6u, 1u, 0u, 7u, 8u, 7u, 0u, 9u, 5u, 4u, 0xffffffffu),
    array(4u, 0u, 10u, 4u, 10u, 5u, 0u, 3u, 10u, 6u, 10u, 7u, 3u, 7u, 10u, 0xffffffffu),
    array(7u, 6u, 10u, 7u, 10u, 8u, 5u, 4u, 10u, 4u, 8u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 9u, 5u, 6u, 11u, 9u, 11u, 8u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 6u, 11u, 0u, 6u, 3u, 0u, 5u, 6u, 0u, 9u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 11u, 8u, 0u, 5u, 11u, 0u, 1u, 5u, 5u, 6u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(6u, 11u, 3u, 6u, 3u, 5u, 5u, 3u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 10u, 9u, 5u, 11u, 9u, 11u, 8u, 11u, 5u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 11u, 3u, 0u, 6u, 11u, 0u, 9u, 6u, 5u, 6u, 9u, 1u, 2u, 10u, 0xffffffffu),
    array(11u, 8u, 5u, 11u, 5u, 6u, 8u, 0u, 5u, 10u, 5u, 2u, 0u, 2u, 5u, 0xffffffffu),
    array(6u, 11u, 3u, 6u, 3u, 5u, 2u, 10u, 3u, 10u, 5u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 8u, 9u, 5u, 2u, 8u, 5u, 6u, 2u, 3u, 8u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 5u, 6u, 9u, 6u, 0u, 0u, 6u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 5u, 8u, 1u, 8u, 0u, 5u, 6u, 8u, 3u, 8u, 2u, 6u, 2u, 8u, 0xffffffffu),
    array(1u, 5u, 6u, 2u, 1u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 3u, 6u, 1u, 6u, 10u, 3u, 8u, 6u, 5u, 6u, 9u, 8u, 9u, 6u, 0xffffffffu),
    array(10u, 1u, 0u, 10u, 0u, 6u, 9u, 5u, 0u, 5u, 6u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 3u, 8u, 5u, 6u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 5u, 6u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(11u, 5u, 10u, 7u, 5u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(11u, 5u, 10u, 11u, 7u, 5u, 8u, 3u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 11u, 7u, 5u, 10u, 11u, 1u, 9u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 7u, 5u, 10u, 11u, 7u, 9u, 8u, 1u, 8u, 3u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(11u, 1u, 2u, 11u, 7u, 1u, 7u, 5u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 3u, 1u, 2u, 7u, 1u, 7u, 5u, 7u, 2u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 7u, 5u, 9u, 2u, 7u, 9u, 0u, 2u, 2u, 11u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(7u, 5u, 2u, 7u, 2u, 11u, 5u, 9u, 2u, 3u, 2u, 8u, 9u, 8u, 2u, 0xffffffffu),
    array(2u, 5u, 10u, 2u, 3u, 5u, 3u, 7u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 2u, 0u, 8u, 5u, 2u, 8u, 7u, 5u, 10u, 2u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 0u, 1u, 5u, 10u, 3u, 5u, 3u, 7u, 3u, 10u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 8u, 2u, 9u, 2u, 1u, 8u, 7u, 2u, 10u, 2u, 5u, 7u, 5u, 2u, 0xffffffffu),
    array(1u, 3u, 5u, 3u, 7u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 7u, 0u, 7u, 1u, 1u, 7u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 0u, 3u, 9u, 3u, 5u, 5u, 3u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 8u, 7u, 5u, 9u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 8u, 4u, 5u, 10u, 8u, 10u, 11u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 0u, 4u, 5u, 11u, 0u, 5u, 10u, 11u, 11u, 3u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 1u, 9u, 8u, 4u, 10u, 8u, 10u, 11u, 10u, 4u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(10u, 11u, 4u, 10u, 4u, 5u, 11u, 3u, 4u, 9u, 4u, 1u, 3u, 1u, 4u, 0xffffffffu),
    array(2u, 5u, 1u, 2u, 8u, 5u, 2u, 11u, 8u, 4u, 5u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 4u, 11u, 0u, 11u, 3u, 4u, 5u, 11u, 2u, 11u, 1u, 5u, 1u, 11u, 0xffffffffu),
    array(0u, 2u, 5u, 0u, 5u, 9u, 2u, 11u, 5u, 4u, 5u, 8u, 11u, 8u, 5u, 0xffffffffu),
    array(9u, 4u, 5u, 2u, 11u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 5u, 10u, 3u, 5u, 2u, 3u, 4u, 5u, 3u, 8u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(5u, 10u, 2u, 5u, 2u, 4u, 4u, 2u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 10u, 2u, 3u, 5u, 10u, 3u, 8u, 5u, 4u, 5u, 8u, 0u, 1u, 9u, 0xffffffffu),
    array(5u, 10u, 2u, 5u, 2u, 4u, 1u, 9u, 2u, 9u, 4u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 4u, 5u, 8u, 5u, 3u, 3u, 5u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 4u, 5u, 1u, 0u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(8u, 4u, 5u, 8u, 5u, 3u, 9u, 0u, 5u, 0u, 3u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 4u, 5u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 11u, 7u, 4u, 9u, 11u, 9u, 10u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 8u, 3u, 4u, 9u, 7u, 9u, 11u, 7u, 9u, 10u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 10u, 11u, 1u, 11u, 4u, 1u, 4u, 0u, 7u, 4u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 1u, 4u, 3u, 4u, 8u, 1u, 10u, 4u, 7u, 4u, 11u, 10u, 11u, 4u, 0xffffffffu),
    array(4u, 11u, 7u, 9u, 11u, 4u, 9u, 2u, 11u, 9u, 1u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 7u, 4u, 9u, 11u, 7u, 9u, 1u, 11u, 2u, 11u, 1u, 0u, 8u, 3u, 0xffffffffu),
    array(11u, 7u, 4u, 11u, 4u, 2u, 2u, 4u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(11u, 7u, 4u, 11u, 4u, 2u, 8u, 3u, 4u, 3u, 2u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 9u, 10u, 2u, 7u, 9u, 2u, 3u, 7u, 7u, 4u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 10u, 7u, 9u, 7u, 4u, 10u, 2u, 7u, 8u, 7u, 0u, 2u, 0u, 7u, 0xffffffffu),
    array(3u, 7u, 10u, 3u, 10u, 2u, 7u, 4u, 10u, 1u, 10u, 0u, 4u, 0u, 10u, 0xffffffffu),
    array(1u, 10u, 2u, 8u, 7u, 4u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 9u, 1u, 4u, 1u, 7u, 7u, 1u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 9u, 1u, 4u, 1u, 7u, 0u, 8u, 1u, 8u, 7u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 0u, 3u, 7u, 4u, 3u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(4u, 8u, 7u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 10u, 8u, 10u, 11u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 0u, 9u, 3u, 9u, 11u, 11u, 9u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 1u, 10u, 0u, 10u, 8u, 8u, 10u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 1u, 10u, 11u, 3u, 10u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 2u, 11u, 1u, 11u, 9u, 9u, 11u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 0u, 9u, 3u, 9u, 11u, 1u, 2u, 9u, 2u, 11u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 2u, 11u, 8u, 0u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(3u, 2u, 11u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 3u, 8u, 2u, 8u, 10u, 10u, 8u, 9u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(9u, 10u, 2u, 0u, 9u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(2u, 3u, 8u, 2u, 8u, 10u, 0u, 1u, 8u, 1u, 10u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 10u, 2u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(1u, 3u, 8u, 9u, 1u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 9u, 1u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0u, 3u, 8u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
    array(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu)
);

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> hash_offsets: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> edge_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> index_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;
@group(0) @binding(6) var<uniform> params: McParams;

fn cell_index(position: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(
        i32(floor(position.x / CELL_SIZE)),
        i32(floor(position.y / CELL_SIZE)),
        i32(floor(position.z / CELL_SIZE)),
    );
}

fn hash_position(cell_id: vec3<i32>) -> u32 {
    let x = u32(cell_id.x);
    let y = u32(cell_id.y);
    let z = u32(cell_id.z);
    let kappa = 9u * (x % 3u) + 3u * (y % 3u) + (z % 3u);
    let beta = ((17u * x) ^ (31u * y) ^ (61u * z)) % HASHTABLE_SIZE;
    return 27u * beta + kappa;
}

fn neighbor_hashes(pos: vec3<f32>) -> array<u32, 27> {
    let cur = cell_index(pos);
    var neighbors: array<u32, 27>;
    var i = 0u;
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                neighbors[i] = hash_position(cur + vec3<i32>(dx, dy, dz));
                i++;
            }
        }
    }
    return neighbors;
}

fn kernel_weight(pos_i: vec3<f32>, pos_j: vec3<f32>) -> f32 {
    let q = distance(pos_j, pos_i) / CELL_SIZE;
    if 0.0 <= q && q <= 0.5 {
        return SIGMA * (6.0 * pow(q, 3.0) - 6.0 * pow(q, 2.0) + 1.0);
    } else if 0.5 < q && q <= 1.0 {
        return SIGMA * 2.0 * pow(1.0 - q, 3.0);
    }
    return 0.0;
}

fn kernel_grad(pos_i: vec3<f32>, pos_j: vec3<f32>) -> vec3<f32> {
    let r = pos_i - pos_j;
    let r_norm = length(r);
    if r_norm < 1e-8 {
        return vec3<f32>(0.0);
    }
    let q = r_norm / CELL_SIZE;
    var dw_dq = 0.0;
    if q <= 0.5 {
        dw_dq = SIGMA * (18.0 * q * q - 12.0 * q);
    } else if q <= 1.0 {
        let t = 1.0 - q;
        dw_dq = SIGMA * -6.0 * t * t;
    } else {
        return vec3<f32>(0.0);
    }
    return (dw_dq / (CELL_SIZE * r_norm)) * r;
}

fn evaluate_density(pos: vec3<f32>) -> f32 {
    let neighbors = neighbor_hashes(pos);
    var density = 0.0;
    for (var i: u32 = 0u; i < 27u; i++) {
        let offset = hash_offsets[neighbors[i]];
        for (var j: u32 = offset.x; j < offset.y; j++) {
            if particles[j].boundary == 0u {
                continue;
            }
            if distance(particles[j].position, pos) <= CELL_SIZE {
                density += kernel_weight(pos, particles[j].position) * particles[j].mass;
            }
        }
    }
    return density;
}

fn evaluate_density_grad(pos: vec3<f32>) -> vec3<f32> {
    let neighbors = neighbor_hashes(pos);
    var g = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 27u; i++) {
        let offset = hash_offsets[neighbors[i]];
        for (var j: u32 = offset.x; j < offset.y; j++) {
            if particles[j].boundary == 0u {
                continue;
            }
            if distance(particles[j].position, pos) <= CELL_SIZE {
                g += kernel_grad(pos, particles[j].position) * particles[j].mass;
            }
        }
    }
    return g;
}

fn cell_corner(grid_x: u32, grid_y: u32, grid_z: u32) -> vec3<f32> {
    return BOTTOM_LEFT + vec3<f32>(
        f32(grid_x) * MARCHING_CUBES_CELL_SIZE,
        f32(grid_y) * MARCHING_CUBES_CELL_SIZE,
        f32(grid_z) * MARCHING_CUBES_CELL_SIZE,
    );
}

fn edge_cell_index(gx: u32, gy: u32, gz: u32) -> u32 {
    return gx + gy * EDGE_N + gz * EDGE_N * EDGE_N;
}

fn mc_edge_slot(edge_cell: u32, mc_edge: u32) -> u32 {
    switch mc_edge {
        case 0u: { return 3u * edge_cell; }
        case 3u: { return 3u * edge_cell + 1u; }
        case 8u: { return 3u * edge_cell + 2u; }
        case 1u: { return 3u * (edge_cell + 1u) + 1u; }
        case 9u: { return 3u * (edge_cell + 1u) + 2u; }
        case 2u: { return 3u * (edge_cell + EDGE_N); }
        case 11u: { return 3u * (edge_cell + EDGE_N) + 2u; }
        case 10u: { return 3u * (edge_cell + EDGE_N + 1u) + 2u; }
        case 4u: { return 3u * (edge_cell + EDGE_N * EDGE_N); }
        case 7u: { return 3u * (edge_cell + EDGE_N * EDGE_N) + 1u; }
        case 5u: { return 3u * (edge_cell + EDGE_N * EDGE_N + 1u) + 1u; }
        case 6u: { return 3u * (edge_cell + EDGE_N * EDGE_N + EDGE_N); }
        default: { return 0u; }
    }
}

fn voxel_cubeindex(gx: u32, gy: u32, gz: u32) -> u32 {
    let c = cell_corner(gx, gy, gz);
    let dx = vec3<f32>(MARCHING_CUBES_CELL_SIZE, 0.0, 0.0);
    let dy = vec3<f32>(0.0, MARCHING_CUBES_CELL_SIZE, 0.0);
    let dz = vec3<f32>(0.0, 0.0, MARCHING_CUBES_CELL_SIZE);
    var cubeindex = 0u;
    if evaluate_density(c) > params.iso_level { cubeindex |= 1u; }
    if evaluate_density(c + dx) > params.iso_level { cubeindex |= 2u; }
    if evaluate_density(c + dx + dy) > params.iso_level { cubeindex |= 4u; }
    if evaluate_density(c + dy) > params.iso_level { cubeindex |= 8u; }
    if evaluate_density(c + dz) > params.iso_level { cubeindex |= 16u; }
    if evaluate_density(c + dx + dz) > params.iso_level { cubeindex |= 32u; }
    if evaluate_density(c + dx + dy + dz) > params.iso_level { cubeindex |= 64u; }
    if evaluate_density(c + dy + dz) > params.iso_level { cubeindex |= 128u; }
    return cubeindex;
}

fn tri_index_count(cubeindex: u32) -> u32 {
    if EDGE_TABLE[cubeindex] == 0u {
        return 0u;
    }
    var n = 0u;
    let edges = TRI_TABLE[cubeindex];
    for (var j = 0u; j < 16u; j++) {
        if edges[j] == 0xffffffffu {
            break;
        }
        n++;
    }
    return n;
}

@compute
@workgroup_size(256)
fn active_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let num_edge_cells = EDGE_N * EDGE_N * EDGE_N;
    if i >= num_edge_cells {
        return;
    }

    let grid_x = i % EDGE_N;
    let grid_y = (i / EDGE_N) % EDGE_N;
    let grid_z = i / (EDGE_N * EDGE_N);
    let start = cell_corner(grid_x, grid_y, grid_z);
    let edge_x_end = start + vec3<f32>(MARCHING_CUBES_CELL_SIZE, 0.0, 0.0);
    let edge_y_end = start + vec3<f32>(0.0, MARCHING_CUBES_CELL_SIZE, 0.0);
    let edge_z_end = start + vec3<f32>(0.0, 0.0, MARCHING_CUBES_CELL_SIZE);

    let density_current = evaluate_density(start);
    let density_x_end = evaluate_density(edge_x_end);
    let density_y_end = evaluate_density(edge_y_end);
    let density_z_end = evaluate_density(edge_z_end);

    edge_mask[3u * i] = 0u;
    edge_mask[3u * i + 1u] = 0u;
    edge_mask[3u * i + 2u] = 0u;
    if (density_current - params.iso_level) * (density_x_end - params.iso_level) < 0.0 {
        edge_mask[3u * i] = 1u;
    }
    if (density_current - params.iso_level) * (density_y_end - params.iso_level) < 0.0 {
        edge_mask[3u * i + 1u] = 1u;
    }
    if (density_current - params.iso_level) * (density_z_end - params.iso_level) < 0.0 {
        edge_mask[3u * i + 2u] = 1u;
    }
}

@compute
@workgroup_size(256)
fn add_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let num_edges = 3u * EDGE_N * EDGE_N * EDGE_N;
    if i >= num_edges {
        return;
    }
    if edge_mask[i] == edge_mask[i + 1u] {
        return;
    }

    let cell = i / 3u;
    let dir = i % 3u;
    let grid_x = cell % EDGE_N;
    let grid_y = (cell / EDGE_N) % EDGE_N;
    let grid_z = cell / (EDGE_N * EDGE_N);
    let start = cell_corner(grid_x, grid_y, grid_z);

    var edge_end = start;
    if dir == 0u {
        edge_end = start + vec3<f32>(MARCHING_CUBES_CELL_SIZE, 0.0, 0.0);
    } else if dir == 1u {
        edge_end = start + vec3<f32>(0.0, MARCHING_CUBES_CELL_SIZE, 0.0);
    } else {
        edge_end = start + vec3<f32>(0.0, 0.0, MARCHING_CUBES_CELL_SIZE);
    }

    let density_current = evaluate_density(start);
    let density_end = evaluate_density(edge_end);
    let denom = density_end - density_current;
    var t = 0.5;
    if abs(denom) > 1e-8 {
        t = clamp((params.iso_level - density_current) / denom, 0.0, 1.0);
    }
    let xhat = start + t * (edge_end - start);

    var n = -evaluate_density_grad(xhat);
    let nlen = length(n);
    if nlen > 1e-8 {
        n = n / nlen;
    } else {
        n = normalize(edge_end - start);
    }

    vertices[edge_mask[i]] = Vertex(xhat, n);
}

@compute
@workgroup_size(256)
fn count_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let num_voxels = N * N * N;
    if i >= num_voxels {
        return;
    }
    let gx = i % N;
    let gy = (i / N) % N;
    let gz = i / (N * N);
    index_counts[i] = tri_index_count(voxel_cubeindex(gx, gy, gz));
}

@compute
@workgroup_size(256)
fn write_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let num_voxels = N * N * N;
    if i >= num_voxels {
        return;
    }

    let gx = i % N;
    let gy = (i / N) % N;
    let gz = i / (N * N);
    let cubeindex = voxel_cubeindex(gx, gy, gz);
    let n = tri_index_count(cubeindex);
    if n == 0u {
        return;
    }

    let base = index_counts[i];
    let edge_cell = edge_cell_index(gx, gy, gz);
    let edges = TRI_TABLE[cubeindex];
    for (var j = 0u; j < n; j++) {
        indices[base + j] = edge_mask[mc_edge_slot(edge_cell, edges[j])];
    }
}
