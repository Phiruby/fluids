struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>,
    density: f32,
    rest_density: f32,
    mass: f32,
    cell_hash: u32,
    // 1 = fluid, 0 = boundary
    boundary: u32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> fluid_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> fluid_particles: array<Particle>;

@compute
@workgroup_size(256)
fn build_fluid_mask(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  if (index >= arrayLength(&particles)) {
    return;
  }

  fluid_mask[index] = u32(particles[index].boundary == 1u);
}

// Assuming fluid_mask has been run through PrefixSum, so it contains
// the offsets
@compute
@workgroup_size(256)
fn fluid_compaction(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  if (index >= arrayLength(&particles)) {
    return;
  }

  let particle = particles[index];
  if (particle.boundary == 1u) {
    let offset = fluid_mask[index];
    fluid_particles[offset] = particle;
  }
}
