#include <../objects/sdf.wgsl>

struct Particle {
    deformation_gradient: mat3x3<f32>,
    B: mat3x3<f32>,
    position: vec3<f32>,
    velocity: vec3<f32>,
    rest_density: f32,
    mass: f32,
}

const SPACING: f32 = 0.01;

struct EmitCounter {
    n: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> rigid_bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> emit_count: EmitCounter;
@group(0) @binding(3) var<storage, read> sources: array<InflowSource>;

#include <../objects/sdf_eval.wgsl>

// under recommendation of chatbot:
// without multiplying by angle, particles fall straight down and collide with boundary.
// this was causing uninteresting behavior
fn spawn_inlet(slot: u32, i: u32, flow: vec3<f32>) -> vec3<f32> {
    let axis = flow / length(flow);
    let h = body_extent(i).y;
    let r = source_radius(i) * 0.75;
    // multiple by angle (= golden angle here) so each slot gets different dir
    let a = f32(slot) * 2.399963;
    let rad = r * sqrt(fract(f32(slot) * 0.618034));
    let tmp = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
    let t1 = normalize(cross(axis, tmp));
    let t2 = cross(axis, t1);
    return -axis * (h - SPACING) + t1 * rad * cos(a) + t2 * rad * sin(a);
}

@compute
@workgroup_size(256)
fn emit_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&particles) { return; }
    var p = particles[idx];
    // i know this is hacky; here's what its trying to do:
    // reservoir fluid particles (i.e particles yet to be pumped) are stored at (1000, 1000, 1000).
    // but rigid bodies waiting to be added to the scene are stored at (1000, -1000, 1000).
    if p.position.x <= 50.0 || p.position.y <= 50.0 { return; }

    let nsrc = arrayLength(&sources);
    // TODO: this won't work for multiple sources
    for (var s = 0u; s < nsrc; s++) {
        let i = sources[s].body;
        let flow = sources[s].flow;
        let fl = length(flow);
        if fl < 1e-8 { continue; }
        if rigid_bodies[i].operand != 0u { continue; }
        if rigid_bodies[i].position.x > 50.0 { continue; }
        let r = source_radius(i);
        let flux = 3.14159265 * r * r * fl * DT / (SPACING * SPACING * SPACING);
        let max_emit = max(u32(flux + 0.5), 1u);
        let n = atomicAdd(&emit_count.n, 1u);
        if n >= max_emit { return; }
        let local = spawn_inlet(n, i, flow);
        p.position = quat_rotate(rigid_bodies[i].orientation, local) + rigid_bodies[i].position;
        p.velocity = quat_rotate(rigid_bodies[i].orientation, flow);
        p.B = mat3x3<f32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        particles[idx] = p;
        return;
    }
}
