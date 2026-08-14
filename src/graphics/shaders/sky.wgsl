struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    var out: VsOut;
    out.clip_position = vec4(pos[vi], 1.0, 1.0);
    out.uv = pos[vi] * 0.5 + vec2(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let zenith = vec3<f32>(0.62, 0.52, 0.42);
    let horizon = vec3<f32>(0.88, 0.82, 0.74);
    let ground = vec3<f32>(0.42, 0.38, 0.34);
    let t = saturate(in.uv.y);
    var color = mix(horizon, zenith, pow(t, 0.75));
    let ground_w = smoothstep(0.42, 0.0, in.uv.y);
    color = mix(color, ground, ground_w * 0.35);
    return vec4(color, 1.0);
}
