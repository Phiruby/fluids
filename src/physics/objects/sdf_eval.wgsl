fn local_of(i: u32, world_p: vec3<f32>) -> vec3<f32> {
    return quat_rotate(quat_conj(rigid_bodies[i].orientation), world_p - rigid_bodies[i].position);
}

fn sd_body_local(i: u32, p: vec3<f32>) -> f32 {
    return sd_primitive(
        rigid_bodies[i].object_type,
        p,
        rigid_bodies[i].scale,
        rigid_bodies[i].params,
    );
}

fn sdf_local(i: u32, p: vec3<f32>) -> f32 {
    if rigid_bodies[i].object_type == SDF_DIFFERENCE {
        let a = u32(rigid_bodies[i].params.x);
        let b = u32(rigid_bodies[i].params.y);
        return max(sd_body_local(a, p), -sd_body_local(b, p));
    }
    return sd_body_local(i, p);
}

fn rigid_sdf(i: u32, world_p: vec3<f32>) -> f32 {
    var d = sdf_local(i, local_of(i, world_p));
    if rigid_bodies[i].invert != 0u {
        d = -d;
    }
    return d;
}

fn source_sdf(i: u32, world_p: vec3<f32>) -> f32 {
    let p = local_of(i, world_p);
    if rigid_bodies[i].object_type == SDF_DIFFERENCE {
        return sd_body_local(u32(rigid_bodies[i].params.y), p);
    }
    return sd_body_local(i, p);
}

fn source_radius(i: u32) -> f32 {
    if rigid_bodies[i].object_type == SDF_DIFFERENCE {
        return rigid_bodies[u32(rigid_bodies[i].params.y)].params.x;
    }
    return rigid_bodies[i].params.x;
}

fn body_extent(i: u32) -> vec3<f32> {
    if rigid_bodies[i].object_type == SDF_DIFFERENCE {
        return rigid_bodies[u32(rigid_bodies[i].params.x)].params.xyz;
    }
    return rigid_bodies[i].params.xyz;
}

fn rigid_sdf_normal(i: u32, world_p: vec3<f32>) -> vec3<f32> {
    let e = 1e-3;
    let n = vec3<f32>(
        rigid_sdf(i, world_p + vec3<f32>(e, 0.0, 0.0)) - rigid_sdf(i, world_p - vec3<f32>(e, 0.0, 0.0)),
        rigid_sdf(i, world_p + vec3<f32>(0.0, e, 0.0)) - rigid_sdf(i, world_p - vec3<f32>(0.0, e, 0.0)),
        rigid_sdf(i, world_p + vec3<f32>(0.0, 0.0, e)) - rigid_sdf(i, world_p - vec3<f32>(0.0, 0.0, e)),
    );
    let len = length(n);
    if len < 1e-8 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return n / len;
}

fn rigid_point_velocity(i: u32, world_p: vec3<f32>) -> vec3<f32> {
    return rigid_bodies[i].v_cm + cross(rigid_bodies[i].omega, world_p - rigid_bodies[i].position);
}

// chapter 12.1 https://alexey.stomakhin.com/research/siggraph2016_mpm.pdf
fn collide_rigid(i: u32, pos: vec3<f32>, v: vec3<f32>, node_mass: f32) -> CollideResult {
    var result: CollideResult;
    result.v = v;
    result.force = vec3<f32>(0.0);
    if rigid_bodies[i].operand != 0u {
        return result;
    }
    if rigid_bodies[i].mass <= 1e-8 {
        return result;
    }
    if rigid_sdf(i, pos) > 0.0 {
        return result;
    }
    let n = rigid_sdf_normal(i, pos);
    let v_body = rigid_point_velocity(i, pos);
    var v_rel = v - v_body;
    let vn = dot(v_rel, n);
    if vn >= 0.0 {
        return result;
    }
    let vt = v_rel - vn * n;
    let vt_len = length(vt);
    if vt_len <= -MU * vn {
        v_rel = vec3<f32>(0.0);
    } else {
        v_rel = vt + MU * vn * vt / vt_len;
    }
    let v_new = v_rel + v_body;
    result.v = v_new;
    result.force = -node_mass * (v_new - v) / DT;
    return result;
}
