/// store a list of analyticsl sdfs supported.
/// all shapes' sdf will be constructed w.r.t origin;
/// it is up to you to translate points to these coordinates.
/// analytical forms from https://iquilezles.org/articles/distfunctions/
enum Sdf {
    Sphere(f32),
    Cylinder{
        radius: f32,
        start: [f32; 3],
        end: [f32; 3],
    },
    Difference(Box<Sdf>, Box<Sdf>),
}

impl Sdf::Sphere {
    pub fn sdf(&self) -> &'static str {
        format!(
            "fn sphere_sdf(p: vec3<f32>) -> f32 {
            let r = {};
            return length(p) - r;
        }",
            self.0
        )
    }
}

impl Sdf::Cylinder {
    pub fn sdf(&self) -> String {
        format!(
            "fn cylinder_sdf(p: vec3<f32>) -> f32 {{
                let a = vec3<f32>({start_x}, {start_y}, {start_z});
                let b = vec3<f32>({finish_x}, {finish_y}, {finish_z});
                let r = {radius};

                let ba = b - a;
                let pa = p - a;
                let baba = dot(ba, ba);
                let paba = dot(pa, ba);
                let x = length(pa * baba - ba * paba) - r * baba;
                let y = abs(paba - baba * 0.5) - baba * 0.5;
                let x2 = x * x;
                let y2 = y * y * baba;
                
                var d: f32;
                if max(x, y) < 0.0 {{
                    d = -min(x2, y2);
                }} else {{
                    let x_part = select(0.0, x2, x > 0.0);
                    let y_part = select(0.0, y2, y > 0.0);
                    d = x_part + y_part;
                }}
                
                return sign(d) * sqrt(abs(d)) / baba;
            }}",
            start_x = self.start[0], start_y = self.start[1], start_z = self.start[2],
            finish_x = self.end[0], finish_y = self.end[1], finish_z = self.end[2],
            radius = self.radius
        )
    }
}

impl Sdf::Difference {
    pub fn sdf(&self) -> String {
        format!(
            "fn difference_sdf(p: vec3<f32>) -> f32 {{
                let sdf1 = {sdf1};
                let sdf2 = {sdf2};
                return max(-sdf1, sdf2);
            }}",
            sdf1 = self.0.sdf_entrypoint(),
            sdf2 = self.1.sdf_entrypoint()
        )
    }
}