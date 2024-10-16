#import wgblas::shape as Shape;

@group(0) @binding(0)
var<uniform> shape_q: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_k: Shape::Shape;
@group(0) @binding(2)
var<uniform> shape: RoPEShape;
@group(0) @binding(3)
var<storage, read_write> in_out_q: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> in_out_k: array<vec2<f32>>;



struct RoPEShape {
    head_size: u32,
    kv_dim: u32,
    pos: u32,
}

struct Rotation2 {
    cos: f32,
    sin: f32,
}

fn rot2(angle: f32) -> Rotation2 {
    return Rotation2(cos(angle), sin(angle));
}

fn rotate2(r: Rotation2, v: vec2<f32>) -> vec2<f32> {
    return vec2(r.cos * v.x - r.sin * v.y, r.sin * v.x + r.cos * v.y);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    let head_dim = f32((i * 2) % shape.head_size);
    let theta = pow(10000.0, -head_dim / f32(shape.head_size));
    let m_theta = f32(shape.pos) * theta;
    let rot = rot2(m_theta);

    let iq = Shape::iv(shape_q, i * 2) / 2;
    let q_unrotated = in_out_q[iq];
    in_out_q[iq] = rotate2(rot, q_unrotated);

    if (i * 2 < shape.kv_dim) {
        let ik = Shape::iv(shape_k, i * 2) / 2;
        let k_unrotated = in_out_k[ik];
        in_out_k[ik] = rotate2(rot, k_unrotated);
    }
}
