#import wgblas::shape as Shape

@group(0) @binding(0)
var<uniform> shape_a: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_b: Shape::Shape;
@group(0) @binding(2)
var<storage, read_write> a: array<f32>;
@group(0) @binding(3)
var<storage, read> b: array<f32>;

const WORKGROUP_SIZE: u32 = 64;

fn add_f32(a: f32, b: f32) -> f32 {
    return a + b;
}

fn sub_f32(a: f32, b: f32) -> f32 {
    return a - b;
}

fn mul_f32(a: f32, b: f32) -> f32 {
    return a * b;
}

fn div_f32(a: f32, b: f32) -> f32 {
    return a / b;
}

fn placeholder(a: f32, b: f32) -> f32 {
    return a + b;
}

// TODO: will the read of a be optimized-out by the shader compiler
//       or do we need to write a dedicated kernel for this?
fn copy_f32(a: f32, b: f32) -> f32 {
    return b;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if invocation_id.x < shape_a.nrows {
        let ia = Shape::iv(shape_a, invocation_id.x);
        let ib = Shape::iv(shape_b, invocation_id.x);
        a[ia] = placeholder(a[ia], b[ib]);
    }
}
