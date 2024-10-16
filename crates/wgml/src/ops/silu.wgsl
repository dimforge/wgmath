#import wgblas::shape as Shape;

@group(0) @binding(0)
var<uniform> shape_a: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_b: Shape::Shape;
@group(0) @binding(2)
var<storage, read_write> in_out_a: array<f32>;
@group(0) @binding(3)
var<storage, read> in_b: array<f32>;

// SwiGLU non-linearity.
fn swish(x: f32, beta: f32) -> f32 {
    // This is the swiglu function from https://youtu.be/Mn_9W1nCFLo?si=LT6puSAfzgpP6ydz&t=3973
    return x / (1.0 + exp(-beta * x));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if (invocation_id.x < shape_a.nrows) {
        let ia = Shape::iv(shape_a, invocation_id.x);
        let ib = Shape::iv(shape_b, invocation_id.x);
        let lhs = in_out_a[ia];
        let rhs = in_b[ib];
        in_out_a[ia] = rhs * swish(lhs, 1.0);
    }
}
