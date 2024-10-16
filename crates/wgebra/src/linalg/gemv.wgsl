#import wgblas::shape as Shape

@group(0) @binding(0)
var<uniform> shape_out: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_m: Shape::Shape;
@group(0) @binding(2)
var<uniform> shape_v: Shape::Shape;
@group(0) @binding(3)
var<storage, read_write> out: array<f32>;
@group(0) @binding(4)
var<storage, read> m: array<f32>;
@group(0) @binding(5)
var<storage, read> v: array<f32>;

const WORKGROUP_SIZE: u32 = 64;

// TODO: needs a lot of optimizations.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv0(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if invocation_id.x < shape_m.nrows {
        // The first iteration overwrites any trash from initial value.
        let v_0 = v[Shape::iv(shape_m, 0u)];
        let i_out = Shape::iv(shape_out, invocation_id.x);

        out[i_out] = m[Shape::im(shape_m, invocation_id.x, 0u)] * v_0;

        for (var j = 1u; j < shape_m.ncols; j++) {
            out[i_out] += m[Shape::im(shape_m, invocation_id.x, j)] * v[Shape::iv(shape_v, j)];
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv1(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if invocation_id.x < shape_m.nrows {
        let i_out = Shape::iv(shape_out, invocation_id.x);

        for (var j = 0u; j < shape_m.ncols; j++) {
            out[i_out] += m[Shape::im(shape_m, invocation_id.x, j)] * v[Shape::iv(shape_v, j)];
        }
    }
}
