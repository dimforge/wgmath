#import wgblas::shape as Shape;

@group(0) @binding(0)
var<uniform> shape_v: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_w: Shape::Shape;
@group(0) @binding(2)
var<uniform> shape_out: Shape::Shape;
@group(0) @binding(3)
var<storage, read> v: array<f32>;
@group(0) @binding(4)
var<storage, read> w: array<f32>;
@group(0) @binding(5)
var<storage, read_write> out: array<f32>; 


const WORKGROUP_SIZE: u32 = 128;
const NUDGE_FACTOR: f32 = 1.0e-5;

/*
 * Magnitude.
 */
var<workgroup> workspace: array<f32, WORKGROUP_SIZE>;

fn reduce_sum(thread_id: u32, stride: u32) {
    if thread_id < stride {
        workspace[thread_id] += workspace[thread_id + stride];
    }
    workgroupBarrier();
}

fn magnitude_squared(thread_id: u32) -> f32 {
    workspace[thread_id] = 0.0;

    for (var i = thread_id; i < shape_v.nrows; i += WORKGROUP_SIZE) {
        let val_i = v[Shape::iv(shape_v, i)];
        workspace[thread_id] += val_i * val_i;
    }

    workgroupBarrier();

    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    return workspace[0];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let magnitude_sq = magnitude_squared(invocation_id.x);

    let len = shape_v.nrows;
    let rms = 1.0 / sqrt((magnitude_sq / f32(len)) + NUDGE_FACTOR);

    for (var i = invocation_id.x; i < len; i += WORKGROUP_SIZE) {
        out[Shape::iv(shape_out, i)] = (v[Shape::iv(shape_v, i)] * rms) * w[Shape::iv(shape_w, i)];
    }
}
