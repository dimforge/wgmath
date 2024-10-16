#import wgblas::shape as Shape;

@group(0) @binding(0)
var<uniform> in_shape: Shape::Shape;
@group(0) @binding(1)
var<uniform> out_shape: Shape::Shape;
@group(0) @binding(2)
var<storage, read_write> in_vec: array<f32>;
@group(0) @binding(3)
var<storage, read_write> out_vec: array<f32>;

const WORKGROUP_SIZE: u32 = 128;
const NUDGE_FACTOR: f32 = 1.0e-5;

var<workgroup> workspace: array<f32, WORKGROUP_SIZE>;
var<workgroup> the_mean: f32;
var<workgroup> scale: f32;

fn reduce_sum(thread_id: u32, stride: u32) {
    workgroupBarrier();

    if thread_id < stride {
        workspace[thread_id] += workspace[thread_id + stride];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let thread_id = invocation_id.x;

    // Compute the MEAN
    let data_len = in_shape.nrows;
    workspace[thread_id] = 0.0;
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let val_i = in_vec[Shape::iv(in_shape, i)];
        workspace[thread_id] += val_i;
    }

    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    if (thread_id == 0) {
        the_mean = workspace[0] / f32(data_len);
    }

    workgroupBarrier();

    // Compute the SQUARED NORM
    workspace[thread_id] = 0.0;
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let val_i = in_vec[Shape::iv(in_shape, i)] - the_mean;
        workspace[thread_id] += val_i * val_i;
    }

    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    if (thread_id == 0) {
        let variance = workspace[0] / f32(data_len);
        scale = 1.0 / sqrt(variance + NUDGE_FACTOR);
    }

    workgroupBarrier();

    // Apply the scale.
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let ii = Shape::iv(in_shape, i);
        let iout = Shape::iv(out_shape, i);
        out_vec[iout] = (in_vec[ii] - the_mean) * scale;
    }
}
