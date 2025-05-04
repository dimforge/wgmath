#import wgblas::shape as Shape

@group(0) @binding(0)
var<uniform> shape_out: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_m1: Shape::Shape;
@group(0) @binding(2)
var<uniform> shape_m2: Shape::Shape;
@group(0) @binding(3)
var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read> m1: array<vec4<f32>>;
@group(0) @binding(5)
var<storage, read> m2: array<vec4<f32>>;

const WORKGROUP_SIZE: u32 = 64;

var<workgroup> sketch: array<mat4x4<f32>, WORKGROUP_SIZE>;


fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemm_fast(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let shape_m1 = Shape::with_vec4_elts(shape_m1);
    let shape_m2 = Shape::with_vec4_elts(shape_m2);
    let shape_out = Shape::with_vec4_elts(shape_out);

    for (var k = 0u; k < shape_m2.ncols; k += 4u) {
        var sum = mat4x4f();

        for (var j = 0u; j < shape_m1.ncols; j += 4u * WORKGROUP_SIZE) {
            var ia1 = Shape::it(shape_m1, workgroup_id.x, j + local_id.x * 4u, workgroup_id.y);
            let ib1 = ia1 + shape_m1.stride;
            let ic1 = ib1 + shape_m1.stride;
            let id1 = ic1 + shape_m1.stride;
            let submat1 = mat4x4(m1[ia1], m1[ib1], m1[ic1], m1[id1]);

            let ia2 = Shape::it(shape_m2, j / 4u + local_id.x, k, workgroup_id.y);
            let ib2 = ia2 + shape_m2.stride;
            let ic2 = ib2 + shape_m2.stride;
            let id2 = ic2 + shape_m2.stride;
            let submat2 = mat4x4(m2[ia2], m2[ib2], m2[ic2], m2[id2]);

            sum += submat1 * submat2;
        }

        sketch[local_id.x] = sum;

        workgroupBarrier();

        reduce_sum(local_id.x, 32u);
        reduce_sum(local_id.x, 16u);
        reduce_sum(local_id.x, 8u);
        reduce_sum(local_id.x, 4u);
        reduce_sum(local_id.x, 2u);
        reduce_sum(local_id.x, 1u);

        if local_id.x == 0u {
            let i_out = Shape::it(shape_out, workgroup_id.x, k, workgroup_id.y);
            let mat = sketch[0];
            out[i_out] = mat[0];
            out[i_out + shape_out.stride] = mat[1];
            out[i_out + shape_out.stride * 2] = mat[2];
            out[i_out + shape_out.stride * 3] = mat[3];
        }

        workgroupBarrier();
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemm(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let shape_m1 = Shape::with_vec4_elts(shape_m1);
    let shape_m2 = Shape::with_vec4_elts(shape_m2);
    let shape_out = Shape::with_vec4_elts(shape_out);

    if invocation_id.x < shape_m1.nrows {
        for (var k = 0u; k < shape_m2.ncols; k += 4u) {
            var sum = mat4x4f();

            for (var j = 0u; j < shape_m1.ncols; j += 4u) {
                let ia1 = Shape::it(shape_m1, invocation_id.x, j, invocation_id.y);
                let ib1 = ia1 + shape_m1.stride;
                let ic1 = ib1 + shape_m1.stride;
                let id1 = ic1 + shape_m1.stride;
                let submat1 = mat4x4(m1[ia1], m1[ib1], m1[ic1], m1[id1]);

                let ia2 = Shape::it(shape_m2, j / 4u, k, invocation_id.y);
                let ib2 = ia2 + shape_m2.stride;
                let ic2 = ib2 + shape_m2.stride;
                let id2 = ic2 + shape_m2.stride;
                let submat2 = mat4x4(m2[ia2], m2[ib2], m2[ic2], m2[id2]);

                sum += submat1 * submat2;
            }

            let i_out = Shape::it(shape_out, invocation_id.x, k, invocation_id.y);
            out[i_out] = sum[0];
            out[i_out + shape_out.stride] = sum[1];
            out[i_out + shape_out.stride * 2] = sum[2];
            out[i_out + shape_out.stride * 3] = sum[3];
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemm_tr(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let shape_m1 = Shape::with_vec4_elts(shape_m1);
    let shape_m2 = Shape::with_vec4_elts(shape_m2);
    let shape_out = Shape::with_vec4_elts(shape_out);

    if invocation_id.x < (shape_m1.ncols + 3u) / 4 {
        for (var k = 0u; k < shape_m2.ncols; k += 4u) {
            var sum = mat4x4f();

            for (var j = 0u; j < shape_m1.nrows; j++) {
                var ia1 = Shape::it(shape_m1, j, invocation_id.x * 4u, invocation_id.y);
                let ib1 = ia1 + shape_m1.stride;
                let ic1 = ib1 + shape_m1.stride;
                let id1 = ic1 + shape_m1.stride;
                let submat1 = mat4x4(m1[ia1], m1[ib1], m1[ic1], m1[id1]);

                let ia2 = Shape::it(shape_m2, j, k, invocation_id.y);
                let ib2 = ia2 + shape_m2.stride;
                let ic2 = ib2 + shape_m2.stride;
                let id2 = ic2 + shape_m2.stride;
                let submat2 = mat4x4(m2[ia2], m2[ib2], m2[ic2], m2[id2]);

                sum += transpose(submat1) * submat2;
            }

            let i_out = Shape::it(shape_out, invocation_id.x, k, invocation_id.y);
            out[i_out] = sum[0];
            out[i_out + shape_out.stride] = sum[1];
            out[i_out + shape_out.stride * 2] = sum[2];
            out[i_out + shape_out.stride * 3] = sum[3];
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemm_tr_fast(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let shape_m1 = Shape::with_vec4_elts(shape_m1);
    let shape_m2 = Shape::with_vec4_elts(shape_m2);
    let shape_out = Shape::with_vec4_elts(shape_out);

    for (var k = 0u; k < shape_m2.ncols; k += 4u) {
        var sum = mat4x4f();

        for (var j = 0u; j < shape_m1.nrows; j += WORKGROUP_SIZE) {
            var ia1 = Shape::it(shape_m1, j + local_id.x, workgroup_id.x * 4u, workgroup_id.y);
            let ib1 = ia1 + shape_m1.stride;
            let ic1 = ib1 + shape_m1.stride;
            let id1 = ic1 + shape_m1.stride;
            let submat1 = mat4x4(m1[ia1], m1[ib1], m1[ic1], m1[id1]);

            let ia2 = Shape::it(shape_m2, j + local_id.x, k, workgroup_id.y);
            let ib2 = ia2 + shape_m2.stride;
            let ic2 = ib2 + shape_m2.stride;
            let id2 = ic2 + shape_m2.stride;
            let submat2 = mat4x4(m2[ia2], m2[ib2], m2[ic2], m2[id2]);

            sum += transpose(submat1) * submat2;
        }

        sketch[local_id.x] = sum;

        workgroupBarrier();

        reduce_sum(local_id.x, 32u);
        reduce_sum(local_id.x, 16u);
        reduce_sum(local_id.x, 8u);
        reduce_sum(local_id.x, 4u);
        reduce_sum(local_id.x, 2u);
        reduce_sum(local_id.x, 1u);

        if local_id.x == 0u {
            let i_out = Shape::it(shape_out, workgroup_id.x, k, workgroup_id.y);
            let mat = sketch[0];
            out[i_out] = mat[0];
            out[i_out + shape_out.stride] = mat[1];
            out[i_out + shape_out.stride * 2] = mat[2];
            out[i_out + shape_out.stride * 3] = mat[3];
        }

        workgroupBarrier();
    }
}