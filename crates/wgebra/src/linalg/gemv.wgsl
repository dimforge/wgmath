#import wgblas::shape as Shape

@group(0) @binding(0)
var<uniform> shape_out: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_m: Shape::Shape;
@group(0) @binding(2)
var<uniform> shape_v: Shape::Shape;
@group(0) @binding(3)
var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read> m: array<vec4<f32>>;
@group(0) @binding(5)
var<storage, read> v: array<vec4<f32>>;

// NOTE: gemv_tr_fast is quite a bit (15%) faster with a workgroup size of 8.
const WORKGROUP_SIZE: u32 = 32;

var<workgroup> sketch: array<vec4<f32>, WORKGROUP_SIZE>;

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_fast(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let shape_m = Shape::with_vec4_elts(shape_m);
    let shape_v = Shape::with_vec4_elts(shape_v);
    let shape_out = Shape::with_vec4_elts(shape_out);

    var sum = vec4(0.0);

    for (var j = 0u; j < shape_m.ncols; j += 4u * WORKGROUP_SIZE) {
        var ia = Shape::it(shape_m, workgroup_id.x, j + local_id.x * 4u, workgroup_id.z);
        let ib = ia + shape_m.stride;
        let ic = ib + shape_m.stride;
        let id = ic + shape_m.stride;
        let submat = mat4x4(m[ia], m[ib], m[ic], m[id]);

        let iv = Shape::it(shape_v, j / 4u + local_id.x, workgroup_id.y, workgroup_id.z);
        sum += submat * v[iv];
    }

    sketch[local_id.x] = sum;

    workgroupBarrier();

//    reduce_sum(local_id.x, 32u);
    reduce_sum(local_id.x, 16u);
    reduce_sum(local_id.x, 8u);
    reduce_sum(local_id.x, 4u);
    reduce_sum(local_id.x, 2u);
    reduce_sum(local_id.x, 1u);

    if local_id.x == 0u {
        let i_out = Shape::it(shape_out, workgroup_id.x, workgroup_id.y, workgroup_id.z);
        out[i_out] = sketch[0];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let shape_m = Shape::with_vec4_elts(shape_m);
    let shape_v = Shape::with_vec4_elts(shape_v);
    let shape_out = Shape::with_vec4_elts(shape_out);

    if invocation_id.x < shape_m.nrows {
        var sum = vec4(0.0);

        for (var j = 0u; j < shape_m.ncols; j += 4u) {
            var ia = Shape::it(shape_m, invocation_id.x, j, invocation_id.z);
            let ib = ia + shape_m.stride;
            let ic = ib + shape_m.stride;
            let id = ic + shape_m.stride;
            let submat = mat4x4(m[ia], m[ib], m[ic], m[id]);

            let iv = Shape::it(shape_v, j / 4u, invocation_id.y, invocation_id.z);
            sum += submat * v[iv];
        }

        let i_out = Shape::it(shape_out, invocation_id.x, invocation_id.y, invocation_id.z);
        out[i_out] = sum;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_tr(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let shape_m = Shape::with_vec4_elts(shape_m);
    let shape_v = Shape::with_vec4_elts(shape_v);
    let shape_out = Shape::with_vec4_elts(shape_out);

    if invocation_id.x < (shape_m.ncols + 3u) / 4 {
        var sum = vec4(0.0);

        for (var j = 0u; j < shape_m.nrows; j++) {
            var ia = Shape::it(shape_m, j, invocation_id.x * 4u, invocation_id.z);
            let ib = ia + shape_m.stride;
            let ic = ib + shape_m.stride;
            let id = ic + shape_m.stride;
            let submat = mat4x4(m[ia], m[ib], m[ic], m[id]);

            let iv = Shape::it(shape_v, j, invocation_id.y, invocation_id.z);
            sum += transpose(submat) * v[iv];
        }

        let i_out = Shape::it(shape_out, invocation_id.x, invocation_id.y, invocation_id.z);
        out[i_out] = sum;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_tr_fast(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let shape_m = Shape::with_vec4_elts(shape_m);
    let shape_v = Shape::with_vec4_elts(shape_v);
    let shape_out = Shape::with_vec4_elts(shape_out);

    var sum = vec4(0.0);

    for (var j = 0u; j < shape_m.nrows; j += WORKGROUP_SIZE) {
        var ia = Shape::it(shape_m, j + local_id.x, workgroup_id.x * 4u, workgroup_id.z);
        let ib = ia + shape_m.stride;
        let ic = ib + shape_m.stride;
        let id = ic + shape_m.stride;
        let submat = mat4x4(m[ia], m[ib], m[ic], m[id]);

        let iv = Shape::it(shape_v, j + local_id.x, workgroup_id.y, workgroup_id.z);
        sum += transpose(submat) * v[iv];
    }

    sketch[local_id.x] = sum;

    workgroupBarrier();

//    reduce_sum(local_id.x, 64u);
//    reduce_sum(local_id.x, 32u);
    reduce_sum(local_id.x, 16u);
    reduce_sum(local_id.x, 8u);
    reduce_sum(local_id.x, 4u);
    reduce_sum(local_id.x, 2u);
    reduce_sum(local_id.x, 1u);

    if local_id.x == 0u {
        let i_out = Shape::it(shape_out, workgroup_id.x, workgroup_id.y, workgroup_id.z);
        out[i_out] = sketch[0];
    }
}