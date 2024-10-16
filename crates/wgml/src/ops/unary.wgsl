// This was mostly ported from ggml-cuda/unary.cu
// (MIT license).

#import wgblas::shape as Shape;

@group(0) @binding(0)
var<uniform> shape_dst: Shape::Shape;
@group(0) @binding(1)
var<uniform> shape_src: Shape::Shape;
@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(3)
var<storage, read_write> src: array<f32>;
@group(0) @binding(4)
var<uniform> args: vec4<f32>;

const GELU_COEF_A: f32 = 0.044715f;
const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876f;
const GELU_QUICK_COEF: f32 = -1.702f;

fn abs_f32(x: f32) -> f32 {
    return abs(x);
}

fn sgn_f32(x: f32) -> f32 {
    return sign(x);
}

fn neg_f32(x: f32) -> f32 {
    return -x;
}

fn step_f32(x: f32) -> f32 {
    if x > 0.0 {
        return 1.0;
    } else {
        return 0.0;
    }
}

fn elu_f32(x: f32) -> f32 {
    if x > 0.0 {
        return x;
    } else {
        return exp(x) - 1.0;
    }
}

fn gelu_f32(x: f32) -> f32 {
    return 0.5f * x * (1.0f + tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

fn gelu_quick_f32(x: f32) -> f32 {
    return x * (1.0f / (1.0f + exp(GELU_QUICK_COEF * x)));
}

fn silu_f32(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn tanh_f32(x: f32) -> f32 {
    return tanh(x);
}

fn relu_f32(x: f32) -> f32 {
    return max(x, 0.0);
}

fn sigmoid_f32(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn hard_sigmoid_f32(x: f32) -> f32 {
    return min(1.0, max(0.0, (x + 3.0) / 6.0));
}

fn hard_swish_f32(x: f32) -> f32 {
    return x * min(1.0, max(0.0, (x + 3.0) / 6.0));
}

fn sqr_f32(x: f32) -> f32 {
    return x * x;
}

fn sqrt_f32(x: f32) -> f32 {
    return sqrt(x);
}

fn log_f32(x: f32) -> f32 {
    return log(x);
}

fn placeholder(x: f32) -> f32 {
    return x;
}

// Unary ops with extra arguments passed through the args uniform.
fn leaky_relu_f32(x: f32) -> f32 {
    return max(x, 0.0) + min(x, 0.0) * args.x;
}

fn clamp_f32(x: f32) -> f32 {
    return min(max(args.x, x), args.y);
}

fn scale_f32(x: f32) -> f32 {
    return x * args.x;
}

fn add_scalar_f32(x: f32) -> f32 {
    return x + args.x;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if (invocation_id.x < shape_src.nrows) {
        let isrc = Shape::iv(shape_src, invocation_id.x);
        let idst = Shape::iv(shape_dst, invocation_id.x);
        dst[idst] = placeholder(src[isrc]);
    }
}
