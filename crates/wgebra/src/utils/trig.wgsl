#define_import_path wgebra::trig

/// The value of pi.
const PI: f32 = 3.14159265358979323846264338327950288;

/// A numerically stable implementation of tanh.
///
/// Metal’s implementation of tanh returns NaN for large values.
/// This function is more numerically stable and should be used as a
/// drop-in replacement.
// Inspired from https://github.com/apache/tvm/pull/16438 (Apache 2.0 license).
fn stable_tanh(x: f32) -> f32 {
  let exp_neg2x = exp(-2.0 * x);
  let exp_pos2x = exp(2.0 * x);
  let tanh_pos = (1.0 - exp_neg2x) / (1.0 + exp_neg2x);
  let tanh_neg = (exp_pos2x - 1.0) / (exp_pos2x + 1.0);
  return select(tanh_neg, tanh_pos, x >= 0.0);
}

/// In some platforms, atan2 has unusable edge cases, e.g., returning NaN when y = 0 and x = 0.
///
/// This is for example the case in Metal/MSL: https://github.com/gfx-rs/wgpu/issues/4319
/// So we need to implement it ourselves to ensure svd always returns reasonable results on some
/// edge cases like the identity.
fn stable_atan2(y: f32, x: f32) -> f32 {
    let ang = atan(y / x);
    if x > 0.0 {
        return ang;
    }
    if x < 0.0 && y > 0.0 {
        return ang + PI;
    }
    if x < 0.0 && y < 0.0 {
        return ang - PI;
    }

    // Force the other ubounded cases to 0.
    return 0.0;
}