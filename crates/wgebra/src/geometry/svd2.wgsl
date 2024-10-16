#define_import_path wgebra::svd2

// The SVD of a 2x2 matrix.
struct Svd {
    U: mat2x2<f32>,
    S: vec2<f32>,
    Vt: mat2x2<f32>,
};

// Computes the SVD of a 3x3 matrix.
fn svd(m: mat2x2<f32>) -> Svd {
    let e = (m.x.x + m.y.y) * 0.5;
    let f = (m.x.x - m.y.y) * 0.5;
    let g = (m.x.y + m.y.x) * 0.5;
    let h = (m.x.y - m.y.x) * 0.5;
    let q = sqrt(e * e + h * h);
    let r = sqrt(f * f + g * g);

    // Note that the singular values are always sorted because sx >= sy
    // because q >= 0 and r >= 0.
    let sx = q + r;
    let sy = q - r;
    let sy_sign = select(1.0, -1.0, sy < 0.0);
    let singular_values = vec2(sx, sy * sy_sign);

    let a1 = atan2_not_nan(g, f);
    let a2 = atan2_not_nan(h, e);
    let theta = (a2 - a1) * 0.5;
    let phi = (a2 + a1) * 0.5;
    let st = sin(theta);
    let ct = cos(theta);
    let sp = sin(phi);
    let cp = cos(phi);

    let u = mat2x2(vec2(cp, sp), vec2(-sp, cp));
    let v_t = mat2x2(vec2(ct, st * sy_sign), vec2(-st, ct * sy_sign));

    return Svd(u, singular_values, v_t);
}

/// THe value of pi.
const PI: f32 = 3.14159265358979323846264338327950288;

/// In some platforms, atan2 has unusable edge cases, e.g., returning NaN when y = 0 and x = 0.
///
/// This is for example the case in Metal/MSL: https://github.com/gfx-rs/wgpu/issues/4319
/// So we need to implement it ourselves to ensure svd always returns reasonable results on some
/// edge cases like the identity.
fn atan2_not_nan(y: f32, x: f32) -> f32 {
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

// Rebuilds the matrix this svd is the decomposition of.
fn recompose(svd: Svd) -> mat2x2<f32> {
    let U_S = mat2x2(svd.U.x * svd.S.x, svd.U.y * svd.S.y);
    return U_S * svd.Vt;
}
