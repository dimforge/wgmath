#define_import_path wgebra::svd2
#import wgebra::trig as Trig

// The SVD of a 2x2 matrix.
struct Svd {
    U: mat2x2<f32>,
    S: vec2<f32>,
    Vt: mat2x2<f32>,
};

// Computes the SVD of a 2x2 matrix.
fn svd(m: mat2x2<f32>) -> Svd {
    let e = (m[0].x + m[1].y) * 0.5;
    let f = (m[0].x - m[1].y) * 0.5;
    let g = (m[0].y + m[1].x) * 0.5;
    let h = (m[0].y - m[1].x) * 0.5;
    let q = sqrt(e * e + h * h);
    let r = sqrt(f * f + g * g);

    // Note that the singular values are always sorted because sx >= sy
    // because q >= 0 and r >= 0.
    let sx = q + r;
    let sy = q - r;
    let sy_sign = select(1.0, -1.0, sy < 0.0);
    let singular_values = vec2(sx, sy * sy_sign);

    let a1 = Trig::stable_atan2(g, f);
    let a2 = Trig::stable_atan2(h, e);
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

// Rebuilds the matrix this svd is the decomposition of.
fn recompose(svd: Svd) -> mat2x2<f32> {
    let U_S = mat2x2(svd.U[0] * svd.S.x, svd.U[1] * svd.S.y);
    return U_S * svd.Vt;
}
