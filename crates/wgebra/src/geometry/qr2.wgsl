#define_import_path wgebra::qr2

// The QR decomposition of a 2x2 matrix.
struct QR {
    q: mat2x2<f32>,
    r: mat2x2<f32>
}

// Computes the QR decomposition of a 2x2 matrix.
fn qr(x: mat2x2<f32>) -> QR {
    const DIM = 2;
    var m = x;
    var diag = vec2<f32>();

    // Apply householder reflections.
    for (var i = 0; i < 2; i++) {
        // Ported from househodler::reflection_axis_mut
        // The axis (or `column`) is `m[i.., i]`.
        var axis_sq_norm = 0.0;
        for (var r = i; r < DIM; r++) {
            axis_sq_norm += m[i][r] * m[i][r];
        }

        let axis_norm = sqrt(axis_sq_norm);
        let modulus = abs(m[i][i]);
        let sgn = sign(m[i][i]);
        var signed_norm = sgn * axis_norm;
        let factor = (axis_sq_norm + modulus * axis_norm) * 2.0;
        m[i][i] += signed_norm;

        if factor != 0.0 {
            let factor_sqrt = sqrt(factor);
            var norm = 0.0;
            for (var r = i; r < DIM; r++) {
                m[i][r] /= factor_sqrt;
                norm += m[i][r] * m[i][r];
            }

            norm = sqrt(norm);

            // Renormalization (see nalgebra’s doc of `householder::reflection_axis_mut`).
            for (var r = i; r < DIM; r++) {
                m[i][r] /= norm;
            }

            diag[i] = -signed_norm;
        } else {
            diag[i] = signed_norm;
        }

        // Apply the reflection.
        if factor != 0.0 {
            // refl.reflect_with_sign(&mut res_rows, signs[i].clone().signum());
            let sgn = sign(diag[i]);
            for (var c = i; c < DIM; c++) {
                let m_two = -2.0 * sgn;
                var factor = 0.0;
                for (var r = i; r < DIM; r++) {
                    factor += m[i][r] * m[c][r];
                }
                for (var r = i; r < DIM; r++) {
                    m[c][r] = m_two * factor * m[i][r] + m[c][r] * sgn;
                }
            }
        }
    }

    // Initialize q from m (see QR::q() in nalgebra).
    var q = mat2x2<f32>(
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
    );
    for (var i = DIM - 1; i >= 0; i--) {
        // axis := m[i.., i]
        // res_rows := q[i.., i..]
        let sgn = sign(diag[i]);

        // refl.reflect_with_sign(&mut res_rows, signs[i].clone().signum());
        for (var c = i; c < DIM; c++) {
            let m_two = -2.0 * sgn;
            var factor = 0.0;
            for (var r = i; r < DIM; r++) {
                factor += m[i][r] * q[c][r];
            }
            for (var r = i; r < DIM; r++) {
                q[c][r] = m_two * factor * m[i][r] + q[c][r] * sgn;
            }
        }

        if i == 0 {
            break;
        }
    }

    // Fill the lower triangle of `m` and set its diagonal to get `r`.
    let r = mat2x2(
        vec2(abs(diag.x), 0.0),
        vec2(m[1][0], abs(diag.y)),
    );

    return QR(q, r);
}
