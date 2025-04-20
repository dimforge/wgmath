#define_import_path wgebra::qr4

/// The QR decomposition of a 4x4 matrix.
///
/// See the [nalgebra](https://nalgebra.rs/docs/user_guide/decompositions_and_lapack#qr) documentation
/// for details on the QR decomposition.
struct QR {
    /// The QR decomposition’s 4x4 unitary matrix.
    q: mat4x4<f32>,
    /// The QR decomposition’s 4x4 upper-triangular matrix.
    r: mat4x4<f32>
}

/// Computes the QR decomposition of a 4x4 matrix.
fn qr(x: mat4x4<f32>) -> QR {
    const DIM = 4;
    var m = x;
    var diag = vec4<f32>();

    // Apply householder reflections.
    for (var i = 0; i < 4; i++) {
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
    var q = mat4x4<f32>(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0),
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
    let r = mat4x4(
        vec4(abs(diag.x), 0.0, 0.0, 0.0),
        vec4(m[1][0], abs(diag.y), 0.0, 0.0),
        vec4(m[2][0], m[2][1], abs(diag.z), 0.0),
        vec4(m[3][0], m[3][1], m[3][2], abs(diag.w))
    );

    return QR(q, r);
}
