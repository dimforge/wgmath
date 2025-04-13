#define_import_path wgebra::eig4

#import wgebra::rot2 as Rot
#import wgebra::eig2 as Eig2
#import wgebra::min_max as MinMax

// The SVD of a 4x4 matrix.
struct SymmetricEigen {
    eigenvectors: mat4x4<f32>,
    eigenvalues: vec4<f32>,
}

struct Tridiag {
    m: mat4x4<f32>,
    off_diag: vec3<f32>
}

// Computes the SVD of a 4x4 matrix.
fn symmetric_eigen(x: mat4x4<f32>) -> SymmetricEigen {
    const DIM: u32 = 4;
    const EPS: f32 = 1.1920929e-7;

    var m = x;
    let m_amax = MinMax::amax4x4(x);

    if m_amax != 0.0 {
        m.x /= m_amax;
        m.y /= m_amax;
        m.z /= m_amax;
        m.w /= m_amax;
    }

    let tridiag = tridiagonalize(m);
    var diag = vec4(tridiag.m.x.x, tridiag.m.y.y, tridiag.m.z.z, tridiag.m.w.w);
    // NOTE: SymmetricTridiagonal::unpack takes the modulus off_diag.
    //       So, here we take the absolute value.
    var off_diag = abs(tridiag.off_diag);

    // Initialize q from tridiag.m (see householder::assample_q in nalgebra).
    var q = mat4x4<f32>(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0),
    );
    for (var i = DIM - 2; i >= 0; i--) {
        // axis := tridiag.m[i + 1.., i]
        // res_rows := q[i + 1.., i..]
        let sgn = sign(tridiag.off_diag[i]);

        // refl.reflect_with_sign(&mut res_rows, signs[i].clone().signum());
        for (var c = i; c < DIM; c++) {
            let m_two = -2.0 * sgn;
            var factor = 0.0;
            for (var r = i + 1; r < DIM; r++) {
                factor += tridiag.m[i][r] * q[c][r];
            }
            for (var r = i + 1; r < DIM; r++) {
                q[c][r] = m_two * factor * tridiag.m[i][r] + q[c][r] * sgn;
            }
        }

        if i == 0 {
            break;
        }
    }

    // Decompose the tridiagonal matrix.
    let start_end = delimit_subproblem(diag, &off_diag, DIM - 1, EPS);
    var start = start_end.x;
    var end = start_end.y;
    var niter = 0;

    while (end != start) {
        let subdim = end - start + 1u;

        if subdim > 2u {
            let m = end - 1u;
            let n = end;

            let shift = wilkinson_shift(diag[m], diag[n], off_diag[m]);
            var v = vec2(diag[start] - shift, off_diag[start]);

            for (var i = start; i < n; i++) {
                let j = i + 1u;
                let rot = Rot::cancel_y(v);
                if Rot::is_valid(rot) {
                    if i > start {
                        // Not the first iteration.
                        off_diag[i - 1] = sign(v.x) * length(v);
                    }

                    let mii = diag[i];
                    let mjj = diag[j];
                    let mij = off_diag[i];

                    let cc = rot.cos_sin.x * rot.cos_sin.x;
                    let ss = rot.cos_sin.y * rot.cos_sin.y;
                    let cs = rot.cos_sin.x * rot.cos_sin.y;

                    let b = cs * 2.0 * mij;

                    diag[i] = (cc * mii + ss * mjj) - b;
                    diag[j] = (ss * mii + cc * mjj) + b;
                    off_diag[i] = cs * (mii - mjj) + mij * (cc - ss);

                    if i != n - 1 {
                        v.x = off_diag[i];
                        v.y = -rot.cos_sin.y * off_diag[i + 1];
                        off_diag[i + 1] *= rot.cos_sin.x;
                    }

                    let inv_rot = Rot::inv(rot);
                    Rot::rotate_rows4(inv_rot, &q, i);
                } else {
                    break;
                }
            }

            if abs(off_diag[m]) <= EPS * (abs(diag[m]) + abs(diag[n])) {
                end -= 1u;
            }
        } else if subdim == 2 {
            let m = mat2x2(
                vec2(diag[start], off_diag[start]),
                vec2(off_diag[start], diag[start + 1]),
            );
            let eigvals = Eig2::eigenvalues(m);
            let basis = vec2(
                eigvals.x - diag[start + 1],
                off_diag[start],
            );

            diag[start] = eigvals[0];
            diag[start + 1] = eigvals[1];

            let basis_len = length(basis);
            if basis_len > EPS {
                let rot = Rot::Rot2(basis * (sign(basis.x) / basis_len));
                Rot::rotate_rows4(rot, &q, start);
            }

            end -= 1u;
        }

        // Re-delimit the subproblem in case some decoupling occurred.
        let start_end = delimit_subproblem(diag, &off_diag, end, EPS);
        start = start_end[0];
        end = start_end[1];

        niter++;
    }

    diag *= m_amax;

    return SymmetricEigen(q, diag);
}

fn delimit_subproblem(
    diag: vec4<f32>,
    off_diag: ptr<function, vec3<f32>>,
    end: u32,
    eps: f32,
) -> vec2<u32>
{
    var n = end;

    while n > 0u {
        let m = n - 1u;

        if abs((*off_diag)[m]) > eps * (abs(diag[n]) + abs(diag[m])) {
            break;
        }

        n -= 1u;
    }

    if n == 0u {
        return vec2(0u, 0u);
    }

    var new_start = n - 1u;
    while new_start > 0u {
        let m = new_start - 1u;

        if (*off_diag)[m] == 0.0 || abs((*off_diag)[m]) <= eps * (abs(diag[new_start]) + abs(diag[m])) {
            (*off_diag)[m] = 0.0;
            break;
        }

        new_start -= 1u;
    }

    return vec2(new_start, n);
}

fn wilkinson_shift(tmm: f32, tnn: f32, tmn: f32) -> f32 {
    let sq_tmn = tmn * tmn;
    if sq_tmn != 0.0 {
        // We have the guarantee that the denominator won't be zero.
        let d = (tmm - tnn) * 0.5;
        return tnn - sq_tmn / (d + sign(d) * sqrt(d * d + sq_tmn));
    } else {
        return tnn;
    }
}

fn tridiagonalize(x: mat4x4<f32>) -> Tridiag {
    const DIM = 4;
    var m = x;
    var off_diagonal = vec3<f32>();

    for (var i = 0; i < DIM - 1; i++) {
        // Ported from househodler::reflection_axis_mut
        // The axis (or `column`) is `m[i + 1.., i]`.
        var axis_sq_norm = 0.0;
        for (var r = i + 1; r < DIM; r++) {
            axis_sq_norm += m[i][r] * m[i][r];
        }

        let axis_norm = sqrt(axis_sq_norm);
        let modulus = abs(m[i][i + 1]);
        let sgn = sign(m[i][i + 1]);
        var signed_norm = sgn * axis_norm;
        let factor = (axis_sq_norm + modulus * axis_norm) * 2.0;
        m[i][i + 1] += signed_norm;

        if factor != 0.0 {
            let factor_sqrt = sqrt(factor);
            var norm = 0.0;
            for (var r = i + 1; r < DIM; r++) {
                m[i][r] /= factor_sqrt;
                norm += m[i][r] * m[i][r];
            }

            norm = sqrt(norm);

            // Renormalization (see nalgebra’s doc of `houselohder::reflection_axis_mut`).
            for (var r = i + 1; r < DIM; r++) {
                m[i][r] /= norm;
            }

            off_diagonal[i] = -signed_norm;
        } else {
            off_diagonal[i] = signed_norm;
        }

        if factor != 0.0 {
            var p = vec4<f32>();

            // p.hegemv(2.0, &m, &axis, T::zero());
            for (var c = i + 1; c < DIM; c++) {
                for (var r = i + 1; r < DIM; r++) {
                    p[r] += 2.0 * m[c][r] * m[i][c];
                }
            }

            // let dot = axis.dotc(&p);
            var dot = 0.0;
            for (var r = i + 1; r < DIM; r++) {
                dot += m[i][r] * p[r];
            }

            // m.hegerc(-1.0, &p, &axis, 1.0);
            // m.hegerc(-1.0, &axis, &p, 1.0);
            // m.hegerc(dot * 2.0, &axis, &axis, 1.0);
            // where `axis := m[i + 1.., i]`
            for (var c = i + 1; c < DIM; c++) {
                for (var r = i + 1; r < DIM; r++) {
                    m[c][r] += 2.0 * dot * m[i][r] * m[i][c]
                            - p[r] * m[i][c]
                            - m[i][r] * p[c];
                }
            }
        }
    }

    return Tridiag(m, off_diagonal);
}
