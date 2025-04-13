#define_import_path IMPORT_PATH

// NOTE: this depends on a preprocessor substituting the following macros:
//       - DIM: the matrix dimension (e.g. `2` for 2x2 matrices).
//       - MAT: the matrix type (e.g. `mat2x2<f32>` for a 2x2 matrix).
//       - IMPORT_PATH: the `define_import_path` path.
fn cholesky(x: MAT) -> MAT {
    var m = x;

    // PERF: consider unrolling the loops?
    for (var j = 0u; j < DIM; j++) {
        for (var k = 0u; k < j; k++) {
            let factor = -m[k][j];

            for (var l = j; l < DIM; l++) {
                m[j][l] += factor * m[k][l];
            }
        }

        let denom = sqrt(m[j][j]);
        m[j][j] = denom;

        for (var l = j + 1u; l < DIM; l++) {
            m[j][l] /= denom;
        }
    }

    return m;
}