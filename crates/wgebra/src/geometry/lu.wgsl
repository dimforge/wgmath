#define_import_path IMPORT_PATH

// NOTE: this shader depends on a preprocessor substituting the following macros:
//       - NROWS: the matrix’s number of rows.
//       - NCOLS: the matrix’s number of columns.
//       - PERM: the vector type for the permutation sequence.
//               Must be a u32 vector of dimension min(NROWS, NCOLS).
//       - MAT: the matrix type (e.g. `mat2x2<f32>` for a 2x2 matrix).
//       - IMPORT_PATH: the `define_import_path` path.

/// Structure describing a permutation sequence applied by the LU decomposition.
struct Permutations {
    /// First permutation indices (row `ia[i]` is permuted with row`ib[i]`].
    ia: PERM,
    /// Second permutation indices (row `ia[i]` is permuted with row`ib[i]`].
    ib: PERM,
    /// The number of permutations in `self`. Only the first `len` elements of
    /// [`Self::ia`] and [`Self::ib`] need to be taken into account.
    len: u32
}

/// GPU representation of a matrix LU decomposition (with partial pivoting).
///
/// See the [nalgebra](https://nalgebra.rs/docs/user_guide/decompositions_and_lapack#lu-with-partial-or-full-pivoting) documentation
/// for details on the LU decomposition.
struct LU {
    /// The LU decomposition where both lower and upper-triangular matrices are stored
    /// in the same matrix. In particular the diagonal full of `1` of the lower-triangular
    /// matrix isn’t stored explicitly.
    lu: MAT,
    /// The row permutations applied during the decomposition.
    p: Permutations
}

/// Computse the LU decomposition of the matrix.
fn lu(x: MAT) -> LU {
    let min_nrows_ncols = min(NROWS, NCOLS);
    var p = Permutations();
    var lu = x;

    for (var i = 0u; i < min_nrows_ncols; i++) {
        // Find the pivot index (maximum absolute value on the
        // column i, on rows [i, NROWS].
        var piv = i;
        var piv_val = abs(lu[i][i]);
        for (var r = i + 1u; r < NROWS; r++) {
            let abs_val = abs(lu[i][r]);
            if abs_val > piv_val {
                piv = r;
                piv_val = abs_val;
            }
        }

        if piv_val == 0.0 {
            // No non-zero entries on this column.
            continue;
        }

        // NOTE: read the diagonal element, not `piv_val` since
        //       the latter involve an absolute value.
        let diag = lu[i][piv];

        if piv != i {
            p.ia[p.len] = i;
            p.ib[p.len] = piv;
            p.len++;

            for (var k = 0u; k < i; k++) {
                let mki = lu[k][i];
                lu[k][i] = lu[k][piv];
                lu[k][piv] = mki;
            }

            gauss_step_swap(&lu, diag, i, piv);
        } else {
            gauss_step(&lu, diag, i);
        }
    }

    return LU(lu, p);
}

/// Executes one step of gaussian elimination on the i-th row and column of `m`. The diagonal
/// element `m[(i, i)]` is provided as argument.
fn gauss_step(m: ptr<function, MAT>, diag: f32, i: u32)
{
    let inv_diag = 1.0 / diag;

    for (var r = i + 1u; r < NROWS; r++) {
        (*m)[i][r] *= inv_diag;
    }

    for (var c = i + 1u; c < NCOLS; c++) {
        let pivot = (*m)[c][i];

        for (var r = i + 1u; r < NROWS; r++) {
            (*m)[c][r] -= pivot * (*m)[i][r];
        }
    }
}

/// Swaps the rows `i` with the row `piv` and executes one step of gaussian elimination on the i-th
/// row and column of `m`. The diagonal element `m[(i, i)]` is provided as argument.
fn gauss_step_swap(
    m: ptr<function, MAT>,
    diag: f32,
    i: u32,
    piv: u32,
)
{
    let inv_diag = 1.0 / diag;

    let mii = (*m)[i][i];
    (*m)[i][i] = (*m)[i][piv];
    (*m)[i][piv] = mii;

    for (var r = i + 1u; r < NROWS; r++) {
        (*m)[i][r] *= inv_diag;
    }

    for (var c = i + 1u; c < NCOLS; c++) {
        let mci = (*m)[c][i];
        (*m)[c][i] = (*m)[c][piv];
        (*m)[c][piv] = mci;

        let pivot = (*m)[c][i];

        for (var r = i + 1u; r < NROWS; r++) {
            (*m)[c][r] -= pivot * (*m)[i][r];
        }
    }
}
