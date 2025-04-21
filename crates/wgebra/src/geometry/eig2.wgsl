#define_import_path wgebra::eig2

// The eigendecomposition of a symmetric 2x2 matrix.
///
/// See the [nalgebra](https://nalgebra.rs/docs/user_guide/decompositions_and_lapack/#eigendecomposition-of-a-hermitian-matrix)
/// documentation for details on the eigendecomposition.
struct SymmetricEigen {
    /// Eigenvectors of the matrix.
    eigenvectors: mat2x2<f32>,
    /// Eigenvalues of the matrix.
    eigenvalues: vec2<f32>,
};

// Computes the eigendecomposition of a symmetric 2x2 matrix.
fn symmetric_eigen(m: mat2x2<f32>) -> SymmetricEigen {
    let a = m.x.x;
    let c = m.x.y;
    let b = m.y.y;

    if c == 0.0 {
        return SymmetricEigen(
            mat2x2(vec2(1.0, 0.0), vec2(0.0, 1.0)),
            vec2(a, b)
        );
    }

    let ab = a - b;
    let sigma = sqrt(4.0 * c * c + ab * ab);
    let eigenvalues = vec2(
        (a + b + sigma) / 2.0,
        (a + b - sigma) / 2.0
    );
    let eigv1 = vec2((a - b + sigma) / (2.0 * c), 1.0);
    let eigv2 = vec2((a - b - sigma) / (2.0 * c), 1.0);

    let eigenvectors = mat2x2(eigv1 / length(eigv1), eigv2 / length(eigv2));

    return SymmetricEigen(eigenvectors, eigenvalues);
}

fn eigenvalues(m: mat2x2<f32>) -> vec2<f32> {
    let a = m.x.x;
    let c = m.x.y;
    let b = m.y.y;

    if c == 0.0 {
        return vec2(a, b);
    }

    let ab = a - b;
    let sigma = sqrt(4.0 * c * c + ab * ab);
    return vec2(
        (a + b + sigma) / 2.0,
        (a + b - sigma) / 2.0
    );
}
