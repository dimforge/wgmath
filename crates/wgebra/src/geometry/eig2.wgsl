#define_import_path wgebra::eig2

// The SVD of a 2x2 matrix.
struct SymmetricEigen {
    eigenvectors: mat2x2<f32>,
    eigenvalues: vec2<f32>,
};

// Computes the SVD of a 2x2 matrix.
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
