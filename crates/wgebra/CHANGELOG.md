## Unreleased

### Modified

- Replaced all lazy kernel invocation queueing by dispatches directly.

### Added

- Added implementation of matrix decompositions (LU, QR, Cholesky, Eigendecomposition)
  for `mat2x2<f32>`, `mat3x3<f32>`, and `mat4x4<f32>` on the GPU.

## v0.2.0

### Modified

- Update to `wgcore` v0.2.0.
