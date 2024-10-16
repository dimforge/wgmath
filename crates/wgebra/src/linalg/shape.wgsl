// Module comment
// And a second line

#define_import_path wgblas::shape

// The shape of a (column-major) matrix.
struct Shape {
    // The number of rows in the matrix.
    nrows: u32,
    // The number of columns in the matrix.
    ncols: u32,
    // The number of elements separating two columns of the matrix.
    // Note that the row stride is always assumed to be 1.
    col_stride: u32,
    // Index of the first element of the matrix.
    offset: u32,
}

// Index of the `i-th` element of a vector.
fn iv(view: Shape, i: u32) -> u32 {
    return view.offset + i;
}

// Index of the element at row `i` and column `j` of a matrix.
fn im(view: Shape, i: u32, j: u32) -> u32 {
    return view.offset + i + j * view.col_stride;
}