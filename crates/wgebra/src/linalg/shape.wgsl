#define_import_path wgblas::shape

// The shape of a matrix.
//
// If the `ROW_MAJOR` constant is defined then this represents the shape of a row-major matrix.
// Otherwise, it represents the shape of a column-major matrix.
struct Shape {
    // The number of rows in each matrix of the tensor.
    nrows: u32,
    // The number of columns in each matrix of the tensor.
    ncols: u32,
    // The number of matrices in the tensor.
    nmats: u32,
    // The number of elements separating two elements along the non-major dimension
    // of each matrix of the tensor.
    //
    // If the matrix is row-major (`ROW_MAJOR` is defined) then this is the number of elements in memory
    // between two consecutive elements from the same column.
    //
    // If the matrix is column-major (`ROW_MAJOR` is undefined) then this is the number of elements in memory
    // between two consecutive elements from the same row.
    //
    // Note that the stride along the other dimension is always assumed to be 1.
    stride: u32,
    // The number of elements separating two elements along the "matrix" direction (i.e. the third
    // tensor direction). This is independent from the matrix ordering (row-major vs. column-major).
    stride_mat: u32,
    // Index of the first element of the tensor.
    offset: u32,
}

// Index of the `i-th` element of a vector.
fn iv(view: Shape, i: u32) -> u32 {
    return view.offset + i;
}

fn div_ceil4(a: u32) -> u32 {
    return (a + 3u) / 4u;
}

// Index of the element at row `i`, column `j` of the matrix `t` in this tensor.
fn it(view: Shape, i: u32, j: u32, t: u32) -> u32 {
    return t * view.stride_mat + im(view, i, j);
}

#ifdef ROW_MAJOR
// Index of the element at row `i` and column `j` of a row-major matrix.
fn im(view: Shape, i: u32, j: u32) -> u32 {
    return view.offset + i * view.stride + j;
}

fn with_vec4_elts(shape: Shape) -> Shape {
    return Shape(shape.nrows, div_ceil4(shape.ncols), shape.nmats, div_ceil4(shape.stride), div_ceil4(shape.stride_mat), shape.offset / 4u);
}
#else
// Index of the element at row `i` and column `j` of a column-major matrix.
fn im(view: Shape, i: u32, j: u32) -> u32 {
    return view.offset + i + j * view.stride;
}

fn with_vec4_elts(shape: Shape) -> Shape {
    return Shape(div_ceil4(shape.nrows), shape.ncols, shape.nmats, div_ceil4(shape.stride), div_ceil4(shape.stride_mat), shape.offset / 4u);
}
#endif
