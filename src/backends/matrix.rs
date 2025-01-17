/// A trait for matrix-like types that can be used with the JIT compiler.
///
/// This trait provides a common interface for different matrix implementations,
/// allowing them to be used interchangeably in JIT-compiled functions. It defines
/// core matrix operations needed for JIT compilation, including accessing raw data
/// and creating zero-initialized matrices.
///
/// # Examples
///
/// ```rust
/// use evalexpr_jit::prelude::Matrix;
/// use ndarray::Array2;
///
/// // Create a zero matrix
/// let mat: Array2<f64> = Matrix::zeros(2, 3);
/// assert_eq!(mat.dims(), (2, 3));
///
/// // Access elements as flat slice
/// let data = mat.flat_slice();
/// assert_eq!(data.len(), 6);
/// assert!(std::ptr::eq(data, mat.as_slice().unwrap()));
/// ```
pub trait Matrix {
    /// Returns a reference to the matrix's data as a flat slice.
    ///
    /// The data is stored in row-major order, meaning elements are arranged row by row.
    /// For a matrix with dimensions (m,n), element (i,j) is at index i*n + j.
    ///
    /// # Returns
    /// A slice containing the matrix elements in row-major order
    fn flat_slice(&self) -> &[f64];

    /// Returns a mutable reference to the matrix's data as a flat slice.
    ///
    /// The data is stored in row-major order, meaning elements are arranged row by row.
    /// For a matrix with dimensions (m,n), element (i,j) is at index i*n + j.
    ///
    /// # Returns
    /// A mutable slice containing the matrix elements in row-major order
    fn flat_mut_slice(&mut self) -> &mut [f64];

    /// Creates a new matrix of the specified dimensions filled with zeros.
    ///
    /// # Arguments
    /// * `rows` - Number of rows in the matrix
    /// * `cols` - Number of columns in the matrix
    ///
    /// # Returns
    /// A new matrix of the specified dimensions with all elements set to 0.0
    ///
    /// # Panics
    /// May panic if the dimensions are invalid (e.g., zero rows/columns) depending
    /// on the specific implementation
    fn zeros(rows: usize, cols: usize) -> Self;

    /// Returns the dimensions of the matrix as (rows, columns).
    ///
    /// # Returns
    /// A tuple containing:
    /// - The number of rows in the matrix
    /// - The number of columns in the matrix
    fn dims(&self) -> (usize, usize);
}

/// Implementation of Matrix trait for ndarray's Array2<f64>.
///
/// This implementation provides an interface between the Matrix trait and ndarray's
/// 2-dimensional array type. It handles the conversion between ndarray's internal
/// representation and the flat slice representation required by the JIT compiler.
///
/// # Examples
///
/// ```rust
/// use evalexpr_jit::prelude::Matrix;
/// use ndarray::Array2;
///
/// let mut mat = Array2::<f64>::zeros((2, 2));
/// let slice = mat.flat_mut_slice();
/// slice[0] = 1.0;
/// assert!(std::ptr::eq(slice, mat.as_slice().unwrap()));
/// ```
#[cfg(feature = "ndarray")]
impl Matrix for ndarray::Array2<f64> {
    fn flat_slice(&self) -> &[f64] {
        self.as_slice().unwrap()
    }

    fn flat_mut_slice(&mut self) -> &mut [f64] {
        self.as_slice_mut().unwrap()
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        ndarray::Array2::zeros((rows, cols))
    }

    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

/// Implementation of Matrix trait for nalgebra's DMatrix<f64>.
///
/// This implementation provides an interface between the Matrix trait and nalgebra's
/// dynamic matrix type. It handles the conversion between nalgebra's internal
/// representation and the flat slice representation required by the JIT compiler.
///
/// # Examples
///
/// ```rust
/// use evalexpr_jit::prelude::Matrix;
/// use nalgebra::DMatrix;
///
/// let mut mat = DMatrix::<f64>::zeros(2, 2);
/// let slice = mat.flat_mut_slice();
/// slice[0] = 1.0;
/// assert!(std::ptr::eq(slice, mat.as_slice()));
/// ```
#[cfg(feature = "nalgebra")]
impl Matrix for nalgebra::DMatrix<f64> {
    fn flat_slice(&self) -> &[f64] {
        self.as_slice()
    }

    fn flat_mut_slice(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        nalgebra::DMatrix::zeros(rows, cols)
    }

    fn dims(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use ndarray::Array2;

    #[test]
    fn test_ndarray_matrix_operations() {
        // Test zeros and dims
        let mat = Array2::<f64>::zeros((2, 3));
        assert_eq!(mat.dims(), (2, 3));
        assert_eq!(mat.flat_slice().len(), 6);
        assert!(mat.flat_slice().iter().all(|&x| x == 0.0));

        // Test mutable operations
        let mut mat = Array2::<f64>::zeros((2, 2));
        {
            let slice = mat.flat_mut_slice();
            slice[0] = 1.0;
            slice[3] = 4.0;
        }
        assert_eq!(mat.flat_slice(), &[1.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn test_nalgebra_matrix_operations() {
        // Test zeros and dims
        let mat = DMatrix::<f64>::zeros(2, 3);
        assert_eq!(mat.dims(), (2, 3));
        assert_eq!(mat.flat_slice().len(), 6);
        assert!(mat.flat_slice().iter().all(|&x| x == 0.0));

        // Test mutable operations
        let mut mat = DMatrix::<f64>::zeros(2, 2);
        {
            let slice = mat.flat_mut_slice();
            slice[0] = 1.0;
            slice[3] = 4.0;
        }
        assert_eq!(mat.flat_slice(), &[1.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn test_matrix_layout() {
        // Test row-major layout for both implementations
        let mut ndarray_mat = Array2::<f64>::zeros((2, 3));
        let mut nalgebra_mat = DMatrix::<f64>::zeros(2, 3);

        // Fill with sequential values
        for i in 0..6 {
            ndarray_mat.flat_mut_slice()[i] = i as f64;
            nalgebra_mat.flat_mut_slice()[i] = i as f64;
        }

        // Check row-major layout
        assert_eq!(ndarray_mat.flat_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(nalgebra_mat.flat_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
