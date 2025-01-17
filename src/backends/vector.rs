/// A trait for vector-like types that can be used with the JIT compiler.
///
/// This trait provides a common interface for different vector implementations,
/// allowing them to be used interchangeably in JIT-compiled functions. It defines
/// core vector operations needed for JIT compilation, including accessing raw data
/// and creating zero-initialized vectors.
///
/// # Examples
///
/// ```rust
/// use evalexpr_jit::prelude::Vector;
///
/// // Create a zero vector
/// let vec: Vec<f64> = Vector::zeros(5);
/// assert_eq!(vec.len(), 5);
///
/// // Access elements
/// let mut vec = vec![1.0, 2.0, 3.0];
/// let slice = vec.as_slice();
/// assert_eq!(slice[0], 1.0);
/// ```
pub trait Vector {
    /// Returns a reference to the vector's data as a slice.
    fn as_slice(&self) -> &[f64];

    /// Returns a mutable reference to the vector's data as a slice.
    fn as_mut_slice(&mut self) -> &mut [f64];

    /// Creates a new vector of the specified length filled with zeros.
    ///
    /// # Arguments
    /// * `len` - The length of the vector to create
    fn zeros(len: usize) -> Self;

    /// Returns the length of the vector.
    fn len(&self) -> usize;

    /// Checks if the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Implementation of Vector trait for standard Vec<f64>.
///
/// This implementation provides an interface between the Vector trait and Rust's
/// standard vector type. Vec<f64> already provides slice access, making this
/// implementation straightforward.
///
/// # Examples
///
/// ```rust
/// use evalexpr_jit::prelude::Vector;
///
/// let mut vec = Vec::<f64>::zeros(3);
/// let slice = vec.as_mut_slice();
/// slice[0] = 1.0;
/// assert_eq!(vec[0], 1.0);
/// ```
impl Vector for Vec<f64> {
    fn as_slice(&self) -> &[f64] {
        self
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        self
    }

    fn zeros(len: usize) -> Self {
        vec![0.0; len]
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Implementation of Vector trait for ndarray's Array1<f64>.
///
/// This implementation provides an interface between the Vector trait and ndarray's
/// 1-dimensional array type. It handles the conversion between ndarray's internal
/// representation and the slice representation required by the JIT compiler.
///
/// # Examples
///
/// ```rust
/// use evalexpr_jit::prelude::Vector;
/// use ndarray::Array1;
///
/// let mut vec = Array1::<f64>::zeros(3);
/// let slice = vec.as_mut_slice();
/// slice[0] = 1.0;
/// assert!(std::ptr::eq(slice, vec.as_slice().unwrap()));
/// ```
#[cfg(feature = "ndarray")]
impl Vector for ndarray::Array1<f64> {
    fn as_slice(&self) -> &[f64] {
        self.as_slice().unwrap()
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.as_slice_mut().unwrap()
    }

    fn zeros(len: usize) -> Self {
        ndarray::Array1::zeros(len)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Implementation of Vector trait for nalgebra's DVector<f64>.
///
/// This implementation provides an interface between the Vector trait and nalgebra's
/// dynamic vector type. It handles the conversion between nalgebra's internal
/// representation and the slice representation required by the JIT compiler.
///
/// # Examples
///
/// ```rust
/// use evalexpr_jit::prelude::Vector;
/// use nalgebra::DVector;
///
/// let mut vec = DVector::<f64>::zeros(3);
/// let slice = vec.as_mut_slice();
/// slice[0] = 1.0;
/// assert!(std::ptr::eq(slice, vec.as_slice()));
/// ```
#[cfg(feature = "nalgebra")]
impl Vector for nalgebra::DVector<f64> {
    fn as_slice(&self) -> &[f64] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }

    fn zeros(len: usize) -> Self {
        nalgebra::DVector::zeros(len)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Implementation of Vector trait for fixed-size arrays.
///
/// This implementation allows fixed-size arrays to be used with the JIT compiler.
/// The array size is specified through the const generic parameter N.
///
/// # Type Parameters
/// * `N` - The fixed size of the array
///
/// # Examples
///
/// ```rust
/// use evalexpr_jit::prelude::Vector;
///
/// let mut arr = <[f64; 3]>::zeros(3);
/// let slice = arr.as_mut_slice();
/// slice[0] = 1.0;
/// assert_eq!(arr[0], 1.0);
/// ```
impl<const N: usize> Vector for [f64; N] {
    fn as_slice(&self) -> &[f64] {
        self
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        self
    }

    fn zeros(len: usize) -> Self {
        assert_eq!(len, N, "Array length must match const generic size");
        [0.0; N]
    }

    fn len(&self) -> usize {
        N
    }
}
