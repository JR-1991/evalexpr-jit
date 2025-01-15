use std::sync::Arc;

/// Type alias for a JIT-compiled function that evaluates a single equation.
///
/// This represents a function that:
/// - Takes a slice of input values corresponding to variables in order
/// - Returns a single f64 result from evaluating the equation
/// - Is both Send and Sync for thread safety
pub type JITFunction = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Type alias for a JIT-compiled function that evaluates multiple equations at once.
///
/// This represents a function that:
/// - Takes a slice of input values corresponding to variables
/// - Takes a mutable slice to store the results
/// - Evaluates multiple equations and writes results into the output slice
/// - Is both Send and Sync for thread safety
pub type CombinedJITFunction = Arc<dyn Fn(&[f64], &mut [f64]) + Send + Sync>;

/// Type alias for a JIT-compiled function that evaluates multiple equations and returns a matrix.
///
/// This represents a function that:
/// - Takes a slice of input values corresponding to variables
/// - Takes a mutable slice of vectors to store the matrix results
/// - Each inner vector represents a row in the result matrix
/// - Is both Send and Sync for thread safety
pub type MatrixJITFunction = Arc<dyn Fn(&[f64], &mut [Vec<f64>]) + Send + Sync>;
