use std::sync::Arc;

/// Type alias for a JIT-compiled function that evaluates multiple equations at once.
///
/// This represents a function that:
/// - Takes a slice of input values corresponding to variables
/// - Takes a mutable slice to store the results
/// - Evaluates multiple equations and writes results into the output slice
/// - Is both Send and Sync for thread safety
pub type CombinedJITFunction = Arc<dyn Fn(&[f64], &mut [f64]) + Send + Sync>;

pub type VectorizedCombinedJITFunction = Arc<dyn Fn(&[f64], &mut [f64]) + Send + Sync>;

/// Type alias for a JIT-compiled function that evaluates multiple equations at once.
///
/// This represents a function that:
/// - Takes a slice of input values corresponding to variables
/// - Takes a mutable slice to store the results
/// - Evaluates multiple equations and writes results into the output slice
/// - Is both Send and Sync for thread safety
pub type VectorizedJITFunction = Arc<dyn Fn(&[f64], &mut [f64]) + Send + Sync>;
