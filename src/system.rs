//! System of equations evaluation with JIT compilation.
//!
//! This module provides functionality for evaluating multiple mathematical equations
//! simultaneously using JIT compilation. The equations are combined into a single
//! optimized function for efficient evaluation.
//!
//! # Features
//!
//! - JIT compilation of multiple equations into a single function
//! - Consistent variable ordering across equations
//! - Automatic variable extraction and mapping
//! - Efficient batch evaluation
//! - Automatic derivative computation and Jacobian matrix generation
//! - Higher-order derivative support
//! - Parallel evaluation capabilities
//!
//! # Example
//!
//! ```
//! use evalexpr_jit::system::EquationSystem;
//!
//! let system = EquationSystem::new(vec![
//!     "2*x + y".to_string(),   // first equation
//!     "x^2 + z".to_string(),   // second equation
//! ]).unwrap();
//!
//! // Variables are automatically sorted (x, y, z)
//! let results = system.eval(&[1.0, 2.0, 3.0]).unwrap();
//! assert_eq!(results, vec![4.0, 4.0]); // [2*1 + 2, 1^2 + 3]
//!
//! // Compute derivatives
//! let dx = system.gradient(&[1.0, 2.0, 3.0], "x").unwrap();
//! assert_eq!(dx, vec![2.0, 2.0]); // [d/dx(2x + y), d/dx(x^2 + z)]
//! ```

use crate::builder::build_combined_function;
use crate::convert::build_ast;
use crate::equation::{extract_all_symbols, extract_symbols};
use crate::errors::EquationError;
use crate::expr::Expr;
use crate::types::{CombinedJITFunction, MatrixJITFunction};
use evalexpr::build_operator_tree;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Represents a system of mathematical equations that can be evaluated together.
pub struct EquationSystem {
    /// The original string representations of the equations
    pub equations: Vec<String>,
    /// The AST representations of the equations
    pub asts: Vec<Box<Expr>>,
    /// Maps variable names to their indices in the input array
    pub variable_map: HashMap<String, u32>,
    /// Variables in sorted order for consistent input ordering
    pub sorted_variables: Vec<String>,
    /// The JIT-compiled function that evaluates all equations
    pub combined_fun: CombinedJITFunction,
    /// Jacobian of the system - maps variable names to their derivative functions
    pub jacobian_funs: HashMap<String, CombinedJITFunction>,
}

impl EquationSystem {
    /// Creates a new equation system from a vector of expression strings.
    ///
    /// This constructor automatically extracts variables from the expressions and assigns them indices
    /// in alphabetical order. The resulting system can evaluate all equations efficiently and compute
    /// derivatives with respect to any variable.
    ///
    /// # Arguments
    /// * `expressions` - Vector of mathematical expressions as strings
    ///
    /// # Returns
    /// A new `EquationSystem` with JIT-compiled evaluation function and derivative capabilities
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "2*x + y".to_string(),
    ///     "x^2 + z".to_string()
    /// ]).unwrap();
    ///
    /// // Evaluate system
    /// let results = system.eval(&[1.0, 2.0, 3.0]).unwrap();
    ///
    /// // Compute derivatives
    /// let dx = system.gradient(&[1.0, 2.0, 3.0], "x").unwrap();
    /// ```
    pub fn new(expressions: Vec<String>) -> Result<Self, EquationError> {
        let sorted_variables = extract_all_symbols(&expressions);
        let variable_map: HashMap<String, u32> = sorted_variables
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i as u32))
            .collect();

        Self::build(expressions, variable_map)
    }

    /// Creates a new equation system from a vector of expressions and a variable map.
    ///
    /// This constructor allows explicit control over variable ordering by providing a map
    /// of variable names to their indices. This is useful when you need to ensure specific
    /// variable ordering or when integrating with other systems that expect variables in
    /// a particular order.
    ///
    /// # Arguments
    /// * `expressions` - Vector of mathematical expressions as strings
    /// * `variable_map` - Map of variable names to their indices, defining input order
    ///
    /// # Returns
    /// A new `EquationSystem` with JIT-compiled evaluation function using the specified variable ordering
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// # use std::collections::HashMap;
    /// let var_map: HashMap<String, u32> = [
    ///     ("x".to_string(), 0),
    ///     ("y".to_string(), 1),
    ///     ("z".to_string(), 2),
    /// ].into_iter().collect();
    ///
    /// let system = EquationSystem::from_var_map(
    ///     vec!["2*x + y".to_string(), "x^2 + z".to_string()],
    ///     &var_map
    /// ).unwrap();
    /// ```
    pub fn from_var_map(
        expressions: Vec<String>,
        variable_map: &HashMap<String, u32>,
    ) -> Result<Self, EquationError> {
        Self::build(expressions, variable_map.clone())
    }

    /// Core builder function used by both `new()` and `from_var_map()`.
    ///
    /// This internal function handles the common construction logic for both public constructors.
    /// It parses expressions, validates variables, builds ASTs, and creates JIT-compiled functions
    /// for both evaluation and derivatives.
    ///
    /// # Arguments
    /// * `expressions` - Original expression strings
    /// * `variable_map` - Map of variable names to indices
    ///
    /// # Returns
    /// A new `EquationSystem` with JIT-compiled evaluation function and derivative capabilities
    fn build(
        expressions: Vec<String>,
        variable_map: HashMap<String, u32>,
    ) -> Result<Self, EquationError> {
        // Convert expressions to ASTs
        let asts: Vec<Box<Expr>> = expressions
            .iter()
            .map(|expr| {
                let node = build_operator_tree(expr)?;

                // Validate variables
                let expr_vars = extract_symbols(&node);
                for var in expr_vars.keys() {
                    if !variable_map.contains_key(var) {
                        return Err(EquationError::VariableNotFound(var.clone()));
                    }
                }

                // Build and simplify AST
                let ast = build_ast(&node, &variable_map)?;
                Ok(ast.simplify())
            })
            .collect::<Result<Vec<_>, EquationError>>()?;

        // Create combined JIT function
        let combined_fun = build_combined_function(asts.clone(), expressions.len())?;

        // Create derivative functions for each variable forming a Jacobian matrix
        let mut jacobian_funs = HashMap::with_capacity(variable_map.len());

        let sorted_variables: Vec<String> = variable_map
            .iter()
            .sorted_by_key(|(_, idx)| *idx)
            .map(|(var, _)| var.clone())
            .collect();

        for var in sorted_variables {
            let derivative_ast = asts
                .iter()
                .map(|ast| ast.derivative(&var))
                .collect::<Vec<Box<Expr>>>();
            let jacobian_fun = build_combined_function(derivative_ast, expressions.len())?;
            jacobian_funs.insert(var, jacobian_fun);
        }

        Ok(Self {
            equations: expressions,
            asts,
            variable_map: variable_map.clone(),
            sorted_variables: variable_map.keys().sorted().cloned().collect(),
            combined_fun,
            jacobian_funs,
        })
    }

    /// Evaluates all equations in the system with the given input values
    /// into a pre-allocated buffer. This should be more efficient than
    /// calling `eval()` and then copying the results.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input values, must match the number of variables
    /// * `results` - Pre-allocated buffer to store results
    ///
    /// # Returns
    /// Reference to the results slice containing the evaluated values
    ///
    /// # Errors
    /// Returns `EquationError::InvalidInputLength` if:
    /// - The number of inputs doesn't match the number of variables
    /// - The results buffer size doesn't match the number of equations
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x + y".to_string(),
    ///     "x * y".to_string(),
    /// ]).unwrap();
    ///
    /// let mut results = vec![0.0; 2];
    /// system.eval_into(&[2.0, 3.0], &mut results).unwrap();
    /// assert_eq!(results, vec![5.0, 6.0]);
    /// ```
    pub fn eval_into<'a>(
        &self,
        inputs: &[f64],
        results: &'a mut [f64],
    ) -> Result<&'a [f64], EquationError> {
        self.validate_input_length(inputs)?;
        if results.len() != self.equations.len() {
            return Err(EquationError::InvalidInputLength {
                expected: self.equations.len(),
                got: results.len(),
            });
        }

        (self.combined_fun)(inputs, results);
        Ok(results)
    }

    /// Evaluates all equations in the system with the given input values.
    /// Allocates a new vector for results.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input values, must match the number of variables
    ///
    /// # Returns
    /// Vector of results, one for each equation in the system
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x + y".to_string(),
    ///     "x * y".to_string(),
    /// ]).unwrap();
    ///
    /// let results = system.eval(&[2.0, 3.0]).unwrap();
    /// assert_eq!(results, vec![5.0, 6.0]);
    /// ```
    pub fn eval(&self, inputs: &[f64]) -> Result<Vec<f64>, EquationError> {
        let mut results = vec![0.0; self.equations.len()];
        self.eval_into(inputs, &mut results)?;
        Ok(results)
    }

    /// Evaluates the equation system in parallel for multiple input sets using explicit threading.
    ///
    /// This method is optimized for batch evaluation of many input sets. It automatically
    /// determines the optimal chunk size based on available CPU cores and distributes the
    /// work across multiple threads.
    ///
    /// # Arguments
    /// * `input_sets` - Slice of input vectors, each must match the number of variables
    ///
    /// # Returns
    /// Vector of result vectors, one for each input set
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x + y".to_string(),
    ///     "x * y".to_string(),
    /// ]).unwrap();
    ///
    /// let input_sets = vec![
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0],
    ///     vec![5.0, 6.0],
    /// ];
    ///
    /// let results = system.eval_parallel(&input_sets).unwrap();
    /// ```
    pub fn eval_parallel(&self, input_sets: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, EquationError> {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        let chunk_size = (input_sets.len() / (num_threads * 4)).max(1);
        let n_equations = self.equations.len();

        let fun = Arc::clone(&self.combined_fun);

        Ok(input_sets
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|inputs| {
                        let mut results = vec![0.0; n_equations];
                        (fun)(inputs, &mut results);
                        results
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect())
    }

    /// Returns the gradient of the equation system with respect to a specific variable.
    ///
    /// The gradient contains the partial derivatives of all equations with respect to the given variable,
    /// evaluated at the provided input values. This is equivalent to one row of the Jacobian matrix.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input values at which to evaluate the gradient
    /// * `variable` - Name of the variable to compute derivatives with respect to
    ///
    /// # Returns
    /// Vector containing the partial derivatives of each equation with respect to
    /// the specified variable, evaluated at the given input values
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y".to_string(),  // f1
    ///     "x*y^2".to_string(),  // f2
    /// ]).unwrap();
    ///
    /// let gradient = system.gradient(&[2.0, 3.0], "x").unwrap();
    /// // gradient contains [12.0, 9.0] (∂f1/∂x, ∂f2/∂x)
    /// ```
    pub fn gradient(&self, inputs: &[f64], variable: &str) -> Result<Vec<f64>, EquationError> {
        self.validate_input_length(inputs)?;
        let n_equations = self.equations.len();
        let mut results = vec![0.0; n_equations];
        self.jacobian_funs
            .get(variable)
            .ok_or(EquationError::VariableNotFound(variable.to_string()))?(
            inputs, &mut results
        );
        Ok(results)
    }

    /// Computes the Jacobian matrix of the equation system at the given input values.
    ///
    /// The Jacobian matrix contains all first-order partial derivatives of the system.
    /// Each row corresponds to an equation, and each column corresponds to a variable.
    /// The entry at position (i,j) is the partial derivative of equation i with respect to variable j.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input values at which to evaluate the Jacobian
    /// * `variables` - Optional slice of variable names to include in the Jacobian.
    ///                If None, includes all variables in sorted order.
    ///
    /// # Returns
    /// The Jacobian matrix as a vector of vectors, where each inner vector
    /// contains the partial derivatives of one equation with respect to all variables
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y".to_string(),  // f1
    ///     "x*y^2".to_string(),  // f2
    /// ]).unwrap();
    ///
    /// let jacobian = system.jacobian(&[2.0, 3.0], None).unwrap();
    /// // jacobian[0] contains [12.0, 4.0]   // ∂f1/∂x, ∂f1/∂y
    /// // jacobian[1] contains [9.0, 12.0]   // ∂f2/∂x, ∂f2/∂y
    /// ```
    pub fn jacobian(
        &self,
        inputs: &[f64],
        variables: Option<&[String]>,
    ) -> Result<Vec<Vec<f64>>, EquationError> {
        self.validate_input_length(inputs)?;

        let sorted_variables = variables.unwrap_or(&self.sorted_variables);
        let mut results = Vec::with_capacity(self.equations.len());
        let n_vars = sorted_variables.len();

        // Initialize vectors for each equation
        for _ in 0..self.equations.len() {
            results.push(Vec::with_capacity(n_vars));
        }

        // Fill the transposed matrix
        let n_equations = self.equations.len();
        for var in sorted_variables {
            let fun = self.jacobian_funs.get(var).unwrap();
            let mut derivatives = vec![0.0; n_equations];
            fun(inputs, &mut derivatives);
            for (eq_idx, &value) in derivatives.iter().enumerate() {
                results[eq_idx].push(value);
            }
        }

        Ok(results)
    }

    /// Creates a JIT-compiled function that computes the Jacobian matrix with respect to specific variables.
    ///
    /// This method is optimized for repeated Jacobian evaluations when you only need derivatives
    /// with respect to a subset of variables. The resulting function computes partial derivatives
    /// of all equations with respect to the specified variables and arranges them in matrix form.
    ///
    /// # Arguments
    /// * `variables` - Slice of variable names to include in the Jacobian matrix
    ///
    /// # Returns
    /// A JIT-compiled function that takes input values and fills a matrix of partial derivatives.
    /// The matrix is represented as a slice of vectors, where each row contains the derivatives
    /// of one equation with respect to the specified variables.
    ///
    /// # Errors
    /// Returns `EquationError::VariableNotFound` if any of the specified variables doesn't exist
    /// in the system.
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y + z".to_string(),    // f1
    ///     "x*y^2 - z^2".to_string(),  // f2
    /// ]).unwrap();
    ///
    /// // Create Jacobian function for x and y only
    /// let jacobian_fn = system.jacobian_wrt(&["x", "y"]).unwrap();
    ///
    /// // Prepare matrix to store results (2 equations × 2 variables)
    /// let mut results = vec![vec![0.0; 2]; 2];
    ///
    /// // Evaluate Jacobian at point (x=2, y=3, z=1)
    /// jacobian_fn(&[2.0, 3.0, 1.0], &mut results);
    ///
    /// // results now contains:
    /// // [
    /// //   [12.0, 4.0],   // [∂f1/∂x, ∂f1/∂y]
    /// //   [9.0,  12.0],  // [∂f2/∂x, ∂f2/∂y]
    /// // ]
    /// ```
    ///
    /// # Performance Notes
    /// - The returned function is JIT-compiled and optimized for repeated evaluations
    /// - Pre-allocate the results matrix and reuse it for better performance
    /// - The matrix dimensions will be `[n_equations × n_variables]`
    pub fn jacobian_wrt(&self, variables: &[&str]) -> Result<MatrixJITFunction, EquationError> {
        // Verify all variables exist
        for var in variables {
            if !self.variable_map.contains_key(*var) {
                return Err(EquationError::VariableNotFound(var.to_string()));
            }
        }

        let mut asts = vec![];
        for ast in self.asts.iter() {
            for var in variables {
                asts.push(ast.derivative(var));
            }
        }

        let fun = build_combined_function(asts, self.equations.len() * variables.len())?;
        let n_vars = variables.len();
        let n_eqs = self.equations.len();

        // Create wrapper closure that rearranges output to matrix form
        let wrapped_fun: MatrixJITFunction =
            Arc::new(move |inputs: &[f64], results: &mut [Vec<f64>]| {
                let mut temp = vec![0.0; n_vars * n_eqs];
                fun(inputs, &mut temp);

                // Fill matrix - each row contains derivatives of one equation wrt all variables
                for eq_idx in 0..n_eqs {
                    for var_idx in 0..n_vars {
                        results[eq_idx][var_idx] = temp[eq_idx * n_vars + var_idx];
                    }
                }
            });

        Ok(wrapped_fun)
    }

    /// Creates a new equation system containing the higher-order derivatives of all equations
    /// with respect to multiple variables.
    ///
    /// This method allows computing mixed partial derivatives by specifying the variables
    /// in the order of differentiation. For example, passing ["x", "y"] computes ∂²f/∂x∂y
    /// for each equation f.
    ///
    /// # Arguments
    /// * `variables` - Slice of variable names to differentiate with respect to, in order
    ///
    /// # Returns
    /// A JIT-compiled function that computes the higher-order derivatives
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y".to_string(),  // f1
    ///     "x*y^2".to_string(),  // f2
    /// ]).unwrap();
    ///
    /// let derivatives = system.derive_wrt(&["x", "y"]).unwrap();
    /// let mut results = vec![0.0; 2];
    /// derivatives(&[2.0, 3.0], &mut results);
    /// assert_eq!(results, vec![4.0, 6.0]); // ∂²f1/∂x∂y = 2x, ∂²f2/∂x∂y = 2y
    /// ```
    pub fn derive_wrt(&self, variables: &[&str]) -> Result<CombinedJITFunction, EquationError> {
        // Verify all variables exist
        for var in variables {
            if !self.variable_map.contains_key(*var) {
                return Err(EquationError::VariableNotFound(var.to_string()));
            }
        }

        // Create higher-order derivatives of all ASTs
        let mut derivative_asts = self.asts.clone();
        for var in variables {
            derivative_asts = derivative_asts
                .into_iter()
                .map(|ast| ast.derivative(var))
                .collect();
        }

        // Create combined JIT function for derivatives
        build_combined_function(derivative_asts.clone(), self.equations.len())
    }

    /// Returns the sorted variables in the system.
    pub fn sorted_variables(&self) -> &[String] {
        &self.sorted_variables
    }

    /// Returns the map of variable names to their indices.
    pub fn variables(&self) -> &HashMap<String, u32> {
        &self.variable_map
    }

    /// Returns the original equation strings.
    pub fn equations(&self) -> &[String] {
        &self.equations
    }

    /// Returns the compiled evaluation function.
    pub fn fun(&self) -> &CombinedJITFunction {
        &self.combined_fun
    }

    /// Returns the map of variable names to their derivative functions.
    pub fn jacobian_funs(&self) -> &HashMap<String, CombinedJITFunction> {
        &self.jacobian_funs
    }

    /// Returns the derivative function for a specific variable.
    pub fn gradient_fun(&self, variable: &str) -> &CombinedJITFunction {
        self.jacobian_funs.get(variable).unwrap()
    }

    /// Returns the number of equations in the system.
    pub fn num_equations(&self) -> usize {
        self.equations.len()
    }

    /// Validates that the number of input values matches the number of variables.
    fn validate_input_length(&self, inputs: &[f64]) -> Result<(), EquationError> {
        if inputs.len() != self.sorted_variables.len() {
            return Err(EquationError::InvalidInputLength {
                expected: self.sorted_variables.len(),
                got: inputs.len(),
            });
        }
        Ok(())
    }
}

impl Clone for EquationSystem {
    fn clone(&self) -> Self {
        Self {
            equations: self.equations.clone(),
            asts: self.asts.clone(),
            variable_map: self.variable_map.clone(),
            sorted_variables: self.sorted_variables.clone(),
            combined_fun: Arc::clone(&self.combined_fun),
            jacobian_funs: self.jacobian_funs.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_with_different_variables() -> Result<(), Box<dyn std::error::Error>> {
        let expressions = vec![
            "2*x + y".to_string(),   // uses x, y
            "z^2".to_string(),       // uses only z
            "x + y + z".to_string(), // uses all
        ];

        let system = EquationSystem::new(expressions)?;

        // Check that all variables are tracked
        assert_eq!(system.sorted_variables, &["x", "y", "z"]);

        // Evaluate with values for all variables
        let results = system.eval(&[1.0, 2.0, 3.0])?;

        // Check results
        assert_eq!(
            results,
            vec![
                4.0, // 2*1 + 2
                9.0, // 3^2
                6.0, // 1 + 2 + 3
            ]
        );

        Ok(())
    }

    #[test]
    fn test_consistent_variable_ordering() -> Result<(), Box<dyn std::error::Error>> {
        let expressions = vec![
            "y + x".to_string(), // variables in different order
            "x + z".to_string(), // different subset of variables
        ];

        let system = EquationSystem::new(expressions)?;

        // Check that ordering is consistent (alphabetical)
        assert_eq!(system.sorted_variables, &["x", "y", "z"]);

        // Values must be provided in the consistent order [x, y, z]
        let results = system.eval(&[1.0, 2.0, 3.0])?;

        assert_eq!(
            results,
            vec![
                3.0, // y(2.0) + x(1.0)
                4.0, // x(1.0) + z(3.0)
            ]
        );

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_invalid_input_length() {
        let system = EquationSystem::new(vec!["x + y".to_string(), "y + z".to_string()]).unwrap();

        // Should panic: providing only 2 values when system needs 3 (x, y, z)
        let _ = system.eval(&[1.0, 2.0]).unwrap();
    }

    #[test]
    fn test_complex_expressions() -> Result<(), Box<dyn std::error::Error>> {
        let expressions = vec![
            "(x + y) * (x - y)".to_string(),     // difference of squares
            "x^3 + y^2 * z".to_string(),         // polynomial
            "(x + y + z) / (x + 1)".to_string(), // division
        ];

        let system = EquationSystem::new(expressions)?;
        let results = system.eval(&[2.0, 3.0, 4.0])?;

        assert_eq!(results[0], -5.0); // (2 + 3) * (2 - 3) = 5 * -1 = -5
        assert_eq!(results[1], 44.0); // 2^3 + 3^2 * 4 = 8 + 9 * 4 = 44
        assert_eq!(results[2], 3.0); // (2 + 3 + 4) / (2 + 1) = 9 / 3 = 3

        Ok(())
    }

    #[test]
    fn test_custom_variable_map() -> Result<(), Box<dyn std::error::Error>> {
        let mut var_map = HashMap::new();
        var_map.insert("alpha".to_string(), 1);
        var_map.insert("beta".to_string(), 0);

        let expressions = vec!["2*alpha + beta".to_string(), "alpha^2 - beta".to_string()];

        let system = EquationSystem::from_var_map(expressions, &var_map)?;
        let results = system.eval(&[2.0, 1.0])?;

        assert_eq!(results, vec![4.0, -1.0]);

        Ok(())
    }

    #[test]
    fn test_error_undefined_variable() {
        let expressions = vec![
            "x + y".to_string(),
            "x + undefined_var".to_string(), // undefined variable
        ];

        let mut var_map = HashMap::new();
        var_map.insert("x".to_string(), 0);
        var_map.insert("y".to_string(), 1);

        let result = EquationSystem::from_var_map(expressions, &var_map);
        assert!(matches!(result, Err(EquationError::VariableNotFound(_))));
    }

    #[test]
    fn test_empty_system() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec![])?;
        let results = system.eval(&[])?;
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn test_derive_wrt() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec!["x^2*y".to_string(), "x*y^2".to_string()])?;

        // First order derivative
        let dx = system.derive_wrt(&["x"])?;
        let mut dx_results = vec![0.0, 0.0];
        dx(&[2.0, 3.0], &mut dx_results);
        assert_eq!(dx_results, vec![12.0, 9.0]); // d/dx[x^2*y] = 2xy, d/dx[x*y^2] = y^2

        // Second order derivative
        let dxy = system.derive_wrt(&["x", "y"])?;
        let mut dxy_results = vec![0.0, 0.0];
        dxy(&[2.0, 3.0], &mut dxy_results);
        assert_eq!(dxy_results, vec![4.0, 6.0]); // d²/dxdy[x^2*y] = 2x, d²/dxdy[x*y^2] = 2y

        Ok(())
    }

    #[test]
    fn test_derive_wrt_invalid_variable() {
        let system =
            EquationSystem::new(vec!["2*x + y^2".to_string(), "x^2 + z".to_string()]).unwrap();

        let result = system.derive_wrt(&["w"]);
        assert!(matches!(result, Err(EquationError::VariableNotFound(_))));
    }

    #[test]
    fn test_jacobian() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec![
            "x^2*y".to_string(), // f1
            "x*y^2".to_string(), // f2
        ])?;

        let jacobian = system.jacobian(&[2.0, 3.0], None)?;

        // Jacobian matrix should be:
        // [∂f1/∂x  ∂f1/∂y] = [12.0  4.0]   // derivatives of first equation
        // [∂f2/∂x  ∂f2/∂y] = [9.0   12.0]  // derivatives of second equation

        assert_eq!(jacobian.len(), 2); // Two rows (one per equation)
        assert_eq!(jacobian[0], vec![12.0, 4.0]); // Derivatives of first equation
        assert_eq!(jacobian[1], vec![9.0, 12.0]); // Derivatives of second equation

        Ok(())
    }

    #[test]
    fn test_jacobian_wrt() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec![
            "x^2*y + z".to_string(),   // f1
            "x*y^2 - z^2".to_string(), // f2
        ])?;

        // Test subset of variables (x and y only)
        let jacobian_fn = system.jacobian_wrt(&["x", "y"])?;
        let mut results = vec![vec![0.0; 2]; 2];
        jacobian_fn(&[2.0, 3.0, 1.0], &mut results);

        // Expected derivatives:
        // ∂f1/∂x = 2xy = 2(2)(3) = 12
        // ∂f1/∂y = x^2 = 4
        // ∂f2/∂x = y^2 = 9
        // ∂f2/∂y = 2xy = 2(2)(3) = 12
        assert_eq!(results[0], vec![12.0, 4.0]); // [∂f1/∂x, ∂f1/∂y]
        assert_eq!(results[1], vec![9.0, 12.0]); // [∂f2/∂x, ∂f2/∂y]

        Ok(())
    }

    #[test]
    fn test_jacobian_wrt_single_variable() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec![
            "x^2*y".to_string(), // f1
            "x*y^2".to_string(), // f2
        ])?;

        // Test with single variable
        let jacobian_fn = system.jacobian_wrt(&["x"])?;
        let mut results = vec![vec![0.0; 1]; 2];
        jacobian_fn(&[2.0, 3.0], &mut results);

        // Expected derivatives:
        // ∂f1/∂x = 2xy = 2(2)(3) = 12
        // ∂f2/∂x = y^2 = 9
        assert_eq!(results[0], vec![12.0]); // [∂f1/∂x]
        assert_eq!(results[1], vec![9.0]); // [∂f2/∂x]

        Ok(())
    }

    #[test]
    fn test_jacobian_wrt_all_variables() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec![
            "x^2*y + z".to_string(),   // f1
            "x*y^2 - z^2".to_string(), // f2
        ])?;

        // Test with all variables
        let jacobian_fn = system.jacobian_wrt(&["x", "y", "z"])?;
        let mut results = vec![vec![0.0; 3]; 2];
        jacobian_fn(&[2.0, 3.0, 1.0], &mut results);

        // Expected derivatives:
        // ∂f1/∂x = 2xy = 2(2)(3) = 12
        // ∂f1/∂y = x^2 = 4
        // ∂f1/∂z = 1
        // ∂f2/∂x = y^2 = 9
        // ∂f2/∂y = 2xy = 2(2)(3) = 12
        // ∂f2/∂z = -2z = -2
        assert_eq!(results[0], vec![12.0, 4.0, 1.0]); // [∂f1/∂x, ∂f1/∂y, ∂f1/∂z]
        assert_eq!(results[1], vec![9.0, 12.0, -2.0]); // [∂f2/∂x, ∂f2/∂y, ∂f2/∂z]

        Ok(())
    }

    #[test]
    fn test_jacobian_wrt_invalid_variable() {
        let system =
            EquationSystem::new(vec!["x^2*y + z".to_string(), "x*y^2 - z^2".to_string()]).unwrap();

        // Test with non-existent variable
        let result = system.jacobian_wrt(&["x", "w"]);
        assert!(matches!(result, Err(EquationError::VariableNotFound(_))));
    }

    #[test]
    fn test_jacobian_wrt_reuse_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec![
            "x^2*y".to_string(), // f1
            "x*y^2".to_string(), // f2
        ])?;

        let jacobian_fn = system.jacobian_wrt(&["x", "y"])?;
        let mut results = vec![vec![0.0; 2]; 2];

        // First evaluation
        jacobian_fn(&[2.0, 3.0], &mut results);
        assert_eq!(results[0], vec![12.0, 4.0]);
        assert_eq!(results[1], vec![9.0, 12.0]);

        // Reuse buffer for second evaluation
        jacobian_fn(&[1.0, 2.0], &mut results);
        assert_eq!(results[0], vec![4.0, 1.0]);
        assert_eq!(results[1], vec![4.0, 4.0]);

        Ok(())
    }
}
