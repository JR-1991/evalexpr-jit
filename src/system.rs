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
//! ```

use crate::builder::build_combined_function;
use crate::convert::build_ast;
use crate::equation::{extract_all_symbols, extract_symbols, CombinedJITFunction};
use crate::errors::EquationError;
use crate::expr::Expr;
use evalexpr::build_operator_tree;
use itertools::Itertools;
use std::collections::HashMap;

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
    /// Jacobian of the system
    pub jacobian_funs: HashMap<String, CombinedJITFunction>,
}

impl EquationSystem {
    /// Creates a new equation system from a vector of expression strings.
    ///
    /// # Arguments
    /// * `expressions` - Vector of mathematical expressions as strings
    ///
    /// # Returns
    /// A new `EquationSystem` with JIT-compiled evaluation function
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "2*x + y".to_string(),
    ///     "x^2 + z".to_string()
    /// ]).unwrap();
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
    /// # Arguments
    /// * `expressions` - Vector of mathematical expressions as strings
    /// * `variable_map` - Map of variable names to their indices
    ///
    /// # Returns
    /// A new `EquationSystem` with JIT-compiled evaluation function
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
    /// # Arguments
    /// * `expressions` - Original expression strings
    /// * `asts` - Parsed and validated ASTs
    /// * `variable_map` - Map of variable names to indices
    ///
    /// # Returns
    /// A new `EquationSystem` with JIT-compiled evaluation function
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

    /// Evaluates all equations in the system with the given input values.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input values, must match the number of variables
    ///
    /// # Returns
    /// Vector of results, one for each equation in the system
    ///
    /// # Errors
    /// Returns `EquationError::InvalidInputLength` if the number of inputs doesn't match
    /// the number of variables
    pub fn eval(&self, inputs: &[f64]) -> Result<Vec<f64>, EquationError> {
        self.validate_input_length(inputs)?;

        Ok((self.combined_fun)(inputs))
    }

    /// Returns the gradient of the equation system with respect to a specific variable.
    ///
    /// The gradient contains the partial derivatives of all equations with respect to the given variable,
    /// evaluated at the provided input values.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input values at which to evaluate the gradient, must match the number of variables
    /// * `variable` - Name of the variable to compute derivatives with respect to
    ///
    /// # Returns
    /// * `Ok(Vec<f64>)` - Vector containing the partial derivatives of each equation with respect to
    ///   the specified variable, evaluated at the given input values
    /// * `Err(EquationError)` - If the number of inputs doesn't match the number of variables,
    ///   or if the specified variable is not found in the system
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

        Ok((self.jacobian_funs.get(variable).ok_or(
            EquationError::VariableNotFound(variable.to_string()),
        )?)(inputs))
    }

    /// Computes the Jacobian matrix of the equation system at the given input values.
    ///
    /// The Jacobian matrix contains all first-order partial derivatives of the system,
    /// where each row corresponds to a variable and each column corresponds to an equation.
    /// The entry at position (i,j) is the partial derivative of equation j with respect to variable i.
    ///
    /// # Dimensions
    /// * If the system has m equations and n variables, the Jacobian will be an n×m matrix
    /// * The outer vector has length n (number of variables)
    /// * Each inner vector has length m (number of equations)
    ///
    /// # Arguments
    /// * `inputs` - Slice of input values at which to evaluate the Jacobian, must match the number of variables
    ///
    /// # Returns
    /// * `Ok(Vec<Vec<f64>>)` - The Jacobian matrix as a vector of vectors, where each inner vector
    ///   contains the partial derivatives with respect to one variable across all equations
    /// * `Err(EquationError)` - If the number of inputs doesn't match the number of variables
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y".to_string(),
    ///     "x*y^2".to_string(),
    /// ]).unwrap();
    ///
    /// let jacobian = system.jacobian(&[2.0, 3.0], None).unwrap();
    /// // jacobian[0] contains [6.0, 9.0] (d/dx[x^2*y], d/dx[xy^2])
    /// // jacobian[1] contains [4.0, 12.0] (d/dy[x^2*y], d/dy[xy^2])
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
        for var in sorted_variables {
            let fun = self.jacobian_funs.get(var).unwrap();
            let derivatives = fun(inputs);
            for (eq_idx, &value) in derivatives.iter().enumerate() {
                results[eq_idx].push(value);
            }
        }

        Ok(results)
    }

    /// Creates a new equation system containing the higher-order derivatives of all equations
    /// with respect to multiple variables.
    ///
    /// # Arguments
    /// * `variables` - Slice of variable names to differentiate with respect to, in order
    ///
    /// # Returns
    /// A new `EquationSystem` containing the higher-order derivatives of all equations
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::system::EquationSystem;
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y".to_string(),
    ///     "x*y^2".to_string(),
    /// ]).unwrap();
    ///
    /// let derivatives = system.derive_wrt(&["x", "y"]).unwrap();
    /// let results = derivatives(&[2.0, 3.0]);
    /// assert_eq!(results, vec![4.0, 6.0]); // d²/dxdy[x^2*y] = 2x, d²/dxdy[x*y^2] = 2y
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

    /// Returns the sorted variables.
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

    /// Returns a vector of gradient functions.
    pub fn jacobian_funs(&self) -> &HashMap<String, CombinedJITFunction> {
        &self.jacobian_funs
    }

    /// Return a specific Jacobian function.
    pub fn gradient_fun(&self, variable: &str) -> &CombinedJITFunction {
        self.jacobian_funs.get(variable).unwrap()
    }

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
        let dx_results = dx(&[2.0, 3.0]);
        assert_eq!(dx_results, vec![12.0, 9.0]); // d/dx[x^2*y] = 2xy, d/dx[x*y^2] = y^2

        // Second order derivative
        let dxy = system.derive_wrt(&["x", "y"])?;
        let dxy_results = dxy(&[2.0, 3.0]);
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
}
