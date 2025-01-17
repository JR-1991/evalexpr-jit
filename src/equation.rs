//! Mathematical equation evaluation and differentiation with JIT compilation.
//!
//! This module provides the core `Equation` type which represents a mathematical expression
//! that can be evaluated and differentiated. Equations are JIT-compiled on creation for
//! efficient evaluation and support automatic differentiation up to second order.
//!
//! # Features
//!
//! - JIT compilation of expressions using Cranelift for fast evaluation
//! - Automatic differentiation up to second order derivatives
//! - Support for multiple variables with flexible ordering
//! - Efficient evaluation using native compiled code
//! - Stack-based evaluation for computing multiple derivatives at once
//!
//! # Example
//!
//! ```
//! use evalexpr_jit::Equation;
//!
//! let eq = Equation::new("2*x + y^2".to_string()).unwrap();
//! let result = eq.eval(&[1.0, 2.0]); // Evaluates to 6.0
//! let gradient = eq.gradient(&[1.0, 2.0]); // Computes [2.0, 4.0]
//! let hessian = eq.hessian(&[1.0, 2.0]); // Computes [[0.0, 0.0], [0.0, 2.0]]
//! ```
//!
//! # Variable Handling
//!
//! Variables can be specified either:
//! - Automatically extracted and sorted alphabetically using `new()`
//! - Explicitly mapped to indices using `from_var_map()`
//!
//! Input arrays must match the variable ordering.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use evalexpr::{build_operator_tree, Node, Operator};

use crate::backends::vector::Vector;
use crate::builder::build_function;
use crate::convert::build_ast;
use crate::errors::EquationError;
use crate::expr::Expr;
use crate::system::OutputType;
use crate::types::JITFunction;
use crate::EquationSystem;
use colored::Colorize;
use itertools::Itertools;

/// Represents a mathematical equation that can be evaluated and differentiated.
///
/// This struct holds both the original equation string and compiled functions for:
/// - Evaluating the equation
/// - Computing first order partial derivatives
/// - Computing second order partial derivatives (Hessian)
/// - Tracking variable names and their indices
///
/// The equation is JIT-compiled on creation for efficient evaluation. All derivatives
/// are also pre-compiled for fast computation.
///
/// Variables can be specified either:
/// - Automatically extracted and sorted alphabetically using `new()`
/// - Explicitly mapped to indices using `from_var_map()`
///
/// Input arrays must match the variable ordering.
pub struct Equation {
    equation_str: String,
    ast: Box<Expr>,
    fun: JITFunction,
    derivatives_first_order: HashMap<String, JITFunction>,
    derivatives_second_order: Vec<Vec<JITFunction>>,
    var_map: HashMap<String, u32>,
    sorted_variables: Vec<String>,
}

impl std::fmt::Debug for Equation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{{\n")?;
        writeln!(f, "    {}: {}\n", "Equation".cyan(), self.equation_str)?;
        writeln!(f, "    {}: {:?}\n", "Variables".cyan(), self.var_map)?;
        writeln!(
            f,
            "    {}: {:?}\n",
            "Sorted Variables".cyan(),
            self.sorted_variables
        )?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl std::fmt::Display for Equation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{{\n")?;
        writeln!(f, "    {}: {}\n", "Equation".cyan(), self.equation_str)?;
        writeln!(f, "    {}: {:?}\n", "Variables".cyan(), self.var_map)?;
        writeln!(
            f,
            "    {}: {:?}\n",
            "Sorted Variables".cyan(),
            self.sorted_variables
        )?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Equation {
    /// Creates a new `Equation` from a string representation.
    ///
    /// This function will automatically extract the variable names from the equation string
    /// and use them to create the `Equation` instance. The variable names will be sorted
    /// alphabetically and it is assumed that the input array of values will be in the same order.
    ///
    /// For more control over variable ordering, use `from_var_map()` instead.
    ///
    /// # Arguments
    /// * `equation_str` - The equation as a string (e.g. "2*x + y^2")
    ///
    /// # Returns
    /// * `Result<Self, EquationError>` - The compiled equation or an error
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("2*x + y^2".to_string()).unwrap();
    /// let result = eq.eval(&[1.0, 2.0]).unwrap(); // x=1, y=2 -> 2*1 + 2^2 = 6
    /// ```
    pub fn new(equation_str: String) -> Result<Self, EquationError> {
        let node = build_operator_tree(&equation_str)?;
        let variables = extract_symbols(&node);
        Self::build(&variables, equation_str)
    }

    /// Creates a new `Equation` from a map of variable names to their indices.
    ///
    /// This function allows explicit control over variable ordering by specifying
    /// the mapping between variable names and their positions in input arrays.
    ///
    /// # Arguments
    /// * `equation_str` - The equation as a string
    /// * `variables` - A map of variable names to their indices in input arrays
    ///
    /// # Returns
    /// * `Result<Self, EquationError>` - The compiled equation or an error
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// # use std::collections::HashMap;
    /// let mut vars = HashMap::new();
    /// vars.insert("y".to_string(), 0); // y will be first in input arrays
    /// vars.insert("x".to_string(), 1); // x will be second
    ///
    /// let eq = Equation::from_var_map("2*x + y^2".to_string(), &vars).unwrap();
    /// let result = eq.eval(&[2.0, 1.0]).unwrap(); // y=2, x=1 -> 2*1 + 2^2 = 6
    /// ```
    pub fn from_var_map(
        equation_str: String,
        variables: &HashMap<String, u32>,
    ) -> Result<Self, EquationError> {
        Self::build(variables, equation_str)
    }

    /// Builds an `Equation` instance from a variable map and equation string.
    ///
    /// This is the core builder function used by both `new()` and `from_var_map()`.
    /// It handles:
    /// - Parsing the equation string into an AST
    /// - JIT compiling the main evaluation function
    /// - Computing and compiling first order partial derivatives
    /// - Computing and compiling second order partial derivatives (Hessian)
    ///
    /// # Arguments
    /// * `variables` - Map of variable names to their indices in input arrays
    /// * `equation_str` - The equation as a string
    ///
    /// # Returns
    /// * `Result<Self, EquationError>` - The compiled equation or an error
    ///
    /// # Errors
    /// Returns `EquationError` if:
    /// - Equation string fails to parse
    /// - AST conversion fails
    /// - JIT compilation fails for any function
    /// - Variables in equation not found in provided map
    fn build(
        variables: &HashMap<String, u32>,
        equation_str: String,
    ) -> Result<Self, EquationError> {
        // Convert the equation string to an AST
        let node = build_operator_tree(&equation_str)?;

        // Validate if the variables are in the equation
        let mut non_defined_variables = HashSet::new();
        let control_variables = extract_symbols(&node);
        for variable in control_variables.keys() {
            if !variables.contains_key(variable) {
                non_defined_variables.insert(variable.clone());
            }
        }

        if !non_defined_variables.is_empty() {
            return Err(EquationError::VariableNotFound(
                non_defined_variables
                    .into_iter()
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        // Sort the variables by their indices
        let sorted_variables: Vec<String> = variables
            .iter()
            .sorted_by_key(|(_, &idx)| idx)
            .map(|(var, _)| var.clone())
            .collect();

        // Build the Expr AST
        let ast = build_ast(&node, variables)?;
        let ast = *ast.simplify();
        let fun = build_function(ast.clone())?;

        // Derive the first order partial derivatives
        let mut derivatives_first_order = HashMap::new();
        for variable in sorted_variables.iter() {
            let derivative = ast.derivative(variable);
            let derivative_func = build_function(*derivative)?;
            derivatives_first_order.insert(variable.clone(), derivative_func);
        }

        // Derive the second order partial derivatives
        let mut derivatives_second_order = Vec::new();
        for variable in sorted_variables.iter() {
            let mut derivatives_second_order_row = Vec::new();
            for variable2 in sorted_variables.iter() {
                let derivative = ast.derivative(variable).derivative(variable2);
                let derivative_func = build_function(*derivative)?;
                derivatives_second_order_row.push(derivative_func);
            }
            derivatives_second_order.push(derivatives_second_order_row);
        }

        Ok(Self {
            equation_str,
            ast: Box::new(ast),
            fun,
            derivatives_first_order,
            derivatives_second_order,
            var_map: variables.clone(),
            sorted_variables,
        })
    }

    /// Evaluates the equation for the given input values.
    ///
    /// # Arguments
    /// * `values` - Array of f64 values corresponding to variables in order
    ///
    /// # Returns
    /// * `Result<f64, EquationError>` - The result of evaluating the equation
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("2*x + y^2".to_string()).unwrap();
    /// let result = eq.eval(&[1.0, 2.0]).unwrap(); // x=1, y=2
    /// assert_eq!(result, 6.0); // 2*1 + 2^2 = 6
    /// ```
    ///
    /// # Errors
    /// Returns `EquationError::InvalidInputLength` if the length of values doesn't match
    /// the number of variables.
    pub fn eval<V: Vector>(&self, values: &V) -> Result<f64, EquationError> {
        self.validate_input_length(values.as_slice())?;
        Ok((self.fun)(values.as_slice()))
    }

    /// Computes the gradient (all first order partial derivatives) at the given point.
    ///
    /// # Arguments
    /// * `values` - Array of f64 values corresponding to variables in order
    ///
    /// # Returns
    /// * `Result<Vec<f64>, EquationError>` - Vector of partial derivatives in variable order
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("2*x + y^2".to_string()).unwrap();
    /// let gradient = eq.gradient(&[1.0, 2.0]).unwrap(); // at point (1,2)
    /// assert_eq!(gradient, vec![2.0, 4.0]); // [∂/∂x, ∂/∂y] = [2, 2y]
    /// ```
    ///
    /// # Errors
    /// Returns `EquationError::InvalidInputLength` if the length of values doesn't match
    /// the number of variables.
    pub fn gradient(&self, values: &[f64]) -> Result<Vec<f64>, EquationError> {
        self.validate_input_length(values)?;
        Ok(self
            .sorted_variables
            .iter()
            .map(|variable| (self.derivatives_first_order[variable])(values))
            .collect())
    }

    /// Computes the Hessian matrix (all second order partial derivatives) at the given point.
    ///
    /// # Arguments
    /// * `values` - Array of f64 values corresponding to variables in order
    ///
    /// # Returns
    /// * `Result<Vec<Vec<f64>>, EquationError>` - Matrix of second order derivatives in variable order
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("2*x + y^2".to_string()).unwrap();
    /// let hessian = eq.hessian(&[1.0, 2.0]).unwrap(); // at point (1,2)
    /// assert_eq!(hessian, vec![vec![0.0, 0.0], vec![0.0, 2.0]]);
    /// // [[∂²/∂x², ∂²/∂x∂y],
    /// //  [∂²/∂y∂x, ∂²/∂y²]]
    /// ```
    ///
    /// # Errors
    /// Returns `EquationError::InvalidInputLength` if the length of values doesn't match
    /// the number of variables.
    pub fn hessian(&self, values: &[f64]) -> Result<Vec<Vec<f64>>, EquationError> {
        self.validate_input_length(values)?;
        Ok(self
            .derivatives_second_order
            .iter()
            .map(|row| row.iter().map(|func| func(values)).collect())
            .collect())
    }

    /// Returns the derivative function for a specific variable.
    ///
    /// The returned function is a pointer to the compiled function. It will be invalidated
    /// if the equation is modified. The input array must not be modified while the function
    /// executes to avoid undefined behavior.
    ///
    /// # Arguments
    /// * `variable` - Name of the variable to get derivative for
    ///
    /// # Returns
    /// * `Result<&JITFunction, EquationError>` - The derivative function or an error if variable not found
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("2*x + y^2".to_string()).unwrap();
    /// let dx = eq.derivative("x").unwrap();
    /// let values = vec![1.0, 2.0];
    /// let result = dx(&values); // evaluate ∂/∂x at (1,2)
    /// assert_eq!(result, 2.0);
    /// ```
    ///
    /// # Errors
    /// Returns `EquationError::DerivativeNotFound` if the variable is not found.
    pub fn derivative(&self, variable: &str) -> Result<&JITFunction, EquationError> {
        self.derivatives_first_order
            .get(variable)
            .ok_or(EquationError::DerivativeNotFound(variable.to_string()))
    }

    /// Computes the higher-order partial derivative of the function with respect to multiple variables.
    ///
    /// The returned function is a pointer to the compiled function. It will be invalidated
    /// if the equation is modified. The input array must not be modified while the function
    /// executes to avoid undefined behavior.
    ///
    /// # Arguments
    /// * `variables` - Slice of variable names to differentiate with respect to, in order
    ///
    /// # Returns
    /// * `Result<JITFunction, EquationError>` - Returns a compiled function for the partial derivative
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("x^2 * y^2".to_string()).unwrap();
    /// let dxdy = eq.derive_wrt(&["x", "y"]).unwrap();
    /// let values = vec![2.0, 3.0];
    /// let result = dxdy(&values); // evaluate ∂²/∂x∂y at (2,3)
    /// assert_eq!(result, 24.0); // ∂²/∂x∂y(x^2 * y^2) = 4xy
    /// ```
    ///
    /// # Errors
    /// Returns `EquationError::DerivativeNotFound` if any variable is not found.
    pub fn derive_wrt(&self, variables: &[&str]) -> Result<JITFunction, EquationError> {
        // Check if the variables are in the equation
        let mut non_defined_variables = HashSet::new();
        for variable in variables.iter() {
            if !self.sorted_variables.contains(&variable.to_string()) {
                non_defined_variables.insert(variable.to_string());
            }
        }

        if !non_defined_variables.is_empty() {
            return Err(EquationError::DerivativeNotFound(
                non_defined_variables
                    .into_iter()
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        let mut expr = self.ast.clone();
        for variable in variables {
            expr = expr.derivative(variable);
        }
        let fun = build_function(*expr)?;
        Ok(fun)
    }

    /// Computes multiple first-order partial derivatives of the function with respect to the given variables,
    /// returning a single compiled function that evaluates all derivatives at once.
    ///
    /// This is more efficient than calling `derive_wrt()` multiple times when you need several
    /// first-order derivatives, as it compiles them into a single function that pushes all results
    /// onto a stack.
    ///
    /// # Arguments
    /// * `variables` - Slice of variable names to differentiate with respect to
    ///
    /// # Returns
    /// * `Result<CombinedJITFunction, EquationError>` - Returns a compiled function that evaluates
    ///   all derivatives and returns them as a Vec<f64>. The derivatives are returned in the same
    ///   order as the input variables.
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("x^2 + y^2 + z^2".to_string()).unwrap();
    /// let derivatives = eq.derive_wrt_stack(&["x", "y"]).unwrap();
    /// let values = vec![2.0, 3.0, 4.0];
    /// let mut results = vec![0.0, 0.0];
    /// derivatives.eval_into(&values, &mut results).unwrap(); // evaluate first derivatives [∂/∂x, ∂/∂y] at (2,3,4)
    /// assert_eq!(results, vec![4.0, 6.0]); // [∂f/∂x = 2x, ∂f/∂y = 2y]
    /// ```
    ///
    /// # Errors
    /// Returns `EquationError::DerivativeNotFound` if any variable is not found.
    pub fn derive_wrt_stack(&self, variables: &[&str]) -> Result<EquationSystem, EquationError> {
        // Derive the derivatives of the equation with respect to the variables
        let mut derivative_asts = Vec::with_capacity(variables.len());
        for variable in variables {
            derivative_asts.push(self.ast.derivative(variable));
        }

        EquationSystem::from_asts(derivative_asts, &self.var_map, OutputType::Vector)
    }

    /// Returns the map of variable names to their indices.
    pub fn variables(&self) -> &HashMap<String, u32> {
        &self.var_map
    }

    /// Returns the original equation string.
    pub fn equation_str(&self) -> &str {
        &self.equation_str
    }

    /// Returns the compiled evaluation function.
    pub fn fun(&self) -> &JITFunction {
        &self.fun
    }

    /// Returns the sorted variables.
    pub fn sorted_variables(&self) -> &[String] {
        &self.sorted_variables
    }

    /// Validates that the input array length matches the number of variables in the equation.
    ///
    /// # Arguments
    /// * `values` - Array of input values to validate
    ///
    /// # Returns
    /// * `Ok(())` if the lengths match
    /// * `Err(EquationError::InvalidInputLength)` if the lengths don't match
    fn validate_input_length(&self, values: &[f64]) -> Result<(), EquationError> {
        if values.len() != self.sorted_variables.len() {
            return Err(EquationError::InvalidInputLength {
                expected: self.sorted_variables.len(),
                got: values.len(),
            });
        }
        Ok(())
    }
}

/// Extracts variables from an expression tree and assigns them indices.
///
/// # Arguments
/// * `node` - Root node of the expression tree
///
/// # Returns
/// HashMap mapping variable names to their indices in the evaluation array
pub fn extract_symbols(node: &Node) -> HashMap<String, u32> {
    let mut symbols = HashSet::new();
    extract_symbols_from_node(node, &mut symbols);

    let mut symbols: Vec<String> = symbols.into_iter().collect();
    symbols.sort();

    symbols
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i as u32))
        .collect()
}

/// Extracts and sorts all unique variables from a collection of equation strings.
///
/// This function takes a slice of equation strings, parses each one into an expression tree,
/// extracts all variable names, and returns them as a sorted vector with duplicates removed.
///
/// # Arguments
/// * `equations` - Slice of strings containing mathematical expressions
///
/// # Returns
/// A sorted `Vec<String>` containing all unique variable names found in the equations
///
/// # Panics
/// Will panic if any equation string cannot be parsed into a valid expression tree
///
/// # Example
/// ```
/// # use evalexpr_jit::equation::extract_all_symbols;
/// let equations = vec!["2*x + y".to_string(), "z + x^2".to_string()];
/// let variables = extract_all_symbols(&equations);
/// assert_eq!(variables, vec!["x".to_string(), "y".to_string(), "z".to_string()]);
/// ```
pub fn extract_all_symbols(equations: &[String]) -> Vec<String> {
    let all_symbols: HashSet<String> = equations
        .iter()
        .flat_map(|e| {
            let tree: Node = build_operator_tree(e).unwrap();
            let symbols = extract_symbols(&tree);

            symbols.keys().cloned().collect::<Vec<String>>()
        })
        .collect();

    let mut all_symbols: Vec<String> = all_symbols.into_iter().collect();
    all_symbols.sort();

    all_symbols
}

/// Recursively extracts variable names from an expression tree node.
///
/// # Arguments
/// * `node` - Current node in the expression tree
/// * `symbols` - Set to store found variable names
fn extract_symbols_from_node(node: &Node, symbols: &mut HashSet<String>) {
    match node.operator() {
        Operator::VariableIdentifierRead { identifier } => {
            symbols.insert(identifier.to_string());
        }
        _ => {
            for child in node.children() {
                extract_symbols_from_node(child, symbols);
            }
        }
    }
}

impl Clone for Equation {
    fn clone(&self) -> Self {
        Self {
            equation_str: self.equation_str.clone(),
            ast: self.ast.clone(),
            fun: Arc::clone(&self.fun),
            derivatives_first_order: self.derivatives_first_order.clone(),
            derivatives_second_order: self.derivatives_second_order.clone(),
            var_map: self.var_map.clone(),
            sorted_variables: self.sorted_variables.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equation() {
        let eq = Equation::new("2*x + y^2".to_string()).unwrap();
        let result = eq.eval(&[1.0, 2.0]).unwrap();
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_gradient() {
        let eq = Equation::new("2*x + y^2".to_string()).unwrap();
        let gradient = eq.gradient(&[1.0, 2.0]).unwrap();
        assert_eq!(gradient, vec![2.0, 4.0]);
    }

    #[test]
    fn test_hessian() {
        let eq = Equation::new("2*x + y^2".to_string()).unwrap();
        let hessian = eq.hessian(&[1.0, 2.0]).unwrap();
        assert_eq!(hessian, vec![vec![0.0, 0.0], vec![0.0, 2.0]]);
    }

    #[test]
    fn test_derivative() {
        let eq = Equation::new("2*x + y^2".to_string()).unwrap();
        let derivative = eq.derivative("x").unwrap();
        let values = vec![1.0, 2.0];
        let result = derivative(&values);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_derive_wrt() {
        let eq = Equation::new("x^2 * y^2".to_string()).unwrap();
        let dxdy = eq.derive_wrt(&["x", "y"]).unwrap();
        let values = vec![2.0, 3.0];
        let result = dxdy(&values);
        assert_eq!(result, 24.0);
    }

    #[test]
    #[should_panic]
    fn test_derive_wrt_invalid() {
        let eq = Equation::new("x^2 * y^2".to_string()).unwrap();
        let _ = eq.derive_wrt(&["x", "z"]).expect("Invalid variable");
    }

    #[test]
    #[should_panic]
    fn test_eval_invalid() {
        let eq = Equation::new("2*x + y^2".to_string()).unwrap();
        let _ = eq.eval(&[1.0]).expect("Invalid input length");
    }

    #[test]
    fn test_from_var_map() {
        let eq = Equation::from_var_map(
            "2*x + y^2".to_string(),
            &HashMap::from([("x".to_string(), 1), ("y".to_string(), 0)]),
        )
        .unwrap();
        let result = eq.eval(&[2.0, 1.0]).unwrap();
        assert_eq!(result, 6.0);
    }

    #[test]
    #[should_panic]
    fn test_from_var_map_invalid() {
        let _ = Equation::from_var_map(
            "2*x + y^2".to_string(),
            &HashMap::from([("x".to_string(), 0), ("z".to_string(), 1)]),
        )
        .expect("Invalid variable");
    }

    #[test]
    fn test_derive_wrt_stack() {
        let eq = Equation::new("x^2 + 2*x*y + y^2 + z^3".to_string()).unwrap();

        // Test getting derivatives with respect to [x, z] (skipping y)
        let derivatives = eq.derive_wrt_stack(&["x", "z"]).unwrap();
        let values = vec![2.0, 3.0, 2.0]; // [x, y, z]
        let mut results = vec![0.0, 0.0];
        derivatives.eval_into(&values, &mut results).unwrap();

        // ∂/∂x = 2x + 2y = 2(2) + 2(3) = 10
        // ∂/∂z = 3z^2 = 3(2^2) = 12
        assert_eq!(results, vec![10.0, 12.0]);

        // Test different order [z, x] to verify order matters
        let derivatives = eq.derive_wrt_stack(&["z", "x"]).unwrap();
        let mut results = vec![0.0, 0.0];
        derivatives.eval_into(&values, &mut results).unwrap();
        assert_eq!(results, vec![12.0, 10.0]);
    }

    #[test]
    fn test_all_backends() {
        use nalgebra::DVector;
        use ndarray::Array1;

        let eq = Equation::new("2*x + y^2".to_string()).unwrap();
        let expected = 6.0; // 2*1 + 2^2 = 6.0

        // Test Vec (standard)
        let vec_input = vec![1.0, 2.0];
        assert_eq!(eq.eval(&vec_input).unwrap(), expected);

        // Test nalgebra
        let nalgebra_input = DVector::from_vec(vec![1.0, 2.0]);
        assert_eq!(eq.eval(&nalgebra_input).unwrap(), expected);

        // Test ndarray
        let ndarray_input = Array1::from_vec(vec![1.0, 2.0]);
        assert_eq!(eq.eval(&ndarray_input).unwrap(), expected);
    }

    #[test]
    fn test_extract_all_symbols() {
        let equations = vec![
            "2*x + y".to_string(),
            "z + x^2".to_string(),
            "y*z".to_string(),
        ];
        let variables = extract_all_symbols(&equations);
        assert_eq!(
            variables,
            vec!["x".to_string(), "y".to_string(), "z".to_string()]
        );
    }

    #[test]
    fn test_debug_and_display_formatting() {
        let eq = Equation::new("2*x + y^2".to_string()).unwrap();

        // Test Debug formatting
        let debug_output = format!("{:?}", eq);
        assert!(debug_output.contains("Equation"));
        assert!(debug_output.contains("2*x + y^2"));

        // Test Display formatting
        let display_output = format!("{}", eq);
        assert!(display_output.contains("Equation"));
        assert!(display_output.contains("2*x + y^2"));
    }

    #[test]
    fn test_equation_clone() {
        let eq = Equation::new("2*x + y^2".to_string()).unwrap();
        let cloned = eq.clone();

        // Test that both evaluate to the same result
        let values = vec![1.0, 2.0];
        assert_eq!(eq.eval(&values).unwrap(), cloned.eval(&values).unwrap());

        // Test that gradients match
        assert_eq!(
            eq.gradient(&values).unwrap(),
            cloned.gradient(&values).unwrap()
        );
    }

    #[test]
    fn test_invalid_expression() {
        let result = Equation::new("2*x + )".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_accessor_methods() {
        let eq = Equation::new("2*x + y^2".to_string()).unwrap();

        assert_eq!(eq.equation_str(), "2*x + y^2");
        assert!(!eq.variables().is_empty());
        assert!(!eq.sorted_variables().is_empty());
    }

    #[test]
    fn test_variable_ordering() {
        let mut vars = HashMap::new();
        vars.insert("z".to_string(), 0);
        vars.insert("y".to_string(), 1);
        vars.insert("x".to_string(), 2);

        let eq = Equation::from_var_map("x + y + z".to_string(), &vars).unwrap();
        assert_eq!(eq.sorted_variables(), &["z", "y", "x"]);

        // Test evaluation with ordered inputs
        let result = eq.eval(&[1.0, 2.0, 3.0]).unwrap(); // z=1, y=2, x=3
        assert_eq!(result, 6.0);
    }
}
