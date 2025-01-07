//! Mathematical equation evaluation and differentiation with JIT compilation.
//!
//! This module provides the core `Equation` type which represents a mathematical expression
//! that can be evaluated and differentiated. Equations are JIT-compiled on creation for
//! efficient evaluation and support automatic differentiation up to second order.
//!
//! # Features
//!
//! - JIT compilation of expressions using Cranelift
//! - Automatic differentiation up to second order derivatives
//! - Support for multiple variables
//! - Efficient evaluation using compiled native code
//!
//! # Example
//!
//! ```
//! use evalexpr_jit::Equation;
//!
//! let eq = Equation::new("2*x + y^2".to_string()).unwrap();
//! let result = eq.eval(&[1.0, 2.0]); // Evaluates to 6.0
//! let gradient = eq.gradient(&[1.0, 2.0]); // Computes [2.0, 4.0]
//! ```

use std::collections::{HashMap, HashSet};

use evalexpr::{build_operator_tree, Node, Operator};

use crate::convert::build_ast;
use crate::errors::EquationError;
use crate::expr::Expr;
use crate::prelude::build_function;

pub type JITFunction = Box<dyn Fn(&[f64]) -> f64>;

/// Represents a mathematical equation that can be evaluated and differentiated.
///
/// This struct holds both the original equation string and compiled functions for:
/// - Evaluating the equation
/// - Computing derivatives with respect to each variable
/// - Tracking variable names and their indices
///
/// The equation is JIT-compiled on creation for efficient evaluation.
pub struct Equation {
    equation_str: String,
    ast: Box<Expr>,
    fun: JITFunction,
    derivatives_first_order: HashMap<String, JITFunction>,
    derivatives_second_order: Vec<Vec<JITFunction>>,
    variables: HashMap<String, u32>,
    sorted_variables: Vec<String>,
}

impl Equation {
    /// Creates a new `Equation` from a string representation.
    ///
    /// This function will automatically extract the variable names from the equation string
    /// and use them to create the `Equation` instance. Please note, the variable names
    /// will be sorted alphabetically and it is assumed that the input array of values
    /// will be in the same order as the variables.
    ///
    /// If you want more control over the variable names, you can use the `from_var_map` function.
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
    /// ```
    pub fn new(equation_str: String) -> Result<Self, EquationError> {
        let node = build_operator_tree(&equation_str)?;
        let variables = extract_variables(&node);
        Self::build(variables, equation_str)
    }

    /// Creates a new `Equation` from a map of variable names to their indices.
    ///
    /// This function is useful when you want to manually specify the variable names
    /// and their indices in the evaluation array. In cases, where you want more control
    /// over the variable names, you can use this function.
    ///
    /// # Arguments
    /// * `variables` - A map of variable names to their indices
    /// * `equation_str` - The equation as a string
    ///
    /// # Returns
    /// * `Result<Self, EquationError>` - The compiled equation or an error
    pub fn from_var_map(
        equation_str: String,
        variables: HashMap<String, u32>,
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
    /// * `variables` - Map of variable names to their indices in the evaluation array
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
    fn build(variables: HashMap<String, u32>, equation_str: String) -> Result<Self, EquationError> {
        // Convert the equation string to an AST
        let node = build_operator_tree(&equation_str)?;

        // Validate if the variables are in the equation
        let mut non_defined_variables = HashSet::new();
        let control_variables = extract_variables(&node);
        for variable in variables.keys() {
            if !control_variables.contains_key(variable) {
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

        // Build the Expr AST
        let ast = build_ast(&node, &variables)?;
        let ast = *ast.simplify();
        let mut sorted_variables: Vec<String> = variables.keys().cloned().collect();
        sorted_variables.sort();

        // Compile the equation
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
            variables,
            sorted_variables,
        })
    }

    /// Evaluates the equation for the given input values.
    ///
    /// # Arguments
    /// * `values` - Array of f64 values corresponding to variables in order
    ///
    /// # Returns
    /// The result of evaluating the equation
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("2*x + y^2".to_string()).unwrap();
    /// let result = eq.eval(&[1.0, 2.0]).unwrap(); // x = 1, y = 2
    /// assert_eq!(result, 6.0); // 2*1 + 2^2 = 6
    /// ```
    pub fn eval(&self, values: &[f64]) -> Result<f64, EquationError> {
        self.validate_input_length(values)?;
        Ok((self.fun)(values))
    }

    /// Computes the gradient (all partial derivatives) at the given point.
    ///
    /// # Arguments
    /// * `values` - Array of f64 values corresponding to variables in order
    ///
    /// # Returns
    /// Vector of partial derivatives in the same order as variables
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("2*x + y^2".to_string()).unwrap();
    /// let gradient = eq.gradient(&[1.0, 2.0]).unwrap(); // at point (1,2)
    /// assert_eq!(gradient, vec![2.0, 4.0]); // [∂/∂x, ∂/∂y] = [2, 2y]
    /// ```
    pub fn gradient(&self, values: &[f64]) -> Result<Vec<f64>, EquationError> {
        self.validate_input_length(values)?;
        Ok(self
            .sorted_variables
            .iter()
            .map(|variable| (self.derivatives_first_order[variable])(values))
            .collect())
    }

    /// Computes the Hessian matrix at the given point.
    ///
    /// # Arguments
    /// * `values` - Array of f64 values corresponding to variables in order
    ///
    /// # Returns
    /// Matrix of second order partial derivatives in the same order as variables
    ///
    /// # Example
    /// ```
    /// # use evalexpr_jit::Equation;
    /// let eq = Equation::new("2*x + y^2".to_string()).unwrap();
    /// let hessian = eq.hessian(&[1.0, 2.0]).unwrap(); // at point (1,2)
    /// assert_eq!(hessian, vec![vec![0.0, 0.0], vec![0.0, 2.0]]);
    /// ```
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
    /// Please note, the returned function is a pointer to the compiled function,
    /// so it will be invalidated if the equation is modified. Also, since the function
    /// accepts a pointer to the input values, the array of input values must not be modified
    /// after the function is created. Otherwise, this will lead to undefined behavior.
    ///
    /// # Arguments
    /// * `variable` - Name of the variable to get derivative for
    ///
    /// # Returns
    /// * `Result<&fn(*const f64) -> f64, EquationError>` - The derivative function or an error if variable not found
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
    pub fn derivative(&self, variable: &str) -> Result<&JITFunction, EquationError> {
        self.derivatives_first_order
            .get(variable)
            .ok_or(EquationError::DerivativeNotFound(variable.to_string()))
    }

    /// Computes the higher-order partial derivative of the function with respect to multiple variables.
    ///
    /// Please note, the returned function is a pointer to the compiled function,
    /// so it will be invalidated if the equation is modified. Also, since the function
    /// accepts a pointer to the input values, the array of input values must not be modified
    /// after the function is created. Otherwise, this will lead to undefined behavior.
    ///
    /// # Arguments
    /// * `variables` - Slice of variable names to differentiate with respect to, in order
    ///
    /// # Returns
    /// * `Result<fn(*const f64) -> f64, EquationError>` - Returns a compiled function for the partial derivative
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

    /// Returns the map of variable names to their indices.
    pub fn variables(&self) -> &HashMap<String, u32> {
        &self.variables
    }

    /// Returns the original equation string.
    pub fn equation_str(&self) -> &str {
        &self.equation_str
    }

    /// Returns the compiled evaluation function.
    pub fn fun(&self) -> &JITFunction {
        &self.fun
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
pub fn extract_variables(node: &Node) -> HashMap<String, u32> {
    let mut variables = HashSet::new();
    extract_variables_from_node(node, &mut variables);

    let mut variables: Vec<String> = variables.into_iter().collect();
    variables.sort();

    variables
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i as u32))
        .collect()
}

/// Recursively extracts variable names from an expression tree node.
///
/// # Arguments
/// * `node` - Current node in the expression tree
/// * `variables` - Set to store found variable names
fn extract_variables_from_node(node: &Node, variables: &mut HashSet<String>) {
    match node.operator() {
        Operator::VariableIdentifierRead { identifier } => {
            variables.insert(identifier.to_string());
        }
        _ => {
            for child in node.children() {
                extract_variables_from_node(child, variables);
            }
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
            HashMap::from([("x".to_string(), 1), ("y".to_string(), 0)]),
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
            HashMap::from([("x".to_string(), 0), ("z".to_string(), 1)]),
        )
        .expect("Invalid variable");
    }
}
