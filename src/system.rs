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
//! let results = system.eval(&vec![1.0, 2.0, 3.0]).unwrap();
//! assert_eq!(results.as_slice(), vec![4.0, 4.0]); // [2*1 + 2, 1^2 + 3]
//!
//! // Compute derivatives
//! let dx = system.gradient(&vec![1.0, 2.0, 3.0], "x").unwrap();
//! assert_eq!(dx.as_slice(), vec![2.0, 2.0]); // [d/dx(2x + y), d/dx(x^2 + z)]
//! ```
//!
//! # Computing Derivatives
//!
//! There are two main ways to create new equation systems with derivatives from existing ones:
//!
//! ## 1. Using `derive_wrt` for Gradients
//!
//! Creates a new equation system that computes derivatives with respect to specified variables:
//!
//! ```
//! # use evalexpr_jit::system::EquationSystem;
//! let system = EquationSystem::new(vec![
//!     "x^2*y".to_string(),     // f1
//!     "x*y^2".to_string(),     // f2
//! ]).unwrap();
//!
//! // First-order derivative with respect to x
//! let dx = system.derive_wrt(&["x"]).unwrap();
//! let results = dx.eval(&[2.0, 3.0]).unwrap();
//! assert_eq!(results.as_slice(), &[12.0, 9.0]);  // [d(x^2*y)/dx = 2xy, d(x*y^2)/dx = y^2]
//!
//! // Second-order mixed derivative (first x, then y)
//! let dxy = system.derive_wrt(&["x", "y"]).unwrap();
//! let results = dxy.eval(&[2.0, 3.0]).unwrap();
//! assert_eq!(results.as_slice(), &[4.0, 6.0]);   // [d²(x^2*y)/dxdy = 2x, d²(x*y^2)/dxdy = 2y]
//! ```
//!
//! ## 2. Using `jacobian_wrt` for Jacobian Matrices
//!
//! Creates a new equation system that computes the Jacobian matrix with respect to specified variables:
//!
//! ```
//! # use evalexpr_jit::system::EquationSystem;
//! # use ndarray::Array2;
//! let system = EquationSystem::new(vec![
//!     "x^2*y + z".to_string(),    // f1
//!     "x*y^2 - z^2".to_string(),  // f2
//! ]).unwrap();
//!
//! // Create Jacobian system for x and y
//! let jacobian_system = system.jacobian_wrt(&["x", "y"]).unwrap();
//!
//! // Prepare matrix to store results (2 equations × 2 variables)
//! let mut results = Array2::zeros((2, 2));
//!
//! // Evaluate Jacobian at point (x=2, y=3, z=1)
//! jacobian_system.eval_into_matrix(&[2.0, 3.0, 1.0], &mut results).unwrap();
//!
//! // results now contains:
//! // [
//! //   [12.0, 4.0],   // [∂f1/∂x, ∂f1/∂y]
//! //   [9.0,  12.0],  // [∂f2/∂x, ∂f2/∂y]
//! // ]
//! ```
//!

use crate::backends::vector::Vector;
use crate::builder::build_combined_function;
use crate::convert::build_ast;
use crate::equation::{extract_all_symbols, extract_symbols};
use crate::errors::EquationError;
use crate::expr::Expr;
use crate::prelude::Matrix;
use crate::types::CombinedJITFunction;
use evalexpr::build_operator_tree;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashMap;

/// Represents a system of mathematical equations that can be evaluated together.
pub struct EquationSystem {
    /// The original string representations of the equations
    pub equations: Vec<String>,
    /// The AST representations of the equations
    pub asts: Vec<Expr>,
    /// Maps variable names to their indices in the input array
    pub variable_map: HashMap<String, u32>,
    /// Variables in sorted order for consistent input ordering
    pub sorted_variables: Vec<String>,
    /// The JIT-compiled function that evaluates all equations
    pub combined_fun: CombinedJITFunction,
    /// Partial derivatives of the system - maps variable names to their derivative functions
    /// E.g. {"x": df(x, y, z)/dx, "y": df(x, y, z)/dy}
    pub partial_derivatives: HashMap<String, CombinedJITFunction>,
    /// The type of output
    output_type: OutputType,
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
    /// let results = system.eval(&vec![1.0, 2.0, 3.0]).unwrap();
    ///
    /// // Compute derivatives
    /// let dx = system.gradient(&vec![1.0, 2.0, 3.0], "x").unwrap();
    /// ```
    pub fn new(expressions: Vec<String>) -> Result<Self, EquationError> {
        let sorted_variables = extract_all_symbols(&expressions);
        let variable_map: HashMap<String, u32> = sorted_variables
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i as u32))
            .collect();
        let asts = Self::create_asts(&expressions, &variable_map)?;
        Self::build(asts, expressions, variable_map, OutputType::Vector)
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
    /// use evalexpr_jit::prelude::*;
    /// use std::collections::HashMap;
    ///
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
        let asts = Self::create_asts(&expressions, variable_map)?;
        Self::build(asts, expressions, variable_map.clone(), OutputType::Vector)
    }

    /// Creates a new equation system directly from AST nodes and a variable map.
    ///
    /// This constructor allows creating a system from pre-built AST nodes rather than parsing
    /// expressions from strings. This is useful when augmenting existing systems or creating derivatives
    /// of existing systems.
    ///
    /// Please note, this constructor is not meant to be used by the end user and is only available for internal use
    /// to create derivatives of existing systems.
    ///
    /// # Arguments
    /// * `asts` - Vector of expression AST nodes
    /// * `variable_map` - Map of variable names to their indices, defining input order
    /// * `output_type` - The type of output. Used to determine the shape of the output vector
    /// # Returns
    /// A new `EquationSystem` with JIT-compiled evaluation function using the specified ASTs
    pub(crate) fn from_asts(
        asts: Vec<Expr>,
        variable_map: &HashMap<String, u32>,
        output_type: OutputType,
    ) -> Result<Self, EquationError> {
        let expressions = asts.iter().map(|ast| ast.to_string()).collect();
        Self::build(asts, expressions, variable_map.clone(), output_type)
    }

    /// Core builder function used by both `new()` and `from_var_map()`.
    ///
    /// This internal function handles the common construction logic for both public constructors.
    /// It builds ASTs and creates JIT-compiled functions for both evaluation and derivatives.
    ///
    /// # Arguments
    /// * `asts` - Vector of AST nodes
    /// * `equations` - Original expression strings
    /// * `variable_map` - Map of variable names to indices
    /// * `output_type` - The type of output (vector or matrix)
    ///
    /// # Returns
    /// A new `EquationSystem` with JIT-compiled evaluation function and derivative capabilities
    fn build(
        asts: Vec<Expr>,
        equations: Vec<String>,
        variable_map: HashMap<String, u32>,
        output_type: OutputType,
    ) -> Result<Self, EquationError> {
        // Create combined JIT function
        let combined_fun = build_combined_function(asts.clone(), equations.len())?;

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
                .map(|ast| *ast.derivative(&var))
                .collect::<Vec<Expr>>();
            let jacobian_fun = build_combined_function(derivative_ast, asts.len())?;
            jacobian_funs.insert(var, jacobian_fun);
        }

        Ok(Self {
            equations,
            asts,
            variable_map: variable_map.clone(),
            sorted_variables: variable_map.keys().sorted().cloned().collect(),
            combined_fun,
            partial_derivatives: jacobian_funs,
            output_type,
        })
    }

    /// Creates abstract syntax trees (ASTs) from a vector of mathematical expressions.
    ///
    /// This internal function parses each expression string into an AST, validates that all
    /// variables used in the expressions exist in the provided variable map, and returns
    /// simplified ASTs ready for compilation.
    ///
    /// # Arguments
    /// * `expressions` - Vector of mathematical expression strings to parse
    /// * `variable_map` - Map of valid variable names to their indices
    ///
    /// # Returns
    /// A vector of simplified ASTs, one for each input expression
    ///
    /// # Errors
    /// Returns `EquationError::VariableNotFound` if an expression uses a variable
    /// that doesn't exist in the variable map
    fn create_asts(
        expressions: &[String],
        variable_map: &HashMap<String, u32>,
    ) -> Result<Vec<Expr>, EquationError> {
        expressions
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
                let ast = build_ast(&node, variable_map)?;
                Ok(*ast.simplify())
            })
            .collect::<Result<Vec<_>, EquationError>>()
    }

    /// Evaluates all equations in the system with the given input values
    /// into a pre-allocated buffer.
    ///
    /// # Arguments
    /// * `inputs` - Input vector implementing the Vector trait
    /// * `results` - Pre-allocated vector to store results
    ///
    /// # Returns
    /// Reference to the results vector containing the evaluated values
    pub fn eval_into<V: Vector, R: Vector>(
        &self,
        inputs: &V,
        results: &mut R,
    ) -> Result<(), EquationError> {
        self.validate_input_length(inputs.as_slice())?;
        if results.len() != self.equations.len() {
            return Err(EquationError::InvalidInputLength {
                expected: self.equations.len(),
                got: results.len(),
            });
        }

        (self.combined_fun)(inputs.as_slice(), results.as_mut_slice());
        Ok(())
    }

    /// Evaluates all equations in the system with the given input values.
    /// Allocates a new vector for results.
    ///
    /// # Arguments
    /// * `inputs` - Input vector implementing the Vector trait
    ///
    /// # Returns
    /// Vector of results, one for each equation in the system
    pub fn eval<V: Vector>(&self, inputs: &V) -> Result<V, EquationError> {
        let mut results = V::zeros(self.equations.len());
        self.eval_into(inputs, &mut results)?;
        Ok(results)
    }

    /// Evaluates all equations in the system with the given input values into a pre-allocated matrix.
    ///
    /// This method is used when the equation system is configured to output a matrix rather than a vector.
    /// The results matrix must have the correct dimensions matching the system's output type.
    ///
    /// # Arguments
    /// * `inputs` - Input vector implementing the Vector trait
    /// * `results` - Pre-allocated matrix to store results
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if the system is not configured for matrix output
    pub fn eval_into_matrix<V: Vector, R: Matrix>(
        &self,
        inputs: &V,
        results: &mut R,
    ) -> Result<(), EquationError> {
        match self.output_type {
            OutputType::Vector => {
                // If the system is not configured to output a matrix, throw an error
                return Err(EquationError::MatrixOutputRequired);
            }
            OutputType::Matrix(n_rows, n_cols) => {
                self.validate_matrix_dimensions(n_rows, n_cols)?;
            }
        }

        (self.combined_fun)(inputs.as_slice(), results.flat_mut_slice());
        Ok(())
    }

    /// Evaluates all equations in the system with the given input values into a new matrix.
    ///
    /// This method allocates a new matrix with the correct dimensions and evaluates the system into it.
    /// It should only be used when the equation system is configured to output a matrix.
    ///
    /// # Arguments
    /// * `inputs` - Input vector implementing the Vector trait
    ///
    /// # Returns
    /// Matrix containing the evaluated results, or an error if the system is not configured for matrix output
    pub fn eval_matrix<V: Vector, R: Matrix>(&self, inputs: &V) -> Result<R, EquationError> {
        match self.output_type {
            OutputType::Vector => Err(EquationError::MatrixOutputRequired),
            OutputType::Matrix(n_rows, n_cols) => {
                let mut results = R::zeros(n_rows, n_cols);
                self.eval_into_matrix(inputs, &mut results)?;
                Ok(results)
            }
        }
    }

    /// Evaluates the equation system in parallel for multiple input sets.
    ///
    /// # Arguments
    /// * `input_sets` - Slice of input vectors, each must match the number of variables
    ///
    /// # Returns
    /// Vector of result vectors, one for each input set
    pub fn eval_parallel<V: Vector + Send + Sync>(
        &self,
        input_sets: &[V],
    ) -> Result<Vec<V>, EquationError> {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        let chunk_size = (input_sets.len() / (num_threads * 4)).max(1);
        let n_equations = self.equations.len();

        // Since we're using Arc, we can efficiently clone the system for parallel processing
        let systems: Vec<_> = (0..num_threads).map(|_| self.clone()).collect();

        Ok(input_sets
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let system = &systems[chunk_idx % systems.len()];
                chunk
                    .iter()
                    .map(|inputs| {
                        let mut results = V::zeros(n_equations);
                        (system.combined_fun)(inputs.as_slice(), results.as_mut_slice());
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
    /// use evalexpr_jit::prelude::*;
    ///
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
        self.partial_derivatives
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
    /// * `variables` - Optional slice of variable names to include in the Jacobian. If None, includes all variables in sorted order.
    ///
    /// # Returns
    /// The Jacobian matrix as a vector of vectors, where each inner vector
    /// contains the partial derivatives of one equation with respect to all variables
    ///
    /// # Example
    /// ```
    /// use evalexpr_jit::prelude::*;
    ///
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y".to_string(),  // f1
    ///     "x*y^2".to_string(),  // f2
    /// ]).unwrap();
    ///
    /// let jacobian = system.eval_jacobian(&[2.0, 3.0], None).unwrap();
    /// // jacobian[0] contains [12.0, 4.0]   // ∂f1/∂x, ∂f1/∂y
    /// // jacobian[1] contains [9.0,  12.0]   // ∂f2/∂x, ∂f2/∂y
    /// ```
    pub fn eval_jacobian(
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
            let fun = self.partial_derivatives.get(var).unwrap();
            let mut derivatives = vec![0.0; n_equations];
            fun(inputs, &mut derivatives);
            for (eq_idx, &value) in derivatives.iter().enumerate() {
                results[eq_idx].push(value);
            }
        }

        Ok(results)
    }

    /// Creates a new equation system that computes the Jacobian matrix with respect to specific variables.
    ///
    /// This method creates a new equation system optimized for computing partial derivatives
    /// with respect to a subset of variables. The resulting system evaluates all equations' derivatives
    /// with respect to the specified variables and arranges them in matrix form.
    ///
    /// # Arguments
    /// * `variables` - Slice of variable names to include in the Jacobian matrix
    ///
    /// # Returns
    /// A new `EquationSystem` that computes the Jacobian matrix when evaluated. The output matrix
    /// has dimensions `[n_equations × n_variables]`, where each row contains the derivatives
    /// of one equation with respect to the specified variables.
    ///
    /// # Errors
    /// Returns `EquationError::VariableNotFound` if any of the specified variables doesn't exist
    /// in the system.
    ///
    /// # Example
    /// ```
    /// use evalexpr_jit::prelude::*;
    /// use ndarray::Array2;
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y + z".to_string(),    // f1
    ///     "x*y^2 - z^2".to_string(),  // f2
    /// ]).unwrap();
    ///
    /// // Create Jacobian system for x and y only
    /// let jacobian_system = system.jacobian_wrt(&["x", "y"]).unwrap();
    ///
    /// // Prepare matrix to store results (2 equations × 2 variables)
    /// let mut results = Array2::zeros((2, 2));
    ///
    /// // Evaluate Jacobian at point (x=2, y=3, z=1)
    /// jacobian_system.eval_into_matrix(&[2.0, 3.0, 1.0], &mut results).unwrap();
    ///
    /// // results now contains:
    /// // [
    /// //   [12.0, 4.0],   // [∂f1/∂x, ∂f1/∂y]
    /// //   [9.0,  12.0],  // [∂f2/∂x, ∂f2/∂y]
    /// // ]
    /// ```
    ///
    /// # Performance Notes
    /// - The returned system is JIT-compiled and optimized for repeated evaluations
    /// - Pre-allocate the results matrix and reuse it for better performance
    /// - The matrix dimensions will be `[n_equations × n_variables]`
    pub fn jacobian_wrt(&self, variables: &[&str]) -> Result<EquationSystem, EquationError> {
        // Verify all variables exist
        for var in variables {
            if !self.variable_map.contains_key(*var) {
                return Err(EquationError::VariableNotFound(var.to_string()));
            }
        }

        let mut asts = vec![];
        for ast in self.asts.iter() {
            for var in variables {
                asts.push(*ast.derivative(var));
            }
        }

        let output_type = OutputType::Matrix(self.num_equations(), variables.len());

        EquationSystem::from_asts(asts, &self.variable_map, output_type)
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
    /// # use ndarray::Array1;
    /// let system = EquationSystem::new(vec![
    ///     "x^2*y".to_string(),  // f1
    ///     "x*y^2".to_string(),  // f2
    /// ]).unwrap();
    ///
    /// let derivatives = system.derive_wrt(&["x", "y"]).unwrap();
    /// let mut results = Array1::zeros(2);
    /// derivatives.eval_into(&vec![2.0, 3.0], &mut results).unwrap();
    /// assert_eq!(results.as_slice().unwrap(), vec![4.0, 6.0]); // ∂²f1/∂x∂y = 2x, ∂²f2/∂x∂y = 2y
    /// ```
    pub fn derive_wrt(&self, variables: &[&str]) -> Result<EquationSystem, EquationError> {
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
                .map(|ast| *ast.derivative(var))
                .collect();
        }

        // Create new system from derivative ASTs
        EquationSystem::from_asts(derivative_asts, &self.variable_map, OutputType::Vector)
    }

    pub fn validate_matrix_dimensions(
        &self,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), EquationError> {
        match self.output_type {
            OutputType::Vector => {
                return Err(EquationError::MatrixOutputRequired);
            }
            OutputType::Matrix(expected_rows, expected_cols) => {
                if n_rows != expected_rows || n_cols != expected_cols {
                    return Err(EquationError::InvalidMatrixDimensions {
                        expected_rows,
                        expected_cols,
                        got_rows: n_rows,
                        got_cols: n_cols,
                    });
                }
            }
        }
        Ok(())
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
        &self.partial_derivatives
    }

    /// Returns the derivative function for a specific variable.
    pub fn gradient_fun(&self, variable: &str) -> &CombinedJITFunction {
        self.partial_derivatives.get(variable).unwrap()
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
        // Since we're using Arc instead of Box, we need to recreate the system
        // This is less efficient but maintains the Arc-based internal structure
        match Self::build(
            self.asts.clone(),
            self.equations.clone(),
            self.variable_map.clone(),
            self.output_type,
        ) {
            Ok(system) => system,
            Err(_) => panic!("Failed to rebuild EquationSystem during clone"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum OutputType {
    Vector,
    Matrix(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use ndarray::{Array1, Array2};

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
            results.as_slice(),
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
        let results = system.eval(&vec![1.0, 2.0, 3.0])?;

        assert_eq!(
            results.as_slice(),
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

        assert_eq!(results.as_slice(), &[4.0, -1.0]);

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
        let dx = system.derive_wrt(&["x"]).unwrap();
        let mut dx_results = vec![0.0, 0.0];
        dx.eval_into(&[2.0, 3.0], &mut dx_results).unwrap();
        assert_eq!(dx_results, vec![12.0, 9.0]); // d/dx[x^2*y] = 2xy, d/dx[x*y^2] = y^2

        // Second order derivative
        let dxy = system.derive_wrt(&["x", "y"]).unwrap();
        let mut dxy_results = vec![0.0, 0.0];
        dxy.eval_into(&[2.0, 3.0], &mut dxy_results).unwrap();
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

        let jacobian = system.eval_jacobian(&[2.0, 3.0], None)?;

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
        let jacobian_fn = system.jacobian_wrt(&["x", "y"]).unwrap();
        let mut results = Array2::zeros((2, 2));
        jacobian_fn
            .eval_into_matrix(&vec![2.0, 3.0, 1.0], &mut results)
            .unwrap();

        // Expected derivatives:
        // ∂f1/∂x = 2xy = 2(2)(3) = 12
        // ∂f1/∂y = x^2 = 4
        // ∂f2/∂x = y^2 = 9
        // ∂f2/∂y = 2xy = 2(2)(3) = 12
        assert_eq!(results[[0, 0]], 12.0); // ∂f1/∂x
        assert_eq!(results[[0, 1]], 4.0); // ∂f1/∂y
        assert_eq!(results[[1, 0]], 9.0); // ∂f2/∂x
        assert_eq!(results[[1, 1]], 12.0); // ∂f2/∂y

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
        let mut results = Array2::zeros((2, 1));
        jacobian_fn
            .eval_into_matrix(&vec![2.0, 3.0], &mut results)
            .unwrap();

        // Expected derivatives:
        // ∂f1/∂x = 2xy = 2(2)(3) = 12
        // ∂f2/∂x = y^2 = 9
        assert_eq!(results[[0, 0]], 12.0); // [∂f1/∂x]
        assert_eq!(results[[1, 0]], 9.0); // [∂f2/∂x]

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
        let mut results = Array2::zeros((2, 3));
        jacobian_fn
            .eval_into_matrix(&vec![2.0, 3.0, 1.0], &mut results)
            .unwrap();

        // Expected derivatives:
        // ∂f1/∂x = 2xy = 2(2)(3) = 12
        // ∂f1/∂y = x^2 = 4
        // ∂f1/∂z = 1
        // ∂f2/∂x = y^2 = 9
        // ∂f2/∂y = 2xy = 2(2)(3) = 12
        // ∂f2/∂z = -2z = -2
        assert_eq!(results[[0, 0]], 12.0); // ∂f1/∂x
        assert_eq!(results[[0, 1]], 4.0); // ∂f1/∂y
        assert_eq!(results[[0, 2]], 1.0); // ∂f1/∂z
        assert_eq!(results[[1, 0]], 9.0); // ∂f2/∂x
        assert_eq!(results[[1, 1]], 12.0); // ∂f2/∂y
        assert_eq!(results[[1, 2]], -2.0); // ∂f2/∂z

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
        let mut results = Array2::zeros((2, 2));

        // First evaluation
        jacobian_fn
            .eval_into_matrix(&vec![2.0, 3.0], &mut results)
            .unwrap();
        assert_eq!(results[[0, 0]], 12.0);
        assert_eq!(results[[0, 1]], 4.0);
        assert_eq!(results[[1, 0]], 9.0);
        assert_eq!(results[[1, 1]], 12.0);

        // Reuse buffer for second evaluation
        jacobian_fn
            .eval_into_matrix(&vec![1.0, 2.0], &mut results)
            .unwrap();
        assert_eq!(results[[0, 0]], 4.0);
        assert_eq!(results[[0, 1]], 1.0);
        assert_eq!(results[[1, 0]], 4.0);
        assert_eq!(results[[1, 1]], 4.0);

        Ok(())
    }

    #[test]
    fn test_different_vector_types() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec!["x^2*y".to_string(), "x*y^2".to_string()])?;

        // Test with Vec<f64>
        let vec_inputs = vec![2.0, 3.0];
        let vec_results = system.eval(&vec_inputs)?;
        assert_eq!(vec_results.as_slice(), &[12.0, 18.0]);

        // Test with ndarray::Array1
        let ndarray_inputs = Array1::from_vec(vec![2.0, 3.0]);
        let ndarray_results = system.eval(&ndarray_inputs)?;
        assert_eq!(ndarray_results.as_slice().unwrap(), &[12.0, 18.0]);

        // Test with nalgebra::DVector
        let nalgebra_inputs = DVector::from_vec(vec![2.0, 3.0]);
        let nalgebra_results = system.eval(&nalgebra_inputs)?;
        assert_eq!(nalgebra_results.as_slice(), &[12.0, 18.0]);

        Ok(())
    }

    #[test]
    fn test_eval_parallel() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec!["x^2*y".to_string(), "x*y^2".to_string()])?;

        let input_sets = vec![
            vec![2.0, 3.0],
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![0.0, 1.0],
        ];

        let results = system.eval_parallel(&input_sets)?;

        assert_eq!(results[0].as_slice(), &[12.0, 18.0]); // [2^2 * 3, 2 * 3^2]
        assert_eq!(results[1].as_slice(), &[2.0, 4.0]); // [1^2 * 2, 1 * 2^2]
        assert_eq!(results[2].as_slice(), &[36.0, 48.0]); // [3^2 * 4, 3 * 4^2]
        assert_eq!(results[3].as_slice(), &[0.0, 0.0]); // [0^2 * 1, 0 * 1^2]

        Ok(())
    }

    #[test]
    fn test_eval_into() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec!["x^2*y".to_string(), "x*y^2".to_string()])?;

        // Test with Vec<f64>
        let mut results = vec![0.0; 2];
        system.eval_into(&vec![2.0, 3.0], &mut results)?;
        assert_eq!(results, vec![12.0, 18.0]);

        // Test with ndarray
        let mut ndarray_results = Array1::zeros(2);
        system.eval_into(&Array1::from_vec(vec![2.0, 3.0]), &mut ndarray_results)?;
        assert_eq!(ndarray_results.as_slice().unwrap(), &[12.0, 18.0]);

        // Test error case: wrong buffer size
        let mut wrong_size = vec![0.0; 3];
        assert!(matches!(
            system.eval_into(&vec![2.0, 3.0], &mut wrong_size),
            Err(EquationError::InvalidInputLength { .. })
        ));

        Ok(())
    }

    #[test]
    fn test_matrix_output_errors() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec!["x^2*y".to_string(), "x*y^2".to_string()])?;

        // Regular vector system should error when trying to output as matrix
        let mut results = Array2::zeros((2, 2));
        assert!(matches!(
            system.eval_into_matrix(&vec![2.0, 3.0], &mut results),
            Err(EquationError::MatrixOutputRequired)
        ));

        assert!(matches!(
            system.eval_matrix::<_, Array2<f64>>(&vec![2.0, 3.0]),
            Err(EquationError::MatrixOutputRequired)
        ));

        Ok(())
    }

    #[test]
    fn test_gradient() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec![
            "x^2*y + z".to_string(),   // f1
            "x*y^2 - z^2".to_string(), // f2
        ])?;

        // Test gradient with respect to x
        let dx = system.gradient(&[2.0, 3.0, 1.0], "x")?;
        assert_eq!(dx, vec![12.0, 9.0]); // [∂f1/∂x = 2xy, ∂f2/∂x = y^2]

        // Test gradient with respect to y
        let dy = system.gradient(&[2.0, 3.0, 1.0], "y")?;
        assert_eq!(dy, vec![4.0, 12.0]); // [∂f1/∂y = x^2, ∂f2/∂y = 2xy]

        // Test gradient with respect to z
        let dz = system.gradient(&[2.0, 3.0, 1.0], "z")?;
        assert_eq!(dz, vec![1.0, -2.0]); // [∂f1/∂z = 1, ∂f2/∂z = -2z]

        // Test error case: invalid variable
        let result = system.gradient(&[2.0, 3.0, 1.0], "w");
        assert!(matches!(result, Err(EquationError::VariableNotFound(_))));

        // Test error case: wrong input length
        let result = system.gradient(&[2.0, 3.0], "x");
        assert!(matches!(
            result,
            Err(EquationError::InvalidInputLength { .. })
        ));

        Ok(())
    }

    #[test]
    fn test_eval_matrix_on_vector_system() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec!["x^2*y".to_string(), "x*y^2".to_string()])?;

        // Attempt to evaluate as matrix should fail
        let mut results = Array2::zeros((2, 2));
        assert!(matches!(
            system.eval_into_matrix(&vec![2.0, 3.0], &mut results),
            Err(EquationError::MatrixOutputRequired)
        ));

        // Direct matrix evaluation should also fail
        assert!(matches!(
            system.eval_matrix::<_, Array2<f64>>(&vec![2.0, 3.0]),
            Err(EquationError::MatrixOutputRequired)
        ));

        Ok(())
    }

    #[test]
    fn test_getters() -> Result<(), Box<dyn std::error::Error>> {
        let system = EquationSystem::new(vec!["x^2*y".to_string(), "x*y^2".to_string()])?;

        // Test sorted_variables()
        assert_eq!(system.sorted_variables(), &["x", "y"]);

        // Test variables()
        let var_map = system.variables();
        assert_eq!(var_map.get("x"), Some(&0));
        assert_eq!(var_map.get("y"), Some(&1));

        // Test equations()
        assert_eq!(system.equations(), &["x^2*y", "x*y^2"]);

        // Test fun() returns a valid function
        let fun = system.fun();
        let mut results = vec![0.0; 2];
        fun(&[2.0, 3.0], &mut results);
        assert_eq!(results, vec![12.0, 18.0]);

        // Test jacobian_funs() returns valid derivative functions
        let jacobian_funs = system.jacobian_funs();
        assert!(jacobian_funs.contains_key("x"));
        assert!(jacobian_funs.contains_key("y"));

        // Test gradient_fun() returns valid derivative function
        let dx_fun = system.gradient_fun("x");
        let mut dx_results = vec![0.0; 2];
        dx_fun(&[2.0, 3.0], &mut dx_results);
        assert_eq!(dx_results, vec![12.0, 9.0]); // [∂(x^2*y)/∂x, ∂(x*y^2)/∂x]

        // Test num_equations()
        assert_eq!(system.num_equations(), 2);

        Ok(())
    }
}
