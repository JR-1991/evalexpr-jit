//! Expression module for representing mathematical expressions.
//!
//! This module defines the core expression types used to represent mathematical expressions
//! in a form that supports both JIT compilation and automatic differentiation. The main types are:
//!
//! - `Expr`: An enum representing different kinds of mathematical expressions
//! - `VarRef`: A struct containing metadata about variables in expressions
//!
//! The expression tree is built recursively using `Box<Expr>` for nested expressions and can be:
//! - JIT compiled into machine code using Cranelift
//! - Symbolically differentiated to compute derivatives
//! - Evaluated efficiently at runtime
//! - Simplified using algebraic rules
//! - Modified by inserting replacement expressions
//!
//! Supported operations include:
//! - Basic arithmetic (+, -, *, /)
//! - Variables and constants
//! - Absolute value
//! - Integer exponentiation
//! - Transcendental functions (exp, ln, sqrt)
//! - Expression caching for optimization
//!
//! # Expression Tree Structure
//! The expression tree is built recursively with each node being one of:
//! - Leaf nodes: Constants and Variables
//! - Unary operations: Abs, Neg, Exp, Ln, Sqrt
//! - Binary operations: Add, Sub, Mul, Div
//! - Special nodes: Pow (with integer exponent), Cached expressions
//!
//! # Symbolic Differentiation
//! The derivative method implements symbolic differentiation by recursively applying
//! calculus rules like:
//! - Product rule
//! - Quotient rule
//! - Chain rule
//! - Power rule
//! - Special function derivatives (exp, ln, sqrt)
//!
//! # Expression Simplification
//! The simplify method performs algebraic simplifications including:
//! - Constant folding (e.g. 2 + 3 → 5)
//! - Identity rules (e.g. x + 0 → x, x * 1 → x)
//! - Exponent rules (e.g. x^0 → 1, x^1 → x)
//! - Expression caching for repeated subexpressions
//! - Special function simplifications
//!
//! # Expression Modification
//! The insert method allows replacing parts of an expression tree by:
//! - Matching nodes using a predicate function
//! - Replacing matched nodes with a new expression
//! - Recursively traversing and rebuilding the tree

use cranelift::prelude::*;
use cranelift_codegen::ir::{immediates::Offset32, Value};
use cranelift_module::Module;

use crate::{errors::EquationError, operators};

/// Represents a reference to a variable in an expression.
///
/// Contains metadata needed to generate code that loads the variable's value:
/// - The variable's name as a string
/// - A Cranelift Value representing the pointer to the input array
/// - The variable's index in the input array
#[derive(Debug, Clone, PartialEq)]
pub struct VarRef {
    pub name: String,
    pub vec_ref: Value,
    pub index: u32,
}

/// An expression tree node representing mathematical operations.
///
/// This enum represents different types of mathematical expressions that can be:
/// - JIT compiled into machine code using Cranelift
/// - Symbolically differentiated to compute derivatives
/// - Simplified using algebraic rules
/// - Modified by inserting replacement expressions
///
/// The expression tree is built recursively using Box<Expr> for nested expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A constant floating point value
    Const(f64),
    /// A reference to a variable
    Var(VarRef),
    /// Addition of two expressions
    Add(Box<Expr>, Box<Expr>),
    /// Multiplication of two expressions
    Mul(Box<Expr>, Box<Expr>),
    /// Subtraction of two expressions
    Sub(Box<Expr>, Box<Expr>),
    /// Division of two expressions
    Div(Box<Expr>, Box<Expr>),
    /// Absolute value of an expression
    Abs(Box<Expr>),
    /// Exponentiation of an expression by an integer constant
    Pow(Box<Expr>, i64),
    /// Exponential function of an expression
    Exp(Box<Expr>),
    /// Natural logarithm of an expression
    Ln(Box<Expr>),
    /// Square root of an expression
    Sqrt(Box<Expr>),
    /// Negation of an expression
    Neg(Box<Expr>),
    /// Cached expression with optional pre-computed value
    Cached(Box<Expr>, Option<f64>),
}

impl Expr {
    /// Generates Cranelift IR code for this expression.
    ///
    /// Recursively traverses the expression tree and generates the appropriate
    /// Cranelift instructions to compute the expression's value at runtime.
    ///
    /// # Arguments
    /// * `builder` - The Cranelift FunctionBuilder to emit instructions into
    /// * `module` - The Cranelift Module to link functions
    ///
    /// # Returns
    /// A Cranelift Value representing the result of this expression
    ///
    /// # Errors
    /// Returns an EquationError if code generation fails
    pub fn codegen(
        &self,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
    ) -> Result<Value, EquationError> {
        match self {
            Expr::Const(val) => Ok(builder.ins().f64const(*val)),
            Expr::Var(VarRef { vec_ref, index, .. }) => {
                let ptr = *vec_ref;
                let offset = *index as i32 * 8;
                Ok(builder
                    .ins()
                    .load(types::F64, MemFlags::new(), ptr, Offset32::new(offset)))
            }
            Expr::Add(left, right) => {
                // Recursively generate code for both sides
                let lhs = left.codegen(builder, module)?; // This could be another nested expression
                let rhs = right.codegen(builder, module)?; // This could be another nested expression
                Ok(builder.ins().fadd(lhs, rhs))
            }
            Expr::Mul(left, right) => {
                let lhs = left.codegen(builder, module)?;
                let rhs = right.codegen(builder, module)?;
                Ok(builder.ins().fmul(lhs, rhs))
            }
            Expr::Sub(left, right) => {
                let lhs = left.codegen(builder, module)?;
                let rhs = right.codegen(builder, module)?;
                Ok(builder.ins().fsub(lhs, rhs))
            }
            Expr::Div(left, right) => {
                let lhs = left.codegen(builder, module)?;
                let rhs = right.codegen(builder, module)?;
                Ok(builder.ins().fdiv(lhs, rhs))
            }
            Expr::Abs(expr) => {
                let expr = expr.codegen(builder, module)?;
                Ok(builder.ins().fabs(expr))
            }
            Expr::Neg(expr) => {
                let expr = expr.codegen(builder, module)?;
                Ok(builder.ins().fneg(expr))
            }
            Expr::Pow(base, exp) => {
                let base_val = base.codegen(builder, module)?;
                match *exp {
                    0 => Ok(builder.ins().f64const(1.0)),
                    1 => Ok(base_val),
                    2 => Ok(builder.ins().fmul(base_val, base_val)), // Special case for squares
                    3 => {
                        // Special case for cubes
                        let square = builder.ins().fmul(base_val, base_val);
                        Ok(builder.ins().fmul(square, base_val))
                    }
                    exp => {
                        let mut result = builder.ins().f64const(1.0);
                        let mut base = base_val;
                        let mut n = exp.abs();

                        while n > 1 {
                            if n & 1 == 1 {
                                result = builder.ins().fmul(result, base);
                            }
                            base = builder.ins().fmul(base, base);
                            n >>= 1;
                        }
                        if n == 1 {
                            result = builder.ins().fmul(result, base);
                        }

                        if exp < 0 {
                            let one = builder.ins().f64const(1.0);
                            Ok(builder.ins().fdiv(one, result))
                        } else {
                            Ok(result)
                        }
                    }
                }
            }
            Expr::Exp(expr) => {
                let arg = expr.codegen(builder, module)?;
                let func_id = operators::exp::link_exp(module).unwrap();
                Ok(operators::exp::call_exp(builder, module, func_id, arg))
            }
            Expr::Ln(expr) => {
                let arg = expr.codegen(builder, module)?;
                let func_id = operators::ln::link_ln(module).unwrap();
                Ok(operators::ln::call_ln(builder, module, func_id, arg))
            }
            Expr::Sqrt(expr) => {
                let arg = expr.codegen(builder, module)?;
                let func_id = operators::sqrt::link_sqrt(module).unwrap();
                Ok(operators::sqrt::call_sqrt(builder, module, func_id, arg))
            }
            Expr::Cached(expr, cached_value) => {
                if let Some(val) = cached_value {
                    Ok(builder.ins().f64const(*val))
                } else {
                    expr.codegen(builder, module)
                }
            }
        }
    }

    /// Computes the symbolic derivative of this expression with respect to a variable.
    ///
    /// Recursively applies the rules of differentiation to build a new expression tree
    /// representing the derivative. The rules implemented are:
    /// - d/dx(c) = 0 for constants
    /// - d/dx(x) = 1 for the variable we're differentiating with respect to
    /// - d/dx(y) = 0 for other variables
    /// - Sum rule: d/dx(f + g) = df/dx + dg/dx
    /// - Product rule: d/dx(f * g) = f * dg/dx + g * df/dx
    /// - Quotient rule: d/dx(f/g) = (g * df/dx - f * dg/dx) / g^2
    /// - Chain rule for abs: d/dx|f| = f/|f| * df/dx
    /// - Power rule: d/dx(f^n) = n * f^(n-1) * df/dx
    /// - Chain rule for exp: d/dx(e^f) = e^f * df/dx
    /// - Chain rule for ln: d/dx(ln(f)) = 1/f * df/dx
    /// - Chain rule for sqrt: d/dx(sqrt(f)) = 1/(2*sqrt(f)) * df/dx
    /// - Negation: d/dx(-f) = -(df/dx)
    ///
    /// # Arguments
    /// * `with_respect_to` - The name of the variable to differentiate with respect to
    ///
    /// # Returns
    /// A new expression tree representing the derivative
    pub fn derivative(&self, with_respect_to: &str) -> Box<Expr> {
        match self {
            Expr::Const(_) => Box::new(Expr::Const(0.0)),

            Expr::Var(var_ref) => {
                if var_ref.name == with_respect_to {
                    Box::new(Expr::Const(1.0))
                } else {
                    Box::new(Expr::Const(0.0))
                }
            }

            Expr::Add(left, right) => {
                // d/dx(f + g) = df/dx + dg/dx
                Box::new(Expr::Add(
                    left.derivative(with_respect_to),
                    right.derivative(with_respect_to),
                ))
            }

            Expr::Sub(left, right) => {
                // d/dx(f - g) = df/dx - dg/dx
                Box::new(Expr::Sub(
                    left.derivative(with_respect_to),
                    right.derivative(with_respect_to),
                ))
            }

            Expr::Mul(left, right) => {
                // d/dx(f * g) = f * dg/dx + g * df/dx
                Box::new(Expr::Add(
                    Box::new(Expr::Mul(left.clone(), right.derivative(with_respect_to))),
                    Box::new(Expr::Mul(right.clone(), left.derivative(with_respect_to))),
                ))
            }

            Expr::Div(left, right) => {
                // d/dx(f/g) = (g * df/dx - f * dg/dx) / g^2
                Box::new(Expr::Div(
                    Box::new(Expr::Sub(
                        Box::new(Expr::Mul(right.clone(), left.derivative(with_respect_to))),
                        Box::new(Expr::Mul(left.clone(), right.derivative(with_respect_to))),
                    )),
                    Box::new(Expr::Pow(right.clone(), 2)),
                ))
            }

            Expr::Abs(expr) => {
                // d/dx|f| = f/|f| * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Div(expr.clone(), Box::new(Expr::Abs(expr.clone())))),
                    expr.derivative(with_respect_to),
                ))
            }

            Expr::Pow(base, exp) => {
                // d/dx(f^n) = n * f^(n-1) * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(*exp as f64)),
                        Box::new(Expr::Pow(base.clone(), exp - 1)),
                    )),
                    base.derivative(with_respect_to),
                ))
            }

            Expr::Exp(expr) => {
                // d/dx(e^f) = e^f * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Exp(expr.clone())),
                    expr.derivative(with_respect_to),
                ))
            }

            Expr::Ln(expr) => {
                // d/dx(ln(f)) = 1/f * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Div(Box::new(Expr::Const(1.0)), expr.clone())),
                    expr.derivative(with_respect_to),
                ))
            }

            Expr::Sqrt(expr) => {
                // d/dx(sqrt(f)) = 1/(2*sqrt(f)) * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Div(
                        Box::new(Expr::Const(1.0)),
                        Box::new(Expr::Sqrt(expr.clone())),
                    )),
                    expr.derivative(with_respect_to),
                ))
            }

            Expr::Neg(expr) => {
                // d/dx(-f) = -(df/dx)
                Box::new(Expr::Neg(expr.derivative(with_respect_to)))
            }

            Expr::Cached(expr, _) => expr.derivative(with_respect_to),
        }
    }

    /// Simplifies the expression by folding constants and applying basic algebraic rules.
    ///
    /// This method performs several types of algebraic simplifications:
    ///
    /// # Constant Folding
    /// - Evaluates constant expressions: 2 + 3 → 5
    /// - Simplifies operations with special constants: x * 0 → 0
    ///
    /// # Identity Rules
    /// - Additive identity: x + 0 → x
    /// - Multiplicative identity: x * 1 → x
    /// - Division identity: x / 1 → x
    /// - Division by self: x / x → 1
    ///
    /// # Exponent Rules
    /// - Zero exponent: x^0 → 1 (except when x = 0)
    /// - First power: x^1 → x
    /// - Nested exponents: (x^a)^b → x^(a*b)
    ///
    /// # Special Function Simplification
    /// - Absolute value: |-3| → 3, ||x|| → |x|
    /// - Double negation: -(-x) → x
    /// - Evaluates constant special functions: ln(1) → 0
    ///
    /// # Expression Caching
    /// - Caches repeated subexpressions to avoid redundant computation
    /// - Preserves existing cached values
    ///
    /// # Returns
    /// A new simplified expression tree
    pub fn simplify(&self) -> Box<Expr> {
        match self {
            // Base cases - constants and variables remain unchanged
            Expr::Const(_) | Expr::Var(_) => Box::new(self.clone()),

            Expr::Add(left, right) => {
                let l = left.simplify();
                let r = right.simplify();
                match (&*l, &*r) {
                    // Fold constants: 1 + 2 -> 3
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a + b)),
                    // Identity: x + 0 -> x
                    (expr, Expr::Const(0.0)) | (Expr::Const(0.0), expr) => Box::new(expr.clone()),
                    // Combine like terms: c1*x + c2*x -> (c1+c2)*x
                    (Expr::Mul(a1, x1), Expr::Mul(a2, x2)) if x1 == x2 => Box::new(Expr::Mul(
                        Expr::Add(a1.clone(), a2.clone()).simplify(),
                        x1.clone(),
                    )),
                    _ => Box::new(Expr::Add(l, r)),
                }
            }

            Expr::Sub(left, right) => {
                let l = left.simplify();
                let r = right.simplify();
                match (&*l, &*r) {
                    // Fold constants: 3 - 2 -> 1
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a - b)),
                    // Identity: x - 0 -> x
                    (expr, Expr::Const(0.0)) => Box::new(expr.clone()),
                    // Zero: x - x -> 0
                    (a, b) if a == b => Box::new(Expr::Const(0.0)),
                    // Combine like terms: c1*x - c2*x -> (c1-c2)*x
                    (Expr::Mul(a1, x1), Expr::Mul(a2, x2)) if x1 == x2 => Box::new(Expr::Mul(
                        Expr::Sub(a1.clone(), a2.clone()).simplify(),
                        x1.clone(),
                    )),
                    _ => Box::new(Expr::Sub(l, r)),
                }
            }

            Expr::Mul(left, right) => {
                let l = left.simplify();
                let r = right.simplify();

                // If we see the same subexpression multiple times, cache it
                if l == r {
                    let cached = Box::new(Expr::Cached(l.clone(), None));
                    return Box::new(Expr::Mul(cached.clone(), cached));
                }

                match (&*l, &*r) {
                    // Fold constants: 2 * 3 -> 6
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a * b)),
                    // Zero property: x * 0 -> 0
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Box::new(Expr::Const(0.0)),
                    // Identity: x * 1 -> x
                    (expr, Expr::Const(1.0)) | (Expr::Const(1.0), expr) => Box::new(expr.clone()),
                    // Combine exponents: x^a * x^b -> x^(a+b)
                    (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) if b1 == b2 => {
                        Box::new(Expr::Pow(b1.clone(), e1 + e2))
                    }
                    _ => Box::new(Expr::Mul(l, r)),
                }
            }

            Expr::Div(left, right) => {
                let l = left.simplify();
                let r = right.simplify();
                match (&*l, &*r) {
                    // Fold constants: 6 / 2 -> 3
                    (Expr::Const(a), Expr::Const(b)) if *b != 0.0 => Box::new(Expr::Const(a / b)),
                    // Zero numerator: 0 / x -> 0
                    (Expr::Const(0.0), _) => Box::new(Expr::Const(0.0)),
                    // Identity: x / 1 -> x
                    (expr, Expr::Const(1.0)) => Box::new(expr.clone()),
                    // Identity: x / x -> 1
                    (a, b) if a == b => Box::new(Expr::Const(1.0)),
                    // Simplify exponents: x^a / x^b -> x^(a-b)
                    (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) if b1 == b2 => {
                        Box::new(Expr::Pow(b1.clone(), e1 - e2))
                    }
                    _ => Box::new(Expr::Div(l, r)),
                }
            }

            Expr::Abs(expr) => {
                let e = expr.simplify();
                match &*e {
                    // Fold constants: abs(3) -> 3
                    Expr::Const(a) => Box::new(Expr::Const(a.abs())),
                    // Nested abs: abs(abs(x)) -> abs(x)
                    Expr::Abs(inner) => Box::new(Expr::Abs(inner.clone())),
                    _ => Box::new(Expr::Abs(e)),
                }
            }

            Expr::Pow(base, exp) => {
                let b = base.simplify();
                match (&*b, exp) {
                    // Fold constants: 2^3 -> 8
                    (Expr::Const(a), exp) => Box::new(Expr::Const(a.powi(*exp as i32))),
                    // x^0 -> 1
                    (base, 0) if !matches!(base, Expr::Const(0.0)) => Box::new(Expr::Const(1.0)),
                    // Identity: x^1 -> x
                    (expr, 1) => Box::new(expr.clone()),
                    // Nested exponents: (x^a)^b -> x^(a*b)
                    (Expr::Pow(inner_base, inner_exp), outer_exp) => {
                        Box::new(Expr::Pow(inner_base.clone(), inner_exp * outer_exp))
                    }
                    _ => Box::new(Expr::Pow(b, *exp)),
                }
            }

            Expr::Exp(expr) => {
                let e = expr.simplify();
                match &*e {
                    Expr::Const(a) => Box::new(Expr::Const(a.exp())),
                    _ => Box::new(Expr::Exp(e)),
                }
            }

            Expr::Ln(expr) => {
                let e = expr.simplify();
                match &*e {
                    Expr::Const(a) => Box::new(Expr::Const(a.ln())),
                    _ => Box::new(Expr::Ln(e)),
                }
            }

            Expr::Sqrt(expr) => {
                let e = expr.simplify();
                match &*e {
                    Expr::Const(a) => Box::new(Expr::Const(a.sqrt())),
                    _ => Box::new(Expr::Sqrt(e)),
                }
            }

            Expr::Neg(expr) => {
                let e = expr.simplify();
                match &*e {
                    // Fold constants: -3 -> -3
                    Expr::Const(a) => Box::new(Expr::Const(-a)),
                    // Double negation: -(-x) -> x
                    Expr::Neg(inner) => inner.clone(),
                    _ => Box::new(Expr::Neg(e)),
                }
            }

            Expr::Cached(expr, cached_value) => {
                if cached_value.is_some() {
                    Box::new(self.clone())
                } else {
                    expr.simplify()
                }
            }
        }
    }

    /// Inserts an expression by replacing nodes that match a predicate.
    ///
    /// Recursively traverses the expression tree and replaces any nodes that match
    /// the given predicate with the replacement expression. This allows for targeted
    /// modifications of the expression tree.
    ///
    /// # Arguments
    /// * `predicate` - A closure that determines which nodes to replace
    /// * `replacement` - The expression to insert where the predicate matches
    ///
    /// # Returns
    /// A new expression tree with the replacements applied
    pub fn insert<F>(&self, predicate: F, replacement: &Expr) -> Box<Expr>
    where
        F: Fn(&Expr) -> bool + Clone,
    {
        if predicate(self) {
            Box::new(replacement.clone())
        } else {
            match self {
                Expr::Const(_) | Expr::Var(_) => Box::new(self.clone()),
                Expr::Add(left, right) => Box::new(Expr::Add(
                    left.insert(predicate.clone(), replacement),
                    right.insert(predicate, replacement),
                )),
                Expr::Mul(left, right) => Box::new(Expr::Mul(
                    left.insert(predicate.clone(), replacement),
                    right.insert(predicate, replacement),
                )),
                Expr::Sub(left, right) => Box::new(Expr::Sub(
                    left.insert(predicate.clone(), replacement),
                    right.insert(predicate, replacement),
                )),
                Expr::Div(left, right) => Box::new(Expr::Div(
                    left.insert(predicate.clone(), replacement),
                    right.insert(predicate, replacement),
                )),
                Expr::Abs(expr) => Box::new(Expr::Abs(expr.insert(predicate, replacement))),
                Expr::Pow(base, exp) => {
                    Box::new(Expr::Pow(base.insert(predicate, replacement), *exp))
                }
                Expr::Exp(expr) => Box::new(Expr::Exp(expr.insert(predicate, replacement))),
                Expr::Ln(expr) => Box::new(Expr::Ln(expr.insert(predicate, replacement))),
                Expr::Sqrt(expr) => Box::new(Expr::Sqrt(expr.insert(predicate, replacement))),
                Expr::Neg(expr) => Box::new(Expr::Neg(expr.insert(predicate, replacement))),
                Expr::Cached(expr, _) => {
                    Box::new(Expr::Cached(expr.insert(predicate, replacement), None))
                }
            }
        }
    }
}

/// Implements string formatting for expressions.
///
/// This implementation converts expressions to their standard mathematical notation:
/// - Constants are formatted as numbers
/// - Variables are formatted as their names
/// - Binary operations (+,-,*,/) are wrapped in parentheses
/// - Functions (exp, ln, sqrt) use function call notation
/// - Absolute value uses |x| notation
/// - Exponents use ^
/// - Negation uses - prefix
/// - Cached expressions display their underlying expression
impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Const(val) => write!(f, "{}", val),
            Expr::Var(var_ref) => write!(f, "{}", var_ref.name),
            Expr::Add(left, right) => write!(f, "({} + {})", left, right),
            Expr::Mul(left, right) => write!(f, "({} * {})", left, right),
            Expr::Sub(left, right) => write!(f, "({} - {})", left, right),
            Expr::Div(left, right) => write!(f, "({} / {})", left, right),
            Expr::Abs(expr) => write!(f, "|{}|", expr),
            Expr::Pow(base, exp) => write!(f, "({}^{})", base, exp),
            Expr::Exp(expr) => write!(f, "exp({})", expr),
            Expr::Ln(expr) => write!(f, "ln({})", expr),
            Expr::Sqrt(expr) => write!(f, "sqrt({})", expr),
            Expr::Neg(expr) => write!(f, "-({})", expr),
            Expr::Cached(expr, _) => write!(f, "{}", expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a variable
    fn var(name: &str) -> Box<Expr> {
        Box::new(Expr::Var(VarRef {
            name: name.to_string(),
            vec_ref: Value::from_u32(0),
            index: 0,
        }))
    }

    #[test]
    fn test_simplify() {
        // Helper function to create a variable
        fn var(name: &str) -> Box<Expr> {
            Box::new(Expr::Var(VarRef {
                name: name.to_string(),
                vec_ref: Value::from_u32(0), // Dummy value for testing
                index: 0,
            }))
        }

        // Test constant folding
        // 2 + 3 → 5
        assert_eq!(
            *Expr::Add(Box::new(Expr::Const(2.0)), Box::new(Expr::Const(3.0))).simplify(),
            Expr::Const(5.0)
        );

        // Test additive identity
        // x + 0 → x
        assert_eq!(
            *Expr::Add(var("x"), Box::new(Expr::Const(0.0))).simplify(),
            *var("x")
        );

        // Test multiplicative identity
        // x * 1 → x
        assert_eq!(
            *Expr::Mul(var("x"), Box::new(Expr::Const(1.0))).simplify(),
            *var("x")
        );

        // Test multiplication by zero
        // x * 0 → 0
        assert_eq!(
            *Expr::Mul(var("x"), Box::new(Expr::Const(0.0))).simplify(),
            Expr::Const(0.0)
        );

        // Test division identity
        // x / 1 → x
        assert_eq!(
            *Expr::Div(var("x"), Box::new(Expr::Const(1.0))).simplify(),
            *var("x")
        );

        // Test division by self
        // x / x → 1
        assert_eq!(*Expr::Div(var("x"), var("x")).simplify(), Expr::Const(1.0));

        // Test exponent simplification
        // x^0 → 1
        assert_eq!(*Expr::Pow(var("x"), 0).simplify(), Expr::Const(1.0));
        // x^1 → x
        assert_eq!(*Expr::Pow(var("x"), 1).simplify(), *var("x"));

        // Test absolute value of constant
        // |-3| → 3
        assert_eq!(
            *Expr::Abs(Box::new(Expr::Const(-3.0))).simplify(),
            Expr::Const(3.0)
        );

        // Test nested absolute value
        // ||x|| → |x|
        assert_eq!(
            *Expr::Abs(Box::new(Expr::Abs(var("x")))).simplify(),
            Expr::Abs(var("x"))
        );
    }

    #[test]
    fn test_insert() {
        // Helper function to create a variable
        fn var(name: &str) -> Box<Expr> {
            Box::new(Expr::Var(VarRef {
                name: name.to_string(),
                vec_ref: Value::from_u32(0),
                index: 0,
            }))
        }

        // Create expression: x + y
        let expr = Box::new(Expr::Add(var("x"), var("y")));

        // Replace all occurrences of 'x' with '2*z'
        let replacement = Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), var("z")));

        let result = expr.insert(|e| matches!(e, Expr::Var(v) if v.name == "x"), &replacement);

        // Expected: (2*z) + y
        assert_eq!(
            *result,
            Expr::Add(
                Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), var("z"),)),
                var("y"),
            )
        );
    }

    #[test]
    fn test_derivative() {
        // Test constant derivative
        assert_eq!(*Expr::Const(5.0).derivative("x"), Expr::Const(0.0));

        // Test variable derivatives (x)' = 1, (y)' = 0
        assert_eq!(*var("x").derivative("x"), Expr::Const(1.0));
        assert_eq!(*var("y").derivative("x"), Expr::Const(0.0));

        // Test sum rule (u+v)' = u' + v'
        let sum = Box::new(Expr::Add(var("x"), var("y")));
        assert_eq!(
            *sum.derivative("x"),
            Expr::Add(Box::new(Expr::Const(1.0)), Box::new(Expr::Const(0.0)))
        );

        // Test product rule (u*v)' = u'*v + u*v'
        let product = Box::new(Expr::Mul(var("x"), var("y")));
        assert_eq!(
            *product.derivative("x"),
            Expr::Add(
                Box::new(Expr::Mul(var("x"), Box::new(Expr::Const(0.0)))),
                Box::new(Expr::Mul(var("y"), Box::new(Expr::Const(1.0))))
            )
        );

        // Test power rule (u^v)' = u'*v*u^(v-1) + ln(u)*u^v*v'
        let power = Box::new(Expr::Pow(var("x"), 3));
        assert_eq!(
            *power.derivative("x"),
            Expr::Mul(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(3.0)),
                    Box::new(Expr::Pow(var("x"), 2))
                )),
                Box::new(Expr::Const(1.0))
            )
        );
    }

    #[test]
    fn test_complex_simplifications() {
        // Test nested operations: (x + 0) * (y + 0) → x * y
        let expr = Box::new(Expr::Mul(
            Box::new(Expr::Add(var("x"), Box::new(Expr::Const(0.0)))),
            Box::new(Expr::Add(var("y"), Box::new(Expr::Const(0.0)))),
        ));
        assert_eq!(*expr.simplify(), Expr::Mul(var("x"), var("y")));

        // Test double negation: -(-x) → x
        let expr = Box::new(Expr::Neg(Box::new(Expr::Neg(var("x")))));
        assert_eq!(*expr.simplify(), *var("x"));

        // Test multiplication by 1: (1 * x) * (y * 1) → x * y
        let expr = Box::new(Expr::Mul(
            Box::new(Expr::Mul(Box::new(Expr::Const(1.0)), var("x"))),
            Box::new(Expr::Mul(var("y"), Box::new(Expr::Const(1.0)))),
        ));
        assert_eq!(*expr.simplify(), Expr::Mul(var("x"), var("y")));

        // Test division simplification: (x/y)/(x/y) → 1
        let div = Box::new(Expr::Div(var("x"), var("y")));
        let expr = Box::new(Expr::Div(div.clone(), div));
        assert_eq!(*expr.simplify(), Expr::Const(1.0));
    }

    #[test]
    fn test_special_functions() {
        // Test abs(abs(x)) simplification to abs(x)
        let expr = Box::new(Expr::Abs(Box::new(Expr::Abs(var("x")))));
        assert_eq!(*expr.simplify(), Expr::Abs(var("x")));

        // Test sqrt(x^2) - should not simplify as it's not always equal to x
        let expr = Box::new(Expr::Sqrt(Box::new(Expr::Pow(var("x"), 2))));
        assert_eq!(*expr.simplify(), *expr);

        // Test constant special functions
        // exp(0) = 1
        assert_eq!(
            *Expr::Exp(Box::new(Expr::Const(0.0))).simplify(),
            Expr::Const(1.0)
        );
        // ln(1) = 0
        assert_eq!(
            *Expr::Ln(Box::new(Expr::Const(1.0))).simplify(),
            Expr::Const(0.0)
        );
    }

    #[test]
    fn test_display() {
        // Test basic expressions
        assert_eq!(format!("{}", Expr::Const(5.0)), "5");
        assert_eq!(format!("{}", *var("x")), "x");

        // Test binary operations
        let sum = Expr::Add(var("x"), var("y"));
        assert_eq!(format!("{}", sum), "(x + y)");

        let product = Expr::Mul(var("x"), var("y"));
        assert_eq!(format!("{}", product), "(x * y)");

        // Test special functions
        let exp = Expr::Exp(var("x"));
        assert_eq!(format!("{}", exp), "exp(x)");

        let abs = Expr::Abs(var("x"));
        assert_eq!(format!("{}", abs), "|x|");

        // Test complex expression
        let complex = Expr::Div(
            Box::new(Expr::Add(Box::new(Expr::Pow(var("x"), 2)), var("y"))),
            var("z"),
        );
        assert_eq!(format!("{}", complex), "(((x^2) + y) / z)");
    }

    #[test]
    fn test_cached_expressions() {
        // Test cached constant
        let cached = Box::new(Expr::Cached(Box::new(Expr::Const(5.0)), Some(5.0)));
        assert_eq!(*cached.simplify(), *cached);

        // Test uncached expression simplification
        let uncached = Box::new(Expr::Cached(
            Box::new(Expr::Add(
                Box::new(Expr::Const(2.0)),
                Box::new(Expr::Const(3.0)),
            )),
            None,
        ));
        assert_eq!(*uncached.simplify(), Expr::Const(5.0));
    }
}
