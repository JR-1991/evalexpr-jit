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

use crate::{errors::EquationError, operators, opt};

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
    /// Exponentiation of an expression by a floating point constant
    PowFloat(Box<Expr>, f64),
    /// Exponentiation of an expression by another expression
    PowExpr(Box<Expr>, Box<Expr>),
    /// Exponential function of an expression
    Exp(Box<Expr>),
    /// Natural logarithm of an expression
    Ln(Box<Expr>),
    /// Square root of an expression
    Sqrt(Box<Expr>),
    /// Sine of an expression (argument in radians)
    Sin(Box<Expr>),
    /// Cosine of an expression (argument in radians)
    Cos(Box<Expr>),
    /// Negation of an expression
    Neg(Box<Expr>),
    /// Cached expression with optional pre-computed value
    Cached(Box<Expr>, Option<f64>),
}

/// Linear operation for flattened expression evaluation
#[derive(Debug, Clone)]
pub enum LinearOp {
    /// Load constant value
    LoadConst(f64),
    /// Load variable by index
    LoadVar(u32),
    /// Add two values from stack positions
    Add,
    /// Subtract two values from stack positions  
    Sub,
    /// Multiply two values from stack positions
    Mul,
    /// Divide two values from stack positions
    Div,
    /// Absolute value of stack top
    Abs,
    /// Negate stack top
    Neg,
    /// Power operation with constant exponent
    PowConst(i64),
    /// Power operation with floating point constant exponent
    PowFloat(f64),
    /// Power operation with expression exponent
    PowExpr,
    /// Exponential of stack top
    Exp,
    /// Natural log of stack top
    Ln,
    /// Square root of stack top
    Sqrt,
    /// Sine of stack top (argument in radians)
    Sin,
    /// Cosine of stack top (argument in radians)
    Cos,
    /// a × b + c  (fused)
    Fma,
    /// a × b − c  (fused)
    Fmsub,
}

/// Flattened expression representation for efficient evaluation
#[derive(Debug, Clone)]
pub struct FlattenedExpr {
    /// Linear sequence of operations
    pub ops: Vec<LinearOp>,
    /// Maximum variable index accessed
    pub max_var_index: Option<u32>,
    /// Pre-computed constant result (if expression is constant)
    pub constant_result: Option<f64>,
}

impl Expr {
    /// Returns the number of variables in the expression
    pub fn n_vars(&self) -> u32 {
        match self {
            Expr::Var(_) => 1,
            _ => 0,
        }
    }
    /// Pre-evaluates constants and caches variable loads for improved performance
    pub fn pre_evaluate(
        &self,
        var_cache: &mut std::collections::HashMap<String, f64>,
    ) -> Box<Expr> {
        match self {
            Expr::Const(_) => Box::new(self.clone()),

            Expr::Var(var_ref) => {
                // Check if we can pre-evaluate this variable
                if let Some(&value) = var_cache.get(&var_ref.name) {
                    Box::new(Expr::Const(value))
                } else {
                    Box::new(self.clone())
                }
            }

            Expr::Add(left, right) => {
                let l = left.pre_evaluate(var_cache);
                let r = right.pre_evaluate(var_cache);
                match (&*l, &*r) {
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a + b)),
                    _ => Box::new(Expr::Add(l, r)),
                }
            }

            Expr::Sub(left, right) => {
                let l = left.pre_evaluate(var_cache);
                let r = right.pre_evaluate(var_cache);
                match (&*l, &*r) {
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a - b)),
                    _ => Box::new(Expr::Sub(l, r)),
                }
            }

            Expr::Mul(left, right) => {
                let l = left.pre_evaluate(var_cache);
                let r = right.pre_evaluate(var_cache);
                match (&*l, &*r) {
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a * b)),
                    _ => Box::new(Expr::Mul(l, r)),
                }
            }

            Expr::Div(left, right) => {
                let l = left.pre_evaluate(var_cache);
                let r = right.pre_evaluate(var_cache);
                match (&*l, &*r) {
                    (Expr::Const(a), Expr::Const(b)) if *b != 0.0 => Box::new(Expr::Const(a / b)),
                    _ => Box::new(Expr::Div(l, r)),
                }
            }

            Expr::Abs(expr) => {
                let e = expr.pre_evaluate(var_cache);
                match &*e {
                    Expr::Const(a) => Box::new(Expr::Const(a.abs())),
                    _ => Box::new(Expr::Abs(e)),
                }
            }

            Expr::Neg(expr) => {
                let e = expr.pre_evaluate(var_cache);
                match &*e {
                    Expr::Const(a) => Box::new(Expr::Const(-a)),
                    _ => Box::new(Expr::Neg(e)),
                }
            }

            Expr::Pow(base, exp) => {
                let b = base.pre_evaluate(var_cache);
                match &*b {
                    Expr::Const(a) => Box::new(Expr::Const(a.powi(*exp as i32))),
                    _ => Box::new(Expr::Pow(b, *exp)),
                }
            }

            Expr::PowFloat(base, exp) => {
                let b = base.pre_evaluate(var_cache);
                match &*b {
                    Expr::Const(a) => Box::new(Expr::Const(a.powf(*exp))),
                    _ => Box::new(Expr::PowFloat(b, *exp)),
                }
            }

            Expr::PowExpr(base, exponent) => {
                let b = base.pre_evaluate(var_cache);
                let e = exponent.pre_evaluate(var_cache);
                match (&*b, &*e) {
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a.powf(*b))),
                    _ => Box::new(Expr::PowExpr(b, e)),
                }
            }

            Expr::Exp(expr) => {
                let e = expr.pre_evaluate(var_cache);
                match &*e {
                    Expr::Const(a) => Box::new(Expr::Const(a.exp())),
                    _ => Box::new(Expr::Exp(e)),
                }
            }

            Expr::Ln(expr) => {
                let e = expr.pre_evaluate(var_cache);
                match &*e {
                    Expr::Const(a) if *a > 0.0 => Box::new(Expr::Const(a.ln())),
                    _ => Box::new(Expr::Ln(e)),
                }
            }

            Expr::Sqrt(expr) => {
                let e = expr.pre_evaluate(var_cache);
                match &*e {
                    Expr::Const(a) if *a >= 0.0 => Box::new(Expr::Const(a.sqrt())),
                    _ => Box::new(Expr::Sqrt(e)),
                }
            }

            Expr::Sin(expr) => {
                let e = expr.pre_evaluate(var_cache);
                match &*e {
                    Expr::Const(a) => Box::new(Expr::Const(a.sin())),
                    _ => Box::new(Expr::Sin(e)),
                }
            }

            Expr::Cos(expr) => {
                let e = expr.pre_evaluate(var_cache);
                match &*e {
                    Expr::Const(a) => Box::new(Expr::Const(a.cos())),
                    _ => Box::new(Expr::Cos(e)),
                }
            }

            Expr::Cached(expr, _) => expr.pre_evaluate(var_cache),
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

            Expr::PowFloat(base, exp) => {
                // d/dx(f^c) = c * f^(c-1) * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(*exp)),
                        Box::new(Expr::PowFloat(base.clone(), exp - 1.0)),
                    )),
                    base.derivative(with_respect_to),
                ))
            }

            Expr::PowExpr(base, exponent) => {
                // d/dx(f^g) = f^g * (g' * ln(f) + g * f'/f)
                // Using the general power rule
                Box::new(Expr::Mul(
                    Box::new(Expr::PowExpr(base.clone(), exponent.clone())),
                    Box::new(Expr::Add(
                        Box::new(Expr::Mul(
                            exponent.derivative(with_respect_to),
                            Box::new(Expr::Ln(base.clone())),
                        )),
                        Box::new(Expr::Mul(
                            exponent.clone(),
                            Box::new(Expr::Div(base.derivative(with_respect_to), base.clone())),
                        )),
                    )),
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

            Expr::Sin(expr) => {
                // d/dx(sin(f)) = cos(f) * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Cos(expr.clone())),
                    expr.derivative(with_respect_to),
                ))
            }

            Expr::Cos(expr) => {
                // d/dx(cos(f)) = -sin(f) * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Neg(Box::new(Expr::Sin(expr.clone())))),
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
                    (Expr::Mul(a1, x1), Expr::Mul(a2, x2)) if x1 == x2 => {
                        let combined_coeff = Expr::Add(a1.clone(), a2.clone()).simplify();
                        Box::new(Expr::Mul(combined_coeff, x1.clone()))
                    }
                    // Associativity: (x + c1) + c2 -> x + (c1 + c2)
                    (Expr::Add(x, c1), c2)
                        if matches!(**c1, Expr::Const(_)) && matches!(*c2, Expr::Const(_)) =>
                    {
                        Box::new(Expr::Add(
                            x.clone(),
                            Expr::Add(c1.clone(), Box::new(c2.clone())).simplify(),
                        ))
                    }
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
                    (Expr::Mul(a1, x1), Expr::Mul(a2, x2)) if x1 == x2 => {
                        let combined_coeff = Expr::Sub(a1.clone(), a2.clone()).simplify();
                        Box::new(Expr::Mul(combined_coeff, x1.clone()))
                    }
                    // Convert subtraction to addition: x - c -> x + (-c)
                    (x, Expr::Const(c)) => {
                        Box::new(Expr::Add(Box::new(x.clone()), Box::new(Expr::Const(-c))))
                    }
                    _ => Box::new(Expr::Sub(l, r)),
                }
            }

            Expr::Mul(left, right) => {
                let l = left.simplify();
                let r = right.simplify();

                // Common subexpression elimination
                if l == r {
                    return Box::new(Expr::Pow(l, 2)); // x * x -> x^2
                }

                match (&*l, &*r) {
                    // Fold constants: 2 * 3 -> 6
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a * b)),
                    // Zero property: x * 0 -> 0
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Box::new(Expr::Const(0.0)),
                    // Identity: x * 1 -> x
                    (expr, Expr::Const(1.0)) | (Expr::Const(1.0), expr) => Box::new(expr.clone()),
                    // Negative one: x * (-1) -> -x
                    (expr, Expr::Const(-1.0)) | (Expr::Const(-1.0), expr) => {
                        Box::new(Expr::Neg(Box::new(expr.clone())))
                    }
                    // Combine exponents: x^a * x^b -> x^(a+b)
                    (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) if b1 == b2 => {
                        Box::new(Expr::Pow(b1.clone(), e1 + e2))
                    }
                    // Distribute constants: c * (x + y) -> c*x + c*y (only if beneficial)
                    (Expr::Const(c), Expr::Add(x, y)) | (Expr::Add(x, y), Expr::Const(c))
                        if c.abs() < 10.0 =>
                    {
                        Expr::Add(
                            Box::new(Expr::Mul(Box::new(Expr::Const(*c)), x.clone())),
                            Box::new(Expr::Mul(Box::new(Expr::Const(*c)), y.clone())),
                        )
                        .simplify()
                    }
                    // Strength reduction: x * 2 -> x + x (only for small integers)
                    (expr, Expr::Const(2.0)) | (Expr::Const(2.0), expr) => {
                        Box::new(Expr::Add(Box::new(expr.clone()), Box::new(expr.clone())))
                    }
                    // Associativity: (c1 * x) * c2 -> (c1 * c2) * x
                    (Expr::Mul(c1, x), c2)
                        if matches!(**c1, Expr::Const(_)) && matches!(*c2, Expr::Const(_)) =>
                    {
                        Box::new(Expr::Mul(
                            Expr::Mul(c1.clone(), Box::new(c2.clone())).simplify(),
                            x.clone(),
                        ))
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
                    // Division by negative one: x / (-1) -> -x
                    (expr, Expr::Const(-1.0)) => Box::new(Expr::Neg(Box::new(expr.clone()))),
                    // Identity: x / x -> 1
                    (a, b) if a == b => Box::new(Expr::Const(1.0)),
                    // Simplify exponents: x^a / x^b -> x^(a-b)
                    (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) if b1 == b2 => {
                        Box::new(Expr::Pow(b1.clone(), e1 - e2))
                    }
                    // Convert division by constant to multiplication: x / c -> x * (1/c)
                    (x, Expr::Const(c)) if *c != 0.0 && c.abs() > 1e-10 => Box::new(Expr::Mul(
                        Box::new(x.clone()),
                        Box::new(Expr::Const(1.0 / c)),
                    )),
                    // Simplify nested divisions: (x/y)/z -> x/(y*z)
                    (Expr::Div(x, y), z) => Box::new(Expr::Div(
                        x.clone(),
                        Box::new(Expr::Mul(y.clone(), Box::new(z.clone()))),
                    )),
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
                    // abs(-x) -> abs(x)
                    Expr::Neg(inner) => Box::new(Expr::Abs(inner.clone())),
                    // abs(x^2) -> x^2 (even powers are always positive)
                    Expr::Pow(_, exp) if exp % 2 == 0 => e,
                    _ => Box::new(Expr::Abs(e)),
                }
            }

            Expr::Pow(base, exp) => {
                let b = base.simplify();
                match (&*b, exp) {
                    // x^0 -> 1 (including 0^0 = 1 by convention)
                    (_, 0) => Box::new(Expr::Const(1.0)),
                    // Fold constants: 2^3 -> 8
                    (Expr::Const(a), exp) => Box::new(Expr::Const(a.powi(*exp as i32))),
                    // Identity: x^1 -> x
                    (expr, 1) => Box::new(expr.clone()),
                    // Simplify negative exponents: x^(-n) -> 1/(x^n)
                    (expr, exp) if *exp < 0 => Box::new(Expr::Div(
                        Box::new(Expr::Const(1.0)),
                        Box::new(Expr::Pow(Box::new(expr.clone()), -exp)),
                    )),
                    // Nested exponents: (x^a)^b -> x^(a*b)
                    (Expr::Pow(inner_base, inner_exp), outer_exp) => {
                        Box::new(Expr::Pow(inner_base.clone(), inner_exp * outer_exp))
                    }
                    // Power of product: (x*y)^n -> x^n * y^n (only for small n)
                    (Expr::Mul(x, y), n) if *n >= 2 && *n <= 4 => Box::new(Expr::Mul(
                        Box::new(Expr::Pow(x.clone(), *n)),
                        Box::new(Expr::Pow(y.clone(), *n)),
                    )),
                    _ => Box::new(Expr::Pow(b, *exp)),
                }
            }

            Expr::PowFloat(base, exp) => {
                let b = base.simplify();
                match (&*b, exp) {
                    // x^0.0 -> 1
                    (_, exp) if exp.abs() < 1e-10 => Box::new(Expr::Const(1.0)),
                    // Fold constants: 2.0^3.5 -> result
                    (Expr::Const(a), exp) => Box::new(Expr::Const(a.powf(*exp))),
                    // Identity: x^1.0 -> x
                    (expr, exp) if (exp - 1.0).abs() < 1e-10 => Box::new(expr.clone()),
                    // Convert to integer power if possible
                    (expr, exp) if exp.fract().abs() < 1e-10 => {
                        Box::new(Expr::Pow(Box::new(expr.clone()), *exp as i64))
                    }
                    _ => Box::new(Expr::PowFloat(b, *exp)),
                }
            }

            Expr::PowExpr(base, exponent) => {
                let b = base.simplify();
                let e = exponent.simplify();
                match (&*b, &*e) {
                    // Fold constants: 2^3 -> 8
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a.powf(*b))),
                    // x^0 -> 1
                    (_, Expr::Const(0.0)) => Box::new(Expr::Const(1.0)),
                    // x^1 -> x
                    (expr, Expr::Const(1.0)) => Box::new(expr.clone()),
                    // Convert to simpler forms if exponent is constant
                    (expr, Expr::Const(exp)) if exp.fract().abs() < 1e-10 => {
                        Box::new(Expr::Pow(Box::new(expr.clone()), *exp as i64))
                    }
                    (expr, Expr::Const(exp)) => {
                        Box::new(Expr::PowFloat(Box::new(expr.clone()), *exp))
                    }
                    _ => Box::new(Expr::PowExpr(b, e)),
                }
            }

            Expr::Exp(expr) => {
                let e = expr.simplify();
                match &*e {
                    // exp(0) -> 1
                    Expr::Const(0.0) => Box::new(Expr::Const(1.0)),
                    // Fold constants: exp(c) -> e^c
                    Expr::Const(a) => Box::new(Expr::Const(a.exp())),
                    // exp(ln(x)) -> x
                    Expr::Ln(inner) => inner.clone(),
                    // exp(x + y) -> exp(x) * exp(y)
                    Expr::Add(x, y) => Box::new(Expr::Mul(
                        Box::new(Expr::Exp(x.clone())),
                        Box::new(Expr::Exp(y.clone())),
                    )),
                    _ => Box::new(Expr::Exp(e)),
                }
            }

            Expr::Ln(expr) => {
                let e = expr.simplify();
                match &*e {
                    // Fold constants: ln(c) -> ln(c)
                    Expr::Const(a) if *a > 0.0 => Box::new(Expr::Const(a.ln())),
                    // ln(1) -> 0
                    Expr::Const(1.0) => Box::new(Expr::Const(0.0)),
                    // ln(exp(x)) -> x
                    Expr::Exp(inner) => inner.clone(),
                    // ln(x*y) -> ln(x) + ln(y)
                    Expr::Mul(x, y) => Box::new(Expr::Add(
                        Box::new(Expr::Ln(x.clone())),
                        Box::new(Expr::Ln(y.clone())),
                    )),
                    // ln(x/y) -> ln(x) - ln(y)
                    Expr::Div(x, y) => Box::new(Expr::Sub(
                        Box::new(Expr::Ln(x.clone())),
                        Box::new(Expr::Ln(y.clone())),
                    )),
                    // ln(x^n) -> n * ln(x)
                    Expr::Pow(x, n) => Box::new(Expr::Mul(
                        Box::new(Expr::Const(*n as f64)),
                        Box::new(Expr::Ln(x.clone())),
                    )),
                    _ => Box::new(Expr::Ln(e)),
                }
            }

            Expr::Sqrt(expr) => {
                let e = expr.simplify();
                match &*e {
                    // Fold constants: sqrt(c) -> sqrt(c)
                    Expr::Const(a) if *a >= 0.0 => Box::new(Expr::Const(a.sqrt())),
                    // sqrt(0) -> 0
                    Expr::Const(0.0) => Box::new(Expr::Const(0.0)),
                    // sqrt(1) -> 1
                    Expr::Const(1.0) => Box::new(Expr::Const(1.0)),
                    // sqrt(x^2) -> abs(x)
                    Expr::Pow(x, 2) => Box::new(Expr::Abs(x.clone())),
                    // sqrt(x*y) -> sqrt(x) * sqrt(y)
                    Expr::Mul(x, y) => Box::new(Expr::Mul(
                        Box::new(Expr::Sqrt(x.clone())),
                        Box::new(Expr::Sqrt(y.clone())),
                    )),
                    _ => Box::new(Expr::Sqrt(e)),
                }
            }

            Expr::Sin(expr) => {
                let e = expr.simplify();
                match &*e {
                    // sin(0) -> 0
                    Expr::Const(0.0) => Box::new(Expr::Const(0.0)),
                    // Fold constants: sin(c) -> sin(c)
                    Expr::Const(a) => Box::new(Expr::Const(a.sin())),
                    _ => Box::new(Expr::Sin(e)),
                }
            }

            Expr::Cos(expr) => {
                let e = expr.simplify();
                match &*e {
                    // cos(0) -> 1
                    Expr::Const(0.0) => Box::new(Expr::Const(1.0)),
                    // Fold constants: cos(c) -> cos(c)
                    Expr::Const(a) => Box::new(Expr::Const(a.cos())),
                    _ => Box::new(Expr::Cos(e)),
                }
            }

            Expr::Neg(expr) => {
                let e = expr.simplify();
                match &*e {
                    // Fold constants: -3 -> -3
                    Expr::Const(a) => Box::new(Expr::Const(-a)),
                    // Double negation: -(-x) -> x
                    Expr::Neg(inner) => inner.clone(),
                    // Distribute negation: -(x + y) -> -x - y
                    Expr::Add(x, y) => {
                        Expr::Sub(Box::new(Expr::Neg(x.clone())), y.clone()).simplify()
                    }
                    // Distribute negation: -(x - y) -> -x + y
                    Expr::Sub(x, y) => {
                        Expr::Add(Box::new(Expr::Neg(x.clone())), y.clone()).simplify()
                    }
                    // Factor out negation: -(c*x) -> (-c)*x
                    Expr::Mul(c, x) if matches!(**c, Expr::Const(_)) => {
                        Expr::Mul(Box::new(Expr::Neg(c.clone())), x.clone()).simplify()
                    }
                    _ => Box::new(Expr::Neg(e)),
                }
            }

            Expr::Cached(expr, cached_value) => {
                if cached_value.is_some() {
                    Box::new(self.clone())
                } else {
                    // Simplify the inner expression directly
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
                Expr::PowFloat(base, exp) => {
                    Box::new(Expr::PowFloat(base.insert(predicate, replacement), *exp))
                }
                Expr::PowExpr(base, exponent) => Box::new(Expr::PowExpr(
                    base.insert(predicate.clone(), replacement),
                    exponent.insert(predicate, replacement),
                )),
                Expr::Exp(expr) => Box::new(Expr::Exp(expr.insert(predicate, replacement))),
                Expr::Ln(expr) => Box::new(Expr::Ln(expr.insert(predicate, replacement))),
                Expr::Sqrt(expr) => Box::new(Expr::Sqrt(expr.insert(predicate, replacement))),
                Expr::Sin(expr) => Box::new(Expr::Sin(expr.insert(predicate, replacement))),
                Expr::Cos(expr) => Box::new(Expr::Cos(expr.insert(predicate, replacement))),
                Expr::Neg(expr) => Box::new(Expr::Neg(expr.insert(predicate, replacement))),
                Expr::Cached(expr, _) => {
                    Box::new(Expr::Cached(expr.insert(predicate, replacement), None))
                }
            }
        }
    }

    /// Converts expression tree to flattened linear operations for efficient evaluation.
    ///
    /// This optimization eliminates:
    /// - Tree traversal overhead
    /// - Function call overhead  
    /// - Memory allocation in hot path
    /// - Variable lookup overhead
    ///
    /// The result is a linear sequence of stack-based operations that can be
    /// executed with minimal overhead.
    pub fn flatten(&self) -> FlattenedExpr {
        let mut ops = Vec::new();
        let mut max_var_index = None;

        // Check if entire expression is constant
        if let Some(constant) = self.try_evaluate_constant() {
            return FlattenedExpr {
                ops: vec![LinearOp::LoadConst(constant)],
                max_var_index: None,
                constant_result: Some(constant),
            };
        }

        self.flatten_recursive(&mut ops, &mut max_var_index);

        FlattenedExpr {
            ops,
            max_var_index,
            constant_result: None,
        }
    }

    /// Tries to evaluate expression as constant (aggressive constant folding)
    fn try_evaluate_constant(&self) -> Option<f64> {
        match self {
            Expr::Const(val) => Some(*val),
            Expr::Var(_) => None,
            Expr::Add(left, right) => {
                Some(left.try_evaluate_constant()? + right.try_evaluate_constant()?)
            }
            Expr::Sub(left, right) => {
                Some(left.try_evaluate_constant()? - right.try_evaluate_constant()?)
            }
            Expr::Mul(left, right) => {
                Some(left.try_evaluate_constant()? * right.try_evaluate_constant()?)
            }
            Expr::Div(left, right) => {
                let r = right.try_evaluate_constant()?;
                if r.abs() < 1e-10 {
                    return None;
                }
                Some(left.try_evaluate_constant()? / r)
            }
            Expr::Abs(expr) => Some(expr.try_evaluate_constant()?.abs()),
            Expr::Neg(expr) => Some(-expr.try_evaluate_constant()?),
            Expr::Pow(base, exp) => Some(base.try_evaluate_constant()?.powi(*exp as i32)),
            Expr::PowFloat(base, exp) => Some(base.try_evaluate_constant()?.powf(*exp)),
            Expr::PowExpr(base, exponent) => Some(
                base.try_evaluate_constant()?
                    .powf(exponent.try_evaluate_constant()?),
            ),
            Expr::Exp(expr) => Some(expr.try_evaluate_constant()?.exp()),
            Expr::Ln(expr) => {
                let val = expr.try_evaluate_constant()?;
                if val <= 0.0 {
                    return None;
                }
                Some(val.ln())
            }
            Expr::Sqrt(expr) => {
                let val = expr.try_evaluate_constant()?;
                if val < 0.0 {
                    return None;
                }
                Some(val.sqrt())
            }
            Expr::Sin(expr) => Some(expr.try_evaluate_constant()?.sin()),
            Expr::Cos(expr) => Some(expr.try_evaluate_constant()?.cos()),
            Expr::Cached(expr, cached_value) => {
                cached_value.or_else(|| expr.try_evaluate_constant())
            }
        }
    }

    /// Recursively flattens expression into linear operations
    fn flatten_recursive(&self, ops: &mut Vec<LinearOp>, max_var_index: &mut Option<u32>) {
        match self {
            Expr::Const(val) => {
                ops.push(LinearOp::LoadConst(*val));
            }

            Expr::Var(var_ref) => {
                let index = var_ref.index;
                *max_var_index = Some(max_var_index.unwrap_or(0).max(index));
                ops.push(LinearOp::LoadVar(index));
            }

            Expr::Add(left, right) => {
                left.flatten_recursive(ops, max_var_index);
                right.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Add);
            }

            Expr::Sub(left, right) => {
                left.flatten_recursive(ops, max_var_index);
                right.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Sub);
            }

            Expr::Mul(left, right) => {
                left.flatten_recursive(ops, max_var_index);
                right.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Mul);
            }

            Expr::Div(left, right) => {
                left.flatten_recursive(ops, max_var_index);
                right.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Div);
            }

            Expr::Abs(expr) => {
                expr.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Abs);
            }

            Expr::Neg(expr) => {
                expr.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Neg);
            }

            Expr::Pow(base, exp) => {
                base.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::PowConst(*exp));
            }

            Expr::PowFloat(base, exp) => {
                base.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::PowFloat(*exp));
            }

            Expr::PowExpr(base, exponent) => {
                base.flatten_recursive(ops, max_var_index);
                exponent.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::PowExpr);
            }

            Expr::Exp(expr) => {
                expr.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Exp);
            }

            Expr::Ln(expr) => {
                expr.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Ln);
            }

            Expr::Sqrt(expr) => {
                expr.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Sqrt);
            }

            Expr::Sin(expr) => {
                expr.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Sin);
            }

            Expr::Cos(expr) => {
                expr.flatten_recursive(ops, max_var_index);
                ops.push(LinearOp::Cos);
            }

            Expr::Cached(expr, cached_value) => {
                if let Some(val) = cached_value {
                    ops.push(LinearOp::LoadConst(*val));
                } else {
                    expr.flatten_recursive(ops, max_var_index);
                }
            }
        }
    }

    /// Generates ultra-optimized linear code from flattened operations.
    ///
    /// This eliminates all function call overhead by generating a single
    /// linear sequence of optimal instructions with direct register allocation.
    /// Includes variable caching optimization to eliminate redundant memory access.
    /// Now supports true Structure of Arrays (SoA) layout for SIMD operations.
    pub fn codegen_flattened(
        &self,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        input_ptr: Value,
        lane: u8,
        vec_ty: Type,
        idx_val: Value,
        n_points: Value,
    ) -> Result<Value, EquationError> {
        // ─── Constant / loop-invariant values ───────────────────────────────
        let i32_ty = types::I32;
        let ptr_ty = builder.func.dfg.value_type(input_ptr); // usually I64
        let eight_i32 = builder.ins().iconst(i32_ty, 8); // literal 8 once

        // points_x8 = n_points * 8  (structure-of-arrays block size in bytes)
        let points_x8 = builder.ins().imul_imm(n_points, 8);

        // ─── Fast-path: whole expression is a known constant ────────────────
        let flattened = opt::optimize(self.flatten());
        if let Some(c) = flattened.constant_result {
            let lit = builder.ins().f64const(c);
            return Ok(if lane == 1 {
                lit
            } else {
                builder.ins().splat(vec_ty, lit)
            });
        }

        // ─── Evaluation stack + per-variable cache ──────────────────────────
        let mut value_stack = Vec::with_capacity(flattened.ops.len());
        let mut var_cache: std::collections::HashMap<u32, Value> = std::collections::HashMap::new();

        // ─── Linear byte-code interpreter ───────────────────────────────────
        for op in &flattened.ops {
            match op {
                // -------- constants -------------------------------------------------
                LinearOp::LoadConst(v) => {
                    let lit = builder.ins().f64const(*v);
                    value_stack.push(if lane == 1 {
                        lit
                    } else {
                        builder.ins().splat(vec_ty, lit)
                    });
                }

                // -------- variable load  -------------------------------------------
                LinearOp::LoadVar(idx) => {
                    let val = *var_cache.entry(*idx).or_insert_with(|| {
                        // var_block_offset = idx * points_x8
                        let idx_i32 = builder.ins().iconst(i32_ty, *idx as i64);
                        let var_block_off = builder.ins().imul(idx_i32, points_x8);

                        // point_offset = idx_val * 8
                        let point_off = builder.ins().imul(idx_val, eight_i32);

                        // final byte offset (still i32) & address
                        let total_off_i32 = builder.ins().iadd(var_block_off, point_off);
                        let total_off_ptr = builder.ins().uextend(ptr_ty, total_off_i32);
                        let addr = builder.ins().iadd(input_ptr, total_off_ptr);

                        let mem = MemFlags::new().with_aligned().with_readonly().with_notrap();
                        if lane == 1 {
                            builder.ins().load(types::F64, mem, addr, Offset32::new(0))
                        } else {
                            builder.ins().load(vec_ty, mem, addr, Offset32::new(0))
                        }
                    });
                    value_stack.push(val);
                }

                // -------- arithmetic -------------------------------------------------
                LinearOp::Add => {
                    let r = value_stack.pop().unwrap();
                    let l = value_stack.pop().unwrap();
                    value_stack.push(builder.ins().fadd(l, r));
                }
                LinearOp::Sub => {
                    /* identical pattern */
                    let r = value_stack.pop().unwrap();
                    let l = value_stack.pop().unwrap();
                    value_stack.push(builder.ins().fsub(l, r));
                }
                LinearOp::Mul => {
                    let r = value_stack.pop().unwrap();
                    let l = value_stack.pop().unwrap();
                    value_stack.push(builder.ins().fmul(l, r));
                }
                LinearOp::Div => {
                    let r = value_stack.pop().unwrap();
                    let l = value_stack.pop().unwrap();
                    value_stack.push(builder.ins().fdiv(l, r));
                }
                LinearOp::Abs => {
                    let v = value_stack.pop().unwrap();
                    value_stack.push(builder.ins().fabs(v));
                }
                LinearOp::Neg => {
                    let v = value_stack.pop().unwrap();
                    value_stack.push(builder.ins().fneg(v));
                }

                // -------- power / transcendental  -----------------------------------
                LinearOp::PowConst(e) => {
                    let base = value_stack.pop().unwrap();
                    value_stack.push(generate_optimized_power(builder, base, *e));
                }
                LinearOp::PowFloat(e) => {
                    let base = value_stack.pop().unwrap();
                    let cst = builder.ins().f64const(*e);
                    let expv = if lane == 1 {
                        cst
                    } else {
                        builder.ins().splat(vec_ty, cst)
                    };

                    let fid = operators::pow::link_powf(module).unwrap();
                    value_stack.push(operators::pow::call_powf(builder, module, fid, base, expv));
                }
                LinearOp::PowExpr => {
                    let expv = value_stack.pop().unwrap();
                    let base = value_stack.pop().unwrap();
                    let fid = operators::pow::link_powf(module).unwrap();
                    value_stack.push(operators::pow::call_powf(builder, module, fid, base, expv));
                }
                LinearOp::Exp => {
                    /* similar wrapper */
                    let v = value_stack.pop().unwrap();
                    let fid = operators::exp::link_exp(module).unwrap();
                    value_stack.push(operators::exp::call_exp(builder, module, fid, v));
                }
                LinearOp::Ln => {
                    let v = value_stack.pop().unwrap();
                    let fid = operators::ln::link_ln(module).unwrap();
                    value_stack.push(operators::ln::call_ln(builder, module, fid, v));
                }
                LinearOp::Sqrt => {
                    let v = value_stack.pop().unwrap();
                    let fid = operators::sqrt::link_sqrt(module).unwrap();
                    value_stack.push(operators::sqrt::call_sqrt(builder, module, fid, v));
                }
                LinearOp::Sin => {
                    let v = value_stack.pop().unwrap();
                    let fid = operators::trigonometric::link_sin(module).unwrap();
                    value_stack.push(operators::trigonometric::call_sin(builder, module, fid, v));
                }
                LinearOp::Cos => {
                    let v = value_stack.pop().unwrap();
                    let fid = operators::trigonometric::link_cos(module).unwrap();
                    value_stack.push(operators::trigonometric::call_cos(builder, module, fid, v));
                }
                LinearOp::Fma => {
                    let c = value_stack.pop().unwrap();
                    let b = value_stack.pop().unwrap();
                    let a = value_stack.pop().unwrap();
                    value_stack.push(builder.ins().fma(a, b, c));
                }
                LinearOp::Fmsub => {
                    let c = value_stack.pop().unwrap();
                    let b = value_stack.pop().unwrap();
                    let a = value_stack.pop().unwrap();
                    // a*b - c  ==  fma(a, b, -c)
                    let neg_c = builder.ins().fneg(c);
                    value_stack.push(builder.ins().fma(a, b, neg_c));
                }
            }
        }

        // Final result – the stack must contain exactly one value.
        Ok(value_stack.pop().unwrap())
    }
}

/// Generates optimized power operation with inlining for common exponents and binary exponentiation
fn generate_optimized_power(builder: &mut FunctionBuilder, base: Value, exp: i64) -> Value {
    match exp {
        0 => builder.ins().f64const(1.0),
        1 => base,
        2 => builder.ins().fmul(base, base),
        3 => {
            let square = builder.ins().fmul(base, base);
            builder.ins().fmul(square, base)
        }
        4 => {
            let square = builder.ins().fmul(base, base);
            builder.ins().fmul(square, square)
        }
        5 => {
            let square = builder.ins().fmul(base, base);
            let fourth = builder.ins().fmul(square, square);
            builder.ins().fmul(fourth, base)
        }
        6 => {
            let square = builder.ins().fmul(base, base);
            let cube = builder.ins().fmul(square, base);
            builder.ins().fmul(cube, cube)
        }
        8 => {
            let square = builder.ins().fmul(base, base);
            let fourth = builder.ins().fmul(square, square);
            builder.ins().fmul(fourth, fourth)
        }
        -1 => {
            let one = builder.ins().f64const(1.0);
            builder.ins().fdiv(one, base)
        }
        -2 => {
            let square = builder.ins().fmul(base, base);
            let one = builder.ins().f64const(1.0);
            builder.ins().fdiv(one, square)
        }
        _ => {
            // Binary exponentiation for other cases
            let abs_exp = exp.abs();
            let mut result = builder.ins().f64const(1.0);
            let mut current_base = base;
            let mut remaining = abs_exp;

            // Binary exponentiation - optimal for any exponent
            while remaining > 0 {
                if remaining & 1 == 1 {
                    result = builder.ins().fmul(result, current_base);
                }
                if remaining > 1 {
                    current_base = builder.ins().fmul(current_base, current_base);
                }
                remaining >>= 1;
            }

            if exp < 0 {
                let one = builder.ins().f64const(1.0);
                builder.ins().fdiv(one, result)
            } else {
                result
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
            Expr::Const(val) => write!(f, "{val}"),
            Expr::Var(var_ref) => write!(f, "{0}", var_ref.name),
            Expr::Add(left, right) => write!(f, "({left} + {right})"),
            Expr::Mul(left, right) => write!(f, "({left} * {right})"),
            Expr::Sub(left, right) => write!(f, "({left} - {right})"),
            Expr::Div(left, right) => write!(f, "({left} / {right})"),
            Expr::Abs(expr) => write!(f, "|{expr}|"),
            Expr::Pow(base, exp) => write!(f, "({base}^{exp})"),
            Expr::PowFloat(base, exp) => write!(f, "({base}^{exp})"),
            Expr::PowExpr(base, exponent) => write!(f, "({base}^{exponent})"),
            Expr::Exp(expr) => write!(f, "exp({expr})"),
            Expr::Ln(expr) => write!(f, "ln({expr})"),
            Expr::Sqrt(expr) => write!(f, "sqrt({expr})"),
            Expr::Sin(expr) => write!(f, "sin({expr})"),
            Expr::Cos(expr) => write!(f, "cos({expr})"),
            Expr::Neg(expr) => write!(f, "-({expr})"),
            Expr::Cached(expr, _) => write!(f, "{expr}"),
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

        // Test sqrt(x^2) - should simplify to abs(x)
        let expr = Box::new(Expr::Sqrt(Box::new(Expr::Pow(var("x"), 2))));
        assert_eq!(*expr.simplify(), Expr::Abs(var("x")));

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
        assert_eq!(format!("{sum}"), "(x + y)");

        let product = Expr::Mul(var("x"), var("y"));
        assert_eq!(format!("{product}"), "(x * y)");

        // Test special functions
        let exp = Expr::Exp(var("x"));
        assert_eq!(format!("{exp}"), "exp(x)");

        let abs = Expr::Abs(var("x"));
        assert_eq!(format!("{abs}"), "|x|");

        // Test complex expression
        let complex = Expr::Div(
            Box::new(Expr::Add(Box::new(Expr::Pow(var("x"), 2)), var("y"))),
            var("z"),
        );
        assert_eq!(format!("{complex}"), "(((x^2) + y) / z)");
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
