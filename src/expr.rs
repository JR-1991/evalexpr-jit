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
//!
//! Supported operations include:
//! - Basic arithmetic (+, -, *, /)
//! - Variables and constants
//! - Absolute value
//! - Integer exponentiation

use cranelift::prelude::*;
use cranelift_codegen::ir::{immediates::Offset32, Value};

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
    Exp(Box<Expr>, i64),
}

impl Expr {
    /// Generates Cranelift IR code for this expression.
    ///
    /// Recursively traverses the expression tree and generates the appropriate
    /// Cranelift instructions to compute the expression's value at runtime.
    ///
    /// # Arguments
    /// * `builder` - The Cranelift FunctionBuilder to emit instructions into
    ///
    /// # Returns
    /// A Cranelift Value representing the result of this expression
    pub fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            Expr::Const(val) => builder.ins().f64const(*val),
            Expr::Var(VarRef { vec_ref, index, .. }) => {
                // Load from the vector at the specified index
                let index_val = builder.ins().iconst(types::I64, *index as i64);
                let offset = builder.ins().imul_imm(index_val, 8); // size of f64
                let vec_ref_val = builder.ins().iadd(*vec_ref, offset);
                builder
                    .ins()
                    .load(types::F64, MemFlags::new(), vec_ref_val, Offset32::new(0))
            }
            Expr::Add(left, right) => {
                // Recursively generate code for both sides
                let lhs = left.codegen(builder); // This could be another nested expression
                let rhs = right.codegen(builder); // This could be another nested expression
                builder.ins().fadd(lhs, rhs)
            }
            Expr::Mul(left, right) => {
                let lhs = left.codegen(builder);
                let rhs = right.codegen(builder);
                builder.ins().fmul(lhs, rhs)
            }
            Expr::Sub(left, right) => {
                let lhs = left.codegen(builder);
                let rhs = right.codegen(builder);
                builder.ins().fsub(lhs, rhs)
            }
            Expr::Div(left, right) => {
                let lhs = left.codegen(builder);
                let rhs = right.codegen(builder);
                builder.ins().fdiv(lhs, rhs)
            }
            Expr::Abs(expr) => {
                let expr = expr.codegen(builder);
                builder.ins().fabs(expr)
            }
            Expr::Exp(base, exp) => {
                let base_val = base.codegen(builder);
                if *exp == 0 {
                    // x^0 = 1
                    builder.ins().f64const(1.0)
                } else if *exp == 1 {
                    // x^1 = x
                    base_val
                } else {
                    let mut result = base_val;
                    // Multiply base_val by itself (exp-1) times
                    for _ in 1..*exp {
                        result = builder.ins().fmul(result, base_val);
                    }
                    result
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
                    Box::new(Expr::Exp(right.clone(), 2)),
                ))
            }

            Expr::Abs(expr) => {
                // d/dx|f| = f/|f| * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Div(expr.clone(), Box::new(Expr::Abs(expr.clone())))),
                    expr.derivative(with_respect_to),
                ))
            }

            Expr::Exp(base, exp) => {
                // d/dx(f^n) = n * f^(n-1) * df/dx
                Box::new(Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(*exp as f64)),
                        Box::new(Expr::Exp(base.clone(), exp - 1)),
                    )),
                    base.derivative(with_respect_to),
                ))
            }
        }
    }

    /// Simplifies the expression by folding constants and applying basic algebraic rules
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
                match (&*l, &*r) {
                    // Fold constants: 2 * 3 -> 6
                    (Expr::Const(a), Expr::Const(b)) => Box::new(Expr::Const(a * b)),
                    // Zero property: x * 0 -> 0
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Box::new(Expr::Const(0.0)),
                    // Identity: x * 1 -> x
                    (expr, Expr::Const(1.0)) | (Expr::Const(1.0), expr) => Box::new(expr.clone()),
                    // Combine exponents: x^a * x^b -> x^(a+b)
                    (Expr::Exp(b1, e1), Expr::Exp(b2, e2)) if b1 == b2 => {
                        Box::new(Expr::Exp(b1.clone(), e1 + e2))
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
                    (Expr::Exp(b1, e1), Expr::Exp(b2, e2)) if b1 == b2 => {
                        Box::new(Expr::Exp(b1.clone(), e1 - e2))
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

            Expr::Exp(base, exp) => {
                let b = base.simplify();
                match (&*b, exp) {
                    // Fold constants: 2^3 -> 8
                    (Expr::Const(a), exp) => Box::new(Expr::Const(a.powi(*exp as i32))),
                    // x^0 -> 1
                    (base, 0) if !matches!(base, Expr::Const(0.0)) => Box::new(Expr::Const(1.0)),
                    // Identity: x^1 -> x
                    (expr, 1) => Box::new(expr.clone()),
                    // Nested exponents: (x^a)^b -> x^(a*b)
                    (Expr::Exp(inner_base, inner_exp), outer_exp) => {
                        Box::new(Expr::Exp(inner_base.clone(), inner_exp * outer_exp))
                    }
                    _ => Box::new(Expr::Exp(b, *exp)),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(*Expr::Exp(var("x"), 0).simplify(), Expr::Const(1.0));
        // x^1 → x
        assert_eq!(*Expr::Exp(var("x"), 1).simplify(), *var("x"));

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
}
