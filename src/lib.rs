//! A library for JIT compilation and symbolic differentiation of mathematical expressions.
//!
//! This crate provides functionality to:
//! - Parse mathematical expressions from strings
//! - Convert them into an internal AST representation
//! - JIT compile expressions into native machine code for fast evaluation
//! - Compute symbolic derivatives of expressions
//!
//! The main components are:
//! - `Equation` - High-level interface for working with mathematical equations
//! - `Expr` - AST representation of mathematical expressions
//! - `build_ast` - Converts parsed expressions into our AST format
//! - `build_function` - JIT compiles expressions into native code

pub use equation::Equation;

pub mod prelude {
    pub use crate::builder::build_function;
    pub use crate::convert::build_ast;
    pub use crate::equation::Equation;
    pub use crate::expr::Expr;
}

/// JIT compilation functionality using Cranelift
pub mod builder;
/// Conversion from parsed expressions to internal AST
pub mod convert;
/// High-level equation handling
pub mod equation;
/// Error types for the various failure modes
pub mod errors;
/// Expression tree representation and symbolic differentiation
pub mod expr;
