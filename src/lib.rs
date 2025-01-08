//! JIT-compiled mathematical expression evaluator with automatic differentiation.
//!
//! This crate provides JIT compilation and automatic differentiation for mathematical expressions.
//! It builds on top of the [evalexpr](https://github.com/ISibboI/evalexpr) crate for parsing and
//! uses [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift) for JIT compilation.
//!
//! # Features
//!
//! - JIT compilation for fast expression evaluation
//! - Automatic differentiation up to second order
//! - Support for multiple variables
//! - Type-safe Rust implementation
//!
//! # Example
//!
//! ```rust
//! use evalexpr_jit::Equation;
//!
//! // Create and compile an equation
//! let eq = Equation::new("2*x + y^2".to_string()).unwrap();
//!
//! // Evaluate at point (x=1, y=2)
//! let result = eq.eval(&[1.0, 2.0]).unwrap(); // Returns 6.0
//!
//! // Compute gradient [∂/∂x, ∂/∂y]
//! let gradient = eq.gradient(&[1.0, 2.0]).unwrap(); // Returns [2.0, 4.0]
//! ```

pub use equation::Equation;
pub use system::EquationSystem;

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
/// System of equations
pub mod system;
/// Functions for linking external functions to the expression tree
pub(crate) mod operators {
    pub(crate) mod exp;
    pub(crate) mod ln;
    pub(crate) mod sqrt;
}
