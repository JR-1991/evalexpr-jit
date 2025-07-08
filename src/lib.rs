//! JIT-compiled mathematical expression evaluator with automatic differentiation.
//!
//! This crate provides JIT compilation and automatic differentiation for mathematical expressions.
//! It parses expressions using [evalexpr](https://github.com/ISibboI/evalexpr) and compiles them to
//! native machine code using [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift).
//!
//! # Features
//!
//! - Fast evaluation through JIT compilation to native code
//! - Automatic differentiation for gradients and Hessians
//! - Support for variables, constants, and mathematical functions
//! - Safe Rust implementation with comprehensive error handling
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
//! let result = eq.eval(&vec![1.0, 2.0]).unwrap(); // Returns 6.0
//!
//! // Compute gradient [∂/∂x, ∂/∂y]
//! let gradient = eq.gradient(&vec![1.0, 2.0]).unwrap(); // Returns [2.0, 4.0]
//! ```

pub use equation::Equation;
pub use expr::Expr;
pub use system::EquationSystem;

pub mod prelude {
    pub use crate::backends::matrix::Matrix;
    pub use crate::backends::vector::Vector;
    pub use crate::equation::Equation;
    pub use crate::expr::Expr;
    pub use crate::system::EquationSystem;
}

/// JIT compilation functionality using Cranelift
pub mod builder;
/// Conversion from evalexpr AST to internal expression format
pub mod convert;
/// High-level equation handling and evaluation
pub mod equation;
/// Error types for parsing, compilation and evaluation
pub mod errors;
/// Expression tree representation and symbolic differentiation
pub mod expr;
/// System of multiple equations
pub mod system;
/// Functions for linking external mathematical operations
pub(crate) mod operators {
    pub(crate) mod exp;
    pub(crate) mod ln;
    pub(crate) mod pow;
    pub(crate) mod sqrt;
    pub(crate) mod trigonometric;
}
/// Type definitions for JIT-compiled functions
pub mod types;

/// Backends for vector operations
pub mod backends {
    pub mod matrix;
    pub mod vector;
}
