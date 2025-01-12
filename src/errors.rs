//! Error types for the evalexpr-jit crate.
//!
//! This module defines the various error types that can occur during expression parsing,
//! AST conversion, and JIT compilation. The main error types are:
//!
//! - `ConvertError`: Errors during conversion from evalexpr AST to internal representation
//! - `BuilderError`: Errors during JIT compilation with Cranelift
//! - `EquationError`: High-level errors when working with equations
//!
//! Each error type implements the standard Error trait and provides detailed error messages.

use cranelift_codegen::CodegenError;
use cranelift_module::ModuleError;
use evalexpr::{DefaultNumericTypes, EvalexprError};
use thiserror::Error;

/// Errors that can occur during conversion from evalexpr AST to our internal AST representation.
///
/// This enum represents various failure modes when converting the evalexpr expression tree
/// into our own AST format that can be used for JIT compilation and symbolic differentiation.
#[derive(Error, Debug)]
pub enum ConvertError {
    /// Error when trying to convert an exponent that is not a valid integer constant
    #[error("Could not convert exponent in Exp operator: {0}")]
    ExpOperator(String),
    /// Error when encountering an operator that is not supported by our implementation
    #[error("Unsupported operator: {0}")]
    UnsupportedOperator(String),
    /// Error when encountering a function that is not supported by our implementation
    #[error("Unsupported function: {0}")]
    UnsupportedFunction(String),
    /// Error when the root node does not have exactly one child
    #[error("Expected single child for root node: {0}")]
    RootNode(String),
    /// Error when a constant value is not a floating point number
    #[error("Expected float constant: {0}")]
    ConstOperator(String),
    /// Error when a variable is not found in the variable map
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
}

/// Errors that can occur during JIT compilation of expressions.
///
/// This enum represents various failure modes in the process of converting our AST
/// into machine code using Cranelift as the JIT compiler backend.
#[derive(Error, Debug)]
pub enum BuilderError {
    /// Error when the target machine architecture is not supported
    #[error("host machine is not supported: {0}")]
    HostMachineNotSupported(String),
    /// Error during Cranelift code generation
    #[error("codegen error: {0}")]
    CodegenError(CodegenError),
    /// Error in the Cranelift JIT module
    #[error("module error: {0}")]
    ModuleError(ModuleError),
    /// Error when defining the JIT function
    #[error("function error: {0}")]
    FunctionError(String),
    /// Error when declaring the JIT function
    #[error("declaration error: {0}")]
    DeclarationError(String),
}

/// High-level errors that can occur when working with mathematical equations.
///
/// This enum represents the various ways that equation parsing, compilation,
/// and differentiation can fail. It wraps lower-level errors from the expression
/// conversion and JIT compilation stages.
#[derive(Debug, Error)]
pub enum EquationError {
    /// Error when parsing the initial expression string with evalexpr
    #[error("Failed to build Evalexpr AST")]
    BuildEvalexprError(#[from] EvalexprError<DefaultNumericTypes>),
    /// Error when converting from evalexpr AST to our internal JIT-compatible AST representation
    #[error("Failed to build JIT AST")]
    BuildJITError(#[from] ConvertError),
    /// Error when JIT compiling the expression
    #[error("Failed to build JIT function")]
    BuildFunctionError(#[from] BuilderError),
    /// Error when trying to get derivative for a variable that doesn't exist
    #[error("Derivative not found for variable: {0}")]
    DerivativeNotFound(String),
    /// Error when the input length is not the same as the number of variables
    #[error("Invalid input length: expected {expected}, got {got}")]
    InvalidInputLength { expected: usize, got: usize },
    /// Error when a variable is not found in the equation
    #[error("Variable not found in equation: {0}")]
    VariableNotFound(String),
    /// Error when the output length is not the same as the number of equations
    #[error("Invalid output length: expected {expected}, got {got}")]
    InvalidOutputLength { expected: usize, got: usize },
}
