//! Conversion module for transforming evalexpr AST nodes into our internal expression representation.
//!
//! This module handles converting the AST nodes from the evalexpr crate into our own
//! expression types that support JIT compilation and automatic differentiation. It maps
//! variables, operators, and function calls into our internal representation.
//!
//! The main entry point is the `build_ast` function which recursively traverses the
//! evalexpr AST and builds up our expression tree.

use std::collections::HashMap;

use crate::{
    errors::ConvertError,
    expr::{Expr, VarRef},
};
use cranelift::prelude::*;
use evalexpr::{Node, Operator};

/// Converts an evalexpr AST node into our internal expression representation.
///
/// This function recursively traverses the evalexpr AST and builds up our own AST
/// representation that can be used for JIT compilation and symbolic differentiation.
///
/// # Arguments
/// * `node` - The evalexpr AST node to convert
/// * `var_map` - A mapping of variable names to their indices in the input array
///
/// # Returns
/// * `Result<Expr, ConvertError>` - The converted expression or an error if conversion fails
///
/// # Examples of supported operations:
/// * Basic arithmetic: +, -, *, /
/// * Variables: x, y, etc.
/// * Constants: floating point numbers
/// * Functions: abs()
/// * Exponentiation: x^n where n is an integer constant
pub fn build_ast(node: &Node, var_map: &HashMap<String, u32>) -> Result<Expr, ConvertError> {
    match node.operator() {
        // Addition operator - combines multiple children into a series of binary Add expressions
        Operator::Add => {
            let children = node.children();
            // Handle multiple children by folding them into a series of Add expressions
            children
                .iter()
                .skip(1)
                .try_fold(build_ast(&children[0], var_map)?, |acc, child| {
                    Ok(Expr::Add(
                        Box::new(acc),
                        Box::new(build_ast(child, var_map)?),
                    ))
                })
        }
        // Multiplication operator - combines multiple children into a series of binary Mul expressions
        Operator::Mul => {
            let children = node.children();
            children.iter().skip(1).try_fold(
                build_ast(&children[0], var_map)?,
                |acc, child| -> Result<Expr, ConvertError> {
                    Ok(Expr::Mul(
                        Box::new(acc),
                        Box::new(build_ast(child, var_map)?),
                    ))
                },
            )
        }
        // Division operator - creates a binary Div expression
        Operator::Div => {
            let children = node.children();
            Ok(Expr::Div(
                Box::new(build_ast(&children[0], var_map)?),
                Box::new(build_ast(&children[1], var_map)?),
            ))
        }
        // Subtraction operator - creates a binary Sub expression
        Operator::Sub => {
            let children = node.children();
            Ok(Expr::Sub(
                Box::new(build_ast(&children[0], var_map)?),
                Box::new(build_ast(&children[1], var_map)?),
            ))
        }
        // Constant value - must be a float
        Operator::Const { value } => match value {
            evalexpr::Value::Float(f) => Ok(Expr::Const(*f)),
            evalexpr::Value::Int(i) => Ok(Expr::Const(*i as f64)),
            _ => Err(ConvertError::ConstOperator(format!(
                "Expected numeric constant: {:?}",
                value
            ))),
        },
        // Variable reference - looks up the variable's index in var_map
        Operator::VariableIdentifierRead { identifier } => {
            let index = var_map
                .get(identifier.as_str())
                .ok_or(ConvertError::VariableNotFound(format!(
                    "Variable not found: {:?}",
                    identifier
                )))?;
            Ok(Expr::Var(VarRef {
                name: identifier.to_string(),
                vec_ref: Value::from_u32(0), // Placeholder value, updated during codegen
                index: *index,
            }))
        }
        // Negation operator - creates a Neg expression
        Operator::Neg => {
            let children = node.children();
            Ok(Expr::Neg(Box::new(build_ast(&children[0], var_map)?)))
        }
        // Function call - currently only supports abs()
        Operator::FunctionIdentifier { identifier } => {
            let children = node.children();
            match identifier.as_str() {
                "abs" => Ok(Expr::Abs(Box::new(build_ast(&children[0], var_map)?))),
                "ln" => Ok(Expr::Ln(Box::new(build_ast(&children[0], var_map)?))),
                "log" => Ok(Expr::Ln(Box::new(build_ast(&children[0], var_map)?))),
                "sqrt" => Ok(Expr::Sqrt(Box::new(build_ast(&children[0], var_map)?))),
                "exp" => Ok(Expr::Exp(Box::new(build_ast(&children[0], var_map)?))),
                _ => Err(ConvertError::UnsupportedFunction(format!(
                    "Unsupported function: {:?}",
                    identifier
                ))),
            }
        }
        // Root node - should have exactly one child
        Operator::RootNode => {
            let children = node.children();
            if children.len() == 1 {
                build_ast(&children[0], var_map)
            } else {
                Err(ConvertError::RootNode(format!(
                    "Expected single child for root node: {:?}",
                    children
                )))
            }
        }

        // Exponentiation - base can be any expression but exponent must be an integer constant
        Operator::Exp => {
            let children = node.children();

            if children.len() != 2 {
                panic!("Expected 2 children for Exp operator");
            }

            // Check if the second child is a constant
            if let Operator::Const { value } = children[1].operator() {
                if let evalexpr::Value::Int(exp) = value {
                    Ok(Expr::Pow(Box::new(build_ast(&children[0], var_map)?), *exp))
                } else {
                    Err(ConvertError::ExpOperator(format!(
                        "Expected integer constant for exponent in Exp operator: {:?}",
                        value
                    )))
                }
            } else {
                Err(ConvertError::ExpOperator(format!(
                    "Expected constant for exponent in Exp operator: {:?}",
                    children[1].operator()
                )))
            }
        }
        // Any other operator is unsupported
        _ => Err(ConvertError::UnsupportedOperator(format!(
            "Unsupported operator: {:?}",
            node.operator()
        ))),
    }
}
