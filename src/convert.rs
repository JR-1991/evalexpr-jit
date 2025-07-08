//! Converts evalexpr AST nodes into our internal expression representation for JIT compilation.
//!
//! This module transforms the AST nodes from evalexpr into our own expression types that support:
//! - JIT compilation to native machine code
//! - Automatic differentiation for gradients and Hessians
//! - Efficient evaluation using Cranelift
//!
//! The main entry point is `build_ast()` which recursively converts an evalexpr AST into our format.
//! The conversion preserves the mathematical structure while adding the ability to compile to native code.

use std::collections::HashMap;

use crate::{
    errors::ConvertError,
    expr::{Expr, VarRef},
};
use cranelift::prelude::*;
use evalexpr::{Node, Operator};

/// Converts an evalexpr AST node into our JIT-compilable expression format.
///
/// Recursively traverses the evalexpr AST and builds an equivalent expression tree that can be:
/// - JIT compiled to native machine code
/// - Symbolically differentiated for gradients
/// - Efficiently evaluated at runtime
///
/// # Arguments
/// * `node` - The evalexpr AST node to convert
/// * `var_map` - Maps variable names to their indices in the input array
///
/// # Returns
/// * `Result<Expr, ConvertError>` - The converted expression or an error
///
/// # Supported Operations
/// - Arithmetic: +, -, *, /
/// - Variables: x, y, etc mapped to array indices
/// - Constants: Floating point numbers
/// - Functions: abs(), ln(), log(), sqrt(), exp(), sin(), cos()
/// - Exponentiation: x^n (integer), x^3.5 (float), x^y (expression)
/// - Unary negation: -x
///
/// # Example
/// ```ignore
/// let node = build_operator_tree("2*x + y^2")?;
/// let var_map = HashMap::from([
///     ("x".to_string(), 0),
///     ("y".to_string(), 1),
/// ]);
/// let expr = build_ast(&node, &var_map)?;
/// ```
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
                "Expected numeric constant: {value:?}"
            ))),
        },
        // Variable reference - looks up the variable's index in var_map
        Operator::VariableIdentifierRead { identifier } => {
            let index = var_map
                .get(identifier.as_str())
                .ok_or(ConvertError::VariableNotFound(format!(
                    "Variable not found: {identifier:?}"
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
                "sin" => Ok(Expr::Sin(Box::new(build_ast(&children[0], var_map)?))),
                "cos" => Ok(Expr::Cos(Box::new(build_ast(&children[0], var_map)?))),
                _ => Err(ConvertError::UnsupportedFunction(format!(
                    "Unsupported function: {identifier:?}"
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
                    "Expected single child for root node: {children:?}"
                )))
            }
        }

        // Exponentiation - base can be any expression, exponent can be integer, float, or expression
        Operator::Exp => {
            let children = node.children();

            if children.len() != 2 {
                panic!("Expected 2 children for Exp operator");
            }

            // Check if the second child is a constant
            if let Operator::Const { value } = children[1].operator() {
                match value {
                    // Integer exponent
                    evalexpr::Value::Int(exp) => {
                        Ok(Expr::Pow(Box::new(build_ast(&children[0], var_map)?), *exp))
                    }
                    // Floating point exponent
                    evalexpr::Value::Float(exp) => {
                        // Check if it's actually an integer disguised as a float
                        if exp.fract().abs() < 1e-10 {
                            Ok(Expr::Pow(
                                Box::new(build_ast(&children[0], var_map)?),
                                *exp as i64,
                            ))
                        } else {
                            Ok(Expr::PowFloat(
                                Box::new(build_ast(&children[0], var_map)?),
                                *exp,
                            ))
                        }
                    }
                    _ => Err(ConvertError::ExpOperator(format!(
                        "Expected numeric constant for exponent in Exp operator: {value:?}"
                    ))),
                }
            } else {
                // Non-constant exponent - general expression exponentiation
                Ok(Expr::PowExpr(
                    Box::new(build_ast(&children[0], var_map)?),
                    Box::new(build_ast(&children[1], var_map)?),
                ))
            }
        }
        // Any other operator is unsupported
        _ => Err(ConvertError::UnsupportedOperator(format!(
            "Unsupported operator: {:?}",
            node.operator()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evalexpr::build_operator_tree;

    fn make_var_map() -> HashMap<String, u32> {
        HashMap::from([
            ("x".to_string(), 0),
            ("y".to_string(), 1),
            ("z".to_string(), 2),
        ])
    }

    #[test]
    fn test_basic_arithmetic() {
        let var_map = make_var_map();

        // Test addition
        let node = build_operator_tree("2 + x").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(
            expr,
            Expr::Add(ref a, ref b) if matches!(**a, Expr::Const(2.0)) && matches!(**b, Expr::Var(_))
        ));

        // Test multiplication
        let node = build_operator_tree("3 * y").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(
            expr,
            Expr::Mul(ref a, ref b) if matches!(**a, Expr::Const(3.0)) && matches!(**b, Expr::Var(_))
        ));

        // Test complex expression
        let node = build_operator_tree("2*x + y/z - 5").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::Sub(_, ref b) if matches!(**b, Expr::Const(5.0))));
    }

    #[test]
    fn test_functions() {
        let var_map = make_var_map();

        // Test abs
        let node = build_operator_tree("abs(x)").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::Abs(ref a) if matches!(**a, Expr::Var(_))));

        // Test ln
        let node = build_operator_tree("ln(y)").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::Ln(ref a) if matches!(**a, Expr::Var(_))));

        // Test sqrt
        let node = build_operator_tree("sqrt(z)").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::Sqrt(ref a) if matches!(**a, Expr::Var(_))));

        // Test sin
        let node = build_operator_tree("sin(x)").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::Sin(ref a) if matches!(**a, Expr::Var(_))));

        // Test cos
        let node = build_operator_tree("cos(y)").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::Cos(ref a) if matches!(**a, Expr::Var(_))));
    }

    #[test]
    fn test_exponentiation() {
        let var_map = make_var_map();

        // Test integer power
        let node = build_operator_tree("x^2").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::Pow(ref a, 2) if matches!(**a, Expr::Var(_))));

        // Test floating point power
        let node = build_operator_tree("x^3.5").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::PowFloat(ref a, 3.5) if matches!(**a, Expr::Var(_))));

        // Test expression power
        let node = build_operator_tree("x^y").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(
            matches!(expr, Expr::PowExpr(ref a, ref b) if matches!(**a, Expr::Var(_)) && matches!(**b, Expr::Var(_)))
        );
    }

    #[test]
    fn test_errors() {
        let var_map = make_var_map();

        // Test undefined variable
        let node = build_operator_tree("w + 1").unwrap();
        assert!(matches!(
            build_ast(&node, &var_map),
            Err(ConvertError::VariableNotFound(_))
        ));

        // Test unsupported function
        let node = build_operator_tree("tan(x)").unwrap();
        assert!(matches!(
            build_ast(&node, &var_map),
            Err(ConvertError::UnsupportedFunction(_))
        ));

        // Test floating point exponent (should now work)
        let node = build_operator_tree("x^2.5").unwrap();
        let expr = build_ast(&node, &var_map).unwrap();
        assert!(matches!(expr, Expr::PowFloat(ref a, 2.5) if matches!(**a, Expr::Var(_))));
    }
}
