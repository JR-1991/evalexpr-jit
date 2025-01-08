//! Performance comparison example between evalexpr-jit and evalexpr.
//!
//! This example demonstrates the performance benefits of using JIT compilation
//! compared to the standard evalexpr interpreter. It evaluates the same expression
//! "2*x + y^2" millions of times using both approaches and measures the execution time.

extern crate evalexpr_jit;

use std::time::Instant;

use evalexpr::{build_operator_tree, ContextWithMutableVariables, HashMapContext, Node};
use evalexpr_jit::Equation;

/// Main function that runs the performance comparison between evalexpr-jit and evalexpr.
///
/// # Steps:
/// 1. Creates a simple mathematical expression "2*x + y^2"
/// 2. Compiles it using evalexpr-jit's JIT compiler
/// 3. Times the execution of 10 million evaluations using evalexpr-jit
/// 4. Creates an equivalent evalexpr AST and context
/// 5. Times the execution of 10 million evaluations using standard evalexpr
///
/// The results demonstrate the significant speedup achieved through JIT compilation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expression = "2*x + y^2 + z^2".to_string();

    // Build JIT function
    let eq = Equation::new(expression.clone())?;

    // Time evalexpr-jit
    let time_jit = time_it(
        "evalexpr-jit",
        || {
            eq.eval(&[1.0, 2.0, 3.0]).unwrap();
        },
        10_000_000,
    );

    // Time evalexpr
    let tree: Node = build_operator_tree(&expression)?;

    let mut context = HashMapContext::new();
    context.set_value("x".to_string(), evalexpr::Value::Float(1.0))?;
    context.set_value("y".to_string(), evalexpr::Value::Float(2.0))?;
    context.set_value("z".to_string(), evalexpr::Value::Float(3.0))?;

    let time_evalexpr = time_it(
        "evalexpr",
        || {
            tree.eval_float_with_context(&context).unwrap();
        },
        10_000_000,
    );

    println!(
        "evalexpr-jit is {}% faster than evalexpr",
        ((time_evalexpr / time_jit) * 100.0).round() as i64
    );

    Ok(())
}

/// Helper function to measure execution time of a given operation.
///
/// # Arguments
/// * `name` - Name of the operation being timed (for display purposes)
/// * `f` - Closure containing the operation to time
/// * `n` - Number of iterations to run
///
/// # Output
/// Prints the total time taken for n iterations of the operation.
///
/// # Returns
/// The total time taken for n iterations of the operation in seconds.
fn time_it<F: Fn()>(name: &str, f: F, n: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..n {
        f();
    }
    let end = Instant::now();
    println!(
        "{}: Takes {:?} for {} runs",
        name,
        end.duration_since(start),
        n
    );

    end.duration_since(start).as_secs_f64()
}
