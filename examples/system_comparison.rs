//! Performance comparison example for equation systems.
//!
//! This example demonstrates the performance difference between evaluating mathematical
//! equations individually versus as a combined system. It compares three approaches:
//!
//! 1. Individual equation evaluation
//! 2. Combined system evaluation (sequential)
//! 3. Combined system evaluation (parallel)
//!
//! The example shows how combining multiple equations into a single system improves
//! performance through:
//! - Shared IR (Intermediate Representation) optimization
//! - Reduced overhead from function calls
//! - Better CPU cache utilization
//! - Parallel evaluation capabilities
//!
//! The test generates 1000 mathematical equations with varying complexity, running them
//! 10,000 times using each approach to get reliable timing data. The equations use
//! 5 variables (x, y, z, w, v) and include operations like:
//! - Basic arithmetic (+, -, *, /)
//! - Exponentiation and roots
//! - Logarithms and exponentials
//! - Compound expressions with multiple terms

use evalexpr_jit::{system::EquationSystem, Equation};
use rayon::prelude::*;
use std::time::Instant;

/// Main function that runs the performance comparison between individual equations,
/// sequential system evaluation, and parallel system evaluation.
///
/// The function:
/// 1. Generates 1000 test equations with varying complexity
/// 2. Compiles them both individually and as a system
/// 3. Times execution using three different approaches
/// 4. Reports compilation times and runtime performance metrics
///
/// # Returns
/// Result indicating success or an error if equation compilation fails
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a set of mathematical expressions with varying complexity
    // Each expression uses three variables: x, y, and z
    let n_runs = 10_000;
    let n_equations = 1000;
    let expressions: Vec<String> = (0..n_equations)
        .map(|i| match i % 10 {
            0 => format!(
                "abs(2*x) + y^{}/z + sqrt(z^{}) + w*v",
                (i % 5) + 2,
                (i % 3) + 2
            ),
            1 => format!(
                "exp(x/{}) + y^{}/sqrt(z) + ln(z+{}*x) + w/v",
                (i % 4) + 2,
                (i % 3) + 2,
                (i % 3) + 1
            ),
            2 => format!(
                "sqrt(x^{}/y + y^{}/z + z^{}/x) + v*w",
                (i % 3) + 2,
                (i % 4) + 2,
                (i % 5) + 2
            ),
            3 => format!(
                "(x+{}*y)^2 + (y+{}*z)^2 + (z+{}*x)^2 + w^2 + v^2",
                (i % 3) + 1,
                (i % 4) + 1,
                (i % 5) + 1
            ),
            4 => format!(
                "ln(x+{}) + exp(y/{}) + z^{} + sqrt(w*v)",
                (i % 4) + 2,
                (i % 3) + 2,
                (i % 5) + 2
            ),
            5 => format!(
                "(x*y)/(z+{}) + (y*z)/(x+{}) + (z*x)/(y+{}) + (w*v)/(x+1)",
                (i % 3) + 1,
                (i % 4) + 1,
                (i % 5) + 1
            ),
            6 => format!(
                "sqrt(x^{}) + sqrt(y^{}) + ln(z^{}) + exp(w) + v^2",
                (i % 3) + 2,
                (i % 4) + 3,
                (i % 5) + 4
            ),
            7 => format!(
                "(x^2 + {}*y^2)/(z + {}) + sqrt({}*x*y*z) + w*v",
                (i % 3) + 1,
                (i % 4) + 1,
                (i % 5) + 1
            ),
            8 => format!(
                "(x^{} + y^{})/(z^{} + 1) + ln(w + v)",
                (i % 4) + 2,
                (i % 3) + 2,
                (i % 5) + 2
            ),
            9 => format!(
                "sqrt(x^2 + {}*y^2)/(z^{} + ln(x + {})) + w/v",
                (i % 5) + 1,
                (i % 3) + 2,
                (i % 4) + 1
            ),
            _ => unreachable!(),
        })
        .collect();

    // Convert string expressions into compiled Equation objects
    let start = std::time::Instant::now();
    let equations: Vec<Equation> = expressions
        .par_iter()
        .map(|expr| Equation::new(expr.clone()))
        .collect::<Result<_, _>>()?;
    let duration = start.elapsed();
    println!("(Individual) Total compilation time: {duration:?} for {n_equations} equations");

    // Create a system that combines all equations into one optimized unit
    let start = std::time::Instant::now();
    let system = EquationSystem::new(expressions)?;
    let duration = start.elapsed();
    println!("(System) Total compilation time: {duration:?} for {n_equations} equations\n");

    // Test input values for x, y, and z respectively
    let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0]; // x, y, z, w, v

    // Time individual evaluations
    let time_individual = time_it(
        "Individual equations",
        || {
            for eq in &equations {
                eq.eval(inputs).unwrap();
            }
        },
        n_runs,
    );

    // Time sequential system evaluation
    let time_system = time_it(
        "System (sequential)",
        || {
            system.eval(inputs).unwrap();
        },
        n_runs,
    );

    // After the sequential system timing block, add parallel evaluation:
    let batch_input = (0..n_runs)
        .map(|i| vec![i as f64, i as f64, i as f64, i as f64, i as f64])
        .collect::<Vec<_>>();

    let start = Instant::now();
    system.eval_parallel(&batch_input).unwrap();
    let duration = start.elapsed();
    println!("System (parallel): {duration:?} for {n_runs} runs");

    // Calculate average time per equation
    let ns_per_eq_individual =
        (time_individual * 1_000_000_000.0) / (n_runs as f64 * equations.len() as f64);
    let ns_per_eq_sequential =
        (time_system * 1_000_000_000.0) / (n_runs as f64 * equations.len() as f64);
    let ns_per_eq_parallel =
        (duration.as_secs_f64() * 1_000_000_000.0) / (n_runs as f64 * equations.len() as f64);

    println!("\nPerformance Analysis:");
    println!("Individual: {ns_per_eq_individual:.2}ns per equation");
    println!("System: {ns_per_eq_sequential:.2}ns per equation");
    println!("System (parallel): {ns_per_eq_parallel:.2}ns per equation");

    Ok(())
}

/// Measures and reports the execution time of a given function over multiple iterations.
///
/// # Arguments
/// * `name` - Description of the operation being timed, used in output message
/// * `f` - The function to benchmark, provided as a closure
/// * `n` - Number of iterations to run
///
/// # Returns
/// The total execution time in seconds
///
/// # Example
/// ```no_run
/// let duration = time_it("My operation", || {
///     // Code to benchmark
/// }, 1000);
/// ```
fn time_it<F: Fn()>(name: &str, f: F, n: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..n {
        f();
    }
    let duration = start.elapsed();
    println!("{name}: Takes {duration:?} for {n} runs");
    duration.as_secs_f64()
}
