//! Performance comparison example for equation systems.
//!
//! This example demonstrates the performance difference between evaluating mathematical
//! equations individually versus as a combined system. It shows how combining multiple
//! equations into a single system can improve performance through:
//! - Shared IR (Intermediate Representation) optimization
//! - Reduced overhead from function calls
//! - Better CPU cache utilization
//!
//! The test evaluates 17 mathematical equations with varying complexity, running them
//! 10 million times both individually and as a system to get reliable timing data.

use evalexpr_jit::{system::EquationSystem, Equation};
use std::time::Instant;

/// Main function that runs the performance comparison
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a set of mathematical expressions with varying complexity
    // Each expression uses three variables: x, y, and z
    let expressions = vec![
        "2*x + y^2/z + sqrt(z^2)".to_string(),
        "x^2/y + 2*y + ln(z^2)".to_string(),
        "sqrt(x^2) + y^2/(z+1) + 2*z".to_string(),
        "3*x + y^3/sqrt(z) + ln(x+1)".to_string(),
        "x/ln(y+1) + 2*y^2 + sqrt(3*z^2)".to_string(),
        "exp(x/2) + y^2/sqrt(z) + ln(z+x)".to_string(),
        "sqrt(x^2/y + y^2/z + z^2/x)".to_string(),
        "x^3 + y^3 + z^3".to_string(),
        "ln(x+2) + exp(y/3) + z^2".to_string(),
        "sqrt(x+y) + sqrt(y+z) + ln(z+x)".to_string(),
        "(x+y)^2 + (y+z)^2 + (z+x)^2".to_string(),
        "exp(x/3) + ln(y+2) + sqrt(z^3)".to_string(),
        "x/(y+1) + y/(z+1) + z/(x+1)".to_string(),
        "(x*y)/(z+1) + (y*z)/(x+1) + (z*x)/(y+1)".to_string(),
        "sqrt(x^2) + sqrt(y^3) + ln(z^4)".to_string(),
        "(x^2 + y^2)/(z + 1) + sqrt(x*y*z)".to_string(),
        "exp(x/4) + (y^2)/(z+2) + ln(x+y+z)".to_string(),
    ];

    // Convert string expressions into compiled Equation objects
    let equations: Vec<Equation> = expressions
        .iter()
        .map(|expr| Equation::new(expr.clone()))
        .collect::<Result<_, _>>()?;

    // Create a system that combines all equations into one optimized unit
    let system = EquationSystem::new(expressions)?;

    // Test input values for x, y, and z respectively
    let inputs = &[1.0, 2.0, 3.0];

    // Time individual evaluations
    let time_individual = time_it(
        "Individual equations",
        || {
            for eq in &equations {
                eq.eval(inputs).unwrap();
            }
        },
        10_000_000,
    );

    // Time sequential system evaluation
    let time_system = time_it(
        "System (sequential)",
        || {
            system.eval(inputs).unwrap();
        },
        10_000_000,
    );

    // Calculate average time per equation
    let ns_per_eq_individual =
        (time_individual * 1_000_000_000.0) / (10_000_000.0 * equations.len() as f64);
    let ns_per_eq_sequential =
        (time_system * 1_000_000_000.0) / (10_000_000.0 * equations.len() as f64);

    println!("\nPerformance Analysis:");
    println!("Individual: {:.2}ns per equation", ns_per_eq_individual);
    println!("System: {:.2}ns per equation", ns_per_eq_sequential);

    Ok(())
}

/// Measures the execution time of a given function over multiple iterations
///
/// # Arguments
/// * `name` - Description of the operation being timed (for logging)
/// * `f` - The function to benchmark
/// * `n` - Number of iterations to run
///
/// # Returns
/// The total execution time in seconds as an f64
///
/// This function runs the provided closure `n` times and measures the total duration.
/// It prints the results and returns the total time in seconds for further analysis.
fn time_it<F: Fn()>(name: &str, f: F, n: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..n {
        f();
    }
    let duration = start.elapsed();
    println!("{}: Takes {:?} for {} runs", name, duration, n);
    duration.as_secs_f64()
}
