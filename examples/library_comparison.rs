//! Benchmarks comparing evalexpr-jit against fasteval for equation evaluation performance.
//!
//! # Overview
//! This example:
//! - Generates a set of complex mathematical expressions
//! - Benchmarks sequential and parallel evaluation using fasteval
//! - Benchmarks sequential and parallel evaluation using evalexpr-jit
//! - Reports timing metrics for compilation and evaluation
//! - Compares performance between libraries and execution modes
//!
//! # Metrics Reported
//! - JIT compilation time for evalexpr-jit
//! - Sequential evaluation time per equation
//! - Parallel evaluation time per equation
//! - Average nanoseconds per equation evaluation

use colored::Colorize;
use evalexpr_jit::system::EquationSystem;
use evalexpr_jit::Equation;
use fasteval::{Compiler, Evaler};
use std::collections::BTreeMap;
use std::time::Instant;

/// Main benchmark runner that compares fasteval and evalexpr-jit performance.
///
/// Runs benchmarks with configurable number of equations and iterations.
/// Reports timing metrics with colored output for readability.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let n_runs = 100_000;
    let n_equations = 500;

    // Generate test expressions
    let expressions = generate_test_expressions(n_equations);

    // Run benchmarks
    println!(
        "\n{}",
        "=== Fasteval Implementation ===".bright_blue().bold()
    );
    benchmark_fasteval(&expressions, n_runs)?;

    println!(
        "\n{}",
        "=== EvalExpr Single Implementation ==="
            .bright_green()
            .bold()
    );
    benchmark_evalexpr_single(&expressions, n_runs)?;

    println!(
        "\n{}",
        "=== EvalExpr JIT Implementation ===".bright_green().bold()
    );
    benchmark_evalexpr_system(&expressions, n_runs)?;

    Ok(())
}

/// Generates a set of test mathematical expressions with varying complexity.
///
/// # Arguments
/// * `n_equations` - Number of unique expressions to generate
///
/// # Returns
/// A vector of strings containing mathematical expressions using variables x, y, z, w, v
fn generate_test_expressions(n_equations: usize) -> Vec<String> {
    (0..n_equations)
        .map(|i| match i % 10 {
            0 => format!("2*x + y^{}/z + sqrt(z^{}) + w*v", (i % 5) + 2, (i % 3) + 2),
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
        .collect()
}

/// Benchmarks expression evaluation using fasteval.
///
/// # Arguments
/// * `expressions` - Vector of mathematical expressions to evaluate
/// * `n_runs` - Number of evaluation iterations to perform
///
/// # Returns
/// Result indicating success or fasteval error
fn benchmark_fasteval(expressions: &[String], n_runs: usize) -> Result<(), fasteval::Error> {
    // Setup
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    let mut map = BTreeMap::new();

    // Compile expressions
    let compiled_exprs: Vec<_> = expressions
        .iter()
        .map(|expr| {
            parser
                .parse(expr, &mut slab.ps)
                .unwrap()
                .from(&slab.ps)
                .compile(&slab.ps, &mut slab.cs)
        })
        .collect();

    // Initialize variables
    map.insert("x", 1.0);
    map.insert("y", 2.0);
    map.insert("z", 3.0);
    map.insert("w", 4.0);
    map.insert("v", 5.0);

    // Benchmark evaluation
    let start = Instant::now();

    for _ in 0..n_runs {
        for compiled in &compiled_exprs {
            let _val = compiled.eval(&slab, &mut map);
        }
    }
    let duration = start.elapsed();

    // Print results
    println!(
        "Time taken: {}",
        format!(
            "{:?} for {} runs and {} equations",
            duration,
            n_runs,
            expressions.len()
        )
        .bright_yellow()
        .italic()
    );

    let ns_per_eq =
        (duration.as_secs_f64() * 1_000_000_000.0) / (n_runs as f64 * expressions.len() as f64);
    println!(
        "Average: {}",
        format!("{:.2}ns per equation", ns_per_eq)
            .bright_cyan()
            .bold()
    );

    Ok(())
}

fn benchmark_evalexpr_single(
    expressions: &[String],
    n_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let equations: Vec<Equation> = expressions
        .iter()
        .map(|e| Equation::new(e.to_string()).unwrap())
        .collect();
    let duration_jit = start.elapsed();
    let duration_str = format!("{:.3}s", duration_jit.as_secs_f64());
    println!("JIT Compilation: {}", duration_str);

    let start = Instant::now();
    for i in 0..n_runs {
        let inputs = &[
            (i + 1) as f64,
            (i + 2) as f64,
            (i + 3) as f64,
            (i + 4) as f64,
            (i + 5) as f64,
        ];
        for eq in &equations {
            let _val = eq.eval(inputs).unwrap();
        }
    }
    let duration = start.elapsed();
    println!(
        "Single Sequential: {}",
        format!(
            "{:?} for {} runs and {} equations",
            duration,
            n_runs,
            expressions.len()
        )
        .bright_yellow()
        .italic()
    );

    let ns_per_eq =
        (duration.as_secs_f64() * 1_000_000_000.0) / (n_runs as f64 * expressions.len() as f64);
    println!(
        "Single Average: {}",
        format!("{:.2}ns per equation", ns_per_eq)
            .bright_cyan()
            .bold()
    );

    Ok(())
}

/// Benchmarks expression evaluation using evalexpr-jit EquationSystem.
///
/// # Arguments
/// * `expressions` - Vector of mathematical expressions to evaluate
/// * `n_runs` - Number of evaluation iterations to perform
///
/// # Returns
/// Result indicating success or error
fn benchmark_evalexpr_system(
    expressions: &[String],
    n_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize system
    let start = Instant::now();
    let system = EquationSystem::new(expressions.to_vec())?;
    let duration_system = start.elapsed();
    println!(
        "JIT Compilation: {}",
        format!("{:.3}s", duration_system.as_secs_f64())
            .bright_yellow()
            .italic()
    );

    // Prepare test data
    let batch_input = (0..n_runs)
        .map(|i| vec![i as f64, i as f64, i as f64, i as f64, i as f64])
        .collect::<Vec<_>>();

    // Benchmark sequential evaluation
    let start = Instant::now();
    for i in 0..n_runs {
        let inputs = &[
            (i + 1) as f64,
            (i + 2) as f64,
            (i + 3) as f64,
            (i + 4) as f64,
            (i + 5) as f64,
        ];
        let _val = system.eval(inputs).unwrap();
    }
    let duration_system = start.elapsed();
    println!(
        "System Sequential: {}",
        format!(
            "{:?} for {} runs and {} equations",
            duration_system,
            n_runs,
            expressions.len()
        )
        .bright_yellow()
        .italic()
    );

    // Benchmark parallel evaluation
    let start = Instant::now();
    system.eval_parallel(&batch_input).unwrap();
    let duration_parallel = start.elapsed();
    println!(
        "System Parallel: {}",
        format!(
            "{:?} for {} runs and {} equations",
            duration_parallel,
            n_runs,
            expressions.len()
        )
        .bright_yellow()
        .italic()
    );

    // Calculate and display metrics
    let ns_per_eq_system = (duration_system.as_secs_f64() * 1_000_000_000.0)
        / (n_runs as f64 * expressions.len() as f64);
    let ns_per_eq_parallel = (duration_parallel.as_secs_f64() * 1_000_000_000.0)
        / (n_runs as f64 * expressions.len() as f64);

    println!(
        "Sequential Average: {}",
        format!("{:.2}ns per equation", ns_per_eq_system)
            .bright_cyan()
            .bold()
    );
    println!(
        "Parallel Average: {}",
        format!("{:.2}ns per equation", ns_per_eq_parallel)
            .bright_magenta()
            .bold()
    );

    Ok(())
}
