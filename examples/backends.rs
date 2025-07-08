//! Benchmarks comparing different vector backend implementations in evalexpr-jit.
//!
//! This example compares the performance of:
//! - Vec<f64>
//! - nalgebra::DVector
//! - ndarray::Array1
//!
//! For each backend, it measures:
//! - JIT compilation time
//! - Sequential evaluation time
//! - Parallel evaluation time
//! - Average nanoseconds per evaluation

use colored::Colorize;
use evalexpr_jit::system::EquationSystem;
use nalgebra::DVector;
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let n_runs = 10_000;
    let n_equations = 10;

    // Generate test expressions
    let expressions = generate_test_expressions(n_equations);

    println!("\n{}", "=== Vec<f64> Backend ===".bright_blue().bold());
    benchmark_vec_backend(&expressions, n_runs)?;

    println!("\n{}", "=== nalgebra Backend ===".bright_green().bold());
    benchmark_nalgebra_backend(&expressions, n_runs)?;

    println!("\n{}", "=== ndarray Backend ===".bright_yellow().bold());
    benchmark_ndarray_backend(&expressions, n_runs)?;

    Ok(())
}

/// Generates test expressions similar to those in library_comparison
fn generate_test_expressions(n_equations: usize) -> Vec<String> {
    (0..n_equations)
        .map(|i| match i % 5 {
            0 => "2*x + y^2/z + sqrt(z^2) + w*v".to_string(),
            1 => "exp(x/2) + y^2/sqrt(z) + ln(z+x) + w/v".to_string(),
            2 => "sqrt(x^2/y + y^2/z + z^2/x) + v*w".to_string(),
            3 => "(x+y)^2 + (y+z)^2 + (z+x)^2 + w^2 + v^2".to_string(),
            4 => "ln(x+2) + exp(y/2) + z^2 + sqrt(w*v)".to_string(),
            _ => unreachable!(),
        })
        .collect()
}

fn benchmark_vec_backend(
    expressions: &[String],
    n_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize system
    let start = Instant::now();
    let system = EquationSystem::new(expressions.to_vec())?;
    let duration_jit = start.elapsed();
    println!(
        "JIT Compilation: {}",
        format!("{:.3}s", duration_jit.as_secs_f64())
            .bright_yellow()
            .italic()
    );

    // Prepare test data
    let batch_input = (0..n_runs)
        .map(|i| vec![i as f64, i as f64, i as f64, i as f64, i as f64])
        .collect::<Vec<_>>();

    // Sequential evaluation
    let start = Instant::now();
    for i in 0..n_runs {
        let inputs = vec![
            (i + 1) as f64,
            (i + 2) as f64,
            (i + 3) as f64,
            (i + 4) as f64,
            (i + 5) as f64,
        ];
        let _result = system.eval(&inputs)?;
    }
    let duration_seq = start.elapsed();

    // Parallel evaluation
    let start = Instant::now();
    let _results = system.eval_parallel(&batch_input)?;
    let duration_par = start.elapsed();

    print_metrics(n_runs, expressions.len(), duration_seq, duration_par);
    Ok(())
}

fn benchmark_nalgebra_backend(
    expressions: &[String],
    n_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize system
    let start = Instant::now();
    let system = EquationSystem::new(expressions.to_vec())?;
    let duration_jit = start.elapsed();
    println!(
        "JIT Compilation: {}",
        format!("{:.3}s", duration_jit.as_secs_f64())
            .bright_yellow()
            .italic()
    );

    // Prepare test data
    let batch_input = (0..n_runs)
        .map(|i| DVector::from_vec(vec![i as f64, i as f64, i as f64, i as f64, i as f64]))
        .collect::<Vec<_>>();

    // Sequential evaluation
    let start = Instant::now();
    for i in 0..n_runs {
        let inputs = DVector::from_vec(vec![
            (i + 1) as f64,
            (i + 2) as f64,
            (i + 3) as f64,
            (i + 4) as f64,
            (i + 5) as f64,
        ]);
        let _result = system.eval(&inputs)?;
    }
    let duration_seq = start.elapsed();

    // Parallel evaluation
    let start = Instant::now();
    let _results = system.eval_parallel(&batch_input)?;
    let duration_par = start.elapsed();

    print_metrics(n_runs, expressions.len(), duration_seq, duration_par);
    Ok(())
}

fn benchmark_ndarray_backend(
    expressions: &[String],
    n_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize system
    let start = Instant::now();
    let system = EquationSystem::new(expressions.to_vec())?;
    let duration_jit = start.elapsed();
    println!(
        "JIT Compilation: {}",
        format!("{:.3}s", duration_jit.as_secs_f64())
            .bright_yellow()
            .italic()
    );

    // Prepare test data
    let batch_input = (0..n_runs)
        .map(|i| Array1::from_vec(vec![i as f64, i as f64, i as f64, i as f64, i as f64]))
        .collect::<Vec<_>>();

    // Sequential evaluation
    let start = Instant::now();
    for i in 0..n_runs {
        let inputs = Array1::from_vec(vec![
            (i + 1) as f64,
            (i + 2) as f64,
            (i + 3) as f64,
            (i + 4) as f64,
            (i + 5) as f64,
        ]);
        let _result = system.eval(&inputs)?;
    }
    let duration_seq = start.elapsed();

    // Parallel evaluation
    let start = Instant::now();
    let _results = system.eval_parallel(&batch_input)?;
    let duration_par = start.elapsed();

    print_metrics(n_runs, expressions.len(), duration_seq, duration_par);
    Ok(())
}

fn print_metrics(
    n_runs: usize,
    n_equations: usize,
    duration_seq: std::time::Duration,
    duration_par: std::time::Duration,
) {
    // Sequential metrics
    println!(
        "Sequential: {}",
        format!("{duration_seq:?} for {n_runs} runs and {n_equations} equations")
            .bright_yellow()
            .italic()
    );

    let ns_per_eq_seq =
        (duration_seq.as_secs_f64() * 1_000_000_000.0) / (n_runs as f64 * n_equations as f64);
    println!(
        "Sequential Average: {}",
        format!("{ns_per_eq_seq:.2}ns per equation")
            .bright_cyan()
            .bold()
    );

    // Parallel metrics
    println!(
        "Parallel: {}",
        format!("{duration_par:?} for {n_runs} runs and {n_equations} equations")
            .bright_yellow()
            .italic()
    );

    let ns_per_eq_par =
        (duration_par.as_secs_f64() * 1_000_000_000.0) / (n_runs as f64 * n_equations as f64);
    println!(
        "Parallel Average: {}",
        format!("{ns_per_eq_par:.2}ns per equation")
            .bright_magenta()
            .bold()
    );
}
