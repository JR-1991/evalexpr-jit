//! Expression Evaluation Benchmarks
//!
//! This benchmark suite compares the performance of JIT-compiled mathematical expressions
//! against direct Rust implementations. It measures both evaluation speed and compilation time
//! to provide insights into the overhead and benefits of JIT compilation for mathematical
//! expression evaluation.
//!
//! ## Benchmark Structure
//!
//! The benchmarks are organized into two main groups:
//!
//! ### 1. Expression Evaluation (`benchmark_expressions`)
//! Compares the runtime performance of evaluating mathematical expressions using:
//! - **Direct Evaluation**: Hand-written Rust functions that directly compute the result
//! - **JIT Evaluation**: Expressions compiled to machine code using Cranelift JIT compiler
//!
//! Each expression is tested with appropriate input parameters and measured for throughput.
//! The JIT compilation overhead is excluded from these measurements since equations are
//! pre-compiled during setup.
//!
//! ### 2. Compilation Time (`benchmark_compilation_time`)
//! Measures the time required to parse, analyze, and JIT-compile expressions from string
//! form into executable machine code. This helps understand the one-time setup cost of
//! using JIT compilation.
//!
//! ## Test Expressions
//!
//! The benchmark includes expressions of varying complexity:
//! - **Simple operations**: Basic arithmetic (addition, multiplication)
//! - **Linear expressions**: Polynomial degree 1
//! - **Quadratic expressions**: Polynomial degree 2 with constants
//! - **Polynomial expressions**: Higher-degree polynomials with multiple variables
//! - **Complex expressions**: Nested operations, division, square roots
//! - **Multi-variable expressions**: Functions of 2-3 variables with complex interactions
//!
//! ## Performance Expectations
//!
//! - **Simple expressions**: Direct evaluation should be faster due to JIT overhead
//! - **Complex expressions**: JIT compilation may show benefits from optimizations
//! - **Compilation time**: Should be reasonable for interactive use cases
//!
//! ## Usage
//!
//! Run with: `cargo bench --bench expressions`
//!
//! The results help determine when JIT compilation provides performance benefits
//! over direct evaluation, particularly for complex mathematical expressions that
//! are evaluated many times.

use std::{f64::consts::PI, hint::black_box};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use evalexpr_jit::Equation;

/// Direct evaluation of mathematical expressions
///
/// This struct provides hand-written Rust implementations of the mathematical
/// expressions being benchmarked. These serve as the baseline for comparison
/// against JIT-compiled versions.
struct DirectEvaluator;

impl DirectEvaluator {
    /// Evaluates: a + 1.1
    /// Simple addition with a constant
    fn evaluate_simple_add(a: f64) -> f64 {
        a + 1.1
    }

    /// Evaluates: a * 2.2
    /// Simple multiplication with a constant
    fn evaluate_simple_mult(a: f64) -> f64 {
        a * 2.2
    }

    /// Evaluates: 2.2 * a + 1.1
    /// Linear expression (degree 1 polynomial)
    fn evaluate_linear(a: f64) -> f64 {
        2.2 * a + 1.1
    }

    /// Evaluates: (2.2 * a + 1.1) * 3.3
    /// Quadratic-like expression with nested operations
    fn evaluate_quadratic(a: f64) -> f64 {
        (2.2 * a + 1.1) * 3.3
    }

    /// Evaluates: a^2 / (2 * π / b) - a / 2.2
    /// Polynomial with division and multiple variables
    fn evaluate_polynomial(a: f64, b: f64) -> f64 {
        a * a / (2.0 * PI / b) - a / 2.2
    }

    /// Evaluates: (a^3 + 2*a^2 - 5*a + 1) / (b^2 + 3*b + 2)
    /// Complex rational function with cubic numerator and quadratic denominator
    fn evaluate_complex_poly(a: f64, b: f64) -> f64 {
        (a * a * a + 2.0 * a * a - 5.0 * a + 1.0) / (b * b + 3.0 * b + 2.0)
    }

    /// Evaluates: ((a + b) * (a - b)) / ((c + 1) * (c - 1))
    /// Nested expression with difference of squares pattern
    fn evaluate_nested_expr(a: f64, b: f64, c: f64) -> f64 {
        ((a + b) * (a - b)) / ((c + 1.0) * (c - 1.0))
    }

    /// Evaluates: a^3 + b^2 - 2*a*b + 5
    /// Multi-variable polynomial with mixed terms
    fn evaluate_power_expr(a: f64, b: f64) -> f64 {
        a.powf(3.0) + b.powf(2.0) - 2.0 * a * b + 5.0
    }

    /// Evaluates: sqrt(1 - 2.2*a + π/b/3.3)
    /// Expression involving square root and division chain
    fn evaluate_sqrt_expr(a: f64, b: f64) -> f64 {
        (1.0 - 2.2 * a + PI / b / 3.3).sqrt()
    }

    /// Evaluates: (a^3 + b^2*c - 2*a*b + c) / ((a+b)*(b+c)*(a+c) + 1) + sqrt(a*b*c) - sqrt((a+b+c)^3)
    /// Very complex expression with multiple operations, nested terms, and square roots
    fn evaluate_very_complex(a: f64, b: f64, c: f64) -> f64 {
        (a * a * a + b * b * c - 2.0 * a * b + c) / ((a + b) * (b + c) * (a + c) + 1.0)
            + (a * b * c).sqrt()
            - ((a + b + c).powi(3)).sqrt()
    }
}

/// JIT evaluator using evalexpr_jit
///
/// This struct manages JIT-compiled versions of the mathematical expressions.
/// All equations are pre-compiled during initialization to exclude compilation
/// overhead from the evaluation benchmarks.
struct JitEvaluator {
    equations: Vec<&'static Equation>,
}

impl JitEvaluator {
    /// Creates a new JIT evaluator with pre-compiled equations
    ///
    /// All equations are compiled once during initialization and stored as
    /// static references to avoid repeated compilation overhead during benchmarking.
    fn new() -> Self {
        // Create and leak all equations to get static references
        // This ensures compilation happens only once, not during each benchmark iteration
        let simple_add = Box::leak(Box::new(
            Equation::new("a + 1.1".to_string()).expect("Failed to create simple add equation"),
        ));

        let simple_mult = Box::leak(Box::new(
            Equation::new("a * 2.2".to_string()).expect("Failed to create simple mult equation"),
        ));

        let linear = Box::leak(Box::new(
            Equation::new("2.2 * a + 1.1".to_string()).expect("Failed to create linear equation"),
        ));

        let quadratic = Box::leak(Box::new(
            Equation::new("(2.2 * a + 1.1) * 3.3".to_string())
                .expect("Failed to create quadratic equation"),
        ));

        let polynomial = Box::leak(Box::new(
            Equation::new("a^2 / (2 * 3.14159 / b) - a / 2.2".to_string())
                .expect("Failed to create polynomial equation"),
        ));

        let complex_poly = Box::leak(Box::new(
            Equation::new("(a^3 + 2*a^2 - 5*a + 1) / (b^2 + 3*b + 2)".to_string())
                .expect("Failed to create complex poly equation"),
        ));

        let nested_expr = Box::leak(Box::new(
            Equation::new("((a + b) * (a - b)) / ((c + 1) * (c - 1))".to_string())
                .expect("Failed to create nested expr equation"),
        ));

        let power_expr = Box::leak(Box::new(
            Equation::new("a^3 + b^2 - 2*a*b + 5".to_string())
                .expect("Failed to create power expr equation"),
        ));

        let sqrt_expr = Box::leak(Box::new(
            Equation::new("sqrt(1 - 2.2*a + 3.14159/b/3.3)".to_string())
                .expect("Failed to create sqrt expr equation"),
        ));

        let very_complex = Box::leak(Box::new(
            Equation::new("(a^3 + b^2*c - 2*a*b + c) / ((a+b)*(b+c)*(a+c) + 1) + sqrt(a*b*c) - sqrt((a+b+c)^3)".to_string())
                .expect("Failed to create very complex equation"),
        ));

        Self {
            equations: vec![
                simple_add,
                simple_mult,
                linear,
                quadratic,
                polynomial,
                complex_poly,
                nested_expr,
                power_expr,
                sqrt_expr,
                very_complex,
            ],
        }
    }

    /// Evaluates a pre-compiled equation with the given parameters
    ///
    /// # Arguments
    /// * `expr_idx` - Index of the equation to evaluate
    /// * `params` - Input parameters for the equation
    ///
    /// # Returns
    /// The result of evaluating the equation with the given parameters
    fn evaluate(&self, expr_idx: usize, params: &[f64]) -> f64 {
        self.equations[expr_idx].fun()(params)
    }
}

/// Benchmarks expression evaluation performance
///
/// Compares the runtime performance of direct Rust implementations against
/// JIT-compiled versions for various mathematical expressions. The JIT compilation
/// overhead is excluded since equations are pre-compiled.
fn benchmark_expressions(c: &mut Criterion) {
    let jit_eval = JitEvaluator::new();

    // Test parameters for expressions with different arities
    let params_1 = [2.5]; // Single variable expressions
    let params_2 = [2.5, 1.8]; // Two variable expressions
    let params_3 = [2.5, 1.8, 0.7]; // Three variable expressions

    // Test cases with expression name, description, and required parameters
    let test_cases = vec![
        ("simple_add", "a + 1.1", &params_1[..]),
        ("simple_mult", "a * 2.2", &params_1[..]),
        ("linear", "2.2*a + 1.1", &params_1[..]),
        ("quadratic", "(2.2*a + 1.1) * 3.3", &params_1[..]),
        ("polynomial", "a^2 / (2*π/b) - a/2.2", &params_2[..]),
        (
            "complex_poly",
            "(a^3 + 2*a^2 - 5*a + 1) / (b^2 + 3*b + 2)",
            &params_2[..],
        ),
        (
            "nested_expr",
            "((a+b)*(a-b)) / ((c+1)*(c-1))",
            &params_3[..],
        ),
        ("power_expr", "a^3 + b^2 - 2*a*b + 5", &params_2[..]),
        ("sqrt_expr", "sqrt(1 - 2.2*a + π/b/3.3)", &params_2[..]),
        (
            "very_complex",
            "Complex multi-term expression",
            &params_3[..],
        ),
    ];

    let mut group = c.benchmark_group("Expression Evaluation");

    for (i, (name, _expr, params)) in test_cases.iter().enumerate() {
        // Direct evaluation benchmarks - hand-written Rust implementations
        group.bench_with_input(
            BenchmarkId::new("Direct", name),
            &(i, params),
            |b, &(expr_idx, params)| {
                b.iter(|| {
                    let result = match expr_idx {
                        0 => DirectEvaluator::evaluate_simple_add(black_box(params[0])),
                        1 => DirectEvaluator::evaluate_simple_mult(black_box(params[0])),
                        2 => DirectEvaluator::evaluate_linear(black_box(params[0])),
                        3 => DirectEvaluator::evaluate_quadratic(black_box(params[0])),
                        4 => DirectEvaluator::evaluate_polynomial(
                            black_box(params[0]),
                            black_box(params[1]),
                        ),
                        5 => DirectEvaluator::evaluate_complex_poly(
                            black_box(params[0]),
                            black_box(params[1]),
                        ),
                        6 => DirectEvaluator::evaluate_nested_expr(
                            black_box(params[0]),
                            black_box(params[1]),
                            black_box(params[2]),
                        ),
                        7 => DirectEvaluator::evaluate_power_expr(
                            black_box(params[0]),
                            black_box(params[1]),
                        ),
                        8 => DirectEvaluator::evaluate_sqrt_expr(
                            black_box(params[0]),
                            black_box(params[1]),
                        ),
                        9 => DirectEvaluator::evaluate_very_complex(
                            black_box(params[0]),
                            black_box(params[1]),
                            black_box(params[2]),
                        ),
                        _ => unreachable!(),
                    };
                    black_box(result)
                })
            },
        );

        // JIT evaluation benchmarks - pre-compiled machine code
        group.bench_with_input(
            BenchmarkId::new("JIT", name),
            &(i, params),
            |b, &(expr_idx, params)| {
                b.iter(|| {
                    let result = jit_eval.evaluate(black_box(expr_idx), black_box(params));
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmarks JIT compilation time
///
/// Measures the time required to parse mathematical expressions from strings
/// and compile them into executable machine code. This represents the one-time
/// setup cost when using JIT compilation.
fn benchmark_compilation_time(c: &mut Criterion) {
    // Expression strings to compile - same as used in evaluation benchmarks
    let expressions = [
        "a + 1.1",
        "a * 2.2",
        "2.2 * a + 1.1",
        "(2.2 * a + 1.1) * 3.3",
        "a^2 / (2 * 3.14159 / b) - a / 2.2",
        "(a^3 + 2*a^2 - 5*a + 1) / (b^2 + 3*b + 2)",
        "((a + b) * (a - b)) / ((c + 1) * (c - 1))",
        "a^3 + b^2 - 2*a*b + 5",
        "sqrt(1 - 2.2*a + 3.14159/b/3.3)",
        "(a^3 + b^2*c - 2*a*b + c) / ((a+b)*(b+c)*(a+c) + 1) + sqrt(a*b*c) - sqrt((a+b+c)^3)",
    ];

    let mut group = c.benchmark_group("Compilation Time");

    for (i, expr) in expressions.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("Compile", format!("expr_{}", i)),
            expr,
            |b, expr| {
                b.iter(|| {
                    // Measure full compilation pipeline: parse -> analyze -> JIT compile
                    let equation = Equation::new(expr.to_string());
                    black_box(equation)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_expressions, benchmark_compilation_time);
criterion_main!(benches);
