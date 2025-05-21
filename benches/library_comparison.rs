use criterion::{criterion_group, criterion_main, Criterion};
use evalexpr_jit::system::EquationSystem;
use evalexpr_jit::Equation;
use fasteval::{Compiler, Evaler};
use std::collections::BTreeMap;

// ... existing code for generate_test_expressions() ...

fn run_benchmarks(c: &mut Criterion) {
    let expressions = generate_test_expressions(500);
    let mut group = c.benchmark_group("Expression Evaluation");

    // FastEval setup
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    let mut map = BTreeMap::new();
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
    map.insert("x", 1.0);
    map.insert("y", 2.0);
    map.insert("z", 3.0);
    map.insert("w", 4.0);
    map.insert("v", 5.0);

    // EvalExpr Single setup
    let equations: Vec<Equation> = expressions
        .iter()
        .map(|e| Equation::new(e.to_string()).unwrap())
        .collect();
    let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0];

    // EvalExpr System setup
    let system = EquationSystem::new(expressions.clone()).unwrap();
    let system_inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Benchmark FastEval
    group.bench_function("FastEval", |b| {
        b.iter(|| {
            for compiled in &compiled_exprs {
                let _val = compiled.eval(&slab, &mut map);
            }
        })
    });

    // Benchmark EvalExpr Single
    group.bench_function("EvalExpr Single", |b| {
        b.iter(|| {
            for eq in &equations {
                let _val = eq.eval(inputs).unwrap();
            }
        })
    });

    // Benchmark EvalExpr System
    group.bench_function("EvalExpr System (no buffer)", |b| {
        b.iter(|| {
            let _val = system.eval(&system_inputs).unwrap();
        })
    });

    let mut buffer = vec![0.0; expressions.len()];
    group.bench_function("EvalExpr System (buffer)", |b| {
        b.iter(|| {
            system.eval_into(&system_inputs, &mut buffer).unwrap();
        })
    });

    group.finish();
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

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);
