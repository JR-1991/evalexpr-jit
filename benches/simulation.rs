//! # Chemical Kinetics Simulation Benchmark
//!
//! This benchmark compares the performance of different approaches for solving
//! ordinary differential equations (ODEs) in chemical kinetics simulations.
//!
//! ## System Description
//!
//! The benchmark simulates a chemical reaction system with:
//! - 7 state variables representing concentrations
//! - Michaelis-Menten enzyme kinetics: rate = (vmax * S) / (km + S)
//! - Product inhibition terms: -kie * P
//! - 21 differential equations total (3 per enzyme: negative rate, positive rate, inhibition)
//!
//! ## Implementations Compared
//!
//! 1. **Direct Implementation**: Hand-coded Rust implementation with explicit
//!    mathematical operations. This represents the theoretical performance ceiling
//!    for this specific system.
//!
//! 2. **Evalexpr-JIT Implementation**: Uses the evalexpr-jit library to parse
//!    mathematical expressions from strings and compile them to optimized machine
//!    code using Cranelift JIT compilation.
//!
//! ## Benchmark Details
//!
//! - **ODE Solver**: Dormand-Prince 5th order (Dopri5) adaptive step-size method
//! - **Time Range**: 0.0 to 150.0 time units
//! - **Initial Step Size**: 0.1
//! - **Tolerances**: Absolute 1e-4, Relative 1e-8
//! - **Initial Conditions**: S=1000.0, P=100.0, E=10.0
//! - **Parameters**: vmax=0.85, km=150.0, kie=0.01 (all enzymes)
//!
//! ## Performance Considerations
//!
//! The evalexpr-jit implementation includes several optimizations:
//! - Pre-populated parameter arrays to minimize allocations
//! - Static lifetime for the equation system to avoid repeated compilation
//! - Efficient parameter passing using slices
//!
//! This benchmark helps evaluate the overhead of JIT compilation versus
//! direct implementation for computationally intensive ODE solving scenarios.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use evalexpr_jit::system::EquationSystem;
use nalgebra::U7;
use ode_solvers::dopri5::*;
use ode_solvers::*;

type State = OVector<f64, U7>;

/// Direct implementation of the chemical kinetics system.
///
/// This struct implements the ODE system using hand-coded mathematical operations.
/// It serves as a performance baseline representing the theoretical maximum
/// performance for this specific system without any parsing or JIT overhead.
struct DirectSystem {
    vmax: [f64; 7], // Maximum reaction velocities for each enzyme
    km: [f64; 7],   // Michaelis constants for each enzyme
    kie: [f64; 7],  // Inhibition constants for each enzyme
}

impl DirectSystem {
    fn new(vmax: [f64; 7], km: [f64; 7], kie: [f64; 7]) -> Self {
        Self { vmax, km, kie }
    }
}

impl System<f64, State> for DirectSystem {
    #[inline(always)]
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        let s = y[0]; // Substrate concentration
        let p = y[1]; // Product concentration

        // Direct calculation matching the jitted system structure
        // Each enzyme contributes 3 equations: negative rate, positive rate, inhibition
        for i in 0..7 {
            match i % 3 {
                0 => {
                    // Negative enzyme rate: -(vmax_i * S) / (km_i + S)
                    let enzyme_idx = i / 3;
                    let rate = (self.vmax[enzyme_idx] * s) / (self.km[enzyme_idx] + s);
                    dy[i] = -rate;
                }
                1 => {
                    // Positive enzyme rate: (vmax_i * S) / (km_i + S)
                    let enzyme_idx = i / 3;
                    let rate = (self.vmax[enzyme_idx] * s) / (self.km[enzyme_idx] + s);
                    dy[i] = rate;
                }
                2 => {
                    // Inhibition term: -kie_i * P
                    let enzyme_idx = i / 3;
                    dy[i] = -self.kie[enzyme_idx] * p;
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Evalexpr-JIT implementation of the chemical kinetics system.
///
/// This struct uses the evalexpr-jit library to evaluate mathematical expressions
/// that were parsed from strings and compiled to machine code. It demonstrates
/// the performance characteristics of JIT-compiled mathematical expressions
/// in a realistic ODE solving context.
struct EvalexprSystem {
    system: &'static EquationSystem, // JIT-compiled equation system
    params: [f64; 25],               // Pre-allocated parameter buffer
}

impl EvalexprSystem {
    /// Creates a new evalexpr system with optimized parameter handling.
    ///
    /// The parameter array is pre-populated with constant values to minimize
    /// runtime overhead. Only dynamic values (S and P concentrations) are
    /// updated during each function call.
    fn new(system: &'static EquationSystem, vmax: [f64; 7], km: [f64; 7], kie: [f64; 7]) -> Self {
        let mut params = [0.0; 25];
        // Pre-populate the constant parameters
        // params[0] and params[1] will be set dynamically for S and P
        let mut idx = 2;
        for i in 0..7 {
            params[idx] = vmax[i]; // vmax_i
            params[idx + 1] = km[i]; // km_i
            params[idx + 2] = kie[i]; // kie_i
            idx += 3;
        }

        Self { system, params }
    }
}

impl System<f64, State> for EvalexprSystem {
    #[inline(always)]
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        // Use pre-populated parameter buffer, only update dynamic values
        let mut params = self.params;
        params[0] = y[0]; // S (substrate concentration)
        params[1] = y[1]; // P (product concentration)

        // Call the JIT-compiled function to evaluate all equations at once
        self.system.fun()(&params, dy.as_mut_slice());
    }
}

/// Runs a complete ODE simulation using the specified system implementation.
///
/// This function sets up initial conditions and integrates the ODE system
/// from t=0 to t=150 using the Dormand-Prince 5th order method with
/// adaptive step size control.
///
/// # Arguments
/// * `system` - The ODE system implementation to benchmark
/// * `s0` - Initial substrate concentration
/// * `p0` - Initial product concentration  
/// * `e0` - Initial enzyme concentration
fn run_simulation<S: System<f64, State>>(system: S, s0: f64, p0: f64, e0: f64) {
    let mut y0 = State::zeros();
    y0[0] = s0; // Substrate
    y0[1] = p0; // Product
    y0[2] = e0; // Enzyme

    let mut stepper = Dopri5::new(
        system, 0.0,    // t0 - initial time
        150.0,  // tf - final time
        0.1,    // dt - initial step size
        y0,     // y0 - initial state
        1.0e-4, // abs_tol - absolute tolerance
        1.0e-8, // rel_tol - relative tolerance
    );

    let _ = stepper.integrate();
}

/// Main benchmark function comparing direct and JIT implementations.
///
/// This function sets up the benchmark parameters, creates the equation system
/// for the JIT implementation, and runs performance comparisons between
/// the direct Rust implementation and the evalexpr-jit implementation.
fn benchmark_simulations(c: &mut Criterion) {
    // System parameters - identical for both implementations
    let vmax = [0.85; 7]; // Maximum velocities
    let km = [150.0; 7]; // Michaelis constants
    let kie = [0.01; 7]; // Inhibition constants

    // Create and leak the equation system for evalexpr-jit implementation
    // The system is leaked to obtain a static reference, avoiding repeated
    // compilation overhead during benchmarking
    let system = Box::leak(Box::new(
        EquationSystem::new({
            let mut equations = Vec::new();
            // Generate equations for each enzyme (3 equations per enzyme)
            for i in 1..=7 {
                equations.push(format!("-(vmax_{} * S) / (km_{} + S)", i, i)); // Negative rate
                equations.push(format!("(vmax_{} * S) / (km_{} + S)", i, i)); // Positive rate
                equations.push(format!("-kie_{} * P", i)); // Inhibition
            }
            equations
        })
        .expect("Failed to create equation system"),
    ));

    let mut group = c.benchmark_group("Chemical Kinetics Simulation");

    // Benchmark the direct implementation
    group.bench_function("Direct Implementation", |b| {
        b.iter(|| {
            let system = DirectSystem::new(vmax, km, kie);
            run_simulation(black_box(system), 1000.0, 100.0, 10.0);
        })
    });

    // Benchmark the evalexpr-jit implementation
    group.bench_function("Evalexpr Implementation", |b| {
        b.iter(|| {
            let system = EvalexprSystem::new(system, vmax, km, kie);
            run_simulation(black_box(system), 1000.0, 100.0, 10.0);
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_simulations);
criterion_main!(benches);
