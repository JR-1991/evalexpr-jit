//! This example demonstrates using evalexpr-jit for efficient chemical kinetics simulation.
//!
//! # Overview
//! The example shows how to:
//! - Create JIT-compiled rate equations using evalexpr-jit's EquationSystem
//! - Integrate equations with an ODE solver (dopri5 from ode_solvers)
//! - Simulate multiple parallel chemical reactions
//! - Achieve high performance through parallel execution
//!
//! # Chemical System
//! The system models multiple parallel reactions following Michaelis-Menten kinetics:
//!
//! ```text
//! S -> P  (7 parallel reactions)
//! ```
//!
//! Each reaction converts substrate S to product P with rate:
//! ```text
//! v = Vmax * [S] / (Km + [S])
//! ```
//! where:
//! - v is reaction rate
//! - Vmax is maximum velocity
//! - [S] is substrate concentration
//! - Km is Michaelis constant
//!
//! # Implementation Details
//! - Uses JIT compilation for efficient rate equation evaluation
//! - Employs Dormand-Prince (DOPRI5) adaptive step-size integrator
//! - Parallelizes simulation across multiple initial conditions
//! - Optimizes memory usage with stack allocation
//! - Handles multiple reactions with shared substrate/product pools

use evalexpr_jit::system::EquationSystem;
use nalgebra::U7;
use ode_solvers::dopri5::*;
use ode_solvers::*;
use rayon::prelude::*;

/// State vector type for the chemical system.
/// Contains concentrations [S] and [P] as a 7D vector (for 7 reactions)
type State = OVector<f64, U7>;

/// Represents a chemical reaction system following Michaelis-Menten kinetics
struct ChemicalSystem {
    /// JIT-compiled system of differential equations (now static)
    system: &'static EquationSystem,
    /// Maximum reaction velocity (Vmax) in concentration/time
    vmax: [f64; 7],
    /// Michaelis constant (Km) in concentration units
    km: [f64; 7],
    /// Kinetic constant for enzyme inhibition (Kie) in concentration units
    kie: [f64; 7],
}

impl ChemicalSystem {
    /// Creates a new ChemicalSystem with specified kinetic parameters
    fn new(system: &'static EquationSystem, vmax: [f64; 7], km: [f64; 7], kie: [f64; 7]) -> Self {
        Self {
            system,
            vmax,
            km,
            kie,
        }
    }
}

/// Implementation of the ODE system trait required by the solver
impl System<f64, State> for ChemicalSystem {
    #[inline(always)]
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        // Stack allocation instead of heap for small arrays
        let mut params = [0.0; 25];

        // Direct array assignments instead of push operations
        params[0] = y[0]; // S
        params[1] = y[1]; // P

        // Unroll the loop for better optimization
        #[allow(clippy::manual_memcpy)]
        {
            let mut idx = 2;
            for i in 0..7 {
                params[idx] = self.vmax[i];
                params[idx + 1] = self.km[i];
                params[idx + 2] = self.kie[i];
                idx += 3;
            }
        }

        // Pre-allocated buffer on stack
        let mut buffer = [0.0; 21];

        // Call the JIT-compiled function
        self.system.fun()(&params, &mut buffer);

        // Replace copy_from_slice with nalgebra's iterator
        for i in 0..7 {
            dy[i] = buffer[i];
        }
    }
}

/// Runs a single simulation with given initial conditions
#[allow(clippy::too_many_arguments)]
#[inline]
fn run_simulation(
    dt: f64,
    s0: f64,
    p0: f64,
    e0: f64,
    vmax: [f64; 7],
    km: [f64; 7],
    kie: [f64; 7],
    system: &'static EquationSystem,
) -> Result<(), String> {
    let mut y0 = State::zeros();
    y0[0] = s0;
    y0[1] = p0;
    y0[2] = e0;

    let t0 = 0.0;
    let tf = 150.0;

    let system = ChemicalSystem::new(system, vmax, km, kie);

    // Configure solver with optimized parameters
    let mut stepper = Dopri5::new(
        system, t0, tf, dt, y0, 1.0e-4, // absolute tolerance
        1.0e-8, // relative tolerance
    );

    match stepper.integrate() {
        Ok(_) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

fn main() {
    // Initialize the thread pool once at the start
    // Use if let to handle the case where it's already initialized
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
    {
        eprintln!(
            "Thread pool initialization warning (may be already initialized): {e}"
        );
    }

    let system = Box::leak(Box::new(
        EquationSystem::new({
            let mut equations = Vec::new();
            // Generate equations for all 7 reactions
            for i in 1..=7 {
                equations.push(format!("-(vmax_{i} * S) / (km_{i} + S)"));
                equations.push(format!("(vmax_{i} * S) / (km_{i} + S)"));
                equations.push(format!("-kie_{i} * P"));
            }
            equations
        })
        .expect("Failed to create equation system"),
    ));

    let dt = 1e-1;

    // Extended parameter arrays for 7 reactions
    let vmax = [0.85; 7]; // Changed from 13 to 7
    let km = [150.0; 7]; // Changed from 13 to 7
    let kie = [0.01; 7]; // Changed from 13 to 7

    let initial_conditions: Vec<(f64, f64, f64)> = (100..=5000)
        .into_par_iter() // Parallelize the initialization itself
        .map(|i| (i as f64 * 0.5, i as f64 * 0.1, i as f64 * 0.01))
        .collect();

    let start = std::time::Instant::now();
    let results: Vec<_> = initial_conditions
        .par_chunks(100) // Process in chunks for better cache utilization
        .flat_map(|chunk| {
            chunk
                .iter()
                .map(|&(s0, p0, e0)| run_simulation(dt, s0, p0, e0, vmax, km, kie, system))
                .collect::<Vec<_>>()
        })
        .collect();

    let duration = start.elapsed();
    println!(
        "Total parallel execution time: {:?} for {} simulations in parallel with dt={} and {} equations",
        duration,
        initial_conditions.len(),
        dt,
        system.num_equations()
    );

    // Print only errors if they occur
    for result in results {
        if let Err(e) = result {
            eprintln!("Error: {e}");
        }
    }
}
