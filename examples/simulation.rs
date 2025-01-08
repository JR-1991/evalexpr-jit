//! This example demonstrates how evalexpr-jit can be used to efficiently simulate chemical kinetics.
//! Specifically, it shows how to:
//! 1. Create a JIT-compiled rate equation using evalexpr-jit's Equation type
//! 2. Integrate that equation into an ODE solver (dopri5 from ode_solvers)
//! 3. Simulate Michaelis-Menten enzyme kinetics as a practical example
//!
//! The Michaelis-Menten model describes the rate of enzyme-catalyzed reactions:
//! v = Vmax * [S] / (Km + [S])
//! where:
//! - v is the reaction rate
//! - Vmax is the maximum reaction velocity
//! - [S] is the substrate concentration
//! - Km is the Michaelis constant

use std::collections::HashMap;

use evalexpr_jit::Equation;
use ode_solvers::dopri5::*;
use ode_solvers::*;

// The state vector represents concentrations of chemical species:
// [S] - Substrate concentration in mM
// [P] - Product concentration in mM
type State = Vector2<f64>;
type Precision = f64;

/// Represents the Michaelis-Menten enzyme kinetics system
struct MichaelisMenten {
    vmax: f64,          // Maximum reaction velocity (mM/s)
    km: f64,            // Michaelis constant (mM)
    equation: Equation, // JIT-compiled rate equation
}

impl MichaelisMenten {
    /// Creates a new instance of the Michaelis-Menten system with given parameters
    ///
    /// # Arguments
    /// * `vmax` - Maximum reaction velocity in mM/s
    /// * `km` - Michaelis constant in mM
    fn new(vmax: f64, km: f64) -> Self {
        // Create a variable mapping for the JIT compiler
        // This maps variable names to their positions in the input array
        let var_map = HashMap::from([
            ("S".to_string(), 0),    // Substrate concentration will be at index 0
            ("vmax".to_string(), 1), // Vmax parameter will be at index 1
            ("km".to_string(), 2),   // Km parameter will be at index 2
        ]);

        // Create a JIT-compiled equation using evalexpr-jit
        // The equation represents the Michaelis-Menten rate law
        let equation = Equation::from_var_map("(vmax * S) / (km + S)".to_string(), var_map)
            .expect("Failed to create Michaelis-Menten equation");

        Self { vmax, km, equation }
    }
}

/// Implementation of the ODE system for the Michaelis-Menten model
impl System<Precision, State> for MichaelisMenten {
    /// Computes the time derivatives for the system state
    ///
    /// # Arguments
    /// * `_t` - Current time (unused in this autonomous system)
    /// * `y` - Current state vector ([S], [P])
    /// * `dy` - Output vector for derivatives (d[S]/dt, d[P]/dt)
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        // Evaluate the JIT-compiled rate equation with current values
        let v = self
            .equation
            .eval(&[y[0], self.vmax, self.km])
            .expect("Failed to evaluate equation");

        dy[0] = -v; // Rate of substrate consumption (negative because it's being consumed)
        dy[1] = v; // Rate of product formation (equal to substrate consumption)
    }
}

fn main() {
    // Define kinetic parameters for the simulation
    let vmax = 1.5; // Maximum reaction velocity (mM/s)
    let km = 2.0; // Michaelis constant (mM)

    // Set initial concentrations
    let s0 = 10.0; // Initial substrate concentration (mM)
    let p0 = 0.0; // Initial product concentration (mM)
    let y0 = State::new(s0, p0);

    // Define simulation time parameters
    let t0 = 0.0; // Start time (s)
    let tf = 150.0; // End time (s)
    let dt = 0.01; // Initial time step (s)

    // Create the Michaelis-Menten system with JIT-compiled rate equation
    let system = MichaelisMenten::new(vmax, km);

    // Initialize the ODE solver with specified tolerances
    let mut stepper = Dopri5::new(system, t0, tf, dt, y0, 1.0e-4, 1.0e-8);

    // Run the simulation
    let res = stepper.integrate();

    // Print simulation statistics
    match res {
        Ok(stats) => {
            println!("Simulation successful!");
            println!("Number of evaluations: {}", stats.num_eval);
            println!("Number of accepted steps: {}", stats.accepted_steps);
            println!("Number of rejected steps: {}", stats.rejected_steps);
        }
        Err(e) => println!("An error occurred: {}", e),
    }
}
