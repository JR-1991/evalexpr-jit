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
//!
//! This example implements a simple chemical reaction system:
//!     S -> P
//! where S is the substrate and P is the product. The reaction follows
//! Michaelis-Menten kinetics, meaning the rate depends on substrate concentration
//! in a non-linear way.
//!
//! The system is described by two differential equations:
//! d[S]/dt = -(Vmax * [S]) / (Km + [S])  # Rate of substrate consumption
//! d[P]/dt = (Vmax * [S]) / (Km + [S])   # Rate of product formation
//!
//! The simulation uses the Dormand-Prince (DOPRI5) method, which is an adaptive
//! step-size Runge-Kutta integrator well-suited for chemical kinetics problems.

use evalexpr_jit::system::EquationSystem;
use ode_solvers::dopri5::*;
use ode_solvers::*;

/// State vector type for the chemical system.
/// Contains concentrations of [S] and [P] as a 2D vector.
type State = Vector2<f64>;

/// Represents a chemical reaction system following Michaelis-Menten kinetics
struct ChemicalSystem {
    /// JIT-compiled system of differential equations
    system: EquationSystem,
    /// Maximum reaction velocity (Vmax) in concentration/time
    vmax: f64,
    /// Michaelis constant (Km) in concentration units
    km: f64,
}

impl ChemicalSystem {
    /// Creates a new ChemicalSystem with specified kinetic parameters
    ///
    /// # Arguments
    /// * `vmax` - Maximum reaction velocity (concentration/time)
    /// * `km` - Michaelis constant (concentration)
    ///
    /// # Returns
    /// A new ChemicalSystem instance with JIT-compiled rate equations
    fn new(vmax: f64, km: f64) -> Self {
        // Define the system equations
        let equations = vec![
            format!("-(vmax * S) / (km + S)"), // d[S]/dt
            format!("(vmax * S) / (km + S)"),  // d[P]/dt
        ];

        let system = EquationSystem::new(equations).expect("Failed to create equation system");

        Self { system, vmax, km }
    }
}

/// Implementation of the ODE system trait required by the solver
impl System<f64, State> for ChemicalSystem {
    /// Computes the time derivatives of the state variables
    ///
    /// # Arguments
    /// * `_t` - Current time (unused in this autonomous system)
    /// * `y` - Current state vector ([S], [P])
    /// * `dy` - Output vector for derivatives (d[S]/dt, d[P]/dt)
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        // Evaluate both equations with current state
        let results = self
            .system
            .eval(&[y[0], self.km, self.vmax])
            .expect("Failed to evaluate system");

        dy[0] = results[0]; // d[S]/dt
        dy[1] = results[1]; // d[P]/dt
    }
}

fn main() {
    // Define kinetic parameters
    let vmax = 1.5; // Maximum reaction velocity (mM/s)
    let km = 2.0; // Michaelis constant (mM)

    // Initial conditions
    let s0 = 10.0; // Initial substrate concentration (mM)
    let p0 = 0.0; // Initial product concentration (mM)
    let y0 = State::new(s0, p0);

    // Simulation parameters
    let t0 = 0.0; // Start time (s)
    let tf = 150.0; // End time (s)
    let dt = 1e-2; // Initial time step (s)

    // Create and run simulation
    let system = ChemicalSystem::new(vmax, km);
    // Initialize the DOPRI5 stepper with:
    // - relative tolerance: 1e-4
    // - absolute tolerance: 1e-8
    let mut stepper = Dopri5::new(system, t0, tf, dt, y0, 1.0e-4, 1.0e-8);

    match stepper.integrate() {
        Ok(stats) => {
            println!("Simulation successful!");
            println!("Number of evaluations: {}", stats.num_eval);
            println!("Number of accepted steps: {}", stats.accepted_steps);
            println!("Number of rejected steps: {}", stats.rejected_steps);
        }
        Err(e) => println!("An error occurred: {}", e),
    }
}
