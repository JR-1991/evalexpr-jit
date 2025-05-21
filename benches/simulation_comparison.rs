use criterion::{black_box, criterion_group, criterion_main, Criterion};
use evalexpr_jit::system::EquationSystem;
use nalgebra::U7;
use ode_solvers::dopri5::*;
use ode_solvers::*;

type State = OVector<f64, U7>;

// Direct implementation of the chemical system
struct DirectSystem {
    vmax: [f64; 7],
    km: [f64; 7],
    kie: [f64; 7],
}

impl DirectSystem {
    fn new(vmax: [f64; 7], km: [f64; 7], kie: [f64; 7]) -> Self {
        Self { vmax, km, kie }
    }
}

impl System<f64, State> for DirectSystem {
    #[inline(always)]
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        let s = y[0];
        let p = y[1];

        // Direct calculation of rates without evalexpr
        for i in 0..7 {
            let rate = (self.vmax[i] * s) / (self.km[i] + s);
            dy[i] = -rate - self.kie[i] * p;
        }
    }
}

// evalexpr-jit implementation (similar to the example)
struct EvalexprSystem {
    system: &'static EquationSystem,
    vmax: [f64; 7],
    km: [f64; 7],
    kie: [f64; 7],
}

impl EvalexprSystem {
    fn new(system: &'static EquationSystem, vmax: [f64; 7], km: [f64; 7], kie: [f64; 7]) -> Self {
        Self {
            system,
            vmax,
            km,
            kie,
        }
    }
}

impl System<f64, State> for EvalexprSystem {
    #[inline(always)]
    fn system(&self, _t: f64, y: &State, dy: &mut State) {
        let mut params = [0.0; 25];
        params[0] = y[0];
        params[1] = y[1];

        let mut idx = 2;
        for i in 0..7 {
            params[idx] = self.vmax[i];
            params[idx + 1] = self.km[i];
            params[idx + 2] = self.kie[i];
            idx += 3;
        }

        let mut buffer = [0.0; 21];
        self.system.fun()(&params, &mut buffer);

        for i in 0..7 {
            dy[i] = buffer[i];
        }
    }
}

fn run_simulation<S: System<f64, State>>(system: S, s0: f64, p0: f64, e0: f64) {
    let mut y0 = State::zeros();
    y0[0] = s0;
    y0[1] = p0;
    y0[2] = e0;

    let mut stepper = Dopri5::new(
        system, 0.0,   // t0
        150.0, // tf
        0.1,   // dt
        y0, 1.0e-4, // abs_tol
        1.0e-8, // rel_tol
    );

    let _ = stepper.integrate();
}

fn benchmark_simulations(c: &mut Criterion) {
    let vmax = [0.85; 7];
    let km = [150.0; 7];
    let kie = [0.01; 7];

    // Create and leak the equation system for evalexpr
    let system = Box::leak(Box::new(
        EquationSystem::new({
            let mut equations = Vec::new();
            for i in 1..=7 {
                equations.push(format!("-(vmax_{} * S) / (km_{} + S)", i, i));
                equations.push(format!("(vmax_{} * S) / (km_{} + S)", i, i));
                equations.push(format!("-kie_{} * P", i));
            }
            equations
        })
        .expect("Failed to create equation system"),
    ));

    let mut group = c.benchmark_group("Chemical Kinetics Simulation");

    group.bench_function("Direct Implementation", |b| {
        b.iter(|| {
            let system = DirectSystem::new(vmax, km, kie);
            run_simulation(black_box(system), 1000.0, 100.0, 10.0);
        })
    });

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
