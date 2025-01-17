# EvalExpr-JIT

![Crates.io Version](https://img.shields.io/crates/v/evalexpr-jit) ![Build Status](https://github.com/JR-1991/evalexpr-jit/actions/workflows/test.yml/badge.svg) 

A high-performance mathematical expression evaluator with JIT compilation and automatic differentiation support. Builds on top of [evalexpr](https://github.com/ISibboI/evalexpr) and [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift).

> This crate is still under development and the API is subject to change.

## Features

- 🚀 JIT compilation for fast expression evaluation
- 📊 Automatic differentiation (up to any order)
- 🔢 Support for multiple variables with consistent ordering
- 🧮 Higher-order partial derivatives
- 📐 Jacobian matrix computation
- 🔄 Batch evaluation of equation systems


Check out the [API Reference](https://docs.rs/evalexpr-jit/latest/evalexpr_jit/) for more details.

## Installation

Install the crate from crates.io:

```sh
cargo add evalexpr-jit
```

or add this to your `Cargo.toml`:

```toml
[dependencies]
evalexpr-jit = "0.1.2"  # Replace with actual version
```

## Quick Start

### Single Equation

The `Equation` struct provides a simple way to evaluate mathematical expressions and compute their derivatives. Variables are automatically detected from the expression and ordered alphabetically.

```rust
use evalexpr_jit::Equation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a single equation
    let eq = Equation::new("2*x + y^2".to_string())?;
    
    // Evaluate at point (x=1, y=2)
    let result = eq.eval(&[1.0, 2.0])?;
    assert_eq!(result, 6.0); // 2*1 + 2^2 = 6
    
    // Compute first-order derivative
    let dx = eq.derivative("x")?;
    let result = dx(&[1.0, 2.0]);
    assert_eq!(result, 2.0); // d/dx[2x + y^2] = 2
    
    // Compute higher-order mixed derivative
    let dxy = eq.derive_wrt(&["x", "y"])?;
    let result = dxy(&[1.0, 2.0]);
    assert_eq!(result, 0.0); // d²/dxdy[2x + y^2] = 0
    
    Ok(())
}
```

### System of Equations

The `EquationSystem` struct allows you to evaluate multiple equations simultaneously, sharing variables across equations for efficient computation. Variables are automatically collected from all equations and consistently ordered.

```rust
use evalexpr_jit::system::EquationSystem;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a system of equations
    let system = EquationSystem::new(vec![
        "x^2*y".to_string(),  // f1
        "x*y^2".to_string(),  // f2
    ])?;
    
    // Evaluate at point (x=2, y=3)
    let results = system.eval(&[2.0, 3.0])?;
    assert_eq!(results.as_slice(), &[
        12.0,  // f1: 2^2 * 3 = 12
        18.0   // f2: 2 * 3^2 = 18
    ]); 
    
    // Get the sorted variable names
    println!("Variables: {:?}", system.sorted_variables()); // ["x", "y"]
    
    Ok(())
}
```

## Advanced Usage

### System Derivatives and Jacobian

```rust
use evalexpr_jit::system::EquationSystem;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = EquationSystem::new(vec![
        "x^2*y".to_string(),  // f1
        "x*y^2".to_string(),  // f2
    ])?;

    // Compute Jacobian matrix at point (2,3)
    let jacobian = system.eval_jacobian(&[2.0, 3.0], None)?;
    assert_eq!(jacobian[0], vec![12.0, 4.0]);  // derivatives of f1 [∂f1/∂x, ∂f1/∂y]
    assert_eq!(jacobian[1], vec![9.0, 12.0]);  // derivatives of f2 [∂f2/∂x, ∂f2/∂y]

    // Create optimized Jacobian computer for specific variables
    let jacobian_fn = system.jacobian_wrt(&["x", "y"])?;
    let mut results = Array2::zeros((2, 2));
    jacobian_fn.eval_into_matrix(&[2.0, 3.0], &mut results)?;

    // Compute higher-order derivatives
    let d2 = system.derive_wrt(&["x", "y"])?;
    let mut results = vec![0.0; 2];
    d2.eval_into(&[2.0, 3.0], &mut results)?;
    assert_eq!(results, vec![
        4.0,  // d²/dxdy[x^2*y] = 2x
        6.0   // d²/dxdy[x*y^2] = 2y
    ]);

    Ok(())
}
```

### Batch Evaluation

```rust
use evalexpr_jit::system::EquationSystem;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = EquationSystem::new(vec![
        "x^2*y".to_string(),
        "x*y^2".to_string(),
    ])?;

    // Evaluate multiple input sets in parallel
    let input_sets = vec![
        vec![2.0, 3.0],
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ];

    let results = system.eval_parallel(&input_sets)?;
    assert_eq!(results[0].as_slice(), &[12.0, 18.0]); // [2^2 * 3, 2 * 3^2]
    assert_eq!(results[1].as_slice(), &[2.0, 4.0]);   // [1^2 * 2, 1 * 2^2]
    assert_eq!(results[2].as_slice(), &[36.0, 48.0]); // [3^2 * 4, 3 * 4^2]

    Ok(())
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
