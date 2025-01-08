# EvalExpr-JIT

A high-performance mathematical expression evaluator with JIT compilation and automatic differentiation support. Builds on top of [evalexpr](https://github.com/ISibboI/evalexpr) and [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift).

## Features

- 🚀 JIT compilation for fast expression evaluation
- 📊 Automatic differentiation (up to any order)
- 🔢 Support for multiple variables with consistent ordering
- 🧮 Higher-order partial derivatives
- 📐 Jacobian matrix computation
- 🔄 Batch evaluation of equation systems

> This crate is still under development and the API is subject to change.

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
    
    // Compute gradient (vector of partial derivatives)
    let gradient = eq.gradient(&[1.0, 2.0])?;
    assert_eq!(gradient, vec![2.0, 4.0]); // [∂/∂x, ∂/∂y] = [2, 2y]
    
    // Compute Hessian matrix (matrix of second derivatives)
    let hessian = eq.hessian(&[1.0, 2.0])?;
    assert_eq!(hessian, vec![
        vec![0.0, 0.0], // [∂²/∂x², ∂²/∂x∂y]
        vec![0.0, 2.0], // [∂²/∂y∂x, ∂²/∂y²]
    ]);
    
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
        "2*x + y".to_string(),   // first equation
        "x^2 + z".to_string(),   // second equation
    ])?;
    
    // Variables are automatically sorted (x, y, z)
    // Input values must be provided in the same order
    let results = system.eval(&[1.0, 2.0, 3.0])?;
    assert_eq!(results, vec![
        4.0,  // eq1: 2*1 + 2 = 4
        4.0   // eq2: 1^2 + 3 = 4
    ]); 
    
    // Get the sorted variable names to ensure correct input ordering
    println!("Variables: {:?}", system.sorted_variables); // ["x", "y", "z"]
    
    Ok(())
}
```

## Advanced Usage

### Single Equation Derivatives

The `Equation` struct provides multiple ways to compute derivatives, from simple partial derivatives to higher-order mixed derivatives:

```rust
use evalexpr_jit::Equation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eq = Equation::new("x^2 * y^2".to_string())?;

    // Get first-order partial derivative
    let dx = eq.derivative("x")?;
    let result = dx(&[2.0, 3.0]);
    assert_eq!(result, 24.0); // d/dx[x^2*y^2] = 2x*y^2 = 2*2*3^2

    // Compute higher-order mixed derivative
    let dxdy = eq.derive_wrt(&["x", "y"])?;
    let result = dxdy(&[2.0, 3.0]);
    assert_eq!(result, 12.0); // d²/dxdy[x^2*y^2] = 4xy
    
    Ok(())
}
```

### System Derivatives and Jacobian

The `EquationSystem` struct provides tools for analyzing the derivatives of multiple equations simultaneously:

```rust
use evalexpr_jit::system::EquationSystem;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = EquationSystem::new(vec![
        "x^2*y".to_string(),   // f1
        "x*y^2".to_string(),   // f2
    ])?;

    // Compute Jacobian matrix at point (2,3)
    let jacobian = system.jacobian(&[2.0, 3.0])?;

    // Jacobian matrix:
    // [∂f1/∂x  ∂f1/∂y] = [12.0  4.0]   // derivatives of f1
    // [∂f2/∂x  ∂f2/∂y] = [9.0   12.0]  // derivatives of f2
    assert_eq!(jacobian[0], vec![12.0, 4.0]);  // derivatives of f1
    assert_eq!(jacobian[1], vec![9.0, 12.0]);  // derivatives of f2

    // Compute higher-order derivatives of the system
    let d2 = system.derive_wrt(&["x", "y"])?;
    let results = d2(&[2.0, 3.0]);
    assert_eq!(results, vec![
        4.0,  // d²/dxdy[x^2*y] = 2x
        6.0   // d²/dxdy[x*y^2] = 2y
    ]);

    Ok(())
}
```

### Notes on Parameters and Variables

When working with expressions that contain both parameters and independent variables, you can use `derive_wrt_stack` to compute derivatives with respect to parameters only. This is particularly useful for parameter estimation and optimization problems.

```rust
use evalexpr_jit::Equation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Expression with parameters (a, b) and variables (x, y)
    let eq = Equation::new("a*x^2 + b*y^2".to_string())?;
    
    // Returns a JITFunction that computes the gradient of the 
    // equation with respect to the parameters only
    let param_gradient = eq.derive_wrt_stack(&["a", "b"])?;
    
    // Values provided in alphabetical order: [a, b, x, y]
    let result = param_gradient(&[2.0, 3.0, 1.0, 2.0]);
    assert_eq!(result, vec![1.0, 4.0]); // [∂/∂a = x^2, ∂/∂b = y^2]
    
    Ok(())
}
```

Note that variables are always sorted alphabetically when providing input values, if no specific order is provided. In the example above, the order is `[a, b, x, y]`. You can check the ordering using `eq.sorted_variables`. If you want to provide a specific order, you can use `from_var_map` and provide a map of variable names to indices:

```rust
let eq = Equation::from_var_map(vec!["a*x^2 + b*y^2".to_string()], &["a", "b", "x", "y"])?;
```

## API Reference

### `Equation`

The basic struct for single equation evaluation:

- `new(equation: String) -> Result<Self, EquationError>`
- `eval(&self, values: &[f64]) -> Result<f64, EquationError>`
- `gradient(&self, values: &[f64]) -> Result<Vec<f64>, EquationError>`
- `hessian(&self, values: &[f64]) -> Result<Vec<Vec<f64>>, EquationError>`
- `derivative(&self, variable: &str) -> Result<JITFunction, EquationError>`
- `derive_wrt(&self, variables: &[&str]) -> Result<JITFunction, EquationError>`
- `derive_wrt_stack(&self, variables: &[&str]) -> Result<JITFunction, EquationError>`

### `EquationSystem`

For evaluating systems of equations:

- `new(expressions: Vec<String>) -> Result<Self, EquationError>`
- `from_var_map(expressions: Vec<String>, variable_map: &HashMap<String, u32>) -> Result<Self, EquationError>`
- `eval(&self, inputs: &[f64]) -> Result<Vec<f64>, EquationError>`
- `gradient(&self, inputs: &[f64], variable: &str) -> Result<Vec<f64>, EquationError>`
- `jacobian(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>, EquationError>`
- `derive_wrt(&self, variables: &[&str]) -> Result<CombinedJITFunction, EquationError>`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
