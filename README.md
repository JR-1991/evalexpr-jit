# EvalExpr-JIT

A high-performance mathematical expression evaluator with JIT compilation and automatic differentiation support. Builds on top of [evalexpr](https://github.com/ISibboI/evalexpr) and [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift).

## Features

- 🚀 JIT compilation for fast expression evaluation
- 📊 Automatic differentiation (up to any order)
- 🔢 Support for multiple variables
- 🧮 Higher-order partial derivatives

> This crate is still under development and the API is subject to change. Not all mathematical operations are supported yet.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
evalexpr-jit = "0.1.0"  # Replace with actual version
```

## Quick Start

```rust
use evalexpr_jit::Equation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new equation
    let eq = Equation::new("2*x + y^2".to_string())?;
    
    // Evaluate at point (x=1, y=2)
    let result = eq.eval(&[1.0, 2.0]);
    assert_eq!(result, 6.0); // 2*1 + 2^2 = 6
    
    // Compute gradient
    let gradient = eq.gradient(&[1.0, 2.0]);
    assert_eq!(gradient, vec![2.0, 4.0]); // [∂/∂x, ∂/∂y] = [2, 2y]
    
    Ok(())
}
```

## Advanced Usage

### Computing Derivatives

```rust
use evalexpr_jit::Equation;

let eq = Equation::new("x^2 * y^2".to_string())?;

// Get specific partial derivative
let dx = eq.derivative("x")?;
let values = vec![2.0, 3.0];
let result = dx(values.as_ptr());

// Compute higher-order derivatives
let dxdy = eq.derive_wrt(&["x", "y"])?;
let result = dxdy(values.as_ptr());

// Compute Hessian matrix
let hessian = eq.hessian(&[2.0, 3.0]);
```

## API Reference

### `Equation`

The main struct representing a mathematical equation. Key methods include:

- `new(equation_str: String) -> Result<Self, EquationError>`
- `eval(&self, values: &[f64]) -> f64`
- `gradient(&self, values: &[f64]) -> Vec<f64>`
- `hessian(&self, values: &[f64]) -> Vec<Vec<f64>>`
- `derivative(&self, variable: &str) -> Result<&JITFunction, EquationError>`
- `derive_wrt(&self, variables: &[&str]) -> Result<JITFunction, EquationError>`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
