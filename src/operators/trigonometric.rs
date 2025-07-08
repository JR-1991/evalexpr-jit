//! Functions for linking and calling trigonometric functions in JIT-compiled code.
//!
//! This module provides functionality to:
//! - Link external trigonometric functions (sin, cos) into JIT-compiled code
//! - Generate Cranelift IR instructions to call trigonometric functions within compiled functions
//!
//! All trigonometric functions operate on 64-bit floating point numbers (f64) and expect
//! arguments in radians.

use cranelift::prelude::FunctionBuilder;
use cranelift_codegen::ir::types::F64;
use cranelift_codegen::ir::{AbiParam, InstBuilder};
use cranelift_module::{FuncId, Linkage, Module};

/// Links the sine function to make it available for JIT compilation.
///
/// This function declares the external sin function to the Cranelift module,
/// making it available for use in JIT-compiled code. It creates a function
/// signature matching the standard sin function: f64 -> f64.
///
/// # Arguments
/// * `module` - The Cranelift module to declare the function in
///
/// # Returns
/// * `Ok(FuncId)` - The function ID that can be used to call sin
/// * `Err(String)` - Error message if declaration fails
pub fn link_sin(module: &mut dyn Module) -> Result<FuncId, String> {
    // Create signature for sin(f64) -> f64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));

    // Declare the function
    let func_id = module
        .declare_function("sin", Linkage::Import, &sig)
        .map_err(|e| e.to_string())?;

    Ok(func_id)
}

/// Generates Cranelift IR instructions to call the sine function.
///
/// This helper function adds instructions to call the previously linked sin
/// function within a function being built with Cranelift.
///
/// # Arguments
/// * `builder` - The Cranelift function builder being used to construct the function
/// * `module` - The Cranelift module containing the function declaration
/// * `func_id` - The function ID returned by link_sin()
/// * `arg` - The Cranelift IR value to pass as the argument to sin (in radians)
///
/// # Returns
/// The Cranelift IR value containing the result of calling sin
pub fn call_sin(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    func_id: cranelift_module::FuncId,
    arg: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let func = module.declare_func_in_func(func_id, builder.func);
    let call = builder.ins().call(func, &[arg]);
    builder.inst_results(call)[0]
}

/// Links the cosine function to make it available for JIT compilation.
///
/// This function declares the external cos function to the Cranelift module,
/// making it available for use in JIT-compiled code. It creates a function
/// signature matching the standard cos function: f64 -> f64.
///
/// # Arguments
/// * `module` - The Cranelift module to declare the function in
///
/// # Returns
/// * `Ok(FuncId)` - The function ID that can be used to call cos
/// * `Err(String)` - Error message if declaration fails
pub fn link_cos(module: &mut dyn Module) -> Result<FuncId, String> {
    // Create signature for cos(f64) -> f64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));

    // Declare the function
    let func_id = module
        .declare_function("cos", Linkage::Import, &sig)
        .map_err(|e| e.to_string())?;

    Ok(func_id)
}

/// Generates Cranelift IR instructions to call the cosine function.
///
/// This helper function adds instructions to call the previously linked cos
/// function within a function being built with Cranelift.
///
/// # Arguments
/// * `builder` - The Cranelift function builder being used to construct the function
/// * `module` - The Cranelift module containing the function declaration
/// * `func_id` - The function ID returned by link_cos()
/// * `arg` - The Cranelift IR value to pass as the argument to cos (in radians)
///
/// # Returns
/// The Cranelift IR value containing the result of calling cos
pub fn call_cos(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    func_id: cranelift_module::FuncId,
    arg: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let func = module.declare_func_in_func(func_id, builder.func);
    let call = builder.ins().call(func, &[arg]);
    builder.inst_results(call)[0]
}
