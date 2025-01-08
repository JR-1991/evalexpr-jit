//! Functions for linking and calling the square root function in JIT-compiled code.
//!
//! This module provides functionality to:
//! - Link the external square root function (sqrt) into JIT-compiled code
//! - Generate Cranelift IR instructions to call sqrt within compiled functions
//!
//! The square root function operates on 64-bit floating point numbers (f64).

use cranelift::prelude::FunctionBuilder;
use cranelift_codegen::ir::types::F64; // since sqrt works with doubles
use cranelift_codegen::ir::{AbiParam, InstBuilder};
use cranelift_module::{FuncId, Linkage, Module};

/// Links the square root function to make it available for JIT compilation.
///
/// This function declares the external sqrt function to the Cranelift module,
/// making it available for use in JIT-compiled code. It creates a function
/// signature matching the standard sqrt function: f64 -> f64.
///
/// # Arguments
/// * `module` - The Cranelift module to declare the function in
///
/// # Returns
/// * `Ok(FuncId)` - The function ID that can be used to call sqrt
/// * `Err(String)` - Error message if declaration fails
pub fn link_sqrt(module: &mut dyn Module) -> Result<FuncId, String> {
    // Create signature for sqrt(f64) -> f64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));

    // Declare the function
    let func_id = module
        .declare_function("sqrt", Linkage::Import, &sig)
        .map_err(|e| e.to_string())?;

    Ok(func_id)
}

/// Generates Cranelift IR instructions to call the square root function.
///
/// This helper function adds instructions to call the previously linked sqrt
/// function within a function being built with Cranelift.
///
/// # Arguments
/// * `builder` - The Cranelift function builder being used to construct the function
/// * `module` - The Cranelift module containing the function declaration
/// * `func_id` - The function ID returned by link_sqrt()
/// * `arg` - The Cranelift IR value to pass as the argument to sqrt
///
/// # Returns
/// The Cranelift IR value containing the result of calling sqrt
pub fn call_sqrt(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    func_id: cranelift_module::FuncId,
    arg: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let func = module.declare_func_in_func(func_id, builder.func);
    let call = builder.ins().call(func, &[arg]);
    builder.inst_results(call)[0]
}
