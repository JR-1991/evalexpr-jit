//! Functions for linking and calling power functions in JIT-compiled code.
//!
//! This module provides functionality to:
//! - Link external power functions (pow, powf) into JIT-compiled code
//! - Generate Cranelift IR instructions to call power functions within compiled functions
//!
//! Power functions operate on 64-bit floating point numbers (f64).

use cranelift::prelude::FunctionBuilder;
use cranelift_codegen::ir::types::F64;
use cranelift_codegen::ir::{AbiParam, InstBuilder};
use cranelift_module::{FuncId, Linkage, Module};

/// Links the floating point power function to make it available for JIT compilation.
///
/// This function declares the external powf function to the Cranelift module,
/// making it available for use in JIT-compiled code. It creates a function
/// signature matching the standard powf function: (f64, f64) -> f64.
///
/// # Arguments
/// * `module` - The Cranelift module to declare the function in
///
/// # Returns
/// * `Ok(FuncId)` - The function ID that can be used to call powf
/// * `Err(String)` - Error message if declaration fails
pub fn link_powf(module: &mut dyn Module) -> Result<FuncId, String> {
    // Create signature for powf(f64, f64) -> f64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(F64)); // base
    sig.params.push(AbiParam::new(F64)); // exponent
    sig.returns.push(AbiParam::new(F64)); // result

    // Declare the function
    let func_id = module
        .declare_function("pow", Linkage::Import, &sig)
        .map_err(|e| e.to_string())?;

    Ok(func_id)
}

/// Generates Cranelift IR instructions to call the floating point power function.
///
/// This helper function adds instructions to call the previously linked powf
/// function within a function being built with Cranelift.
///
/// # Arguments
/// * `builder` - The Cranelift function builder being used to construct the function
/// * `module` - The Cranelift module containing the function declaration
/// * `func_id` - The function ID returned by link_powf()
/// * `base` - The Cranelift IR value for the base
/// * `exponent` - The Cranelift IR value for the exponent
///
/// # Returns
/// The Cranelift IR value containing the result of calling powf
pub fn call_powf(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    func_id: cranelift_module::FuncId,
    base: cranelift_codegen::ir::Value,
    exponent: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let func = module.declare_func_in_func(func_id, builder.func);
    let call = builder.ins().call(func, &[base, exponent]);
    builder.inst_results(call)[0]
}
