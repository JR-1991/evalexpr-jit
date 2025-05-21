//! Functions for linking and calling the exponential function in JIT-compiled code.
//!
//! This module provides functionality to:
//! - Link the external exponential function (exp) into JIT-compiled code
//! - Generate IR instructions to call exp within compiled functions
//!
//! The exponential function operates on 64-bit floating point numbers (f64).

#[cfg(feature = "cranelift-backend")]
use cranelift::prelude::FunctionBuilder;
#[cfg(feature = "cranelift-backend")]
use cranelift_codegen::ir::types::F64; // since exp works with doubles
#[cfg(feature = "cranelift-backend")]
use cranelift_codegen::ir::{AbiParam, InstBuilder};
#[cfg(feature = "cranelift-backend")]
use cranelift_module::{FuncId, Linkage, Module as CraneliftModule};

#[cfg(feature = "llvm-backend")]
use inkwell::{module::Module as LLVMModule, types::FunctionType, values::FunctionValue};

#[cfg(feature = "cranelift-backend")]
/// Links the exponential function to make it available for JIT compilation.
///
/// This function declares the external exp function to the Cranelift module,
/// making it available for use in JIT-compiled code. It creates a function
/// signature matching the standard exp function: f64 -> f64.
///
/// # Arguments
/// * `module` - The Cranelift module to declare the function in
///
/// # Returns
/// * `Ok(FuncId)` - The function ID that can be used to call exp
/// * `Err(String)` - Error message if declaration fails
pub fn link_exp(module: &mut dyn CraneliftModule) -> Result<FuncId, String> {
    // Create signature for exp(f64) -> f64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));

    // Declare the function
    let func_id = module
        .declare_function("exp", Linkage::Import, &sig)
        .map_err(|e| e.to_string())?;

    Ok(func_id)
}

#[cfg(feature = "cranelift-backend")]
/// Generates Cranelift IR instructions to call the exponential function.
///
/// This helper function adds instructions to call the previously linked exp
/// function within a function being built with Cranelift.
///
/// # Arguments
/// * `builder` - The Cranelift function builder being used to construct the function
/// * `module` - The Cranelift module containing the function declaration
/// * `func_id` - The function ID returned by link_exp()
/// * `arg` - The Cranelift IR value to pass as the argument to exp
///
/// # Returns
/// The Cranelift IR value containing the result of calling exp
pub fn call_exp(
    builder: &mut FunctionBuilder,
    module: &mut dyn CraneliftModule,
    func_id: cranelift_module::FuncId,
    arg: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let func = module.declare_func_in_func(func_id, builder.func);
    let call = builder.ins().call(func, &[arg]);
    builder.inst_results(call)[0]
}

#[cfg(feature = "llvm-backend")]
/// Gets or inserts the exponential function in an LLVM module.
///
/// This function checks if the exp function is already declared in the module.
/// If not, it adds a declaration. Returns the LLVM function value for the exp function.
///
/// # Arguments
/// * `module` - The LLVM module to get or insert the function in
///
/// # Returns
/// * `Ok(FunctionValue)` - The LLVM function value for the exp function
/// * `Err(String)` - Error message if declaration fails
pub fn get_or_insert_function<'ctx>(
    module: &LLVMModule<'ctx>,
) -> Result<FunctionValue<'ctx>, String> {
    // Check if function already exists in module
    if let Some(func) = module.get_function("exp") {
        return Ok(func);
    }

    // Function doesn't exist, so declare it
    let context = module.get_context();
    let f64_type = context.f64_type();
    let fn_type = f64_type.fn_type(&[f64_type.into()], false);

    let function = module.add_function("exp", fn_type, None);

    Ok(function)
}
