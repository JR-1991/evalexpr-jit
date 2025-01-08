//! This module provides functionality for JIT compilation of mathematical expressions.
//! It uses Cranelift as the backend compiler to generate native machine code.

use std::sync::Arc;

use crate::{
    equation::{CombinedJITFunction, JITFunction},
    errors::{BuilderError, EquationError},
    expr::{Expr, VarRef},
};
use cranelift::prelude::*;
use cranelift_codegen::{ir::immediates::Offset32, Context};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use isa::TargetIsa;

/// Builds a JIT-compiled function from an expression tree.
///
/// # Arguments
/// * `expr` - The expression tree to compile
///
/// # Returns
/// A wrapped function that safely takes a slice of f64 values and returns an f64
pub fn build_function(expr: Expr) -> Result<JITFunction, EquationError> {
    let isa = create_isa()?;
    let (mut module, mut ctx) = create_module_and_context(isa);
    build_function_body(&mut ctx, expr, &mut module)?;
    let raw_fn = compile_and_finalize(&mut module, &mut ctx)?;

    // Wrap the unsafe function in a safe interface
    Ok(Box::new(move |input: &[f64]| {
        // The raw pointer is only used within this scope and we ensure
        // it's valid because it comes from a slice reference
        raw_fn(input.as_ptr())
    }))
}

/// Creates an Instruction Set Architecture (ISA) target for code generation.
pub(crate) fn create_isa() -> Result<Arc<dyn TargetIsa>, BuilderError> {
    let mut flag_builder = settings::builder();

    // Get target triple to detect architecture
    let target_triple = target_lexicon::Triple::host();
    let is_x86 = matches!(
        target_triple.architecture,
        target_lexicon::Architecture::X86_64
    );

    // Set flags based on architecture
    if is_x86 {
        flag_builder.set("use_colocated_libcalls", "true").unwrap();
        flag_builder.set("is_pic", "true").unwrap();
    } else {
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
    }

    let isa_builder = cranelift_native::builder()
        .map_err(|msg| BuilderError::HostMachineNotSupported(msg.to_string()))?;

    isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(BuilderError::CodegenError)
}

/// Creates a new JIT module and function context.
///
/// # Arguments
/// * `isa` - The target instruction set architecture
///
/// # Returns
/// A tuple containing the JIT module and function context. The context is initialized
/// with a signature taking an i64 (pointer to f64 array) and returning an f64.
pub(crate) fn create_module_and_context(isa: Arc<dyn TargetIsa>) -> (JITModule, Context) {
    let mut flags_builder = settings::builder();
    flags_builder.set("opt_level", "speed").unwrap();
    #[cfg(debug_assertions)]
    flags_builder.set("enable_verifier", "true").unwrap();
    #[cfg(not(debug_assertions))]
    flags_builder.set("enable_verifier", "false").unwrap();

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    builder.symbol("exp", f64::exp as *const u8);
    builder.symbol("ln", f64::log as *const u8);
    builder.symbol("sqrt", f64::sqrt as *const u8);
    builder.symbol("powi", f64::powi as *const u8);

    let module = JITModule::new(builder);
    let mut ctx = module.make_context();

    // Create signature
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::F64));
    ctx.func.signature = sig;

    (module, ctx)
}

/// Updates all variable references in the AST with the vector pointer.
///
/// # Arguments
/// * `ast` - The expression tree to update
/// * `vec_ptr` - The Value representing the pointer to the input array
///
/// This recursively traverses the AST and updates all variable nodes with the
/// pointer to the input array, which will be used during code generation.
fn update_ast_vec_refs(ast: &mut Expr, vec_ptr: Value) {
    match ast {
        // Update vector reference in variable nodes
        Expr::Var(VarRef { vec_ref, .. }) => {
            *vec_ref = vec_ptr;
        }
        // Recursively traverse binary operations
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            update_ast_vec_refs(left, vec_ptr);
            update_ast_vec_refs(right, vec_ptr);
        }
        Expr::Abs(expr) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
        Expr::Pow(base, _) => {
            update_ast_vec_refs(base, vec_ptr);
        }
        Expr::Exp(expr) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
        Expr::Ln(expr) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
        Expr::Sqrt(expr) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
        Expr::Neg(expr) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
        // Handle leaf nodes or other expression types that don't contain variables
        Expr::Const(_) => {}
        Expr::Cached(_, _) => {}
    }
}

/// Builds the function body by generating Cranelift IR from the expression tree.
///
/// # Arguments
/// * `ctx` - The function context to build into
/// * `ast` - The expression tree to generate code from
///
/// This creates a single basic block, updates variable references with the input
/// array pointer, generates code from the AST, and adds a return instruction.
fn build_function_body(
    ctx: &mut Context,
    mut ast: Expr,
    module: &mut dyn Module,
) -> Result<(), EquationError> {
    let mut builder_ctx = FunctionBuilderContext::new();
    let mut func_builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let entry_block = func_builder.create_block();
    func_builder.switch_to_block(entry_block);

    // Add parameter and update AST
    let vec_ptr = func_builder.append_block_param(entry_block, types::I64);
    update_ast_vec_refs(&mut ast, vec_ptr);

    // Generate code and return
    let result = ast.codegen(&mut func_builder, module)?;
    func_builder.ins().return_(&[result]);

    func_builder.seal_block(entry_block);
    func_builder.finalize();

    Ok(())
}

/// Compiles and finalizes the function, returning a callable function pointer.
///
/// # Arguments
/// * `module` - The JIT module to compile into
/// * `ctx` - The function context containing the IR to compile
///
/// # Returns
/// A function pointer that can be called with a pointer to an array of f64 values
fn compile_and_finalize(
    module: &mut JITModule,
    ctx: &mut Context,
) -> Result<fn(*const f64) -> f64, BuilderError> {
    let func_id = module
        .declare_function("my_jit_func", Linkage::Local, &ctx.func.signature)
        .map_err(|msg| BuilderError::DeclarationError(msg.to_string()))?;

    module
        .define_function(func_id, ctx)
        .map_err(|msg| BuilderError::FunctionError(msg.to_string()))?;

    module.clear_context(ctx);
    module
        .finalize_definitions()
        .map_err(BuilderError::ModuleError)?;

    let code_ptr = module.get_finalized_function(func_id);
    let func = unsafe { std::mem::transmute::<*const u8, fn(*const f64) -> f64>(code_ptr) };
    Ok(func)
}

/// Builds a JIT-compiled function that evaluates multiple expressions together.
///
/// # Arguments
/// * `exprs` - Vector of expression ASTs to compile
/// * `results_len` - Number of results to expect (must match number of expressions)
///
/// # Returns
/// A boxed function that takes input values and returns a vector of results
///
/// This function generates machine code that:
/// 1. Takes pointers to input and output arrays
/// 2. Evaluates all expressions sequentially
/// 3. Stores results directly in the output array
/// 4. Returns the filled output array
pub fn build_combined_function(
    exprs: Vec<Box<Expr>>,
    results_len: usize,
) -> Result<CombinedJITFunction, EquationError> {
    // Set up JIT compilation context
    let mut builder_context = FunctionBuilderContext::new();
    let mut codegen_context = Context::new();
    let isa = create_isa()?;
    let (mut module, _) = create_module_and_context(isa);

    // Create function signature: fn(input_ptr: *const f64, output_ptr: *mut f64)
    let mut sig = module.make_signature();
    sig.params
        .push(AbiParam::new(module.target_config().pointer_type())); // input_ptr
    sig.params
        .push(AbiParam::new(module.target_config().pointer_type())); // output_ptr
                                                                     // Remove return value since we'll write directly to output buffer

    // Create function
    let func_id = module
        .declare_function("combined", Linkage::Export, &sig)
        .unwrap();

    codegen_context.func.signature = sig; // Set signature before moving
    let func = &mut codegen_context.func; // Borrow instead of move
    let mut builder = FunctionBuilder::new(func, &mut builder_context);

    // Create entry block
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Get input array and output array parameters
    let output_ptr = builder.block_params(entry_block)[1];

    // Generate code for each expression
    for (i, expr) in exprs.iter().enumerate() {
        let result = expr.codegen(&mut builder, &mut module)?;

        // Store result in output array
        let offset = i as i64 * 8;
        builder.ins().store(
            MemFlags::new(),
            result,
            output_ptr,
            Offset32::new(offset as i32),
        );
    }

    // Return void since we wrote directly to output buffer
    builder.ins().return_(&[]);
    builder.finalize();

    // Finalize the function
    module
        .define_function(func_id, &mut codegen_context)
        .unwrap();
    module
        .finalize_definitions()
        .map_err(BuilderError::ModuleError)?;

    // Get function pointer
    let code = module.get_finalized_function(func_id);

    // Create wrapper function
    let wrapper = Box::new(move |inputs: &[f64]| -> Vec<f64> {
        let mut results = vec![0.0; results_len];
        unsafe {
            let f: extern "C" fn(*const f64, *mut f64) = std::mem::transmute(code);
            f(inputs.as_ptr(), results.as_mut_ptr());
        }
        results
    });

    Ok(wrapper)
}
