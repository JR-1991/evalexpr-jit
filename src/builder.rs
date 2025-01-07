//! This module provides functionality for JIT compilation of mathematical expressions.
//! It uses Cranelift as the backend compiler to generate native machine code.

use std::sync::Arc;

use crate::{
    equation::JITFunction,
    errors::BuilderError,
    expr::{Expr, VarRef},
};
use cranelift::prelude::*;
use cranelift_codegen::Context;
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
pub fn build_function(expr: Expr) -> Result<JITFunction, BuilderError> {
    let isa = create_isa()?;
    let (mut module, mut ctx) = create_module_and_context(isa);
    build_function_body(&mut ctx, expr);
    let raw_fn = compile_and_finalize(&mut module, &mut ctx)?;

    // Wrap the unsafe function in a safe interface
    Ok(Box::new(move |input: &[f64]| {
        // The raw pointer is only used within this scope and we ensure
        // it's valid because it comes from a slice reference
        raw_fn(input.as_ptr())
    }))
}

/// Creates an Instruction Set Architecture (ISA) target for code generation.
fn create_isa() -> Result<Arc<dyn TargetIsa>, BuilderError> {
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
fn create_module_and_context(isa: Arc<dyn TargetIsa>) -> (JITModule, Context) {
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
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
        Expr::Exp(base, _) => {
            update_ast_vec_refs(base, vec_ptr);
        }
        // Handle leaf nodes or other expression types that don't contain variables
        Expr::Const(_) => {}
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
fn build_function_body(ctx: &mut Context, mut ast: Expr) {
    let mut builder_ctx = FunctionBuilderContext::new();
    let mut func_builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let entry_block = func_builder.create_block();
    func_builder.switch_to_block(entry_block);

    // Add parameter and update AST
    let vec_ptr = func_builder.append_block_param(entry_block, types::I64);
    update_ast_vec_refs(&mut ast, vec_ptr);

    // Generate code and return
    let result = ast.codegen(&mut func_builder);
    func_builder.ins().return_(&[result]);

    func_builder.seal_block(entry_block);
    func_builder.finalize();
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
