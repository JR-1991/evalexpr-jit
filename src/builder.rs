//! This module provides functionality for JIT compilation of mathematical expressions.
//! It uses Cranelift as the backend compiler to generate native machine code.
//!
//! The main entry points are:
//! - `build_function()` - Compiles a single expression into a JIT function
//! - `build_combined_function()` - Compiles multiple expressions into a single JIT function

use std::sync::Arc;

use crate::{
    errors::{BuilderError, EquationError},
    expr::{Expr, VarRef},
    types::{CombinedJITFunction, JITFunction},
};
use cranelift::prelude::*;
use cranelift_codegen::{ir::immediates::Offset32, Context};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use isa::TargetIsa;
use rayon::prelude::*;

struct ThreadSafeFunction(*const u8);
unsafe impl Send for ThreadSafeFunction {}
unsafe impl Sync for ThreadSafeFunction {}

/// Builds a JIT-compiled function from an expression tree.
///
/// This function takes an expression AST and compiles it to native machine code using Cranelift.
/// The resulting function is wrapped in a safe interface that handles pointer safety.
///
/// # Arguments
/// * `expr` - The expression AST to compile
///
/// # Returns
/// A thread-safe function that takes a slice of f64 values and returns an f64 result.
/// The function is wrapped in an Arc to allow sharing between threads.
///
/// # Errors
/// Returns an EquationError if compilation fails for any reason.
pub fn build_function(expr: Expr) -> Result<JITFunction, EquationError> {
    let isa = create_isa()?;
    let (mut module, mut ctx) = create_module_and_context(isa);
    build_function_body(&mut ctx, expr, &mut module)?;
    let raw_fn = compile_and_finalize(&mut module, &mut ctx)?;

    // Wrap the unsafe function in a safe interface
    Ok(Arc::new(move |input: &[f64]| raw_fn(input.as_ptr())))
}

/// Creates an Instruction Set Architecture (ISA) target for code generation.
///
/// This function detects the host machine architecture and configures appropriate
/// compilation flags for optimal code generation.
///
/// # Returns
/// An Arc-wrapped TargetIsa configured for the host machine.
///
/// # Errors
/// Returns a BuilderError if:
/// - The host machine architecture is not supported
/// - Code generation configuration fails
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
/// This function initializes a Cranelift JIT module and context with:
/// - Optimization settings configured for speed
/// - Debug verification enabled in debug builds
/// - Standard math functions (exp, ln, sqrt, powi) linked in
/// - A function signature taking a pointer to f64 array and returning f64
///
/// # Arguments
/// * `isa` - The target instruction set architecture to compile for
///
/// # Returns
/// A tuple containing:
/// - JITModule: The module that will contain the compiled code
/// - Context: The function context initialized with the correct signature
pub(crate) fn create_module_and_context(isa: Arc<dyn TargetIsa>) -> (JITModule, Context) {
    let mut flags_builder = settings::builder();
    flags_builder.set("opt_level", "speed").unwrap();

    #[cfg(debug_assertions)]
    {
        flags_builder.set("enable_verifier", "true").unwrap();
        flags_builder.set("enable_alias_analysis", "true").unwrap();
    }
    #[cfg(not(debug_assertions))]
    {
        flags_builder.set("enable_verifier", "false").unwrap();
        flags_builder.set("enable_alias_analysis", "false").unwrap();
    }

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
/// This function recursively traverses the AST and updates all variable nodes with a
/// pointer to the input array. This pointer will be used during code generation to
/// load values from the input array.
///
/// # Arguments
/// * `ast` - The expression tree to update
/// * `vec_ptr` - The Cranelift Value representing the pointer to the input array
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
/// This function:
/// 1. Creates a new function builder and entry block
/// 2. Adds the input array pointer parameter
/// 3. Updates all variable references in the AST with the pointer
/// 4. Generates code from the AST
/// 5. Adds a return instruction
///
/// # Arguments
/// * `ctx` - The function context to build into
/// * `ast` - The expression tree to generate code from
/// * `module` - The module to compile into
///
/// # Errors
/// Returns an EquationError if code generation fails
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
/// This function:
/// 1. Declares the function in the module
/// 2. Defines the function body from the context
/// 3. Finalizes all definitions
/// 4. Extracts the function pointer
///
/// # Arguments
/// * `module` - The JIT module to compile into
/// * `ctx` - The function context containing the IR to compile
///
/// # Returns
/// A function pointer that can be called with a pointer to an array of f64 values
///
/// # Errors
/// Returns a BuilderError if:
/// - Function declaration fails
/// - Function definition fails
/// - Module finalization fails
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

    // SAFETY: This transmute is safe because:
    // - The function was compiled with signature fn(*const f64) -> f64
    // - The module is kept alive via Arc in the calling function
    // - The function pointer remains valid as long as the module exists
    let func = unsafe {
        std::mem::transmute::<*const u8, fn(*const f64) -> f64>(
            module.get_finalized_function(func_id),
        )
    };
    Ok(func)
}

/// Builds a JIT-compiled function that evaluates multiple expressions together.
///
/// This function generates optimized machine code that evaluates multiple expressions
/// in a single function call, storing results directly in an output buffer. This is
/// more efficient than calling multiple single-expression functions.
///
/// # Arguments
/// * `exprs` - Vector of expression ASTs to compile together
/// * `results_len` - Expected length of the results array (must match number of expressions)
///
/// # Returns
/// A thread-safe function that:
/// - Takes a slice of input values
/// - Takes a mutable slice for results
/// - Evaluates all expressions
/// - Stores results directly in the output slice
///
/// # Errors
/// Returns an EquationError if compilation fails
///
/// # Panics
/// The returned function will panic if the results slice length doesn't match results_len
pub fn build_combined_function(
    exprs: Vec<Expr>,
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

    // Pre-process expressions in parallel
    let prepared_exprs: Vec<_> = exprs
        .par_iter()
        .map(|expr| expr.clone()) // Or any other thread-safe preparation
        .collect();

    // Sequential codegen
    let results: Vec<_> = prepared_exprs
        .iter()
        .map(|expr| expr.codegen(&mut builder, &mut module))
        .collect::<Result<_, _>>()?;

    // Store results in output array
    for (i, result) in results.iter().enumerate() {
        let offset = i as i64 * 8;
        builder.ins().store(
            MemFlags::new(),
            *result,
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
    let code = Arc::new(ThreadSafeFunction(module.get_finalized_function(func_id)));
    let wrapper = Box::new(move |inputs: &[f64], results: &mut [f64]| {
        debug_assert_eq!(
            results.len(),
            results_len,
            "Results buffer has incorrect length"
        );
        unsafe {
            let f: extern "C" fn(*const f64, *mut f64) = std::mem::transmute(code.0);
            f(inputs.as_ptr(), results.as_mut_ptr());
        }
    });

    Ok(Arc::new(wrapper))
}
