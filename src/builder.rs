//! This module provides functionality for JIT compilation of mathematical expressions.
//! It supports both Cranelift and LLVM as backend compilers to generate native machine code.
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

#[cfg(feature = "cranelift-backend")]
use cranelift::prelude::*;
#[cfg(feature = "cranelift-backend")]
use cranelift_codegen::{ir::immediates::Offset32, Context};
#[cfg(feature = "cranelift-backend")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "cranelift-backend")]
use cranelift_module::{Linkage, Module};
#[cfg(feature = "cranelift-backend")]
use isa::TargetIsa;
use rayon::prelude::*;

#[cfg(feature = "llvm-backend")]
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction as InkwellJitFunction},
    module::Module,
    types::BasicType,
    values::{BasicValue, FloatValue, FunctionValue, PointerValue},
    AddressSpace, FloatPredicate, OptimizationLevel,
};

#[cfg(feature = "llvm-backend")]
use rayon::prelude::*;

/// Builds a JIT-compiled function from an expression tree.
///
/// This function takes an expression AST and compiles it to native machine code using
/// either Cranelift or LLVM depending on the selected backend.
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
    #[cfg(feature = "cranelift-backend")]
    {
        let isa = create_isa()?;
        let (mut module, mut ctx) = create_module_and_context(isa);
        build_function_body(&mut ctx, expr, &mut module)?;
        let raw_fn = compile_and_finalize(&mut module, &mut ctx)?;

        // Extract the memory address which is thread-safe (Send + Sync)
        let fn_addr = raw_fn as usize;

        // Create a closure that captures the memory address instead of the raw function pointer
        let result = Arc::new(move |input: &[f64]| {
            if input.is_empty() {
                return 0.0;
            }

            // Convert the address back to a function pointer only when needed
            let f: fn(*const f64) -> f64 = unsafe { std::mem::transmute(fn_addr) };
            f(input.as_ptr())
        });

        // Keep the module alive for the lifetime of the program
        std::mem::forget(module);

        Ok(result)
    }
    #[cfg(all(feature = "llvm-backend", not(feature = "cranelift-backend")))]
    {
        // Create LLVM context, module, and builder
        let context = Context::create();
        let module = context.create_module("jit_function");
        let builder = context.create_builder();

        // Create execution engine
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| {
                BuilderError::FunctionError(format!("Failed to create JIT engine: {}", e))
            })?;

        // Create function type: fn(*const f64) -> f64
        let f64_type = context.f64_type();
        let ptr_type = f64_type.ptr_type(AddressSpace::default());
        let function_type = f64_type.fn_type(&[ptr_type.into()], false);

        // Add function to module
        let function = module.add_function("jit_function", function_type, None);
        let basic_block = context.append_basic_block(function, "entry");

        // Set builder's current block
        builder.position_at_end(basic_block);

        // Get input array pointer parameter
        let input_ptr = function.get_nth_param(0).unwrap().into_pointer_value();

        // Clone expression to modify variable references
        let mut modified_expr = expr.clone();

        // Generate code from expression tree
        let result =
            build_llvm_expression(&context, &builder, &module, &mut modified_expr, input_ptr)?;

        // Return the result
        builder
            .build_return(Some(&result))
            .map_err(|e| BuilderError::FunctionError(format!("Failed to build return: {}", e)))?;

        // Get function pointer
        type FnType = unsafe extern "C" fn(*const f64) -> f64;
        let jit_function = unsafe {
            execution_engine
                .get_function::<FnType>("jit_function")
                .map_err(|e| {
                    BuilderError::FunctionError(format!("Failed to get JIT function: {}", e))
                })?
        };

        // Extract the raw function pointer to a thread-safe format (usize)
        let fn_addr = unsafe { jit_function.as_raw() as usize };

        // Create a thread-safe wrapper that uses the function address
        let result = Arc::new(move |input: &[f64]| -> f64 {
            if input.is_empty() {
                return 0.0;
            }
            // Convert the address back to a function pointer only when needed
            let func: fn(*const f64) -> f64 = unsafe { std::mem::transmute(fn_addr) };
            unsafe { func(input.as_ptr()) }
        });

        // Keep execution engine alive for the lifetime of the program
        std::mem::forget(execution_engine);

        Ok(result)
    }
    #[cfg(not(any(feature = "cranelift-backend", feature = "llvm-backend")))]
    {
        Err(EquationError::JITError(
            "No JIT backend enabled. Enable either 'cranelift-backend' or 'llvm-backend' features."
                .to_string(),
        ))
    }
}

#[cfg(feature = "llvm-backend")]
/// Updates all variable references in the AST with the vector pointer for LLVM.
///
/// This is a helper function for LLVM backend to prepare the expression for code generation.
fn update_llvm_vec_refs(ast: &mut Expr, vec_ptr: PointerValue) {
    // Recursively traverse the AST and mark all variables
    // The expression tree itself will handle the variable accesses during code generation
    match ast {
        Expr::Var(VarRef { .. }) => {
            // No need to modify the reference, as we'll use the vec_ptr directly during codegen
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            update_llvm_vec_refs(left, vec_ptr);
            update_llvm_vec_refs(right, vec_ptr);
        }
        Expr::Abs(expr) => {
            update_llvm_vec_refs(expr, vec_ptr);
        }
        Expr::Pow(base, _) => {
            update_llvm_vec_refs(base, vec_ptr);
        }
        Expr::Exp(expr) => {
            update_llvm_vec_refs(expr, vec_ptr);
        }
        Expr::Ln(expr) => {
            update_llvm_vec_refs(expr, vec_ptr);
        }
        Expr::Sqrt(expr) => {
            update_llvm_vec_refs(expr, vec_ptr);
        }
        Expr::Neg(expr) => {
            update_llvm_vec_refs(expr, vec_ptr);
        }
        Expr::Cached(expr, _) => {
            update_llvm_vec_refs(expr, vec_ptr);
        }
        Expr::Const(_) => {}
    }
}

#[cfg(feature = "llvm-backend")]
/// Generates LLVM code for an expression tree.
///
/// This is the LLVM-specific implementation of expression code generation.
fn build_llvm_expression<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    module: &Module<'ctx>,
    expr: &mut Expr,
    input_ptr: PointerValue<'ctx>,
) -> Result<FloatValue<'ctx>, EquationError> {
    match expr {
        Expr::Const(val) => {
            let float_val = context.f64_type().const_float(*val);
            Ok(float_val)
        }
        Expr::Var(VarRef { index, .. }) => {
            // Calculate the offset in the input array
            let idx = context.i32_type().const_int(*index as u64, false);

            // Get pointer to the variable
            let var_ptr = unsafe {
                builder
                    .build_in_bounds_gep(context.f64_type(), input_ptr, &[idx], "var_ptr")
                    .map_err(|e| {
                        BuilderError::FunctionError(format!("Failed to build GEP: {}", e))
                    })?
            };

            // Load the value
            let loaded = builder
                .build_load(context.f64_type(), var_ptr, "var_value")
                .map_err(|e| BuilderError::FunctionError(format!("Failed to build load: {}", e)))?;
            Ok(loaded.into_float_value())
        }
        Expr::Add(left, right) => {
            let lhs = build_llvm_expression(context, builder, module, left, input_ptr)?;
            let rhs = build_llvm_expression(context, builder, module, right, input_ptr)?;

            Ok(builder
                .build_float_add(lhs, rhs, "addtmp")
                .map_err(|e| BuilderError::FunctionError(format!("Failed to build add: {}", e)))?)
        }
        Expr::Mul(left, right) => {
            let lhs = build_llvm_expression(context, builder, module, left, input_ptr)?;
            let rhs = build_llvm_expression(context, builder, module, right, input_ptr)?;

            Ok(builder
                .build_float_mul(lhs, rhs, "multmp")
                .map_err(|e| BuilderError::FunctionError(format!("Failed to build mul: {}", e)))?)
        }
        Expr::Sub(left, right) => {
            let lhs = build_llvm_expression(context, builder, module, left, input_ptr)?;
            let rhs = build_llvm_expression(context, builder, module, right, input_ptr)?;

            Ok(builder
                .build_float_sub(lhs, rhs, "subtmp")
                .map_err(|e| BuilderError::FunctionError(format!("Failed to build sub: {}", e)))?)
        }
        Expr::Div(left, right) => {
            let lhs = build_llvm_expression(context, builder, module, left, input_ptr)?;
            let rhs = build_llvm_expression(context, builder, module, right, input_ptr)?;

            Ok(builder
                .build_float_div(lhs, rhs, "divtmp")
                .map_err(|e| BuilderError::FunctionError(format!("Failed to build div: {}", e)))?)
        }
        Expr::Abs(expr) => {
            let val = build_llvm_expression(context, builder, module, expr, input_ptr)?;

            // Implement abs using select: if val < 0 then -val else val
            let zero = context.f64_type().const_float(0.0);
            let is_negative = builder
                .build_float_compare(FloatPredicate::OLT, val, zero, "is_neg")
                .map_err(|e| {
                    BuilderError::FunctionError(format!("Failed to build float compare: {}", e))
                })?;

            let neg_val = builder.build_float_neg(val, "negtmp").map_err(|e| {
                BuilderError::FunctionError(format!("Failed to build negation: {}", e))
            })?;

            Ok(builder
                .build_select(is_negative, neg_val, val, "abs_result")
                .map_err(|e| BuilderError::FunctionError(format!("Failed to build select: {}", e)))?
                .into_float_value())
        }
        Expr::Neg(expr) => {
            let val = build_llvm_expression(context, builder, module, expr, input_ptr)?;

            Ok(builder.build_float_neg(val, "negtmp").map_err(|e| {
                BuilderError::FunctionError(format!("Failed to build negation: {}", e))
            })?)
        }
        Expr::Pow(base, exp) => {
            let base_val = build_llvm_expression(context, builder, module, base, input_ptr)?;

            match *exp {
                0 => Ok(context.f64_type().const_float(1.0)),
                1 => Ok(base_val),
                2 => {
                    // Special case for x^2
                    Ok(builder
                        .build_float_mul(base_val, base_val, "pow2")
                        .map_err(|e| {
                            BuilderError::FunctionError(format!("Failed to build mul: {}", e))
                        })?)
                }
                3 => {
                    // Special case for x^3
                    let square = builder
                        .build_float_mul(base_val, base_val, "square")
                        .map_err(|e| {
                            BuilderError::FunctionError(format!("Failed to build mul: {}", e))
                        })?;

                    Ok(builder
                        .build_float_mul(square, base_val, "pow3")
                        .map_err(|e| {
                            BuilderError::FunctionError(format!("Failed to build mul: {}", e))
                        })?)
                }
                exp => {
                    // For other exponents, use a binary exponentiation algorithm
                    if exp < 0 {
                        // For negative exponents: 1.0 / (x^abs(exp))
                        let mut result = context.f64_type().const_float(1.0);
                        let mut base = base_val;
                        let mut n = (-exp) as u64;

                        // Binary exponentiation
                        while n > 0 {
                            if n & 1 == 1 {
                                result = builder.build_float_mul(result, base, "powmul").map_err(
                                    |e| {
                                        BuilderError::FunctionError(format!(
                                            "Failed to build mul: {}",
                                            e
                                        ))
                                    },
                                )?;
                            }
                            base =
                                builder
                                    .build_float_mul(base, base, "powsquare")
                                    .map_err(|e| {
                                        BuilderError::FunctionError(format!(
                                            "Failed to build mul: {}",
                                            e
                                        ))
                                    })?;
                            n >>= 1;
                        }

                        let one = context.f64_type().const_float(1.0);
                        Ok(builder
                            .build_float_div(one, result, "invpow")
                            .map_err(|e| {
                                BuilderError::FunctionError(format!("Failed to build div: {}", e))
                            })?)
                    } else {
                        // For positive exponents
                        let mut result = context.f64_type().const_float(1.0);
                        let mut base = base_val;
                        let mut n = exp as u64;

                        // Binary exponentiation
                        while n > 0 {
                            if n & 1 == 1 {
                                result = builder.build_float_mul(result, base, "powmul").map_err(
                                    |e| {
                                        BuilderError::FunctionError(format!(
                                            "Failed to build mul: {}",
                                            e
                                        ))
                                    },
                                )?;
                            }
                            base =
                                builder
                                    .build_float_mul(base, base, "powsquare")
                                    .map_err(|e| {
                                        BuilderError::FunctionError(format!(
                                            "Failed to build mul: {}",
                                            e
                                        ))
                                    })?;
                            n >>= 1;
                        }

                        Ok(result)
                    }
                }
            }
        }
        Expr::Exp(expr) => {
            let val = build_llvm_expression(context, builder, module, expr, input_ptr)?;

            // Get the exp function
            let exp_fn = crate::operators::exp::get_or_insert_function(module)
                .map_err(|e| BuilderError::FunctionError(e))?;

            // Call exp function
            let result = builder
                .build_call(exp_fn, &[val.into()], "expcall")
                .map_err(|e| {
                    BuilderError::FunctionError(format!("Failed to build exp call: {}", e))
                })?;

            // Extract return value
            Ok(result
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_float_value())
        }
        Expr::Ln(expr) => {
            let val = build_llvm_expression(context, builder, module, expr, input_ptr)?;

            // Get the ln function
            let ln_fn = crate::operators::ln::get_or_insert_function(module)
                .map_err(|e| BuilderError::FunctionError(e))?;

            // Call ln function
            let result = builder
                .build_call(ln_fn, &[val.into()], "lncall")
                .map_err(|e| {
                    BuilderError::FunctionError(format!("Failed to build ln call: {}", e))
                })?;

            // Extract return value
            Ok(result
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_float_value())
        }
        Expr::Sqrt(expr) => {
            let val = build_llvm_expression(context, builder, module, expr, input_ptr)?;

            // Get the sqrt function
            let sqrt_fn = crate::operators::sqrt::get_or_insert_function(module)
                .map_err(|e| BuilderError::FunctionError(e))?;

            // Call sqrt function
            let result = builder
                .build_call(sqrt_fn, &[val.into()], "sqrtcall")
                .map_err(|e| {
                    BuilderError::FunctionError(format!("Failed to build sqrt call: {}", e))
                })?;

            // Extract return value
            Ok(result
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_float_value())
        }
        Expr::Cached(expr, cached_value) => {
            if let Some(val) = cached_value {
                // Use the cached value
                Ok(context.f64_type().const_float(*val))
            } else {
                // Generate code for the expression
                build_llvm_expression(context, builder, module, expr, input_ptr)
            }
        }
        _ => {
            // For all other expression types, use the codegen_llvm method
            expr.codegen_llvm(context, builder, module, input_ptr)
        }
    }
}

#[cfg(feature = "cranelift-backend")]
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

#[cfg(feature = "cranelift-backend")]
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

#[cfg(feature = "cranelift-backend")]
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
            *vec_ref = vec_ptr.as_u32() as usize;
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

#[cfg(feature = "cranelift-backend")]
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

#[cfg(feature = "cranelift-backend")]
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
    #[cfg(feature = "cranelift-backend")]
    {
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
            .map_err(|msg| BuilderError::DeclarationError(msg.to_string()))?;

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
            .map_err(|msg| BuilderError::FunctionError(msg.to_string()))?;
        module
            .finalize_definitions()
            .map_err(BuilderError::ModuleError)?;

        // Get the raw function pointer once
        let func_ptr = module.get_finalized_function(func_id);

        // We need to extract the actual memory address to make it thread-safe
        // Convert the raw pointer to a usize which is Send + Sync
        let func_addr = func_ptr as usize;

        // Create a closure that captures the memory address instead of the raw pointer
        let wrapper = Box::new(move |inputs: &[f64], results: &mut [f64]| {
            // Validation code
            if inputs.is_empty() || results.is_empty() {
                return;
            }

            if results.len() != results_len {
                results.iter_mut().for_each(|r| *r = 0.0);
                return;
            }

            // Convert the address back to a function pointer only when needed
            let f: extern "C" fn(*const f64, *mut f64) = unsafe { std::mem::transmute(func_addr) };
            f(inputs.as_ptr(), results.as_mut_ptr())
        });

        // We need to keep the module alive as long as the function is in use
        // Leak the module to ensure it stays alive for the program duration
        // This is acceptable for JIT functions that live for the entire program
        std::mem::forget(module);

        // The wrapped function satisfies CombinedJITFunction
        Ok(Arc::new(wrapper))
    }
    #[cfg(all(feature = "llvm-backend", not(feature = "cranelift-backend")))]
    {
        // Create LLVM context, module, and builder
        let context = Context::create();
        let module = context.create_module("combined_function");
        let builder = context.create_builder();

        // Create execution engine
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| {
                BuilderError::FunctionError(format!("Failed to create JIT engine: {}", e))
            })?;

        // Create function type: fn(input_ptr: *const f64, output_ptr: *mut f64) -> void
        let f64_type = context.f64_type();
        let input_ptr_type = f64_type.ptr_type(AddressSpace::default());
        let output_ptr_type = f64_type.ptr_type(AddressSpace::default());
        let void_type = context.void_type();
        let function_type =
            void_type.fn_type(&[input_ptr_type.into(), output_ptr_type.into()], false);

        // Add function to module
        let function = module.add_function("combined_function", function_type, None);
        let basic_block = context.append_basic_block(function, "entry");

        // Set builder's current block
        builder.position_at_end(basic_block);

        // Get input and output array pointers
        let input_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let output_ptr = function.get_nth_param(1).unwrap().into_pointer_value();

        // Generate code for each expression and store results
        for (i, expr) in exprs.iter().enumerate() {
            // Clone expression to modify variable references
            let mut modified_expr = expr.clone();

            // Generate code for the expression
            let result =
                build_llvm_expression(&context, &builder, &module, &mut modified_expr, input_ptr)?;

            // Calculate offset into output array
            let idx = context.i32_type().const_int(i as u64, false);

            // Get pointer to the result location
            let result_ptr = unsafe {
                builder
                    .build_in_bounds_gep(
                        context.f64_type(),
                        output_ptr,
                        &[idx],
                        &format!("result_{}", i),
                    )
                    .map_err(|e| {
                        BuilderError::FunctionError(format!("Failed to build GEP: {}", e))
                    })?
            };

            // Store result in output array
            builder.build_store(result_ptr, result).map_err(|e| {
                BuilderError::FunctionError(format!("Failed to build store: {}", e))
            })?;
        }

        // Return void
        builder
            .build_return(None)
            .map_err(|e| BuilderError::FunctionError(format!("Failed to build return: {}", e)))?;

        // Get function pointer
        type CombinedFnType = unsafe extern "C" fn(*const f64, *mut f64);
        let jit_function = unsafe {
            execution_engine
                .get_function::<CombinedFnType>("combined_function")
                .map_err(|e| {
                    BuilderError::FunctionError(format!("Failed to get JIT function: {}", e))
                })?
        };

        // Extract the raw function pointer to a thread-safe format (usize)
        let fn_addr = unsafe { jit_function.as_raw() as usize };

        // Create a thread-safe wrapper
        let wrapper = Arc::new(move |inputs: &[f64], results: &mut [f64]| {
            // Validation code
            if inputs.is_empty() || results.is_empty() {
                return;
            }

            if results.len() != results_len {
                results.iter_mut().for_each(|r| *r = 0.0);
                return;
            }

            // Convert the address back to a function pointer only when needed
            let func: fn(*const f64, *mut f64) = unsafe { std::mem::transmute(fn_addr) };
            unsafe { func(inputs.as_ptr(), results.as_mut_ptr()) }
        });

        // Keep execution engine alive
        std::mem::forget(execution_engine);

        Ok(wrapper)
    }
    #[cfg(not(any(feature = "cranelift-backend", feature = "llvm-backend")))]
    {
        Err(EquationError::JITError(
            "No JIT backend enabled. Enable either 'cranelift-backend' or 'llvm-backend' features."
                .to_string(),
        ))
    }
}
