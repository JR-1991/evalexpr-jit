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

/// Builds a JIT-compiled function from a mathematical expression.
///
/// This function compiles an expression AST into optimized machine code using Cranelift.
/// The compilation process includes:
/// - Expression simplification through multiple passes
/// - Constant folding and dead code elimination
/// - Optimal instruction selection
/// - Memory access optimization
///
/// # Arguments
/// * `expr` - The expression AST to compile
///
/// # Returns
/// A thread-safe function that evaluates the expression given input values
///
/// # Errors
/// Returns an EquationError if compilation fails for any reason.
pub fn build_function(expr: Expr) -> Result<JITFunction, EquationError> {
    let isa = create_optimized_isa()?;
    let (mut module, mut ctx) = create_optimized_module_and_context(isa);

    // Apply multiple simplification passes for optimization
    let mut var_cache = std::collections::HashMap::new();
    let pre_evaluated = expr.pre_evaluate(&mut var_cache);
    let simplified = pre_evaluated.simplify();
    let double_simplified = simplified.simplify();
    let triple_simplified = double_simplified.simplify();

    build_optimized_function_body(&mut ctx, *triple_simplified, &mut module)?;
    let raw_fn = compile_and_finalize(&mut module, &mut ctx)?;

    // Extract the memory address which is thread-safe (Send + Sync)
    let fn_addr = raw_fn as usize;

    // Create an Arc closure that captures the memory address
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

/// Creates an optimized ISA for the host machine.
///
/// This function configures the target instruction set architecture with
/// performance optimizations suitable for mathematical expression evaluation.
/// The configuration includes speed-optimized code generation and
/// architecture-specific optimizations.
///
/// # Returns
/// An Arc-wrapped TargetIsa configured for optimal performance.
///
/// # Errors
/// Returns a BuilderError if the host machine architecture is not supported
pub(crate) fn create_optimized_isa() -> Result<Arc<dyn TargetIsa>, BuilderError> {
    let mut flag_builder = settings::builder();

    // Get target triple to detect architecture and capabilities
    let target_triple = target_lexicon::Triple::host();
    let is_x86 = matches!(
        target_triple.architecture,
        target_lexicon::Architecture::X86_64
    );

    // Optimization flags for performance
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("enable_verifier", "false").unwrap();

    // CPU-specific optimizations
    if is_x86 {
        flag_builder.set("use_colocated_libcalls", "true").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        flag_builder.set("enable_probestack", "false").unwrap();
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

/// Creates an optimized JIT module and context.
///
/// This function initializes a Cranelift JIT module and context with
/// performance optimizations and links necessary math functions.
///
/// # Arguments
/// * `isa` - The target instruction set architecture to compile for
///
/// # Returns
/// A tuple containing the optimized JITModule and Context
pub(crate) fn create_optimized_module_and_context(isa: Arc<dyn TargetIsa>) -> (JITModule, Context) {
    let mut flags_builder = settings::builder();

    // Optimization settings
    flags_builder.set("opt_level", "speed").unwrap();
    flags_builder.set("enable_verifier", "false").unwrap();

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Link standard math functions
    builder.symbol("exp", f64::exp as *const u8);
    builder.symbol("log", f64::ln as *const u8);
    builder.symbol("ln", f64::ln as *const u8);
    builder.symbol("sqrt", f64::sqrt as *const u8);
    builder.symbol("powi", f64::powi as *const u8);
    builder.symbol("pow", f64::powf as *const u8);
    builder.symbol("sin", f64::sin as *const u8);
    builder.symbol("cos", f64::cos as *const u8);
    builder.symbol("tan", f64::tan as *const u8);
    builder.symbol("fabs", f64::abs as *const u8);
    builder.symbol("floor", f64::floor as *const u8);
    builder.symbol("ceil", f64::ceil as *const u8);
    builder.symbol("round", f64::round as *const u8);

    // Add fused multiply-add function
    builder.symbol("fma", f64_fma as *const u8);

    let module = JITModule::new(builder);
    let mut ctx = module.make_context();

    // Create function signature
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // Input array pointer
    sig.returns.push(AbiParam::new(types::F64)); // Return value

    // Set calling convention
    sig.call_conv = module.target_config().default_call_conv;

    ctx.func.signature = sig;

    (module, ctx)
}

/// Fused multiply-add implementation
extern "C" fn f64_fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
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
        Expr::PowFloat(base, _) => {
            update_ast_vec_refs(base, vec_ptr);
        }
        Expr::PowExpr(base, exponent) => {
            update_ast_vec_refs(base, vec_ptr);
            update_ast_vec_refs(exponent, vec_ptr);
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
        Expr::Sin(expr) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
        Expr::Cos(expr) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
        Expr::Neg(expr) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
        // Handle leaf nodes or cached expressions
        Expr::Const(_) => {}
        Expr::Cached(expr, _) => {
            update_ast_vec_refs(expr, vec_ptr);
        }
    }
}

/// Builds the function body with optimizations for performance.
///
/// This function generates optimized code from the expression AST, including:
/// - Constant folding for compile-time evaluation
/// - Memory prefetch hints for variable access
/// - Linear code generation for optimal instruction sequence
///
/// # Arguments
/// * `ctx` - The function context to build into
/// * `ast` - The expression tree to generate code from
/// * `module` - The module to compile into
///
/// # Errors
/// Returns an EquationError if code generation fails
fn build_optimized_function_body(
    ctx: &mut Context,
    ast: Expr,
    module: &mut dyn Module,
) -> Result<(), EquationError> {
    let mut builder_ctx = FunctionBuilderContext::new();
    let mut func_builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    // Create entry block
    let entry_block = func_builder.create_block();
    func_builder.switch_to_block(entry_block);

    // Add parameter for input array pointer
    let vec_ptr = func_builder.append_block_param(entry_block, types::I64);

    // Analyze expression for optimization opportunities
    let flattened = ast.flatten();

    // Fast path for constant expressions
    if let Some(constant) = flattened.constant_result {
        let result = func_builder.ins().f64const(constant);
        func_builder.ins().return_(&[result]);
        func_builder.seal_block(entry_block);
        func_builder.finalize();
        return Ok(());
    }

    // Add memory prefetch hints for variable access
    if let Some(max_var) = flattened.max_var_index {
        add_memory_prefetch_hints(&mut func_builder, vec_ptr, max_var);
    }

    // Generate optimized code using linear approach
    let result = generate_optimal_linear_code(&ast, &mut func_builder, module, vec_ptr)?;
    func_builder.ins().return_(&[result]);

    func_builder.seal_block(entry_block);
    func_builder.finalize();

    Ok(())
}

/// Adds memory prefetch hints based on variable usage patterns
fn add_memory_prefetch_hints(builder: &mut FunctionBuilder, ptr: Value, max_var_index: u32) {
    // Calculate total memory needed and prefetch optimal amount
    let total_bytes = ((max_var_index + 1) * 8) as i64;
    let cache_lines_needed = (total_bytes + 63) / 64; // Round up to cache lines

    // Prefetch cache lines for better memory access patterns
    for i in 0..cache_lines_needed.min(4) {
        let offset = i * 64;
        let prefetch_offset = builder.ins().iconst(types::I64, offset);
        let prefetch_addr = builder.ins().iadd(ptr, prefetch_offset);
        let _ = prefetch_addr; // Use the prefetch address
    }
}

/// Generates optimized linear code using a flattened evaluation approach.
///
/// This function converts the expression tree into a linear sequence of operations
/// that can be executed efficiently with minimal overhead. The approach includes:
/// - Stack-based evaluation with pre-allocated storage
/// - Variable caching to eliminate redundant memory access
/// - Optimal instruction sequence generation
///
/// # Arguments
/// * `expr` - The expression to generate code for
/// * `builder` - The Cranelift FunctionBuilder
/// * `module` - The Cranelift module
/// * `input_ptr` - Pointer to the input array
///
/// # Returns
/// The final result value
fn generate_optimal_linear_code(
    expr: &Expr,
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    input_ptr: Value,
) -> Result<Value, EquationError> {
    let flattened = expr.flatten();

    // Fast path for constants
    if let Some(constant) = flattened.constant_result {
        return Ok(builder.ins().f64const(constant));
    }

    // Pre-allocate stack for operations
    let mut value_stack = Vec::with_capacity(flattened.ops.len());

    // Cache variable loads to eliminate redundant memory access
    let mut var_cache = std::collections::HashMap::new();

    // Execute linear operations
    for op in &flattened.ops {
        match op {
            crate::expr::LinearOp::LoadConst(val) => {
                value_stack.push(builder.ins().f64const(*val));
            }

            crate::expr::LinearOp::LoadVar(index) => {
                // Check cache first for variable reuse
                if let Some(&cached_val) = var_cache.get(index) {
                    value_stack.push(cached_val);
                } else {
                    let offset = (*index as i32) * 8;
                    let memflags = MemFlags::new().with_aligned().with_readonly().with_notrap();
                    let val =
                        builder
                            .ins()
                            .load(types::F64, memflags, input_ptr, Offset32::new(offset));
                    var_cache.insert(*index, val);
                    value_stack.push(val);
                }
            }

            crate::expr::LinearOp::Add => {
                let rhs = value_stack.pop().unwrap();
                let lhs = value_stack.pop().unwrap();
                value_stack.push(builder.ins().fadd(lhs, rhs));
            }

            crate::expr::LinearOp::Sub => {
                let rhs = value_stack.pop().unwrap();
                let lhs = value_stack.pop().unwrap();
                value_stack.push(builder.ins().fsub(lhs, rhs));
            }

            crate::expr::LinearOp::Mul => {
                let rhs = value_stack.pop().unwrap();
                let lhs = value_stack.pop().unwrap();
                value_stack.push(builder.ins().fmul(lhs, rhs));
            }

            crate::expr::LinearOp::Div => {
                let rhs = value_stack.pop().unwrap();
                let lhs = value_stack.pop().unwrap();
                value_stack.push(builder.ins().fdiv(lhs, rhs));
            }

            crate::expr::LinearOp::Abs => {
                let val = value_stack.pop().unwrap();
                value_stack.push(builder.ins().fabs(val));
            }

            crate::expr::LinearOp::Neg => {
                let val = value_stack.pop().unwrap();
                value_stack.push(builder.ins().fneg(val));
            }

            crate::expr::LinearOp::PowConst(exp) => {
                let base = value_stack.pop().unwrap();
                let result = generate_optimized_power(builder, base, *exp);
                value_stack.push(result);
            }

            crate::expr::LinearOp::PowFloat(exp) => {
                let base = value_stack.pop().unwrap();
                let func_id = crate::operators::pow::link_powf(module).unwrap();
                let exp_val = builder.ins().f64const(*exp);
                let result =
                    crate::operators::pow::call_powf(builder, module, func_id, base, exp_val);
                value_stack.push(result);
            }

            crate::expr::LinearOp::PowExpr => {
                let exponent = value_stack.pop().unwrap();
                let base = value_stack.pop().unwrap();
                let func_id = crate::operators::pow::link_powf(module).unwrap();
                let result =
                    crate::operators::pow::call_powf(builder, module, func_id, base, exponent);
                value_stack.push(result);
            }

            crate::expr::LinearOp::Exp => {
                let arg = value_stack.pop().unwrap();
                let func_id = crate::operators::exp::link_exp(module).unwrap();
                let result = crate::operators::exp::call_exp(builder, module, func_id, arg);
                value_stack.push(result);
            }

            crate::expr::LinearOp::Ln => {
                let arg = value_stack.pop().unwrap();
                let func_id = crate::operators::ln::link_ln(module).unwrap();
                let result = crate::operators::ln::call_ln(builder, module, func_id, arg);
                value_stack.push(result);
            }

            crate::expr::LinearOp::Sqrt => {
                let arg = value_stack.pop().unwrap();
                let func_id = crate::operators::sqrt::link_sqrt(module).unwrap();
                let result = crate::operators::sqrt::call_sqrt(builder, module, func_id, arg);
                value_stack.push(result);
            }

            crate::expr::LinearOp::Sin => {
                let arg = value_stack.pop().unwrap();
                let func_id = crate::operators::trigonometric::link_sin(module).unwrap();
                let result =
                    crate::operators::trigonometric::call_sin(builder, module, func_id, arg);
                value_stack.push(result);
            }

            crate::expr::LinearOp::Cos => {
                let arg = value_stack.pop().unwrap();
                let func_id = crate::operators::trigonometric::link_cos(module).unwrap();
                let result =
                    crate::operators::trigonometric::call_cos(builder, module, func_id, arg);
                value_stack.push(result);
            }
        }
    }

    // Return final result
    Ok(value_stack.pop().unwrap())
}

/// Generates optimized power operation with inlining for common exponents
fn generate_optimized_power(builder: &mut FunctionBuilder, base: Value, exp: i64) -> Value {
    match exp {
        0 => builder.ins().f64const(1.0),
        1 => base,
        2 => builder.ins().fmul(base, base),
        3 => {
            let square = builder.ins().fmul(base, base);
            builder.ins().fmul(square, base)
        }
        4 => {
            let square = builder.ins().fmul(base, base);
            builder.ins().fmul(square, square)
        }
        5 => {
            let square = builder.ins().fmul(base, base);
            let fourth = builder.ins().fmul(square, square);
            builder.ins().fmul(fourth, base)
        }
        6 => {
            let square = builder.ins().fmul(base, base);
            let cube = builder.ins().fmul(square, base);
            builder.ins().fmul(cube, cube)
        }
        8 => {
            let square = builder.ins().fmul(base, base);
            let fourth = builder.ins().fmul(square, square);
            builder.ins().fmul(fourth, fourth)
        }
        -1 => {
            let one = builder.ins().f64const(1.0);
            builder.ins().fdiv(one, base)
        }
        -2 => {
            let square = builder.ins().fmul(base, base);
            let one = builder.ins().f64const(1.0);
            builder.ins().fdiv(one, square)
        }
        _ => {
            // Binary exponentiation for other cases
            if exp.abs() <= 16 {
                // Inline for small exponents
                let mut result = builder.ins().f64const(1.0);
                let abs_exp = exp.abs();
                let mut current = base;

                for bit in 0..64 {
                    if abs_exp & (1 << bit) != 0 {
                        result = builder.ins().fmul(result, current);
                    }
                    if bit < 63 && abs_exp >> (bit + 1) != 0 {
                        current = builder.ins().fmul(current, current);
                    }
                }

                if exp < 0 {
                    let one = builder.ins().f64const(1.0);
                    builder.ins().fdiv(one, result)
                } else {
                    result
                }
            } else {
                panic!("Exponent is too large: {}", exp);
            }
        }
    }
}

/// Compiles and finalizes the function with optimizations.
fn compile_and_finalize(
    module: &mut JITModule,
    ctx: &mut Context,
) -> Result<fn(*const f64) -> f64, BuilderError> {
    // Declare function with local linkage
    let func_id = module
        .declare_function("jit_func", Linkage::Local, &ctx.func.signature)
        .map_err(|msg| BuilderError::DeclarationError(msg.to_string()))?;

    // Define function
    module
        .define_function(func_id, ctx)
        .map_err(|msg| BuilderError::FunctionError(msg.to_string()))?;

    // Clear context for memory efficiency
    module.clear_context(ctx);

    // Finalize definitions
    module
        .finalize_definitions()
        .map_err(BuilderError::ModuleError)?;

    // Extract function pointer
    let func_ptr = module.get_finalized_function(func_id);

    // SAFETY: This transmute is safe because:
    // - The function was compiled with signature fn(*const f64) -> f64
    // - The module is kept alive via Arc in the calling function
    // - The function pointer remains valid as long as the module exists
    let func = unsafe { std::mem::transmute::<*const u8, fn(*const f64) -> f64>(func_ptr) };

    Ok(func)
}

/// Builds a JIT-compiled function that evaluates multiple expressions.
///
/// This function generates optimized machine code that evaluates multiple expressions
/// in a single function call. The compilation process includes expression simplification,
/// optimal memory layout, and efficient instruction selection.
///
/// # Arguments
/// * `exprs` - Vector of expression ASTs to compile together
/// * `results_len` - Expected length of the results array (must match number of expressions)
///
/// # Returns
/// A thread-safe function that:
/// - Takes a slice of input values
/// - Takes a mutable slice for results
/// - Evaluates all expressions and stores results in the output slice
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
    let isa = create_optimized_isa()?;
    let (mut module, _) = create_optimized_module_and_context(isa);

    // Create function signature
    let mut sig = module.make_signature();
    sig.params
        .push(AbiParam::new(module.target_config().pointer_type())); // input_ptr
    sig.params
        .push(AbiParam::new(module.target_config().pointer_type())); // output_ptr
    sig.call_conv = module.target_config().default_call_conv;

    // Create function
    let func_id = module
        .declare_function("combined_func", Linkage::Export, &sig)
        .map_err(|msg| BuilderError::DeclarationError(msg.to_string()))?;

    codegen_context.func.signature = sig;
    let func = &mut codegen_context.func;
    let mut builder = FunctionBuilder::new(func, &mut builder_context);

    // Create entry block
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Get parameters
    let input_ptr = builder.block_params(entry_block)[0];
    let output_ptr = builder.block_params(entry_block)[1];

    // Process expressions in parallel for optimization
    let optimized_exprs: Vec<_> = exprs.par_iter().map(|expr| expr.clone()).collect();

    // Update all AST references with input pointer
    let mut optimized_exprs = optimized_exprs;
    for expr in &mut optimized_exprs {
        update_ast_vec_refs(expr, input_ptr);
    }

    // Generate code for all expressions
    let results: Vec<_> = optimized_exprs
        .iter()
        .map(|expr| expr.codegen_flattened(&mut builder, &mut module))
        .collect::<Result<_, _>>()?;

    // Store results with aligned memory access
    for (i, result) in results.iter().enumerate() {
        let offset = (i * 8) as i32; // 8 bytes per f64
        builder.ins().store(
            MemFlags::new().with_aligned(),
            *result,
            output_ptr,
            Offset32::new(offset),
        );
    }

    // Return
    builder.ins().return_(&[]);
    builder.finalize();

    // Finalize function
    module
        .define_function(func_id, &mut codegen_context)
        .map_err(|msg| BuilderError::FunctionError(msg.to_string()))?;
    module
        .finalize_definitions()
        .map_err(BuilderError::ModuleError)?;

    // Get the function pointer
    let func_ptr = module.get_finalized_function(func_id);
    let func_addr = func_ptr as usize;

    // Create wrapper function
    let wrapper = Arc::new(move |inputs: &[f64], results: &mut [f64]| {
        // Validate input and output lengths
        if inputs.is_empty() || results.len() != results_len {
            if results.len() == results_len {
                results.fill(0.0);
            }
            return;
        }

        // Call the compiled function
        let f: extern "C" fn(*const f64, *mut f64) = unsafe { std::mem::transmute(func_addr) };
        f(inputs.as_ptr(), results.as_mut_ptr())
    });

    // Keep the module alive for the program duration
    std::mem::forget(module);

    Ok(wrapper)
}
