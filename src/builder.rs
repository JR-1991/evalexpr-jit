//! This module provides functionality for JIT compilation of mathematical expressions.
//! It uses Cranelift as the backend compiler to generate native machine code.
//!
//! The main entry points are:
//! - `build_function()` - Compiles a single expression into a JIT function
//! - `build_vector_function()` - Compiles an expression into a SIMD-aware vector function
//! - `build_combined_function()` - Compiles multiple expressions into a single JIT function
//!
//! # Architecture
//!
//! The compilation pipeline consists of:
//! 1. **Expression Optimization** - Multiple passes of algebraic simplification
//! 2. **ISA Configuration** - Target-specific optimizations and feature detection
//! 3. **Module Setup** - JIT module creation with math function linking
//! 4. **Code Generation** - Cranelift IR generation from optimized expressions
//! 5. **Compilation** - Native machine code generation and finalization
//!
//! # Performance Optimizations
//!
//! - **Multi-pass simplification** for constant folding and dead code elimination
//! - **Memory prefetch hints** for optimal cache utilization
//! - **SIMD detection** and vectorization for supported architectures
//! - **Variable caching** to eliminate redundant memory access
//! - **Fused operations** for mathematical expressions

use std::sync::Arc;

use crate::{
    errors::{BuilderError, EquationError},
    expr::{Expr, VarRef},
    types::{CombinedJITFunction, VectorizedCombinedJITFunction, VectorizedJITFunction},
};
use cranelift::prelude::*;
use cranelift_codegen::{ir::immediates::Offset32, Context};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use isa::TargetIsa;
use rayon::prelude::*;

// ================================================================================================
// OPTIMIZATION PIPELINE
// ================================================================================================

/// Applies multiple optimization passes to an expression for better performance.
///
/// This function performs the following optimizations:
/// - Pre-evaluation with variable caching
/// - Multiple simplification passes for constant folding and algebraic simplification
/// - Dead code elimination
///
/// # Arguments
/// * `expr` - The expression to optimize
///
/// # Returns
/// An optimized expression ready for code generation
fn optimize_expression(expr: Expr) -> Expr {
    // Apply multiple simplification passes for optimization
    let mut var_cache = std::collections::HashMap::new();
    let pre_evaluated = expr.pre_evaluate(&mut var_cache);
    let simplified = pre_evaluated.simplify();
    let double_simplified = simplified.simplify();
    let triple_simplified = double_simplified.simplify();
    *triple_simplified
}

// ================================================================================================
// ISA AND MODULE CONFIGURATION
// ================================================================================================

/// Configuration for JIT compilation settings.
#[derive(Debug, Clone)]
pub struct JITConfig {
    /// Whether to enable function verification (slower but safer)
    pub enable_verifier: bool,
    /// Whether to enable alias analysis optimization
    pub enable_alias_analysis: bool,
    /// Whether to use position-independent code
    pub use_pic: bool,
    /// Whether to enable probestack for stack overflow protection
    pub enable_probestack: bool,
}

impl Default for JITConfig {
    fn default() -> Self {
        Self {
            enable_verifier: false,
            enable_alias_analysis: true,
            use_pic: false,
            enable_probestack: false,
        }
    }
}

/// Creates an optimized ISA for the host machine.
///
/// This function configures the target instruction set architecture with
/// performance optimizations suitable for mathematical expression evaluation.
/// The configuration includes speed-optimized code generation and
/// architecture-specific optimizations.
///
/// # Arguments
/// * `config` - Optional configuration for compilation settings
///
/// # Returns
/// An Arc-wrapped TargetIsa configured for optimal performance.
///
/// # Errors
/// Returns a BuilderError if the host machine architecture is not supported
pub(crate) fn create_optimized_isa(
    config: Option<JITConfig>,
) -> Result<(Arc<dyn TargetIsa>, settings::Flags), BuilderError> {
    let config = config.unwrap_or_default();
    let mut flag_builder = settings::builder();

    // Get target triple to detect architecture and capabilities
    let target_triple = target_lexicon::Triple::host();
    let is_x86 = matches!(
        target_triple.architecture,
        target_lexicon::Architecture::X86_64
    );

    // Optimization flags for performance
    flag_builder.set("opt_level", "speed_and_size").unwrap();
    flag_builder
        .set(
            "enable_verifier",
            if config.enable_verifier {
                "true"
            } else {
                "false"
            },
        )
        .unwrap();
    flag_builder
        .set(
            "enable_alias_analysis",
            if config.enable_alias_analysis {
                "true"
            } else {
                "false"
            },
        )
        .unwrap();
    flag_builder.set("use_colocated_libcalls", "true").unwrap();
    flag_builder
        .set("is_pic", if config.use_pic { "true" } else { "false" })
        .unwrap();

    // CPU-specific optimizations
    if is_x86 && !config.enable_probestack {
        flag_builder.set("enable_probestack", "false").unwrap();
    }

    let isa_builder = cranelift_native::builder()
        .map_err(|msg| BuilderError::HostMachineNotSupported(msg.to_string()))?;

    let flags = settings::Flags::new(flag_builder);
    let isa = isa_builder
        .finish(flags.clone())
        .map_err(BuilderError::CodegenError)?;

    Ok((isa, flags))
}

/// Creates and configures a JIT module with optimized settings and math function links.
///
/// This function sets up a Cranelift JIT module with:
/// - Performance-optimized compilation flags
/// - Standard math function symbols for external calls
/// - Proper calling convention configuration
///
/// # Arguments
/// * `isa` - The target instruction set architecture
///
/// # Returns
/// A configured JITModule ready for function compilation
fn create_jit_module(isa: Arc<dyn TargetIsa>) -> JITModule {
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Link standard math functions for external calls
    link_math_functions(&mut builder);

    JITModule::new(builder)
}

/// Links standard math functions to the JIT module.
///
/// This function makes standard library math functions available for external calls
/// from JIT-compiled code. Functions are linked by symbol name to their implementations.
///
/// # Arguments
/// * `builder` - The JIT builder to add symbol links to
fn link_math_functions(builder: &mut JITBuilder) {
    // Basic math functions
    builder.symbol("exp", f64::exp as *const u8);
    builder.symbol("log", f64::ln as *const u8);
    builder.symbol("ln", f64::ln as *const u8);
    builder.symbol("sqrt", f64::sqrt as *const u8);
    builder.symbol("powi", f64::powi as *const u8);
    builder.symbol("pow", f64::powf as *const u8);

    // Trigonometric functions
    builder.symbol("sin", f64::sin as *const u8);
    builder.symbol("cos", f64::cos as *const u8);
    builder.symbol("tan", f64::tan as *const u8);

    // Utility functions
    builder.symbol("fabs", f64::abs as *const u8);
    builder.symbol("floor", f64::floor as *const u8);
    builder.symbol("ceil", f64::ceil as *const u8);
    builder.symbol("round", f64::round as *const u8);

    // Fused multiply-add function
    builder.symbol("fma", f64_fma as *const u8);
}

/// Fused multiply-add implementation for external linking.
extern "C" fn f64_fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// Creates a function signature for vector expression evaluation.
///
/// Creates a signature for functions that process multiple points in
/// structure-of-arrays layout.
///
/// # Arguments
/// * `module` - The module to create the signature for
///
/// # Returns
/// A function signature for vector evaluation
fn create_vector_signature(module: &JITModule) -> Signature {
    let mut sig = module.make_signature();
    let ptr_ty = module.target_config().pointer_type();

    sig.params.push(AbiParam::new(ptr_ty)); // input ptr
    sig.params.push(AbiParam::new(ptr_ty)); // output ptr
    sig.params.push(AbiParam::new(types::I32)); // n_points
    sig.params.push(AbiParam::new(types::I32)); // n_vars
    sig.call_conv = module.target_config().default_call_conv;

    sig
}

/// Creates a function signature for combined expression evaluation.
///
/// Creates a signature for functions that evaluate multiple expressions
/// and store results in an output array.
///
/// # Arguments
/// * `module` - The module to create the signature for
///
/// # Returns
/// A function signature for combined evaluation
fn create_combined_signature(module: &JITModule) -> Signature {
    let mut sig = module.make_signature();
    let ptr_ty = module.target_config().pointer_type();

    sig.params.push(AbiParam::new(ptr_ty)); // input_ptr
    sig.params.push(AbiParam::new(ptr_ty)); // output_ptr
    sig.call_conv = module.target_config().default_call_conv;

    sig
}

// ================================================================================================
// SIMD CONFIGURATION
// ================================================================================================

/// Determines the optimal SIMD lane count and vector type for the host architecture.
///
/// This function detects the widest SIMD instruction set available on the host
/// and returns the appropriate lane count and Cranelift vector type.
///
/// # Returns
/// A tuple of (lane_count, vector_type) for optimal SIMD operations
fn simd_lane_and_type() -> (u8, Type) {
    // Fallback for architectures without SIMD support
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return (1, types::F64);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // 512-bit vectors → 8 × f64
        if std::is_x86_feature_detected!("avx512f") {
            return (8, types::F64X8);
        }
        // 256-bit vectors → 4 × f64
        if std::is_x86_feature_detected!("avx") {
            return (4, types::F64X4);
        }
        // Fallback to scalar for x86 without SIMD
        return (1, types::F64);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is fixed 128-bit → 2 × f64
        (2, types::F64X2)
    }
}

// ================================================================================================
// MEMORY OPTIMIZATION
// ================================================================================================

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

// ================================================================================================
// FUNCTION COMPILATION
// ================================================================================================

/// Compiles and finalizes a function with error handling.
///
/// This function handles the complete compilation pipeline from function declaration
/// to native code generation with proper error handling and cleanup.
///
/// # Arguments
/// * `module` - The JIT module to compile in
/// * `ctx` - The function context containing the IR
/// * `func_name` - The name to give the compiled function
/// * `linkage` - The linkage type for the function
///
/// # Returns
/// A function pointer to the compiled native code
///
/// # Errors
/// Returns a BuilderError if compilation fails at any stage
fn compile_and_finalize_function(
    module: &mut JITModule,
    ctx: &mut Context,
    func_name: &str,
    linkage: Linkage,
) -> Result<*const u8, BuilderError> {
    // Declare function
    let func_id = module
        .declare_function(func_name, linkage, &ctx.func.signature)
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
    Ok(module.get_finalized_function(func_id))
}

// ================================================================================================
// PUBLIC API FUNCTIONS
// ================================================================================================

/// Builds a SIMD-aware vector function for efficient batch evaluation.
///
/// Evaluates `expr` for `n` points stored in *structure-of-arrays* layout
/// (`[x0 … xN][y0 … yN] …`) and writes the results back into `out`.
///
/// # Arguments
/// * `expr` - The expression to compile for vector evaluation
///
/// # Returns
/// A thread-safe function that evaluates the expression for multiple points
///
/// # Errors
/// Returns an EquationError if compilation fails
pub fn build_vector_function(expr: Expr) -> Result<VectorizedJITFunction, EquationError> {
    let (isa, _) = create_optimized_isa(None)?;
    let mut module = create_jit_module(isa);
    let mut ctx = Context::new();

    // Set up function signature
    ctx.func.signature = create_vector_signature(&module);

    // Apply expression optimization passes
    let optimized_expr = optimize_expression(expr);

    // Build vector function body
    build_vector_function_body(&mut ctx, optimized_expr, &mut module)?;

    // Compile and get function pointer
    let func_ptr =
        compile_and_finalize_function(&mut module, &mut ctx, "vec_kernel", Linkage::Export)?;

    // Wrap as safe Rust closure
    let f: extern "C" fn(*const f64, *mut f64, i32, i32) = unsafe { std::mem::transmute(func_ptr) };

    // Keep the module alive
    std::mem::forget(module);

    Ok(Arc::new(move |inp, out| {
        let n_points = out.len() as i32;
        let n_vars = if n_points > 0 {
            (inp.len() / out.len()) as i32
        } else {
            0
        };
        f(inp.as_ptr(), out.as_mut_ptr(), n_points, n_vars);
    }))
}

/// Builds a SIMD‑aware *combined* vector function that evaluates **all expressions** for each point
/// in a structure‑of‑arrays layout.
///
/// Layout expectations:
/// * **Inputs**: identical to `build_vector_function` – every variable has its own contiguous slice
///   (`[x0 … xN][y0 … yN] …`).
/// * **Outputs**: expression‑major SoA – the results for *expr₀* come first (`[f₀(x0)…f₀(xN)]`),
///   followed by *expr₁*, etc. Total length must therefore be `exprs.len() * n_points`.
///
/// Wrapper will panic if `out.len() % exprs.len() != 0`.
pub fn build_combined_vector_function(
    exprs: Vec<Expr>,
) -> Result<VectorizedCombinedJITFunction, EquationError> {
    let (isa, _) = create_optimized_isa(None)?;
    let mut module = create_jit_module(isa);
    let mut ctx = Context::new();

    // Re‑use the standard SIMD signature (in_ptr, out_ptr, n_points, n_vars)
    ctx.func.signature = create_vector_signature(&module);

    // Generate body
    build_combined_vector_function_body(&mut ctx, exprs.clone(), &mut module)?;

    // Finalise & obtain raw fn ptr
    let func_ptr = compile_and_finalize_function(
        &mut module,
        &mut ctx,
        "combined_vec_kernel",
        Linkage::Export,
    )?;
    let f: extern "C" fn(*const f64, *mut f64, i32, i32) = unsafe { std::mem::transmute(func_ptr) };

    // Keep module alive for the lifetime of the closure
    std::mem::forget(module);

    let expr_count = exprs.len() as i32;
    Ok(Arc::new(move |inp: &[f64], out: &mut [f64]| {
        assert_eq!(
            out.len() as i32 % expr_count,
            0,
            "Output slice has wrong length"
        );
        let n_points = (out.len() as i32) / expr_count;
        let n_vars = if n_points > 0 {
            (inp.len() as i32) / n_points
        } else {
            0
        };
        if n_points == 0 {
            return;
        }
        f(inp.as_ptr(), out.as_mut_ptr(), n_points, n_vars);
    }))
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
    let (isa, _) = create_optimized_isa(None)?;
    let mut module = create_jit_module(isa);
    let mut ctx = Context::new();

    // Set up function signature
    ctx.func.signature = create_combined_signature(&module);

    // Build combined function body
    build_combined_function_body(&mut ctx, exprs, &mut module)?;

    // Compile and get function pointer
    let func_ptr =
        compile_and_finalize_function(&mut module, &mut ctx, "combined_func", Linkage::Export)?;
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

// ================================================================================================
// FUNCTION BODY BUILDERS
// ================================================================================================

/// Builds the function body for vector expression evaluation.
///
/// This function generates SIMD-optimized code that processes multiple data points
/// in structure-of-arrays layout with both vectorized and scalar fallback paths.
///
/// # Arguments
/// * `ctx` - The function context to build into
/// * `expr` - The expression tree to generate code from
/// * `module` - The module to compile into
///
/// # Errors
/// Returns an EquationError if code generation fails
fn build_vector_function_body(
    ctx: &mut Context,
    expr: Expr,
    module: &mut dyn Module,
) -> Result<(), EquationError> {
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

    // Select the widest SIMD width the host supports.
    let (lane, vec_ty) = simd_lane_and_type();

    // ───── Control-flow graph ─────
    let entry = b.create_block();
    let vec_head = b.create_block();
    let vec_body = b.create_block();
    let tail_head = b.create_block();
    let tail_body = b.create_block();
    let exit = b.create_block();

    // Add block parameters immediately after creating blocks
    let idx_param_vec = b.append_block_param(vec_head, types::I32);
    let idx_param_vec_body = b.append_block_param(vec_body, types::I32);
    let idx_param_tail = b.append_block_param(tail_head, types::I32);
    let idx_param_tail_body = b.append_block_param(tail_body, types::I32);

    // ───── Entry block ─────
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);

    let inputs = b.block_params(entry)[0];
    let outputs = b.block_params(entry)[1];
    let n_points = b.block_params(entry)[2];

    let lane_val = b.ins().iconst(types::I32, lane as i64);
    let zero = b.ins().iconst(types::I32, 0);

    // jump → vector loop
    b.ins().jump(vec_head, &[zero.into()]);

    // ───── Vector-loop header ─────
    b.switch_to_block(vec_head);
    let idx_vec = idx_param_vec;
    let idx_next = b.ins().iadd(idx_vec, lane_val);
    let vec_ok = b
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, idx_next, n_points);
    b.ins().brif(
        vec_ok,
        vec_body,
        &[idx_vec.into()],
        tail_head,
        &[idx_vec.into()],
    );
    // Don't seal vec_head yet - we'll jump back to it from vec_body

    // ───── Vector-loop body ─────
    b.switch_to_block(vec_body);
    let i = idx_param_vec_body;

    let vec_res = expr.codegen_flattened(&mut b, module, inputs, lane, vec_ty, i, n_points)?;

    let byte_off = b.ins().imul_imm(i, 8);
    let dst_ptr = b.ins().iadd(outputs, byte_off);
    b.ins().store(
        MemFlags::new().with_aligned(),
        vec_res,
        dst_ptr,
        Offset32::new(0),
    );

    // next iteration
    b.ins().jump(vec_head, &[idx_next.into()]);
    b.seal_block(vec_body);
    // Now we can seal vec_head since all jumps to it are complete
    b.seal_block(vec_head);

    // ───── Tail-loop header ─────
    b.switch_to_block(tail_head);
    let idx_tail = idx_param_tail;
    let tail_ok = b.ins().icmp(IntCC::UnsignedLessThan, idx_tail, n_points);
    b.ins()
        .brif(tail_ok, tail_body, &[idx_tail.into()], exit, &[]);
    // Don't seal tail_head yet - we'll jump back to it from tail_body

    // ───── Scalar tail body ─────
    b.switch_to_block(tail_body);
    let i_s = idx_param_tail_body;

    let scalar_res = expr.codegen_flattened(
        &mut b,
        module,
        inputs,
        1,
        types::F64,
        i_s, // keep point index for correct SoA loads
        n_points,
    )?;
    let imul_imm_ins = b.ins().imul_imm(i_s, 8);
    let dst_ptr_s = b.ins().iadd(outputs, imul_imm_ins);
    b.ins().store(
        MemFlags::new().with_aligned(),
        scalar_res,
        dst_ptr_s,
        Offset32::new(0),
    );

    let next_s = b.ins().iadd_imm(i_s, 1);
    b.ins().jump(tail_head, &[next_s.into()]);
    b.seal_block(tail_body);
    // Now we can seal tail_head since all jumps to it are complete
    b.seal_block(tail_head);

    // ───── Exit ─────
    b.switch_to_block(exit);
    b.ins().return_(&[]);
    b.seal_block(exit);
    b.finalize();

    Ok(())
}

fn build_combined_vector_function_body(
    ctx: &mut Context,
    exprs: Vec<Expr>,
    module: &mut dyn Module,
) -> Result<(), EquationError> {
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

    // SIMD config
    let (lane, vec_ty) = simd_lane_and_type();

    // ───── CFG blocks ─────
    let entry = b.create_block();
    let vec_head = b.create_block();
    let vec_body = b.create_block();
    let tail_head = b.create_block();
    let tail_body = b.create_block();
    let exit = b.create_block();

    // Loop‑index block parameters
    let idx_vec_head = b.append_block_param(vec_head, types::I32);
    let idx_vec_body = b.append_block_param(vec_body, types::I32);
    let idx_tail_head = b.append_block_param(tail_head, types::I32);
    let idx_tail_body = b.append_block_param(tail_body, types::I32);

    // ───── Entry ─────
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);

    let inputs = b.block_params(entry)[0];
    let outputs = b.block_params(entry)[1];
    let n_points = b.block_params(entry)[2];

    let lane_val = b.ins().iconst(types::I32, lane as i64);
    let zero = b.ins().iconst(types::I32, 0);

    // Jump into vector loop
    b.ins().jump(vec_head, &[zero.into()]);

    // ───── Vector‑loop header ─────
    b.switch_to_block(vec_head);
    let idx_vec = idx_vec_head;
    let idx_next = b.ins().iadd(idx_vec, lane_val);
    let vec_ok = b
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, idx_next, n_points);
    b.ins().brif(
        vec_ok,
        vec_body,
        &[idx_vec.into()],
        tail_head,
        &[idx_vec.into()],
    );
    // vec_head sealed later

    // ───── Vector‑loop body ─────
    b.switch_to_block(vec_body);
    let i = idx_vec_body; // current point index (vectorised)

    // For each expression generate vector code & store
    for (e_idx, expr) in exprs.iter().enumerate() {
        let vec_res = expr.codegen_flattened(&mut b, module, inputs, lane, vec_ty, i, n_points)?;

        // offset_bytes = ((e_idx * n_points) + i) * 8
        let e_const = b.ins().iconst(types::I32, e_idx as i64);
        let e_base_pts = b.ins().imul(e_const, n_points);
        let elem_idx = b.ins().iadd(e_base_pts, i);
        let byte_off = b.ins().imul_imm(elem_idx, 8);
        let dst_ptr = b.ins().iadd(outputs, byte_off);
        b.ins().store(
            MemFlags::new().with_aligned(),
            vec_res,
            dst_ptr,
            Offset32::new(0),
        );
    }

    // Next vector iteration
    b.ins().jump(vec_head, &[idx_next.into()]);
    b.seal_block(vec_body);
    b.seal_block(vec_head);

    // ───── Tail‑loop header ─────
    b.switch_to_block(tail_head);
    let idx_tail = idx_tail_head;
    let tail_ok = b.ins().icmp(IntCC::UnsignedLessThan, idx_tail, n_points);
    b.ins()
        .brif(tail_ok, tail_body, &[idx_tail.into()], exit, &[]);
    // tail_head sealed later

    // ───── Scalar tail body ─────
    b.switch_to_block(tail_body);
    let i_s = idx_tail_body;

    for (e_idx, expr) in exprs.iter().enumerate() {
        let scalar_res =
            expr.codegen_flattened(&mut b, module, inputs, 1, types::F64, i_s, n_points)?;

        let e_const = b.ins().iconst(types::I32, e_idx as i64);
        let e_base_pts = b.ins().imul(e_const, n_points);
        let elem_idx = b.ins().iadd(e_base_pts, i_s);
        let byte_off = b.ins().imul_imm(elem_idx, 8);
        let dst_ptr = b.ins().iadd(outputs, byte_off);
        b.ins().store(
            MemFlags::new().with_aligned(),
            scalar_res,
            dst_ptr,
            Offset32::new(0),
        );
    }

    // Next scalar iteration
    let next_s = b.ins().iadd_imm(i_s, 1);
    b.ins().jump(tail_head, &[next_s.into()]);
    b.seal_block(tail_body);
    b.seal_block(tail_head);

    // ───── Exit ─────
    b.switch_to_block(exit);
    b.ins().return_(&[]);
    b.seal_block(exit);
    b.finalize();

    Ok(())
}

/// Builds the function body for combined expression evaluation.
///
/// This function generates code that evaluates multiple expressions in a single
/// function call, storing results in an output array with optimal memory layout.
///
/// # Arguments
/// * `ctx` - The function context to build into
/// * `exprs` - The vector of expressions to evaluate
/// * `module` - The module to compile into
///
/// # Errors
/// Returns an EquationError if code generation fails
fn build_combined_function_body(
    ctx: &mut Context,
    exprs: Vec<Expr>,
    module: &mut dyn Module,
) -> Result<(), EquationError> {
    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

    // Create entry block
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Get parameters
    let input_ptr = builder.block_params(entry_block)[0];
    let output_ptr = builder.block_params(entry_block)[1];

    // Process expressions in parallel for optimization
    let mut optimized_exprs: Vec<_> = exprs
        .par_iter()
        .map(|expr| optimize_expression(expr.clone()))
        .collect();

    // Update all AST references with input pointer
    for expr in &mut optimized_exprs {
        update_ast_vec_refs(expr, input_ptr);
    }

    // Generate code for all expressions
    let results: Vec<_> = optimized_exprs
        .iter()
        .map(|expr| {
            // For scalar operations, use dummy values for n_points and n_vars
            let dummy_n_points = builder.ins().iconst(types::I32, 1);
            expr.codegen_flattened(
                &mut builder,
                module,
                input_ptr,
                1,
                types::F64,
                input_ptr,
                dummy_n_points,
            )
        })
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

    Ok(())
}

// ======================================
// FUNCTION BODY BUILDERS
// ======================================

/// Returns the CLIF IR for a given expression.
pub fn get_clif_ir(expr: Expr) -> Result<String, EquationError> {
    let (isa, _) = create_optimized_isa(None)?;
    let mut module = create_jit_module(isa);
    let mut ctx = Context::new();

    // Set up function signature
    ctx.func.signature = create_vector_signature(&module);

    // Apply expression optimization passes
    let optimized_expr = optimize_expression(expr);

    // Build vector function body
    build_vector_function_body(&mut ctx, optimized_expr, &mut module)?;

    Ok(ctx.func.display().to_string())
}
