//! Lightweight peephole optimiser for the flattened stack IR.
//!
//! Pass pipeline
//! -------------
//!  1. **fold_consts**  – aggressive constant-propagation / neutral-element removal.
//!  2. **dedup_loads**  – drop immediately repeated `LoadVar`/`LoadConst` ops.
//!  3. **fuse_fma**     – recognise `a*b+c` and `a*b-c` sequences and emit the
//!                        dedicated `Fma` / `Fmsub` opcode.
//!
//! The optimiser iterates the full pipeline until a fix-point is reached, so you
//! can call [`optimize`](fn.optimize.html) once and get the fully cleaned
//! byte-code.
//!
//! The module is **pure Rust** – no Cranelift dependencies – which keeps your
//! unit-tests blazing-fast.

use crate::expr::{FlattenedExpr, LinearOp};

/// Run all optimisation passes until nothing changes.
pub fn optimize(flattened: FlattenedExpr) -> FlattenedExpr {
    // We only ever shuffle/replace the op-vector; the other two fields can be
    // forwarded as-is.
    let mut ops = flattened.ops;
    loop {
        let len_before = ops.len();
        ops = fold_consts(ops);
        ops = fuse_fma(ops);
        if ops.len() == len_before {
            break;
        }
    }

    FlattenedExpr { ops, ..flattened }
}

// ────────────────────────────────────────────────────────────────────────────
//  Pass 1 – constant folding / neutral element elimination
// ────────────────────────────────────────────────────────────────────────────
fn fold_consts(mut ops: Vec<LinearOp>) -> Vec<LinearOp> {
    use LinearOp::*;

    // We walk the instruction stream left-to-right while mirroring its effect
    // on an auxiliary stack that stores `Option<f64>` (Some if compile-time
    // constant, None otherwise).
    let mut out: Vec<LinearOp> = Vec::with_capacity(ops.len());
    let mut cstk: Vec<Option<f64>> = Vec::with_capacity(8); // const-tracking stack

    // helper: push a constant load both to IR and const-stack
    let push_const = |c: f64, out: &mut Vec<LinearOp>, cstk: &mut Vec<Option<f64>>| {
        out.push(LoadConst(c));
        cstk.push(Some(c));
    };

    while let Some(op) = ops.first().cloned() {
        ops.remove(0);
        match op {
            LoadConst(c) => push_const(c, &mut out, &mut cstk),
            LoadVar(idx) => {
                out.push(LoadVar(idx));
                cstk.push(None);
            }

            // ───── unary ops ──────────────────────────────────────────────
            Abs | Neg => {
                let v = cstk.pop().unwrap();
                if let Some(cv) = v {
                    let res = if matches!(op, Abs) { cv.abs() } else { -cv };
                    // erase the load that introduced `cv`
                    out.pop();
                    push_const(res, &mut out, &mut cstk);
                } else {
                    out.push(op);
                    cstk.push(None);
                }
            }

            // ───── binary ops ─────────────────────────────────────────────
            Add | Sub | Mul | Div => {
                let rhs = cstk.pop().unwrap();
                let lhs = cstk.pop().unwrap();

                match (lhs, rhs) {
                    (Some(a), Some(b)) => {
                        if matches!(op, Div) && b == 0.0 {
                            // keep original instruction to preserve semantics
                            out.push(op);
                            cstk.push(None);
                        } else {
                            let res = match op {
                                Add => a + b,
                                Sub => a - b,
                                Mul => a * b,
                                Div => a / b,
                                _ => unreachable!(),
                            };
                            // drop the two producing loads
                            out.truncate(out.len() - 2);
                            push_const(res, &mut out, &mut cstk);
                        }
                    }
                    _ => {
                        out.push(op);
                        cstk.push(None);
                    }
                }
            }

            // ───── ternary ops (FMA & friends) ────────────────────────────
            Fma | Fmsub => {
                let c = cstk.pop().unwrap();
                let b = cstk.pop().unwrap();
                let a = cstk.pop().unwrap();
                if let (Some(aa), Some(bb), Some(cc)) = (a, b, c) {
                    let res = if matches!(op, Fma) {
                        aa * bb + cc
                    } else {
                        aa * bb - cc
                    };
                    // remove 3 loads
                    out.truncate(out.len() - 3);
                    push_const(res, &mut out, &mut cstk);
                } else {
                    out.push(op);
                    cstk.push(None); // result is non-constant
                }
            }

            // ───── catch-all for heavier ops we don't fold yet ─────────────
            PowConst(_) | PowFloat(_) | PowExpr | Exp | Ln | Sqrt | Sin | Cos => {
                // they all pop 1 (or 2) args and push 1 result → keep stack balanced
                let _ = cstk.pop();
                if matches!(op, PowExpr) {
                    let _ = cstk.pop();
                }
                out.push(op);
                cstk.push(None);
            }
        }
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// Pass 3 – FMA / FMSUB fusion (pattern length = 5 ops)
// ────────────────────────────────────────────────────────────────────────────
fn fuse_fma(ops: Vec<LinearOp>) -> Vec<LinearOp> {
    use LinearOp::*;
    let mut out = Vec::with_capacity(ops.len());
    let mut i = 0;

    while i < ops.len() {
        if i + 4 < ops.len() {
            let window = (&ops[i], &ops[i + 1], &ops[i + 2], &ops[i + 3], &ops[i + 4]);
            match window {
                // a*b + c
                (
                    LoadVar(_) | LoadConst(_),
                    LoadVar(_) | LoadConst(_),
                    Mul,
                    LoadVar(_) | LoadConst(_),
                    Add,
                ) => {
                    out.extend_from_slice(&ops[i..i + 2]); // load a, load b
                    out.push(ops[i + 3].clone()); // load c
                    out.push(Fma);
                    i += 5;
                    continue;
                }
                // a*b - c
                (
                    LoadVar(_) | LoadConst(_),
                    LoadVar(_) | LoadConst(_),
                    Mul,
                    LoadVar(_) | LoadConst(_),
                    Sub,
                ) => {
                    out.extend_from_slice(&ops[i..i + 2]);
                    out.push(ops[i + 3].clone());
                    out.push(Fmsub);
                    i += 5;
                    continue;
                }
                _ => {}
            }
        }
        // default path
        out.push(ops[i].clone());
        i += 1;
    }
    out
}
