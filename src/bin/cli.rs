use clap::Parser;
use evalexpr::build_operator_tree;
use evalexpr_jit::{builder::get_clif_ir, convert::build_ast, equation::extract_symbols};
use std::process;

#[derive(Parser)]
#[command(name = "evalexpr-clif")]
#[command(about = "Generate CLIF IR from mathematical expressions")]
#[command(version)]
struct Args {
    /// Mathematical expression to convert to CLIF IR
    expression: String,
}

fn main() {
    let args = Args::parse();

    match generate_clif_ir(&args.expression) {
        Ok(clif_ir) => {
            println!("CLIF IR for expression '{}':", args.expression);
            println!("{}", clif_ir);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    }
}

fn generate_clif_ir(expression: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Parse the expression string into an AST using evalexpr
    let node = build_operator_tree(expression)?;

    // Extract variables from the expression
    let var_map = extract_symbols(&node);

    // Convert the evalexpr AST to our internal expression format
    let expr = build_ast(&node, &var_map)?;

    // Generate CLIF IR from the expression
    let clif_ir = get_clif_ir(expr)?;

    Ok(clif_ir)
}
