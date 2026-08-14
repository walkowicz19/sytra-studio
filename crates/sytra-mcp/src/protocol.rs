use std::io::Write;

use serde_json::{json, Value};

pub const PROTOCOL_VERSION: &str = "2025-06-18";

pub fn respond(id: &Value, result: Value) {
    let msg = json!({ "jsonrpc": "2.0", "id": id, "result": result });
    println!("{msg}");
    if let Err(err) = std::io::stdout().flush() {
        eprintln!("failed to flush MCP stdout: {err}");
    }
}

pub fn respond_error(id: &Value, code: i64, message: &str) {
    let msg = json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } });
    println!("{msg}");
    if let Err(err) = std::io::stdout().flush() {
        eprintln!("failed to flush MCP stdout: {err}");
    }
}
