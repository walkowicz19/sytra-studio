//! Spawns the real sytra-mcp binary and exercises the MCP handshake and
//! tool calls over stdio, exactly as Claude Code / Cursor / Codex would.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl McpClient {
    fn spawn() -> Self {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let mut child = Command::new(env!("CARGO_BIN_EXE_sytra-mcp"))
            .env("SYTRA_WORKSPACE", &workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sytra-mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
            next_id: 0,
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let msg =
            json!({ "jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params });
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();

        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let response: Value = serde_json::from_str(&line).expect("response must be JSON");
        assert_eq!(
            response["id"],
            json!(self.next_id),
            "response id must match request"
        );
        response
    }

    fn call_tool(&mut self, name: &str, args: Value) -> Value {
        self.request("tools/call", json!({ "name": name, "arguments": args }))
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn handshake_lists_tools_and_answers_calls() {
    let mut client = McpClient::spawn();

    // 1. initialize
    let init = client.request(
        "initialize",
        json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0" }
        }),
    );
    assert_eq!(init["result"]["serverInfo"]["name"], "sytra-studio");
    assert!(init["result"]["capabilities"]["tools"].is_object());

    // 2. tools/list carries the frozen tool set and input schemas
    let tools = client.request("tools/list", json!({}));
    let listed = tools["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = listed.iter().map(|t| t["name"].as_str().unwrap()).collect();
    const FROZEN_TOOLS: &[&str] = &[
        "get_status",
        "get_settings",
        "set_cache_dir",
        "set_main_memory_limit",
        "configure_fast_cache",
        "download_model",
        "list_catalog",
        "guider_recommend",
        "merge_check",
        "list_runs",
        "get_run",
        "start_train",
        "start_merge",
        "stop_op",
        "preview_dataset",
        "export_guide",
        "plan_inference",
    ];
    assert_eq!(names, FROZEN_TOOLS, "MCP tool names/order must stay stable");
    let required_by_name: std::collections::BTreeMap<&str, Vec<&str>> = listed
        .iter()
        .map(|t| {
            let name = t["name"].as_str().unwrap();
            let required = t["inputSchema"]["required"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            (name, required)
        })
        .collect();
    assert_eq!(required_by_name["download_model"], vec!["model"]);
    assert_eq!(required_by_name["merge_check"], vec!["models", "method"]);
    assert_eq!(required_by_name["get_run"], vec!["op_id"]);
    assert_eq!(required_by_name["start_train"], vec!["config"]);
    assert_eq!(required_by_name["start_merge"], vec!["config"]);
    assert_eq!(required_by_name["preview_dataset"], vec!["source"]);
    assert_eq!(required_by_name["plan_inference"], vec!["model_path"]);

    // 3. get_status returns hardware info
    let status = client.call_tool("get_status", json!({}));
    assert_eq!(status["result"]["isError"], false);
    let text = status["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert!(parsed["backend"].is_string());
    assert_eq!(parsed["running"], false);
    assert_eq!(parsed["envs"]["refuses_system_python"], true);
    assert_eq!(parsed["envs"]["schema"], 2);

    let settings = client.call_tool("get_settings", json!({}));
    assert_eq!(settings["result"]["isError"], false);
    let text = settings["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert!(parsed["hf_cache_dir"].is_string());
    assert!(parsed["effective_main_memory_mb"].as_u64().unwrap() >= 2048);

    // 4. merge_check validates a real pair
    let check = client.call_tool(
        "merge_check",
        json!({ "models": ["a/x", "b/y"], "method": "dare_ties" }),
    );
    assert_eq!(check["result"]["isError"], false);

    // 5. unknown tool is a tool-level error, not a protocol failure
    let bad = client.call_tool("does_not_exist", json!({}));
    assert_eq!(bad["result"]["isError"], true);

    // 6. list_runs answers (possibly empty) without error
    let runs = client.call_tool("list_runs", json!({}));
    assert_eq!(runs["result"]["isError"], false);
}

#[test]
fn preview_dataset_reads_local_fixture() {
    let mut client = McpClient::spawn();
    client.request(
        "initialize",
        json!({ "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "t", "version": "0" } }),
    );

    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let fixture = workspace.join("fixtures").join("data.golden.jsonl");

    let preview = client.call_tool(
        "preview_dataset",
        json!({
            "source": {
                "source": "local",
                "local": {
                    "path": fixture.to_string_lossy(),
                    "format": "jsonl",
                    "mapping": { "prompt": "prompt", "completion": "completion" }
                }
            },
            "rows": 2
        }),
    );
    assert_eq!(
        preview["result"]["isError"], false,
        "preview failed: {preview}"
    );
    let text = preview["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["rows"].as_array().unwrap().len(), 2);
}
