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
        Self::spawn_in(&workspace)
    }

    fn spawn_in(workspace: &std::path::Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_sytra-mcp"))
            .env("SYTRA_WORKSPACE", workspace)
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
        "sytra_dataset_merge",
        "sytra_dataset_split",
        "export_guide",
        "plan_inference",
        "list_local_models",
        "get_download_status",
        "cancel_download",
        "build_moe_index",
        "generate_moe_oracle",
        "set_inference_settings",
        "get_model_settings",
        "set_model_settings",
        "start_server",
        "stop_server",
        "server_status",
        "convert_model",
        "export_model",
        "list_chat_skills",
        "list_chat_agents",
        "get_voice_status",
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
    assert_eq!(required_by_name["sytra_dataset_merge"], vec!["inputs", "output_dir"]);
    assert_eq!(required_by_name["sytra_dataset_split"], vec!["input", "output_dir"]);
    assert_eq!(required_by_name["plan_inference"], vec!["model_path"]);
    assert_eq!(required_by_name["build_moe_index"], vec!["model_path"]);
    assert_eq!(required_by_name["generate_moe_oracle"], vec!["model_path"]);
    assert_eq!(required_by_name["get_model_settings"], vec!["model_path"]);
    assert_eq!(required_by_name["set_model_settings"], vec!["model_path"]);
    assert_eq!(required_by_name["start_server"], vec!["model_path"]);
    assert_eq!(required_by_name["convert_model"], vec!["model"]);
    assert_eq!(required_by_name["export_model"], vec!["model"]);

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

#[test]
fn cli_flags_configure_project_mcp_without_npm() {
    let bin = env!("CARGO_BIN_EXE_sytra-mcp");

    let ver = Command::new(bin).arg("--version").output().expect("version");
    assert!(ver.status.success());
    let stdout = String::from_utf8_lossy(&ver.stdout);
    assert!(stdout.contains("sytra-mcp 1.3.1"), "unexpected version output: {stdout}");

    let cfg = Command::new(bin)
        .args(["print-config", "--client", "generic"])
        .output()
        .expect("print-config");
    assert!(cfg.status.success());
    let cfg_json: Value = serde_json::from_slice(&cfg.stdout).expect("valid config json");
    let cmd = cfg_json["mcpServers"]["sytra-studio"]["command"]
        .as_str()
        .expect("command string");
    assert!(!cmd.contains("npx"), "direct MCP config must not rely on npx");

    let tmp = std::env::temp_dir().join(format!("sytra-mcp-cli-proj-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let install = Command::new(bin)
        .current_dir(&tmp)
        .args([
            "install",
            "--client",
            "project",
            "--workspace",
            tmp.to_str().unwrap(),
        ])
        .output()
        .expect("install project");
    assert!(
        install.status.success(),
        "install failed: {}",
        String::from_utf8_lossy(&install.stderr)
    );

    let mcp_file = tmp.join(".mcp.json");
    assert!(mcp_file.is_file());
    let written: Value = serde_json::from_str(&std::fs::read_to_string(&mcp_file).unwrap()).unwrap();
    assert_eq!(
        written["mcpServers"]["sytra-studio"]["env"]["SYTRA_WORKSPACE"],
        tmp.to_string_lossy().as_ref()
    );
    assert!(tmp.join("runner").join("sytra_runner").join("__init__.py").is_file());

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn inference_setup_settings_and_chat_tools_work_over_stdio() {
    let ws = std::env::temp_dir().join(format!("sytra-mcp-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).unwrap();
    let model_file = ws.join("sample.gguf");
    std::fs::write(&model_file, b"GGUF").unwrap();

    let mut client = McpClient::spawn_in(&ws);
    client.request(
        "initialize",
        json!({ "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "t", "version": "0" } }),
    );

    let local = client.call_tool(
        "list_local_models",
        json!({ "custom_dir": ws.to_string_lossy() }),
    );
    assert_eq!(local["result"]["isError"], false, "local: {local}");
    let local_json: Value = serde_json::from_str(local["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert!(local_json["count"].as_u64().unwrap() >= 1);

    let inf_set = client.call_tool(
        "set_inference_settings",
        json!({
            "default_context_window": 8192,
            "default_temperature": 0.3,
            "kv_cache_quant": "fp16",
            "cpu_kv_cache": true
        }),
    );
    assert_eq!(inf_set["result"]["isError"], false, "set_inference_settings: {inf_set}");
    let inf_json: Value = serde_json::from_str(inf_set["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(inf_json["default_context_window"], 8192);
    assert_eq!(inf_json["kv_cache_quant"], "fp16");
    assert_eq!(inf_json["cpu_kv_cache"], true);

    let model_set = client.call_tool(
        "set_model_settings",
        json!({
            "model_path": model_file.to_string_lossy(),
            "context_length": 4096,
            "allow_unverified": true
        }),
    );
    assert_eq!(model_set["result"]["isError"], false, "set_model_settings: {model_set}");
    let model_get = client.call_tool(
        "get_model_settings",
        json!({ "model_path": model_file.to_string_lossy() }),
    );
    assert_eq!(model_get["result"]["isError"], false);
    let mjson: Value = serde_json::from_str(model_get["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(mjson["context_length"], 4096);
    assert_eq!(mjson["allow_unverified"], true);

    for tool_name in [
        "get_download_status",
        "server_status",
        "list_chat_skills",
        "list_chat_agents",
        "get_voice_status",
    ] {
        let res = client.call_tool(tool_name, json!({}));
        assert_eq!(res["result"]["isError"], false, "{tool_name} failed: {res}");
    }

    let _ = std::fs::remove_dir_all(&ws);
}
