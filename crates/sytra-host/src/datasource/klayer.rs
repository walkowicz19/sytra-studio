use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use async_trait::async_trait;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{DataSource, DataSourceError, DatasetSpec, Materialized, PreviewRows, Provenance};

const EXPORT_TOOL: &str = "export_dataset";
const PROTOCOL_VERSION: &str = "2025-06-18";

#[derive(Debug, Clone, serde::Deserialize)]
struct KlayerSourceParams {
    #[serde(default)]
    domain: Option<String>,
}

pub struct KlayerDataSource;

impl KlayerDataSource {
    fn parse_params(spec: &DatasetSpec) -> Result<KlayerSourceParams, DataSourceError> {
        serde_json::from_value(spec.params.clone())
            .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))
    }

    fn mcp_argv() -> Result<Vec<String>, DataSourceError> {
        if let Ok(raw) = std::env::var("KLAYER_MCP_ARGV") {
            let argv: Vec<String> = serde_json::from_str(&raw).map_err(|e| {
                DataSourceError::InvalidSpec(format!("KLAYER_MCP_ARGV must be a JSON string array: {e}"))
            })?;
            if argv.is_empty() {
                return Err(DataSourceError::InvalidSpec(
                    "KLAYER_MCP_ARGV is empty".into(),
                ));
            }
            return Ok(argv);
        }
        Ok(vec![
            "npx".into(),
            "-y".into(),
            "klayer-mcp@latest".into(),
        ])
    }
}

#[async_trait]
impl DataSource for KlayerDataSource {
    fn id(&self) -> &'static str {
        "klayer"
    }

    fn validate(&self, spec: &DatasetSpec) -> Result<(), DataSourceError> {
        let _params = Self::parse_params(spec)?;
        Ok(())
    }

    async fn preview(&self, spec: &DatasetSpec, n: usize) -> Result<PreviewRows, DataSourceError> {
        let temp = std::env::temp_dir().join(format!("sytra-klayer-preview-{}", uuid::Uuid::new_v4()));
        let materialized = self.materialize(spec, &temp).await?;
        let content = std::fs::read_to_string(&materialized.jsonl_path)?;
        let rows: Vec<_> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(n)
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .map(preview_row)
            .collect();
        let _ = std::fs::remove_dir_all(&temp);
        Ok(PreviewRows {
            rows,
            total_estimate: Some(materialized.row_count),
        })
    }

    async fn materialize(
        &self,
        spec: &DatasetSpec,
        out_dir: &Path,
    ) -> Result<Materialized, DataSourceError> {
        let params = Self::parse_params(spec)?;
        std::fs::create_dir_all(out_dir)?;
        let export_dir = out_dir.join("klayer-export");
        std::fs::create_dir_all(&export_dir)?;

        let argv = Self::mcp_argv()?;
        call_export_dataset(&argv, &export_dir, params.domain.as_deref())?;

        let jsonl_path = out_dir.join("data.jsonl");
        let row_count = concat_exported_jsonl(&export_dir, &jsonl_path)?;
        if row_count == 0 {
            return Err(DataSourceError::InvalidSpec(
                "klayer export_dataset wrote no reviewed/user training rows. Promote examples in klayer first; proposed stubs are not exported."
                    .into(),
            ));
        }

        Ok(Materialized {
            jsonl_path,
            fingerprint: self.fingerprint(spec)?,
            row_count,
            provenance: Some(Provenance {
                query: params.domain.clone().unwrap_or_else(|| "*".into()),
                min_trust_tier: "reviewed+user".into(),
                snapshot: format!("mcp:{EXPORT_TOOL}"),
            }),
        })
    }

    fn fingerprint(&self, spec: &DatasetSpec) -> Result<String, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let mut hasher = Sha256::new();
        hasher.update(EXPORT_TOOL.as_bytes());
        hasher.update(params.domain.as_deref().unwrap_or("*").as_bytes());
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }
}

fn preview_row(value: Value) -> Value {
    if value.get("prompt").is_some() {
        return value;
    }
    let Some(messages) = value.get("messages").and_then(|m| m.as_array()) else {
        return value;
    };
    let mut prompt = String::new();
    let mut completion = String::new();
    for message in messages {
        let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = message.get("content").and_then(|c| c.as_str()).unwrap_or("");
        match role {
            "assistant" => completion = content.to_string(),
            _ => {
                if !prompt.is_empty() {
                    prompt.push('\n');
                }
                prompt.push_str(content);
            }
        }
    }
    json!({
        "prompt": prompt,
        "completion": completion,
        "messages": messages,
    })
}

fn concat_exported_jsonl(export_dir: &Path, dest: &Path) -> Result<usize, DataSourceError> {
    let mut out = String::new();
    let mut row_count = 0usize;
    let mut files: Vec<PathBuf> = std::fs::read_dir(export_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    files.sort();
    for path in files {
        let content = std::fs::read_to_string(&path)?;
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            out.push_str(line);
            out.push('\n');
            row_count += 1;
        }
    }
    std::fs::write(dest, out)?;
    Ok(row_count)
}

fn call_export_dataset(
    argv: &[String],
    out_dir: &Path,
    domain: Option<&str>,
) -> Result<(), DataSourceError> {
    let mut client = KlayerMcpClient::spawn(argv)?;
    client.initialize()?;
    let mut arguments = json!({ "out_dir": out_dir.display().to_string() });
    if let Some(domain) = domain.filter(|d| !d.is_empty()) {
        arguments["domain"] = json!(domain);
    }
    let result = client.call_tool(EXPORT_TOOL, arguments)?;
    if result.get("isError").and_then(|v| v.as_bool()) == Some(true) {
        let text = result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|rows| rows.first())
            .and_then(|row| row.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("export_dataset returned isError");
        return Err(DataSourceError::InvalidSpec(text.into()));
    }
    Ok(())
}

struct KlayerMcpClient {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: i64,
}

impl KlayerMcpClient {
    fn spawn(argv: &[String]) -> Result<Self, DataSourceError> {
        let (program, args) = argv.split_first().ok_or_else(|| {
            DataSourceError::InvalidSpec("klayer MCP command is empty".into())
        })?;
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                DataSourceError::InvalidSpec(format!(
                    "failed to spawn klayer MCP (`{}`): {e}. Set KLAYER_MCP_ARGV to a JSON array such as [\"npx\",\"-y\",\"klayer-mcp@latest\"]. kl-train is not used.",
                    argv.join(" ")
                ))
            })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            DataSourceError::InvalidSpec("klayer MCP stdin missing".into())
        })?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| {
            DataSourceError::InvalidSpec("klayer MCP stdout missing".into())
        })?);
        Ok(Self {
            child,
            stdin,
            stdout,
            next_id: 0,
        })
    }

    fn initialize(&mut self) -> Result<(), DataSourceError> {
        let _ = self.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "sytra-studio", "version": "1.2.0" }
            }),
        )?;
        self.notify("notifications/initialized", json!({}))
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value, DataSourceError> {
        let response = self.request(
            "tools/call",
            json!({ "name": name, "arguments": arguments }),
        )?;
        if let Some(error) = response.get("error") {
            return Err(DataSourceError::InvalidSpec(format!(
                "klayer MCP {name} failed: {error}"
            )));
        }
        response
            .get("result")
            .cloned()
            .ok_or_else(|| DataSourceError::InvalidSpec("klayer MCP returned no result".into()))
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), DataSourceError> {
        write_rpc(
            &mut self.stdin,
            &json!({ "jsonrpc": "2.0", "method": method, "params": params }),
        )
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value, DataSourceError> {
        self.next_id += 1;
        let id = self.next_id;
        write_rpc(
            &mut self.stdin,
            &json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }),
        )?;
        loop {
            let msg = read_rpc(&mut self.stdout)?;
            if msg.get("id") == Some(&json!(id)) {
                return Ok(msg);
            }
        }
    }
}

impl Drop for KlayerMcpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn write_rpc(stdin: &mut impl Write, value: &Value) -> Result<(), DataSourceError> {
    let body = serde_json::to_vec(value)
        .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))?;
    write!(stdin, "Content-Length: {}\r\n\r\n", body.len())
        .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))?;
    stdin
        .write_all(&body)
        .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))?;
    stdin
        .flush()
        .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))
}

fn read_rpc(stdout: &mut BufReader<std::process::ChildStdout>) -> Result<Value, DataSourceError> {
    let mut header = String::new();
    stdout
        .read_line(&mut header)
        .map_err(|e| DataSourceError::InvalidSpec(format!("klayer MCP closed stdout: {e}")))?;
    if header.is_empty() {
        return Err(DataSourceError::InvalidSpec(
            "klayer MCP returned an empty response".into(),
        ));
    }
    if header.to_ascii_lowercase().starts_with("content-length:") {
        let len: usize = header
            .split(':')
            .nth(1)
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| DataSourceError::InvalidSpec("invalid Content-Length".into()))?;
        loop {
            let mut line = String::new();
            stdout.read_line(&mut line)?;
            if line.trim().is_empty() {
                break;
            }
        }
        let mut buf = vec![0u8; len];
        stdout.read_exact(&mut buf)?;
        return serde_json::from_slice(&buf)
            .map_err(|e| DataSourceError::InvalidSpec(format!("klayer MCP JSON: {e}")));
    }
    serde_json::from_str(header.trim())
        .map_err(|e| DataSourceError::InvalidSpec(format!("klayer MCP JSON: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasource::{DatasetSpec, SourceKind};
    use sytra_contracts::run_config::TrainMode;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn fake_argv() -> Vec<String> {
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("fixtures")
            .join("fake_klayer_mcp.py");
        vec!["python".into(), script.display().to_string()]
    }

    #[tokio::test]
    async fn materialize_calls_export_dataset_not_kl_train() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(
            "KLAYER_MCP_ARGV",
            serde_json::to_string(&fake_argv()).unwrap(),
        );
        let spec = DatasetSpec {
            source: SourceKind::Klayer,
            train_mode: TrainMode::Sft,
            params: json!({ "domain": "demo" }),
        };
        let out = std::env::temp_dir().join(format!("sytra-klayer-mcp-{}", uuid::Uuid::new_v4()));
        let source = KlayerDataSource;
        let materialized = source.materialize(&spec, &out).await.expect("export");
        assert_eq!(materialized.row_count, 1);
        assert_eq!(materialized.provenance.as_ref().unwrap().snapshot, "mcp:export_dataset");
        let line = std::fs::read_to_string(&materialized.jsonl_path).unwrap();
        assert!(line.contains("\"messages\""));
        let preview = source.preview(&spec, 1).await.unwrap();
        assert_eq!(preview.rows[0]["completion"], "4");
        let _ = std::fs::remove_dir_all(&out);
        std::env::remove_var("KLAYER_MCP_ARGV");
    }
}
