use std::collections::BTreeSet;
use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::workspace::user_home_dir;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpClientKind {
    ClaudeDesktop,
    ClaudeCode,
    Cursor,
    Codex,
    VsCode,
    Antigravity,
    Project,
}

impl McpClientKind {
    pub const ALL_GLOBAL: [McpClientKind; 6] = [
        Self::ClaudeDesktop,
        Self::ClaudeCode,
        Self::Cursor,
        Self::Codex,
        Self::VsCode,
        Self::Antigravity,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::ClaudeDesktop => "claude-desktop",
            Self::ClaudeCode => "claude-code",
            Self::Cursor => "cursor",
            Self::Codex => "codex",
            Self::VsCode => "vscode",
            Self::Antigravity => "antigravity",
            Self::Project => "project",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ClaudeDesktop => "Claude Desktop",
            Self::ClaudeCode => "Claude Code",
            Self::Cursor => "Cursor",
            Self::Codex => "Codex CLI",
            Self::VsCode => "VS Code",
            Self::Antigravity => "Antigravity",
            Self::Project => "Workspace (.mcp.json)",
        }
    }

    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "claude-desktop" | "claude_desktop" | "claude" => Ok(Self::ClaudeDesktop),
            "claude-code" | "claude_code" => Ok(Self::ClaudeCode),
            "cursor" => Ok(Self::Cursor),
            "codex" | "codex-cli" => Ok(Self::Codex),
            "vscode" | "vs-code" | "code" => Ok(Self::VsCode),
            "antigravity" | "agy" => Ok(Self::Antigravity),
            "project" | "local" | "repo" => Ok(Self::Project),
            other => Err(anyhow!("Unsupported MCP client '{other}'")),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct McpClientStatus {
    pub id: String,
    pub label: String,
    pub config_path: String,
    pub detected: bool,
    pub configured: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct McpStatusReport {
    pub mcp_binary_path: String,
    pub mcp_binary_exists: bool,
    pub workspace: String,
    pub runner_ready: bool,
    pub clients: Vec<McpClientStatus>,
    pub generic_json: String,
    pub vscode_json: String,
    pub codex_toml: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct McpActionOutcome {
    pub id: String,
    pub label: String,
    pub config_path: String,
    pub action: String,
}

pub fn mcp_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "sytra-mcp.exe"
    } else {
        "sytra-mcp"
    }
}

pub fn studio_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "sytra-studio.exe"
    } else {
        "sytra-studio"
    }
}

pub fn resolve_mcp_binary(workspace: &Path) -> PathBuf {
    let exe_name = mcp_binary_name();
    let mut candidates = Vec::new();
    if let Ok(current) = std::env::current_exe() {
        if let Some(dir) = current.parent() {
            candidates.push(dir.join(exe_name));
        }
    }
    if let Some(home) = user_home_dir() {
        candidates.push(home.join(".sytra").join("bin").join(exe_name));
    }
    candidates.push(workspace.join("bin").join(exe_name));
    candidates.push(workspace.join("target-build").join("release").join(exe_name));
    candidates.push(workspace.join("target-build").join("debug").join(exe_name));
    let platform_dir = if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    };
    candidates.push(workspace.join("binaries").join(platform_dir).join(exe_name));

    for candidate in &candidates {
        if candidate.is_file() {
            return candidate.clone();
        }
    }
    if let Some(path_var) = std::env::var_os("PATH") {
        for entry in std::env::split_paths(&path_var) {
            let candidate = entry.join(exe_name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    user_home_dir()
        .map(|h| h.join(".sytra").join("bin").join(exe_name))
        .unwrap_or_else(|| workspace.join("bin").join(exe_name))
}

fn app_data_dir() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| user_home_dir().map(|h| h.join("AppData").join("Roaming")))
    } else if cfg!(target_os = "macos") {
        user_home_dir().map(|h| h.join("Library").join("Application Support"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| user_home_dir().map(|h| h.join(".config")))
    }
}

pub fn client_config_path(kind: McpClientKind, cwd: &Path) -> Result<PathBuf> {
    if kind == McpClientKind::Project {
        return Ok(cwd.join(".mcp.json"));
    }
    let home = user_home_dir().ok_or_else(|| anyhow!("Could not locate user home directory"))?;
    match kind {
        McpClientKind::ClaudeDesktop => {
            let base = app_data_dir().ok_or_else(|| anyhow!("Could not locate app config dir"))?;
            Ok(base.join("Claude").join("claude_desktop_config.json"))
        }
        McpClientKind::ClaudeCode => Ok(home.join(".claude.json")),
        McpClientKind::Cursor => Ok(home.join(".cursor").join("mcp.json")),
        McpClientKind::Codex => {
            let codex_home = std::env::var_os("CODEX_HOME")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".codex"));
            Ok(codex_home.join("config.toml"))
        }
        McpClientKind::VsCode => {
            let base = app_data_dir().ok_or_else(|| anyhow!("Could not locate VS Code config dir"))?;
            Ok(base.join("Code").join("User").join("mcp.json"))
        }
        McpClientKind::Antigravity => Ok(home.join(".gemini").join("antigravity").join("mcp_config.json")),
        McpClientKind::Project => unreachable!(),
    }
}

fn is_client_detected(kind: McpClientKind, config_path: &Path) -> bool {
    if config_path.is_file() {
        return true;
    }
    match kind {
        McpClientKind::ClaudeCode => user_home_dir().is_some_and(|h| h.join(".claude").is_dir()),
        McpClientKind::Project => false,
        _ => config_path.parent().is_some_and(|p| p.is_dir()),
    }
}

pub fn mcp_server_entry(binary: &Path, workspace: &Path, vscode: bool) -> Value {
    let mut obj = Map::new();
    if vscode {
        obj.insert("type".to_string(), Value::String("stdio".to_string()));
    }
    obj.insert(
        "command".to_string(),
        Value::String(binary.display().to_string()),
    );
    obj.insert("args".to_string(), Value::Array(Vec::new()));
    obj.insert(
        "env".to_string(),
        json!({
            "SYTRA_WORKSPACE": workspace.display().to_string()
        }),
    );
    Value::Object(obj)
}

pub fn codex_toml_snippet(binary: &Path, workspace: &Path) -> String {
    let cmd = toml_basic_string(&binary.display().to_string());
    let ws = toml_basic_string(&workspace.display().to_string());
    format!(
        "[mcp_servers.sytra-studio]\ncommand = \"{cmd}\"\nargs = []\n\n[mcp_servers.sytra-studio.env]\nSYTRA_WORKSPACE = \"{ws}\"\n"
    )
}

fn toml_basic_string(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn atomic_write_text(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config");
    let tmp = path.with_file_name(format!(".{file_name}.sytra-tmp-{}", std::process::id()));
    fs::write(&tmp, content.as_bytes())
        .with_context(|| format!("Failed to write temporary config {}", tmp.display()))?;
    if let Err(err) = fs::rename(&tmp, path) {
        if err.kind() == ErrorKind::AlreadyExists || cfg!(target_os = "windows") {
            let _ = fs::remove_file(path);
            fs::rename(&tmp, path)
                .with_context(|| format!("Failed to replace {}", path.display()))?;
        } else {
            let _ = fs::remove_file(&tmp);
            return Err(err).with_context(|| format!("Failed to write {}", path.display()));
        }
    }
    Ok(())
}

fn read_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }
    let raw = fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let trimmed = raw.trim_start_matches('\u{feff}').trim();
    if trimmed.is_empty() {
        return Ok(Map::new());
    }
    let parsed: Value = serde_json::from_str(trimmed).with_context(|| {
        format!(
            "Existing config {} is not valid JSON; refusing to overwrite",
            path.display()
        )
    })?;
    match parsed {
        Value::Object(map) => Ok(map),
        _ => Err(anyhow!(
            "Existing config {} must contain a JSON object at top level",
            path.display()
        )),
    }
}

fn json_root_key(kind: McpClientKind) -> &'static str {
    if kind == McpClientKind::VsCode {
        "servers"
    } else {
        "mcpServers"
    }
}

pub fn upsert_json_mcp_config(
    path: &Path,
    root_key: &str,
    entry: Value,
) -> Result<()> {
    let mut root = read_json_object(path)?;
    let servers = root
        .entry(root_key.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let Value::Object(map) = servers else {
        return Err(anyhow!(
            "Existing config {} has a non-object '{root_key}' field",
            path.display()
        ));
    };
    map.insert("sytra-studio".to_string(), entry);
    let mut serialized = serde_json::to_string_pretty(&Value::Object(root))?;
    serialized.push('\n');
    atomic_write_text(path, &serialized)
}

pub fn remove_json_mcp_config(path: &Path, root_key: &str) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let mut root = read_json_object(path)?;
    let Some(Value::Object(map)) = root.get_mut(root_key) else {
        return Ok(false);
    };
    if map.remove("sytra-studio").is_none() {
        return Ok(false);
    }
    let mut serialized = serde_json::to_string_pretty(&Value::Object(root))?;
    serialized.push('\n');
    atomic_write_text(path, &serialized)?;
    Ok(true)
}

fn is_sytra_toml_header(trimmed_line: &str) -> bool {
    if !trimmed_line.starts_with('[') || !trimmed_line.ends_with(']') {
        return false;
    }
    let inner = trimmed_line[1..trimmed_line.len() - 1].trim();
    matches!(
        inner,
        "mcp_servers.sytra-studio"
            | "mcp_servers.\"sytra-studio\""
            | "mcp_servers.'sytra-studio'"
            | "mcp_servers.sytra-studio.env"
            | "mcp_servers.\"sytra-studio\".env"
            | "mcp_servers.'sytra-studio'.env"
    )
}

pub fn strip_codex_sytra_sections(raw: &str) -> (String, bool) {
    let mut kept = Vec::new();
    let mut skipping = false;
    let mut removed = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && !trimmed.starts_with("[[") && trimmed.ends_with(']') {
            if is_sytra_toml_header(trimmed) {
                skipping = true;
                removed = true;
                continue;
            }
            skipping = false;
        }
        if !skipping {
            kept.push(line);
        }
    }
    let mut joined = kept.join("\n");
    while joined.ends_with("\n\n") {
        joined.pop();
    }
    if !joined.is_empty() && !joined.ends_with('\n') {
        joined.push('\n');
    }
    (joined, removed)
}

pub fn upsert_codex_toml(path: &Path, binary: &Path, workspace: &Path) -> Result<()> {
    let existing = if path.is_file() {
        fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?
    } else {
        String::new()
    };
    let (mut base, _) = strip_codex_sytra_sections(&existing);
    if !base.trim().is_empty() {
        base.push('\n');
    }
    base.push_str(&codex_toml_snippet(binary, workspace));
    atomic_write_text(path, &base)
}

pub fn remove_codex_toml(path: &Path) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let existing = fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let (updated, removed) = strip_codex_sytra_sections(&existing);
    if !removed {
        return Ok(false);
    }
    atomic_write_text(path, &updated)?;
    Ok(true)
}

fn is_client_configured(kind: McpClientKind, path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    if kind == McpClientKind::Codex {
        return fs::read_to_string(path)
            .map(|text| text.lines().any(|l| is_sytra_toml_header(l.trim())))
            .unwrap_or(false);
    }
    read_json_object(path)
        .ok()
        .and_then(|root| root.get(json_root_key(kind)).cloned())
        .and_then(|v| v.as_object().cloned())
        .is_some_and(|map| map.contains_key("sytra-studio"))
}

pub fn status_report(workspace: &Path, explicit_binary: Option<&Path>) -> McpStatusReport {
    let cwd = std::env::current_dir().unwrap_or_else(|_| workspace.to_path_buf());
    let binary = explicit_binary
        .map(PathBuf::from)
        .unwrap_or_else(|| resolve_mcp_binary(workspace));
    let mcp_binary_exists = binary.is_file();
    let runner_ready = workspace.join("runner").join("sytra_runner").is_dir();

    let mut clients = Vec::new();
    for kind in McpClientKind::ALL_GLOBAL {
        if let Ok(path) = client_config_path(kind, &cwd) {
            clients.push(McpClientStatus {
                id: kind.id().to_string(),
                label: kind.label().to_string(),
                config_path: path.display().to_string(),
                detected: is_client_detected(kind, &path),
                configured: is_client_configured(kind, &path),
            });
        }
    }

    let generic_json = serde_json::to_string_pretty(&json!({
        "mcpServers": {
            "sytra-studio": mcp_server_entry(&binary, workspace, false)
        }
    }))
    .unwrap_or_default();
    let vscode_json = serde_json::to_string_pretty(&json!({
        "servers": {
            "sytra-studio": mcp_server_entry(&binary, workspace, true)
        }
    }))
    .unwrap_or_default();
    let codex_toml = codex_toml_snippet(&binary, workspace);

    McpStatusReport {
        mcp_binary_path: binary.display().to_string(),
        mcp_binary_exists,
        workspace: workspace.display().to_string(),
        runner_ready,
        clients,
        generic_json,
        vscode_json,
        codex_toml,
    }
}

pub fn resolve_client_selection(
    selection: Option<&[String]>,
    cwd: &Path,
) -> Result<Vec<McpClientKind>> {
    let Some(items) = selection else {
        return Ok(detected_clients(cwd));
    };
    if items.is_empty() {
        return Ok(detected_clients(cwd));
    }
    let mut out = BTreeSet::new();
    for raw in items {
        for token in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match token.to_ascii_lowercase().as_str() {
                "auto" | "detected" => {
                    for k in detected_clients(cwd) {
                        out.insert(k);
                    }
                }
                "all" => {
                    for k in McpClientKind::ALL_GLOBAL {
                        out.insert(k);
                    }
                }
                _ => {
                    out.insert(McpClientKind::parse(token)?);
                }
            }
        }
    }
    Ok(out.into_iter().collect())
}

fn detected_clients(cwd: &Path) -> Vec<McpClientKind> {
    McpClientKind::ALL_GLOBAL
        .into_iter()
        .filter(|&kind| {
            client_config_path(kind, cwd)
                .map(|p| is_client_detected(kind, &p))
                .unwrap_or(false)
        })
        .collect()
}

pub fn install_mcp_clients(
    workspace: &Path,
    selection: Option<&[String]>,
    explicit_binary: Option<&Path>,
    dry_run: bool,
) -> Result<Vec<McpActionOutcome>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| workspace.to_path_buf());
    let targets = resolve_client_selection(selection, &cwd)?;
    let binary = explicit_binary
        .map(PathBuf::from)
        .unwrap_or_else(|| resolve_mcp_binary(workspace));

    let mut outcomes = Vec::new();
    for kind in targets {
        let path = client_config_path(kind, &cwd)?;
        if !dry_run {
            match kind {
                McpClientKind::Codex => upsert_codex_toml(&path, &binary, workspace)?,
                _ => upsert_json_mcp_config(
                    &path,
                    json_root_key(kind),
                    mcp_server_entry(&binary, workspace, kind == McpClientKind::VsCode),
                )?,
            }
        }
        outcomes.push(McpActionOutcome {
            id: kind.id().to_string(),
            label: kind.label().to_string(),
            config_path: path.display().to_string(),
            action: if dry_run { "would_configure" } else { "configured" }.to_string(),
        });
    }
    Ok(outcomes)
}

pub fn uninstall_mcp_clients(
    workspace: &Path,
    selection: Option<&[String]>,
    dry_run: bool,
) -> Result<Vec<McpActionOutcome>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| workspace.to_path_buf());
    let targets = resolve_client_selection(selection, &cwd)?;
    let mut outcomes = Vec::new();
    for kind in targets {
        let path = client_config_path(kind, &cwd)?;
        let changed = if dry_run {
            is_client_configured(kind, &path)
        } else if kind == McpClientKind::Codex {
            remove_codex_toml(&path)?
        } else {
            remove_json_mcp_config(&path, json_root_key(kind))?
        };
        if changed {
            outcomes.push(McpActionOutcome {
                id: kind.id().to_string(),
                label: kind.label().to_string(),
                config_path: path.display().to_string(),
                action: if dry_run { "would_remove" } else { "removed" }.to_string(),
            });
        }
    }
    Ok(outcomes)
}

pub(crate) fn safe_relative_components(rel: &str) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(part) => out.push(part),
            _ => return None,
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sytra-mcp-install-test-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn json_upsert_preserves_other_servers_and_uninstall_keeps_file_clean() {
        let dir = temp_dir("json-preserve");
        let cfg = dir.join("claude_desktop_config.json");
        fs::write(
            &cfg,
            r#"{"theme":"dark","mcpServers":{"other":{"command":"foo"}}}"#,
        )
        .unwrap();

        let bin = Path::new("C:\\Users\\alice\\.sytra\\bin\\sytra-mcp.exe");
        let ws = Path::new("C:\\Users\\alice\\.sytra");
        upsert_json_mcp_config(&cfg, "mcpServers", mcp_server_entry(bin, ws, false)).unwrap();

        let parsed: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert_eq!(parsed["theme"], "dark");
        assert_eq!(parsed["mcpServers"]["other"]["command"], "foo");
        assert_eq!(
            parsed["mcpServers"]["sytra-studio"]["command"],
            "C:\\Users\\alice\\.sytra\\bin\\sytra-mcp.exe"
        );
        assert_eq!(
            parsed["mcpServers"]["sytra-studio"]["env"]["SYTRA_WORKSPACE"],
            "C:\\Users\\alice\\.sytra"
        );

        assert!(remove_json_mcp_config(&cfg, "mcpServers").unwrap());
        let after: Value = serde_json::from_str(&fs::read_to_string(&cfg).unwrap()).unwrap();
        assert!(after["mcpServers"].get("sytra-studio").is_none());
        assert_eq!(after["mcpServers"]["other"]["command"], "foo");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_upsert_refuses_corrupt_config() {
        let dir = temp_dir("json-corrupt");
        let cfg = dir.join("broken.json");
        fs::write(&cfg, "{ not valid json").unwrap();
        let err = upsert_json_mcp_config(
            &cfg,
            "mcpServers",
            mcp_server_entry(Path::new("sytra-mcp"), Path::new("ws"), false),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not valid JSON"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn codex_toml_replaces_npx_entry_and_preserves_other_sections() {
        let dir = temp_dir("codex-toml");
        let cfg = dir.join("config.toml");
        fs::write(
            &cfg,
            "model = \"gpt-5\"\n\n[mcp_servers.sytra-studio]\ncommand = \"npx\"\nargs = [\"-y\", \"github:walkowicz19/sytra-studio\", \"mcp\"]\n\n[mcp_servers.other]\ncommand = \"other-mcp\"\n",
        )
        .unwrap();

        upsert_codex_toml(
            &cfg,
            Path::new("C:\\Users\\alice\\.sytra\\bin\\sytra-mcp.exe"),
            Path::new("C:\\Users\\alice\\.sytra"),
        )
        .unwrap();

        let text = fs::read_to_string(&cfg).unwrap();
        assert!(text.contains("model = \"gpt-5\""));
        assert!(text.contains("[mcp_servers.other]\ncommand = \"other-mcp\""));
        assert!(!text.contains("npx"));
        assert!(text.contains("C:\\\\Users\\\\alice\\\\.sytra\\\\bin\\\\sytra-mcp.exe"));

        assert!(remove_codex_toml(&cfg).unwrap());
        let after = fs::read_to_string(&cfg).unwrap();
        assert!(!after.contains("sytra-studio"));
        assert!(after.contains("[mcp_servers.other]"));
        let _ = fs::remove_dir_all(&dir);
    }
}
