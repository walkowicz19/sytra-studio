//! Custom agents, isolated 24/7 workspaces, host-PC gated tools, persistent cross-session memory,
//! and native DuckDuckGo web search independent of which LLM server is selected.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::process::apply_hidden_window;

const MAX_AGENTS: usize = 64;
const MAX_MEMORY_ENTRIES: usize = 2000;
const MAX_WORKSPACE_FILE_BYTES: usize = 256 * 1024;
const MAX_READ_BYTES: usize = 64 * 1024;
const MAX_CMD_OUTPUT_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentProfile {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default = "default_avatar_kind")]
    pub avatar_kind: String,
    #[serde(default)]
    pub avatar_url: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub endpoint_url: Option<String>,
    #[serde(default)]
    pub skill_ids: Vec<String>,
    /// `"sandbox"` (isolated 24/7 workspace), `"pc_ask"` (host PC per-call approval), or `"pc_auto"`
    #[serde(default = "default_computer_mode")]
    pub computer_mode: String,
    #[serde(default)]
    pub routine_enabled: bool,
    #[serde(default = "default_routine_interval")]
    pub routine_interval_mins: u64,
    #[serde(default)]
    pub routine_prompt: String,
    #[serde(default)]
    pub last_routine_at: Option<u64>,
}

fn default_avatar_kind() -> String {
    "pochi".to_string()
}

fn default_computer_mode() -> String {
    "sandbox".to_string()
}

fn default_routine_interval() -> u64 {
    30
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryEntry {
    pub id: String,
    pub agent_id: String,
    pub session_id: String,
    pub session_title: String,
    /// `"fact"`, `"compaction"`, or `"routine"`
    pub kind: String,
    pub summary: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WebSearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WebSearchResult {
    pub query: String,
    pub provider: String,
    pub results: Vec<WebSearchHit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn valid_id(raw: &str) -> bool {
    !raw.is_empty()
        && raw.len() <= 64
        && raw
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

pub fn normalize_id(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if valid_id(trimmed) {
        return Ok(trimmed.to_string());
    }
    let slug: String = trimmed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else if c.is_whitespace() || c == '.' {
                '-'
            } else {
                '\0'
            }
        })
        .filter(|&c| c != '\0')
        .collect();
    let cleaned = slug.trim_matches('-').to_string();
    if !valid_id(&cleaned) {
        return Err("Agent identifier must contain letters, digits, '-' or '_'.".into());
    }
    Ok(cleaned)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Cannot create directory: {e}"))?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&tmp, bytes).map_err(|e| format!("Cannot write temporary file: {e}"))?;
    if let Err(err) = fs::rename(&tmp, path) {
        if err.kind() == std::io::ErrorKind::AlreadyExists || cfg!(windows) {
            let _ = fs::remove_file(path);
            fs::rename(&tmp, path).map_err(|e| format!("Cannot finalize file write: {e}"))?;
        } else {
            let _ = fs::remove_file(&tmp);
            return Err(format!("Cannot finalize file write: {err}"));
        }
    }
    Ok(())
}

fn agents_registry_path(workspace: &Path) -> PathBuf {
    workspace.join("agents").join("registry.json")
}

fn memories_path(workspace: &Path) -> PathBuf {
    workspace.join("memory").join("memories.jsonl")
}

pub fn agent_workspace_dir(workspace: &Path, agent_id: &str) -> Result<PathBuf, String> {
    if !valid_id(agent_id) {
        return Err("Invalid agent identifier.".into());
    }
    let dir = workspace.join("agents").join(agent_id).join("workspace");
    fs::create_dir_all(&dir).map_err(|e| format!("Cannot initialize agent workspace: {e}"))?;
    dir.canonicalize()
        .map_err(|e| format!("Cannot resolve agent workspace: {e}"))
}

fn resolve_agent_rel_path(root: &Path, raw_rel: &str, must_exist: bool) -> Result<PathBuf, String> {
    let trimmed = raw_rel.trim();
    if trimmed.is_empty() || trimmed == "." {
        return Ok(root.to_path_buf());
    }
    let rel = Path::new(trimmed);
    for comp in rel.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err("Path escapes the agent's isolated workspace.".into());
            }
        }
    }
    let candidate = root.join(rel);
    if must_exist {
        let canonical = candidate
            .canonicalize()
            .map_err(|e| format!("Path not found in agent workspace: {e}"))?;
        if !canonical.starts_with(root) {
            return Err("Path escapes the agent's isolated workspace.".into());
        }
        Ok(canonical)
    } else {
        if let Some(parent) = candidate.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Cannot create directory in agent workspace: {e}"))?;
            let c_parent = parent
                .canonicalize()
                .map_err(|e| format!("Cannot resolve parent path: {e}"))?;
            if !c_parent.starts_with(root) {
                return Err("Path escapes the agent's isolated workspace.".into());
            }
        }
        Ok(candidate)
    }
}

pub fn list_agents(workspace: &Path) -> Result<Vec<AgentProfile>, String> {
    let path = agents_registry_path(workspace);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(&path).map_err(|e| format!("Cannot read agent registry: {e}"))?;
    let mut parsed: Vec<AgentProfile> =
        serde_json::from_str(&raw).map_err(|e| format!("Corrupt agent registry: {e}"))?;
    parsed.retain(|a| valid_id(&a.id));
    Ok(parsed)
}

pub fn save_agent(workspace: &Path, mut agent: AgentProfile) -> Result<AgentProfile, String> {
    agent.id = normalize_id(&agent.id)?;
    agent.name = agent.name.trim().chars().take(80).collect();
    if agent.name.is_empty() {
        agent.name = agent.id.clone();
    }
    agent.system_prompt = agent.system_prompt.chars().take(16_000).collect();
    agent.routine_prompt = agent.routine_prompt.chars().take(4_000).collect();
    agent.routine_interval_mins = agent.routine_interval_mins.clamp(1, 1440);
    if !matches!(
        agent.computer_mode.as_str(),
        "sandbox" | "pc_ask" | "pc_auto"
    ) {
        agent.computer_mode = "sandbox".to_string();
    }
    if !matches!(
        agent.avatar_kind.as_str(),
        "pochi" | "kitsune" | "mochi" | "owl" | "custom"
    ) {
        agent.avatar_kind = "pochi".to_string();
    }
    if let Some(url) = &agent.avatar_url {
        let trimmed = url.trim();
        let safe = trimmed.starts_with("data:image/")
            || trimmed.starts_with("https://")
            || trimmed.starts_with("http://");
        if trimmed.is_empty() || !safe || trimmed.len() > 512 * 1024 {
            agent.avatar_url = None;
        }
    }

    let _ = agent_workspace_dir(workspace, &agent.id)?;

    let mut existing = list_agents(workspace)?;
    if let Some(pos) = existing.iter().position(|a| a.id == agent.id) {
        existing[pos] = agent.clone();
    } else {
        if existing.len() >= MAX_AGENTS {
            return Err("Maximum number of custom agents reached.".into());
        }
        existing.push(agent.clone());
    }
    let encoded = serde_json::to_vec_pretty(&existing)
        .map_err(|e| format!("Cannot serialize agents: {e}"))?;
    atomic_write(&agents_registry_path(workspace), &encoded)?;
    Ok(agent)
}

pub fn delete_agent(workspace: &Path, id: &str) -> Result<(), String> {
    if !valid_id(id) {
        return Err("Invalid agent identifier.".into());
    }
    let mut existing = list_agents(workspace)?;
    existing.retain(|a| a.id != id);
    let encoded = serde_json::to_vec_pretty(&existing)
        .map_err(|e| format!("Cannot serialize agents: {e}"))?;
    atomic_write(&agents_registry_path(workspace), &encoded)
}

pub fn list_memories(workspace: &Path, agent_id: Option<&str>) -> Result<Vec<MemoryEntry>, String> {
    let path = memories_path(workspace);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file =
        fs::File::open(&path).map_err(|e| format!("Cannot open memory store: {e}"))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<MemoryEntry>(&line) {
            if let Some(filter) = agent_id {
                if entry.agent_id != filter && entry.agent_id != "default" {
                    continue;
                }
            }
            out.push(entry);
        }
    }
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

pub fn save_memory(
    workspace: &Path,
    agent_id: &str,
    session_id: &str,
    session_title: &str,
    kind: &str,
    summary: &str,
    tags: Vec<String>,
) -> Result<MemoryEntry, String> {
    let clean_agent = if agent_id.trim().is_empty() || agent_id.trim() == "default" {
        "default".to_string()
    } else {
        normalize_id(agent_id)?
    };
    let clean_summary: String = summary.trim().chars().take(8192).collect();
    if clean_summary.is_empty() {
        return Err("Memory summary cannot be empty.".into());
    }
    let clean_kind = if matches!(kind, "fact" | "compaction" | "routine") {
        kind.to_string()
    } else {
        "fact".to_string()
    };
    let entry = MemoryEntry {
        id: uuid::Uuid::new_v4().to_string(),
        agent_id: clean_agent,
        session_id: session_id.trim().chars().take(80).collect(),
        session_title: session_title.trim().chars().take(120).collect(),
        kind: clean_kind,
        summary: clean_summary,
        tags: tags
            .into_iter()
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .take(12)
            .collect(),
        created_at: now_secs(),
    };

    let mut all = list_memories(workspace, None)?;
    all.insert(0, entry.clone());
    if all.len() > MAX_MEMORY_ENTRIES {
        all.truncate(MAX_MEMORY_ENTRIES);
    }
    rewrite_memories(workspace, &all)?;
    Ok(entry)
}

pub fn delete_memory(workspace: &Path, id: &str) -> Result<(), String> {
    let mut all = list_memories(workspace, None)?;
    all.retain(|e| e.id != id);
    rewrite_memories(workspace, &all)
}

fn rewrite_memories(workspace: &Path, entries: &[MemoryEntry]) -> Result<(), String> {
    let mut buf = String::new();
    for entry in entries {
        if let Ok(line) = serde_json::to_string(entry) {
            buf.push_str(&line);
            buf.push('\n');
        }
    }
    atomic_write(&memories_path(workspace), buf.as_bytes())
}

pub fn recall_memories(
    workspace: &Path,
    agent_id: Option<&str>,
    query: &str,
    limit: usize,
) -> Result<Vec<MemoryEntry>, String> {
    let cap = limit.clamp(1, 25);
    let entries = list_memories(workspace, agent_id)?;
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .map(|s| s.to_ascii_lowercase())
        .filter(|s| s.len() >= 2)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if tokens.is_empty() {
        return Ok(entries.into_iter().take(cap).collect());
    }

    let mut scored: Vec<(usize, u64, MemoryEntry)> = entries
        .into_iter()
        .map(|entry| {
            let hay = format!(
                "{} {} {}",
                entry.summary.to_ascii_lowercase(),
                entry.session_title.to_ascii_lowercase(),
                entry.tags.join(" ")
            );
            let mut score = 0usize;
            for tok in &tokens {
                if hay.contains(tok) {
                    score += 3;
                }
                if entry.tags.iter().any(|tag| tag == tok) {
                    score += 4;
                }
            }
            if entry.kind == "compaction" {
                score += 1;
            }
            (score, entry.created_at, entry)
        })
        .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    Ok(scored.into_iter().take(cap).map(|(_, _, e)| e).collect())
}

pub fn execute_agent_tool(
    workspace: &Path,
    agent_id: &str,
    tool_name: &str,
    arguments: &Value,
) -> Result<Value, String> {
    let clean_agent = if agent_id.trim().is_empty() || agent_id.trim() == "default" {
        "default".to_string()
    } else {
        normalize_id(agent_id)?
    };
    let agents = list_agents(workspace)?;
    let agent = agents.iter().find(|a| a.id == clean_agent);
    let pc_allowed = agent.map(|a| a.computer_mode != "sandbox").unwrap_or(false);

    match tool_name {
        "workspace_list" => {
            let root = agent_workspace_dir(workspace, &clean_agent)?;
            let subdir = arguments.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let target = resolve_agent_rel_path(&root, subdir, true)?;
            if !target.is_dir() {
                return Err("Target path is not a directory.".into());
            }
            let mut items = Vec::new();
            for entry in fs::read_dir(&target)
                .map_err(|e| format!("Cannot read workspace directory: {e}"))?
                .flatten()
                .take(200)
            {
                let meta = entry.metadata().ok();
                items.push(json!({
                    "name": entry.file_name().to_string_lossy(),
                    "is_dir": meta.as_ref().map(|m| m.is_dir()).unwrap_or(false),
                    "bytes": meta.as_ref().map(|m| m.len()).unwrap_or(0),
                }));
            }
            Ok(json!({ "workspace": root.to_string_lossy(), "entries": items }))
        }
        "workspace_read" => {
            let root = agent_workspace_dir(workspace, &clean_agent)?;
            let rel = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing `path` argument.".to_string())?;
            let target = resolve_agent_rel_path(&root, rel, true)?;
            read_bounded_text(&target)
        }
        "workspace_write" => {
            let root = agent_workspace_dir(workspace, &clean_agent)?;
            let rel = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing `path` argument.".to_string())?;
            let content = arguments
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing `content` argument.".to_string())?;
            if content.len() > MAX_WORKSPACE_FILE_BYTES {
                return Err("File content exceeds 256 KiB workspace limit.".into());
            }
            let target = resolve_agent_rel_path(&root, rel, false)?;
            atomic_write(&target, content.as_bytes())?;
            Ok(json!({ "written": rel, "bytes": content.len() }))
        }
        "workspace_delete" => {
            let root = agent_workspace_dir(workspace, &clean_agent)?;
            let rel = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing `path` argument.".to_string())?;
            if let Ok(target) = resolve_agent_rel_path(&root, rel, true) {
                if target.is_file() {
                    let _ = fs::remove_file(&target);
                }
            }
            Ok(json!({ "removed": rel }))
        }
        "pc_list_dir" => {
            if !pc_allowed {
                return Err("Host PC access is disabled for this agent (running in isolated sandbox mode).".into());
            }
            let raw_path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from)
                .or_else(crate::workspace::user_home_dir)
                .ok_or_else(|| "Cannot resolve directory path.".to_string())?;
            let mut items = Vec::new();
            for entry in fs::read_dir(&raw_path)
                .map_err(|e| format!("Cannot read directory: {e}"))?
                .flatten()
                .take(200)
            {
                let meta = entry.metadata().ok();
                items.push(json!({
                    "name": entry.file_name().to_string_lossy(),
                    "is_dir": meta.as_ref().map(|m| m.is_dir()).unwrap_or(false),
                    "bytes": meta.as_ref().map(|m| m.len()).unwrap_or(0),
                }));
            }
            Ok(json!({ "path": raw_path.to_string_lossy(), "entries": items }))
        }
        "pc_read_file" => {
            if !pc_allowed {
                return Err("Host PC access is disabled for this agent (running in isolated sandbox mode).".into());
            }
            let raw_path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing `path` argument.".to_string())?;
            read_bounded_text(Path::new(raw_path))
        }
        "pc_run_command" => {
            if !pc_allowed {
                return Err("Host PC access is disabled for this agent (running in isolated sandbox mode).".into());
            }
            let command = arguments
                .get("command")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| "Missing `command` argument.".to_string())?;
            let cwd = arguments
                .get("cwd")
                .and_then(|v| v.as_str())
                .map(PathBuf::from);
            run_bounded_shell(command, cwd.as_deref())
        }
        _ => Err(format!("Unknown agent tool `{tool_name}`.")),
    }
}

fn read_bounded_text(path: &Path) -> Result<Value, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("Cannot open file: {e}"))?;
    let mut buf = vec![0u8; MAX_READ_BYTES + 1];
    let n = file
        .read(&mut buf)
        .map_err(|e| format!("Cannot read file: {e}"))?;
    let truncated = n > MAX_READ_BYTES;
    buf.truncate(n.min(MAX_READ_BYTES));
    let text = String::from_utf8_lossy(&buf).to_string();
    Ok(json!({
        "path": path.to_string_lossy(),
        "content": text,
        "truncated": truncated,
    }))
}

fn run_bounded_shell(command: &str, cwd: Option<&Path>) -> Result<Value, String> {
    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = Command::new("powershell");
        apply_hidden_window(&mut c);
        c.args(["-NoProfile", "-NonInteractive", "-Command", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };
    if let Some(dir) = cwd {
        if dir.is_dir() {
            cmd.current_dir(dir);
        }
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start command: {e}"))?;
    let deadline = Instant::now() + Duration::from_secs(25);
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|e| format!("Error waiting on command: {e}"))?
        {
            let mut out_bytes = Vec::new();
            let mut err_bytes = Vec::new();
            if let Some(mut s) = child.stdout.take() {
                let _ = s.by_ref().take(MAX_CMD_OUTPUT_BYTES as u64).read_to_end(&mut out_bytes);
            }
            if let Some(mut s) = child.stderr.take() {
                let _ = s.by_ref().take(MAX_CMD_OUTPUT_BYTES as u64).read_to_end(&mut err_bytes);
            }
            return Ok(json!({
                "exit_code": status.code().unwrap_or(-1),
                "stdout": String::from_utf8_lossy(&out_bytes).to_string(),
                "stderr": String::from_utf8_lossy(&err_bytes).to_string(),
            }));
        }
        if Instant::now() >= deadline {
            crate::process::kill_process_tree(child.id());
            let _ = child.wait();
            return Err("Host PC command timed out after 25s.".into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn strip_html_tags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_search_response(data: &Value, provider: &str, limit: usize) -> Result<Vec<WebSearchHit>, String> {
    let malformed = || "provider returned an invalid response".to_string();
    if !data.is_object() || data.get("error").is_some() {
        return Err(malformed());
    }
    let (section, list) = if provider == "brave" { ("web", "results") } else { ("query", "search") };
    let Some(rows) = data.get(section).and_then(|value| value.get(list)) else {
        return Ok(Vec::new());
    };
    let mut hits = Vec::new();
    for row in rows.as_array().ok_or_else(malformed)? {
        let title = strip_html_tags(row["title"].as_str().ok_or_else(malformed)?);
        let (url, description) = if provider == "brave" {
            (row["url"].as_str().ok_or_else(malformed)?.to_string(), "description")
        } else {
            let id = row["pageid"].as_u64().ok_or_else(malformed)?;
            (format!("https://en.wikipedia.org/?curid={id}"), "snippet")
        };
        if title.is_empty()
            || url.chars().count() > 512
            || !reqwest::Url::parse(&url).is_ok_and(|parsed| {
                matches!(parsed.scheme(), "http" | "https") && parsed.host_str().is_some()
            })
        {
            continue;
        }
        hits.push(WebSearchHit {
            title: title.chars().take(160).collect(),
            url,
            snippet: strip_html_tags(row[description].as_str().unwrap_or_default())
                .chars().take(280).collect(),
        });
        if hits.len() == limit {
            break;
        }
    }
    Ok(hits)
}

/// Native JSON API search keeps the desktop route available without a Python proxy.
pub fn search_web_native(query: &str, max_results: usize) -> WebSearchResult {
    let cleaned: String = query.split_whitespace().collect::<Vec<_>>().join(" ");
    let key = std::env::var("BRAVE_SEARCH_API_KEY").unwrap_or_default();
    let key = key.trim();
    let provider = if key.is_empty() { "wikipedia" } else { "brave" };
    let mut result = WebSearchResult {
        query: cleaned,
        provider: provider.to_string(),
        results: Vec::new(),
        error: None,
    };
    if result.query.is_empty() || result.query.chars().count() > 512 {
        result.error = Some("Search query must be 1 to 512 characters.".into());
        return result;
    }

    let count = max_results.clamp(1, 8).to_string();
    let search = || -> Result<Vec<WebSearchHit>, String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(8))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("SytraStudio/1.3 (https://github.com/walkowicz19/sytra-studio)")
            .build()
            .map_err(|_| "could not initialize the HTTP client".to_string())?;
        let request = if key.is_empty() {
            client.get("https://en.wikipedia.org/w/api.php").query(&[
                ("action", "query"), ("format", "json"), ("list", "search"),
                ("srsearch", &result.query), ("srlimit", &count),
            ])
        } else {
            client.get("https://api.search.brave.com/res/v1/web/search")
                .query(&[("q", result.query.as_str()), ("count", &count)])
                .header("X-Subscription-Token", key)
        };
        let response = request.header("Accept", "application/json").send()
            .map_err(|_| "could not connect".to_string())?;
        if !response.status().is_success() {
            return Err(format!("failed (HTTP {})", response.status().as_u16()));
        }
        const MAX_BODY: u64 = 512 * 1024;
        let mut body = Vec::new();
        response.take(MAX_BODY + 1).read_to_end(&mut body)
            .map_err(|_| "could not read the response".to_string())?;
        if body.len() as u64 > MAX_BODY {
            return Err("response exceeded the size limit".into());
        }
        let data = serde_json::from_slice(&body).map_err(|_| "returned invalid JSON".to_string())?;
        parse_search_response(&data, provider, max_results.clamp(1, 8))
    };
    match search() {
        Ok(hits) => result.results = hits,
        Err(error) => result.error = Some(format!("{provider} search: {error}.")),
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sytra-chat-agents-{name}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn saves_lists_and_deletes_custom_agents() {
        let ws = temp_workspace("agents");
        let saved = save_agent(
            &ws,
            AgentProfile {
                id: "Pochi Helper".into(),
                name: "Pochi".into(),
                role: "Research pet".into(),
                system_prompt: "Be helpful.".into(),
                avatar_kind: "pochi".into(),
                avatar_url: None,
                provider: Some("ollama".into()),
                model_id: Some("qwen2.5:7b".into()),
                endpoint_url: None,
                skill_ids: vec!["workspace:reviewer".into()],
                computer_mode: "sandbox".into(),
                routine_enabled: true,
                routine_interval_mins: 15,
                routine_prompt: "Check workspace notes.".into(),
                last_routine_at: None,
            },
        )
        .unwrap();
        assert_eq!(saved.id, "pochi-helper");
        assert!(agent_workspace_dir(&ws, &saved.id).unwrap().exists());
        assert_eq!(list_agents(&ws).unwrap().len(), 1);
        delete_agent(&ws, &saved.id).unwrap();
        assert!(list_agents(&ws).unwrap().is_empty());
        let _ = fs::remove_dir_all(&ws);
    }

    #[test]
    fn enforces_isolated_workspace_and_blocks_host_pc_in_sandbox_mode() {
        let ws = temp_workspace("sandbox");
        let agent = save_agent(
            &ws,
            AgentProfile {
                id: "sandboxed".into(),
                name: "Sandboxed".into(),
                role: "".into(),
                system_prompt: "".into(),
                avatar_kind: "mochi".into(),
                avatar_url: None,
                provider: None,
                model_id: None,
                endpoint_url: None,
                skill_ids: vec![],
                computer_mode: "sandbox".into(),
                routine_enabled: false,
                routine_interval_mins: 30,
                routine_prompt: "".into(),
                last_routine_at: None,
            },
        )
        .unwrap();

        let written = execute_agent_tool(
            &ws,
            &agent.id,
            "workspace_write",
            &json!({ "path": "notes/todo.md", "content": "Hello 24/7" }),
        )
        .unwrap();
        assert_eq!(written["written"], "notes/todo.md");

        let read = execute_agent_tool(
            &ws,
            &agent.id,
            "workspace_read",
            &json!({ "path": "notes/todo.md" }),
        )
        .unwrap();
        assert_eq!(read["content"], "Hello 24/7");

        execute_agent_tool(
            &ws,
            &agent.id,
            "workspace_delete",
            &json!({ "path": "notes/todo.md" }),
        )
        .unwrap();
        assert!(execute_agent_tool(
            &ws,
            &agent.id,
            "workspace_read",
            &json!({ "path": "notes/todo.md" }),
        )
        .is_err());

        assert!(execute_agent_tool(
            &ws,
            &agent.id,
            "workspace_read",
            &json!({ "path": "../../secret.txt" }),
        )
        .is_err());

        assert!(execute_agent_tool(
            &ws,
            &agent.id,
            "pc_run_command",
            &json!({ "command": "echo hi" }),
        )
        .is_err());

        let _ = fs::remove_dir_all(&ws);
    }

    #[test]
    fn persists_and_recalls_cross_session_memories() {
        let ws = temp_workspace("memory");
        let e1 = save_memory(
            &ws,
            "pochi",
            "s1",
            "Latency Audit",
            "compaction",
            "User prefers dark theme and Qwen2.5 7B quantized Q4_K_M on RTX GPU.",
            vec!["gpu".into(), "qwen".into()],
        )
        .unwrap();
        let _e2 = save_memory(
            &ws,
            "pochi",
            "s2",
            "Grocery List",
            "fact",
            "Buy oat milk and coffee beans.",
            vec!["shopping".into()],
        )
        .unwrap();

        let recalled = recall_memories(&ws, Some("pochi"), "which qwen gpu quantization?", 5).unwrap();
        assert_eq!(recalled[0].id, e1.id);
        delete_memory(&ws, &e1.id).unwrap();
        assert_eq!(list_memories(&ws, Some("pochi")).unwrap().len(), 1);
        let _ = fs::remove_dir_all(&ws);
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OllamaModelInfo {
    pub name: String,
    pub size_gb: f64,
    pub parameter_size: String,
    pub quantization: String,
    #[serde(default = "default_lang_scope")]
    pub language_scope: String,
}

fn default_lang_scope() -> String {
    "unknown".to_string()
}

pub fn infer_language_scope(model_name: &str, family: &str) -> String {
    let hay = format!("{} {}", model_name, family).to_ascii_lowercase();
    if hay.contains("-en")
        || hay.contains("_en")
        || hay.contains("english")
        || hay.contains("tinyllama")
        || hay.contains("phi-2")
        || hay.contains("phi2")
        || hay.contains("codellama")
        || hay.contains("stablelm-zephyr")
    {
        return "english".to_string();
    }
    const MULTI: &[&str] = &[
        "qwen", "llama3", "llama-3", "llama4", "gemma", "mistral", "mixtral", "aya",
        "command-r", "deepseek", "glm", "yi-", "internlm", "lfm", "eurollm", "multilingual",
        "bilingual", "nemo", "minimax", "kimi",
    ];
    if MULTI.iter().any(|k| hay.contains(k)) {
        return "multilingual".to_string();
    }
    "unknown".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiveProvidersStatus {
    pub sytra_running: bool,
    pub sytra_models: Vec<String>,
    pub ollama_running: bool,
    pub ollama_installed: bool,
    pub ollama_models: Vec<OllamaModelInfo>,
    pub lmstudio_running: bool,
    pub lmstudio_models: Vec<String>,
}

fn find_ollama_binary() -> Option<PathBuf> {
    if let Ok(status) = {
        let mut c = Command::new("ollama");
        apply_hidden_window(&mut c);
        c.arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status()
    } {
        if status.success() {
            return Some(PathBuf::from("ollama"));
        }
    }
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let candidate = PathBuf::from(local).join("Programs").join("Ollama").join("ollama.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn parse_size_gb(size_str: &str) -> f64 {
    let lower = size_str.trim().to_ascii_lowercase();
    if let Some(num) = lower.strip_suffix("gb") {
        if let Ok(v) = num.trim().parse::<f64>() {
            return (v * 100.0).round() / 100.0;
        }
    }
    if let Some(num) = lower.strip_suffix("mb") {
        if let Ok(v) = num.trim().parse::<f64>() {
            return (v / 1024.0 * 100.0).round() / 100.0;
        }
    }
    0.0
}

fn discover_ollama_cli_models(bin: &Path) -> Vec<OllamaModelInfo> {
    let mut cmd = Command::new(bin);
    apply_hidden_window(&mut cmd);
    let Ok(out) = cmd.arg("list").output() else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut items = Vec::new();
    for line in text.lines().skip(1) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let name = parts[0].to_string();
        let size_gb = parts
            .get(2)
            .zip(parts.get(3))
            .map(|(n, u)| parse_size_gb(&format!("{n}{u}")))
            .unwrap_or(0.0);
        let lang_scope = infer_language_scope(&name, "");
        items.push(OllamaModelInfo {
            name,
            size_gb,
            parameter_size: String::new(),
            quantization: String::new(),
            language_scope: lang_scope,
        });
    }
    items
}

fn extract_openai_model_ids(client: &reqwest::blocking::Client, url: &str) -> Option<Vec<String>> {
    let resp = client.get(url).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let val: Value = resp.json().ok()?;
    let ids = val
        .get("data")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.get("id").and_then(|id| id.as_str()).map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(ids)
}

pub fn probe_chat_endpoints() -> LiveProvidersStatus {
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_millis(800))
        .timeout(Duration::from_millis(2000))
        .build();

    let Ok(client) = client else {
        return LiveProvidersStatus {
            sytra_running: false,
            sytra_models: Vec::new(),
            ollama_running: false,
            ollama_installed: false,
            ollama_models: Vec::new(),
            lmstudio_running: false,
            lmstudio_models: Vec::new(),
        };
    };

    let sytra_models = extract_openai_model_ids(&client, "http://127.0.0.1:8080/v1/models");
    let lmstudio_models = extract_openai_model_ids(&client, "http://127.0.0.1:1234/v1/models");

    let mut ollama_running = false;
    let mut ollama_models = Vec::new();
    if let Ok(resp) = client.get("http://127.0.0.1:11434/api/tags").send() {
        if resp.status().is_success() {
            ollama_running = true;
            if let Ok(val) = resp.json::<Value>() {
                if let Some(arr) = val.get("models").and_then(|v| v.as_array()) {
                    for m in arr {
                        let Some(name) = m.get("name").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let bytes = m.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
                        let details = m.get("details");
                        let family = details
                            .and_then(|d| d.get("family"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        ollama_models.push(OllamaModelInfo {
                            name: name.to_string(),
                            size_gb: ((bytes as f64) / (1024.0 * 1024.0 * 1024.0) * 100.0).round()
                                / 100.0,
                            parameter_size: details
                                .and_then(|d| d.get("parameter_size"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            quantization: details
                                .and_then(|d| d.get("quantization_level"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            language_scope: infer_language_scope(name, family),
                        });
                    }
                }
            }
        }
    }

    let ollama_bin = find_ollama_binary();
    let ollama_installed = ollama_running || ollama_bin.is_some();
    if !ollama_running {
        if let Some(bin) = ollama_bin {
            ollama_models = discover_ollama_cli_models(&bin);
        }
    }

    LiveProvidersStatus {
        sytra_running: sytra_models.is_some(),
        sytra_models: sytra_models.unwrap_or_default(),
        ollama_running,
        ollama_installed,
        ollama_models,
        lmstudio_running: lmstudio_models.is_some(),
        lmstudio_models: lmstudio_models.unwrap_or_default(),
    }
}

pub fn start_ollama_serve() -> Result<bool, String> {
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .timeout(Duration::from_millis(400))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;
    if client
        .get("http://127.0.0.1:11434/api/tags")
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
    {
        return Ok(true);
    }
    let bin = find_ollama_binary().ok_or_else(|| "Ollama is not installed on this PC.".to_string())?;
    let mut cmd = Command::new(bin);
    apply_hidden_window(&mut cmd);
    cmd.arg("serve")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start Ollama: {e}"))?;

    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if client
            .get("http://127.0.0.1:11434/api/tags")
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return Ok(true);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err("Timed out waiting for Ollama daemon on port 11434.".into())
}

pub fn ollama_model_context(base_url: &str, model: &str) -> Option<usize> {
    let clean_base = base_url.trim_end_matches('/');
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .timeout(Duration::from_millis(2500))
        .build()
        .ok()?;
    let resp = client
        .post(format!("{clean_base}/api/show"))
        .json(&json!({ "name": model }))
        .send()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let val: Value = resp.json().ok()?;
    if let Some(params) = val.get("parameters").and_then(|v| v.as_str()) {
        for line in params.lines() {
            let mut parts = line.split_whitespace();
            if parts.next() == Some("num_ctx") {
                if let Some(n) = parts.next().and_then(|s| s.parse::<usize>().ok()) {
                    if n > 0 {
                        return Some(n);
                    }
                }
            }
        }
    }
    if let Some(obj) = val.get("model_info").and_then(|v| v.as_object()) {
        for (k, v) in obj {
            if k.ends_with(".context_length") {
                if let Some(n) = v.as_u64() {
                    if n > 0 {
                        return Some(n as usize);
                    }
                }
            }
        }
    }
    None
}

pub fn stream_http_relay(
    url: &str,
    body: &Value,
    mut on_line: impl FnMut(&str) -> bool,
) -> Result<(), String> {
    let trimmed = url.trim();
    if !(trimmed.starts_with("http://") || trimmed.starts_with("https://")) {
        return Err("Relay endpoint must start with http:// or https://".into());
    }
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("Relay HTTP client error: {e}"))?;
    let resp = client
        .post(trimmed)
        .header("Accept", "text/event-stream, application/json")
        .json(body)
        .send()
        .map_err(|e| format!("Cannot reach {trimmed}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let code = status.as_u16();
        let text = resp.text().unwrap_or_default();
        let snippet: String = text.chars().take(400).collect();
        return Err(format!("HTTP {code}: {snippet}"));
    }
    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if !on_line(&line) {
                    break;
                }
            }
            Err(e) => return Err(format!("Stream interrupted: {e}")),
        }
    }
    Ok(())
}


/// Unload an active model from GPU VRAM / system RAM so the user can reclaim resources
/// during inactivity or when working outside Chat.
pub fn unload_chat_model(
    chat: &crate::serve::ChatServer,
    provider: &str,
    base_url: &str,
    model: &str,
) -> Result<String, String> {
    let prov = provider.trim().to_ascii_lowercase();
    let clean_base = base_url.trim().trim_end_matches('/').trim_end_matches("/v1");

    if prov == "sytra" || clean_base.contains(":8080") {
        let was_running = chat.is_running();
        chat.stop()?;
        return Ok(if was_running {
            "Stopped Sytra model server and released GPU/RAM.".to_string()
        } else {
            "Sytra model server was already idle.".to_string()
        });
    }

    if prov == "ollama" || clean_base.contains(":11434") {
        let target_base = if clean_base.is_empty() {
            "http://127.0.0.1:11434"
        } else {
            clean_base
        };
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(3))
            .build()
            .map_err(|e| format!("HTTP client error: {e}"))?;

        let mut models_to_unload: Vec<String> = Vec::new();
        if !model.trim().is_empty() {
            models_to_unload.push(model.trim().to_string());
        } else if let Ok(resp) = client.get(format!("{target_base}/api/ps")).send() {
            if let Ok(val) = resp.json::<Value>() {
                if let Some(arr) = val.get("models").and_then(|v| v.as_array()) {
                    for m in arr {
                        if let Some(n) = m.get("name").and_then(|v| v.as_str()) {
                            models_to_unload.push(n.to_string());
                        }
                    }
                }
            }
        }
        if models_to_unload.is_empty() {
            return Ok("No loaded Ollama model found in GPU memory.".to_string());
        }
        for m_name in &models_to_unload {
            let _ = client
                .post(format!("{target_base}/api/generate"))
                .json(&json!({ "model": m_name, "prompt": "", "keep_alive": 0, "stream": false }))
                .send();
        }
        return Ok(format!(
            "Unloaded Ollama model(s) from GPU/RAM: {}.",
            models_to_unload.join(", ")
        ));
    }

    if prov == "lmstudio" || clean_base.contains(":1234") {
        let mut cmd = Command::new("lms");
        apply_hidden_window(&mut cmd);
        if let Ok(out) = cmd.args(["unload", "--all"]).output() {
            if out.status.success() {
                return Ok("Unloaded LM Studio model(s) from GPU/RAM.".to_string());
            }
        }
        return Ok("Requested idle state; unload inside LM Studio if lms CLI is not installed.".to_string());
    }

    Ok("Cleared active model focus.".to_string())
}

#[cfg(test)]
mod language_tests {
    use super::*;

    #[test]
    fn infers_multilingual_vs_english_focused_models() {
        assert_eq!(infer_language_scope("qwen2.5:7b", "qwen2"), "multilingual");
        assert_eq!(infer_language_scope("lfm3-1.7b:latest", ""), "multilingual");
        assert_eq!(infer_language_scope("phi-2:latest", ""), "english");
        assert_eq!(infer_language_scope("tinyllama-1.1b", ""), "english");
        assert_eq!(infer_language_scope("custom-alpha-v1", ""), "unknown");
    }
}
