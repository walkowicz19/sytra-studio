//! Explicitly selected local chat skills. Paths never come from the frontend.
use serde::Serialize;
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_SKILL_BYTES: u64 = 128 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct ChatSkill {
    pub id: String,
    pub name: String,
    pub scope: String,
}

fn roots(workspace: &Path) -> Vec<(&'static str, PathBuf)> {
    let mut result = vec![("workspace", workspace.join(".agents").join("skills"))];
    if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
        let home = PathBuf::from(home);
        result.push(("user", home.join(".sytra").join("skills")));
        result.push(("user-agents", home.join(".agents").join("skills")));
        result.push(("user-codex", home.join(".codex").join("skills")));
    }
    result
}

fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 100 && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

fn collect(roots: &[(impl AsRef<str>, PathBuf)]) -> Vec<(ChatSkill, PathBuf)> {
    let mut found = Vec::new();
    for (scope, root) in roots {
        let Ok(canonical_root) = root.canonicalize() else { continue };
        let Ok(entries) = std::fs::read_dir(&canonical_root) else { continue };
        for entry in entries.flatten() {
            let Some(name) = entry.file_name().to_str().map(str::to_string) else { continue };
            if !valid_name(&name) { continue; }
            let Ok(file) = entry.path().join("SKILL.md").canonicalize() else { continue };
            if !file.starts_with(&canonical_root) || !file.is_file() { continue; }
            found.push((ChatSkill { id: format!("{}:{name}", scope.as_ref()), name, scope: scope.as_ref().to_string() }, file));
        }
    }
    found.sort_by(|a, b| a.0.id.cmp(&b.0.id));
    found
}

pub fn list(workspace: &Path) -> Vec<ChatSkill> {
    collect(&roots(workspace)).into_iter().map(|(skill, _)| skill).collect()
}

pub fn read(workspace: &Path, id: &str) -> Result<String, String> {
    read_from(&roots(workspace), id)
}

fn atomic_write_skill(dir: &Path, content: &[u8]) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("Cannot create skill directory: {e}"))?;
    let dest = dir.join("SKILL.md");
    let tmp = dir.join(format!(".SKILL.md.tmp-{}", std::process::id()));
    std::fs::write(&tmp, content).map_err(|e| format!("Cannot write skill file: {e}"))?;
    if let Err(err) = std::fs::rename(&tmp, &dest) {
        if err.kind() == std::io::ErrorKind::AlreadyExists || cfg!(windows) {
            let _ = std::fs::remove_file(&dest);
            std::fs::rename(&tmp, &dest).map_err(|e| format!("Cannot finalize skill file: {e}"))?;
        } else {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("Cannot finalize skill file: {err}"));
        }
    }
    Ok(())
}

pub fn normalize_skill_name(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if valid_name(trimmed) {
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
    if !valid_name(&cleaned) {
        return Err("Skill name must contain letters, digits, '-' or '_' (max 100 chars).".into());
    }
    Ok(cleaned)
}

pub fn install(workspace: &Path, name: &str, content: &str) -> Result<ChatSkill, String> {
    let clean_name = normalize_skill_name(name)?;
    if content.trim().is_empty() {
        return Err("SKILL.md content cannot be empty.".into());
    }
    let bytes = content.as_bytes();
    if bytes.len() as u64 > MAX_SKILL_BYTES {
        return Err("Skill exceeds 128 KiB".into());
    }
    let base = workspace.join(".agents").join("skills");
    std::fs::create_dir_all(&base).map_err(|e| format!("Cannot prepare skills root: {e}"))?;
    let canonical_root = base
        .canonicalize()
        .map_err(|e| format!("Cannot resolve skills root: {e}"))?;
    let skill_dir = canonical_root.join(&clean_name);
    atomic_write_skill(&skill_dir, bytes)?;
    Ok(ChatSkill {
        id: format!("workspace:{clean_name}"),
        name: clean_name,
        scope: "workspace".into(),
    })
}

pub fn import_file(
    workspace: &Path,
    source_path: &Path,
    custom_name: Option<&str>,
) -> Result<ChatSkill, String> {
    let canonical_src = source_path
        .canonicalize()
        .map_err(|e| format!("Cannot open skill file: {e}"))?;
    if !canonical_src.is_file() {
        return Err("Selected SKILL.md path is not a file.".into());
    }
    let file =
        std::fs::File::open(&canonical_src).map_err(|e| format!("Cannot read skill file: {e}"))?;
    let mut bytes = Vec::new();
    file.take(MAX_SKILL_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Cannot read skill file: {e}"))?;
    if bytes.len() as u64 > MAX_SKILL_BYTES {
        return Err("Skill exceeds 128 KiB".into());
    }
    let content = String::from_utf8(bytes).map_err(|_| "SKILL.md must be valid UTF-8".to_string())?;
    let inferred = if let Some(explicit) = custom_name.map(str::trim).filter(|s| !s.is_empty()) {
        explicit.to_string()
    } else if canonical_src
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.eq_ignore_ascii_case("SKILL.md"))
    {
        canonical_src
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("custom-skill")
            .to_string()
    } else {
        canonical_src
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("custom-skill")
            .to_string()
    };
    install(workspace, &inferred, &content)
}

pub fn remove(workspace: &Path, id: &str) -> Result<(), String> {
    let Some(name) = id.strip_prefix("workspace:") else {
        return Err("Only workspace-installed skills can be removed here.".into());
    };
    if !valid_name(name) {
        return Err("Invalid skill identifier.".into());
    }
    let base = workspace.join(".agents").join("skills");
    let Ok(canonical_root) = base.canonicalize() else {
        return Ok(());
    };
    let target = canonical_root.join(name);
    let Ok(canonical_target) = target.canonicalize() else {
        return Ok(());
    };
    if !canonical_target.starts_with(&canonical_root) || canonical_target == canonical_root {
        return Err("Skill path escapes skills directory.".into());
    }
    std::fs::remove_dir_all(&canonical_target).map_err(|e| format!("Cannot remove skill: {e}"))
}

fn read_from(roots: &[(impl AsRef<str>, PathBuf)], id: &str) -> Result<String, String> {
    let path = collect(roots).into_iter().find(|(skill, _)| skill.id == id)
        .map(|(_, path)| path).ok_or("Skill is no longer available")?;
    let file = std::fs::File::open(path).map_err(|e| format!("Cannot read skill: {e}"))?;
    let mut bytes = Vec::new();
    file.take(MAX_SKILL_BYTES + 1).read_to_end(&mut bytes).map_err(|e| format!("Cannot read skill: {e}"))?;
    if bytes.len() as u64 > MAX_SKILL_BYTES { return Err("Skill exceeds 128 KiB".into()); }
    String::from_utf8(bytes).map_err(|_| "Skill must be UTF-8".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_only_discovered_skills_and_enforces_size() {
        let root = std::env::temp_dir().join(format!("sytra-chat-skills-{}", uuid::Uuid::new_v4()));
        let skill = root.join("explain-code");
        std::fs::create_dir_all(&skill).unwrap();
        std::fs::write(skill.join("SKILL.md"), "Explain the code.").unwrap();
        let roots = [("workspace", root.clone())];
        assert_eq!(collect(&roots)[0].0.id, "workspace:explain-code");
        assert_eq!(read_from(&roots, "workspace:explain-code").unwrap(), "Explain the code.");
        assert!(read_from(&roots, "workspace:../secret").is_err());
        std::fs::write(skill.join("SKILL.md"), vec![b'x'; MAX_SKILL_BYTES as usize + 1]).unwrap();
        assert!(read_from(&roots, "workspace:explain-code").unwrap_err().contains("128 KiB"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn installs_imports_and_removes_workspace_skill_safely() {
        let ws = std::env::temp_dir().join(format!("sytra-ws-skills-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&ws).unwrap();

        let installed = install(&ws, "My Custom Skill", "# Role\nBe helpful.").unwrap();
        assert_eq!(installed.id, "workspace:my-custom-skill");
        assert_eq!(read(&ws, &installed.id).unwrap(), "# Role\nBe helpful.");

        let src_dir = ws.join("source-pack").join("reviewer-skill");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src_file = src_dir.join("SKILL.md");
        std::fs::write(&src_file, "Review diffs carefully.").unwrap();
        let imported = import_file(&ws, &src_file, None).unwrap();
        assert_eq!(imported.id, "workspace:reviewer-skill");

        assert!(install(&ws, "../escape", "bad").is_err() || list(&ws).iter().all(|s| !s.name.contains("..")));
        remove(&ws, &imported.id).unwrap();
        assert!(read(&ws, &imported.id).is_err());

        let _ = std::fs::remove_dir_all(ws);
    }
}
