use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

fn should_skip_dir(name: &str) -> bool {
    matches!(
        name,
        "__pycache__"
            | ".pytest_cache"
            | ".sytra-envs"
            | ".hf-cache"
            | ".venv"
            | ".ruff_cache"
            | ".mypy_cache"
    ) || name.starts_with(".test")
        || name.starts_with(".tmp")
        || name.ends_with(".egg-info")
}

fn should_skip_file(name: &str) -> bool {
    name.ends_with(".pyc") || name.ends_with(".pyo") || name.ends_with(".pyd")
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();

    for path in paths {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if path.is_dir() {
            if !should_skip_dir(name) {
                println!("cargo:rerun-if-changed={}", path.display());
                collect_files(root, &path, out);
            }
        } else if path.is_file() && !should_skip_file(name) {
            if let Ok(rel) = path.strip_prefix(root) {
                let rel_unix = rel.to_string_lossy().replace('\\', "/");
                println!("cargo:rerun-if-changed={}", path.display());
                out.push((rel_unix, path));
            }
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let generated_path = out_dir.join("embedded_runner.rs");
    let mut out = File::create(&generated_path).expect("create embedded_runner.rs");

    let runner_root = manifest_dir.join("..").join("..").join("runner");
    let mut files = Vec::new();
    if runner_root.join("sytra_runner").is_dir() {
        let pyproject = runner_root.join("pyproject.toml");
        if pyproject.is_file() {
            println!("cargo:rerun-if-changed={}", pyproject.display());
            files.push(("pyproject.toml".to_string(), pyproject));
        }
        for sub in ["sytra_runner", "scripts"] {
            let dir = runner_root.join(sub);
            if dir.is_dir() {
                println!("cargo:rerun-if-changed={}", dir.display());
                collect_files(&runner_root, &dir, &mut files);
            }
        }
    }

    writeln!(out, "pub static EMBEDDED_RUNNER_FILES: &[(&str, &[u8])] = &[")
        .expect("write slice header");
    for (rel, abs) in files {
        let canonical = abs.canonicalize().unwrap_or(abs);
        let rel_esc = rel.escape_default();
        let abs_esc = canonical.to_string_lossy().into_owned();
        writeln!(
            out,
            "    (\"{rel_esc}\", include_bytes!(r#\"{abs_esc}\"#)),"
        )
        .expect("write embedded entry");
    }
    writeln!(out, "];").expect("write slice footer");
}
