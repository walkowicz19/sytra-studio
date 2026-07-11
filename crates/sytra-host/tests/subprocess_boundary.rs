//! Spawns the real Python runner as a subprocess and feeds its stdout
//! through the Rust telemetry parser — the actual headless-run proof the
//! Phase 0 freeze checklist asks for. Skips (rather than fails) if no
//! `python` is on PATH, since this crate doesn't own the Python toolchain.

use std::process::Command;

use sytra_contracts::telemetry::{parse_line, EventName, TelemetryLine};

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn run_module_and_parse(module: &str, config_rel_path: &str) -> Option<Vec<TelemetryLine>> {
    let root = repo_root();
    let output = Command::new("python")
        .arg("-m")
        .arg(module)
        .arg(config_rel_path)
        .current_dir(&root)
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(
        stdout
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(parse_line)
            .collect(),
    )
}

fn assert_well_formed_and_no_log_fallback(lines: &[TelemetryLine]) {
    assert!(!lines.is_empty(), "subprocess produced no telemetry lines");

    let mut terminal_count = 0;
    for line in lines {
        match line {
            TelemetryLine::Log { line: raw, .. } => {
                panic!("a real telemetry line fell back to Log (parser regression?): {raw}")
            }
            TelemetryLine::Event { event, .. } => {
                let event_name: EventName =
                    serde_json::from_value(serde_json::Value::String(event.clone()))
                        .unwrap_or_else(|_| panic!("unknown event name: {event}"));
                if event_name.is_terminal() {
                    terminal_count += 1;
                }
            }
            TelemetryLine::Metric { .. } => {}
        }
    }
    assert_eq!(terminal_count, 1, "expected exactly one terminal event");
}

#[test]
fn headless_train_subprocess_produces_valid_transcript() {
    let Some(lines) = run_module_and_parse("sytra_runner", "fixtures/run.golden.yaml") else {
        eprintln!("skipping: python not available on PATH");
        return;
    };
    assert_well_formed_and_no_log_fallback(&lines);
}

#[test]
fn headless_merge_subprocess_produces_valid_transcript() {
    let Some(lines) = run_module_and_parse("sytra_runner.merge", "fixtures/merge.golden.yaml")
    else {
        eprintln!("skipping: python not available on PATH");
        return;
    };
    assert_well_formed_and_no_log_fallback(&lines);
}
