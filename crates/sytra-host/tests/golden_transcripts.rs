use sytra_contracts::telemetry::{parse_line, EventName, TelemetryLine};

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join(name)
}

fn assert_transcript_has_exactly_one_terminal_event(name: &str) {
    let raw = std::fs::read_to_string(fixture_path(name)).unwrap();
    let mut terminal_count = 0;
    let mut saw_starting = false;

    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        match parse_line(line) {
            TelemetryLine::Log { .. } => panic!("line unexpectedly fell back to Log: {line}"),
            TelemetryLine::Event { event, .. } => {
                let event_name: EventName =
                    serde_json::from_value(serde_json::Value::String(event.clone()))
                        .unwrap_or_else(|_| {
                            panic!("unknown event name in closed vocabulary: {event}")
                        });
                if event_name == EventName::Starting {
                    saw_starting = true;
                }
                if event_name.is_terminal() {
                    terminal_count += 1;
                }
            }
            TelemetryLine::Metric { .. } => {}
        }
    }

    assert!(saw_starting, "{name} must start with a `starting` event");
    assert_eq!(
        terminal_count, 1,
        "{name} must contain exactly one terminal done/error event"
    );
}

#[test]
fn train_golden_transcript_is_well_formed() {
    assert_transcript_has_exactly_one_terminal_event("run.golden.transcript.jsonl");
}

#[test]
fn merge_golden_transcript_is_well_formed() {
    assert_transcript_has_exactly_one_terminal_event("merge.golden.transcript.jsonl");
}

#[tokio::test]
async fn local_datasource_materializes_golden_run_yaml_dataset() {
    use sytra_contracts::run_config::RunConfig;
    use sytra_host::datasource::local::LocalDataSource;
    use sytra_host::{DataSource, DatasetSpec, SourceKind};

    let raw = std::fs::read_to_string(fixture_path("run.golden.yaml")).unwrap();
    let config = RunConfig::from_yaml_str(&raw).unwrap();

    let local_params = match &config.data {
        sytra_contracts::run_config::DataSpec::Local { local, .. } => local.clone(),
        other => panic!("expected local data source in fixture, got {other:?}"),
    };

    // Resolve the fixture-relative path against the repo root, mirroring
    // how the host would resolve a path written relative to the yaml.
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let mut local_params = local_params;
    local_params.path = repo_root.join(&local_params.path);

    let spec = DatasetSpec {
        source: SourceKind::Local,
        train_mode: config.train_mode,
        params: serde_json::to_value(&local_params).unwrap(),
    };

    let source = LocalDataSource;
    source
        .validate(&spec)
        .expect("golden dataset must validate");

    let out_dir = std::env::temp_dir().join("sytra-phase0-test-materialize");
    let materialized = source
        .materialize(&spec, &out_dir)
        .await
        .expect("materialize must succeed");

    assert_eq!(materialized.row_count, 2);
    assert!(materialized.fingerprint.starts_with("sha256:"));
    assert!(materialized.jsonl_path.exists());

    std::fs::remove_dir_all(&out_dir).ok();
}

#[tokio::test]
async fn local_datasource_preserves_conversational_messages() {
    use std::collections::BTreeMap;

    use sytra_contracts::run_config::TrainMode;
    use sytra_host::datasource::local::LocalDataSource;
    use sytra_host::{DataSource, DatasetSpec, SourceKind};

    let temp_root = std::env::temp_dir().join(format!(
        "sytra-conversational-materialize-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp_root).unwrap();
    let source_path = temp_root.join("source.jsonl");
    std::fs::write(
        &source_path,
        r#"{"instruction":"Question","output":"Answer","messages":[{"role":"system","content":"Be accurate"},{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}"#,
    )
    .unwrap();

    let spec = DatasetSpec {
        source: SourceKind::Local,
        train_mode: TrainMode::Sft,
        params: serde_json::json!({
            "path": source_path,
            "format": "jsonl",
            "mapping": BTreeMap::from([
                ("prompt".to_string(), "instruction".to_string()),
                ("completion".to_string(), "output".to_string()),
            ]),
        }),
    };

    let materialized = LocalDataSource
        .materialize(&spec, &temp_root.join("out"))
        .await
        .unwrap();
    let row: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(materialized.jsonl_path).unwrap()).unwrap();

    assert_eq!(row["prompt"], "Question");
    assert_eq!(row["completion"], "Answer");
    assert_eq!(row["messages"][0]["role"], "system");
    assert_eq!(row["messages"][0]["content"], "Be accurate");

    std::fs::remove_dir_all(&temp_root).ok();
}
