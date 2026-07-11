use sytra_contracts::{MergeConfig, RunConfig};

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join(name)
}

#[test]
fn run_yaml_round_trips() {
    let raw = std::fs::read_to_string(fixture_path("run.golden.yaml")).unwrap();
    let config = RunConfig::from_yaml_str(&raw).expect("golden run.yaml must parse");
    assert_eq!(config.version, RunConfig::SUPPORTED_VERSION);

    let reserialized = config.to_yaml_string().unwrap();
    let reparsed = RunConfig::from_yaml_str(&reserialized).expect("reserialized yaml must parse");
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        serde_json::to_value(&reparsed).unwrap()
    );
}

#[test]
fn merge_yaml_round_trips() {
    let raw = std::fs::read_to_string(fixture_path("merge.golden.yaml")).unwrap();
    let config = MergeConfig::from_yaml_str(&raw).expect("golden merge.yaml must parse");
    assert_eq!(config.version, MergeConfig::SUPPORTED_VERSION);
    assert_eq!(config.models.len(), 2);

    let reserialized = config.to_yaml_string().unwrap();
    let reparsed = MergeConfig::from_yaml_str(&reserialized).expect("reserialized yaml must parse");
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        serde_json::to_value(&reparsed).unwrap()
    );
}
