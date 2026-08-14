use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::json;

fn fixture() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("sytra-engine-cli-{nonce}"));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("experts.bin"), b"EXPERT").unwrap();
    fs::write(
        root.join("config.json"),
        serde_json::to_vec_pretty(&json!({
            "model_type": "kimi_k3",
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "moe_intermediate_size": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2
        }))
        .unwrap(),
    )
    .unwrap();
    let experts: Vec<_> = (0..8)
        .map(|expert| {
            json!({
                "layer": 0,
                "expert": expert,
                "segments": [{
                    "tensor": format!("experts.{expert}.gate_proj"),
                    "shard": "experts.bin",
                    "offset": 0,
                    "length": 6
                }]
            })
        })
        .collect();
    fs::write(
        root.join(".sytra-runtime.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "architecture": {
                "adapter": "sytra-kimi-k3",
                "model_type": "kimi_k3",
                "attention": "mla",
                "router": "top_k_weighted",
                "expert_format": "mxfp4",
                "family": "kimi_k3",
                "expert_layout": "discrete",
                "activation": "silu",
                "hidden_size": 8,
                "expert_intermediate_size": 4,
                "moe_layers": [0],
                "num_layers": 1,
                "experts_per_layer": 8,
                "experts_per_token": 2,
                "forward_verified": true
            },
            "dense_bytes": 10,
            "storage": {
                "contiguous_experts": true,
                "experts": experts
            }
        }))
        .unwrap(),
    )
    .unwrap();
    root
}

#[test]
fn doctor_does_not_trust_forward_verified_from_model_metadata() {
    let root = fixture();
    let output = Command::new(env!("CARGO_BIN_EXE_sytra-engine"))
        .args(["doctor", "--model"])
        .arg(&root)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("\"ready\":false"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot serve tokens"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn doctor_rejects_manifest_that_disagrees_with_config() {
    let root = fixture();
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "kimi_k3",
            "num_hidden_layers": 1,
            "hidden_size": 9,
            "moe_intermediate_size": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2
        }))
        .unwrap(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sytra-engine"))
        .args(["doctor", "--model"])
        .arg(&root)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("hidden_size"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn plan_reports_storage_residency_without_loading_the_model() {
    let root = fixture();
    let output = Command::new(env!("CARGO_BIN_EXE_sytra-engine"))
        .args([
            "plan",
            "--model",
            root.to_str().unwrap(),
            "--ram-expert-mb",
            "0",
            "--accelerator-expert-mb",
            "0",
            "--storage-bandwidth-mbps",
            "1",
            "--target-tps",
            "1000000",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["storage_expert_bytes"], 48);
    let ram_cache = plan["memory_envelope"]["ram_cache_bytes"].as_u64().unwrap();
    let planned_ram_cache = plan["ram_dense_budget_bytes"].as_u64().unwrap()
        + plan["ram_expert_budget_bytes"].as_u64().unwrap();
    assert!(planned_ram_cache <= ram_cache);
    assert_eq!(plan["io_performance"]["target_io_feasible"], false);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn benchmark_refuses_storage_only_or_uncompiled_forward_contracts() {
    let root = fixture();
    let output = Command::new(env!("CARGO_BIN_EXE_sytra-engine"))
        .args(["benchmark", "--model"])
        .arg(&root)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no token-verified forward kernel"));
    fs::remove_dir_all(root).unwrap();
}
