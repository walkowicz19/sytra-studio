use sytra_contracts::{
    guider::{Guider, HardwareCapabilities},
    merge_config::{MergeMethod, Verdict},
    operation::{OpRecord, OpStatus, TrainSpec},
    run_config::{
        AdapterConfig, AdapterType, BackendConfig, BackendKind, DataSpec, HfParams, OptimConfig,
        RunConfig, RunOutput, Schedule, TrainMode, TrainParams,
    },
};
use sytra_host::{BackendResolver, ResourceGuard, RunArchive};
use uuid::Uuid;

#[test]
fn test_guider_and_resource_guard() {
    let guider = Guider::new();
    let catalog = guider.catalog();
    assert!(!catalog.is_empty(), "model catalog should not be empty");

    // Resolve models
    let mistral = guider.resolve_model("mistralai/Mistral-7B-v0.1").unwrap();
    assert_eq!(mistral.architecture, "MistralForCausalLM");

    // Recommend train recipes
    let hw = HardwareCapabilities {
        accelerator: "cuda".to_string(),
        total_vram_mb: 16192, // 16GB VRAM
        total_ram_mb: 32768,  // 32GB RAM
    };
    let recipes = guider.recommend(&hw);
    assert!(!recipes.is_empty(), "should recommend recipes");

    // Test merge check (green case)
    let comp_green = guider.merge_check(
        &[
            "org/knowledge-ft".to_string(),
            "org/toolcalling-ft".to_string(),
        ],
        MergeMethod::DareTies,
    );
    assert_eq!(comp_green.verdict, Verdict::Green);

    // Test merge check (amber tokenizer mismatch - let's simulate by matching Mistral with Llama 3)
    let comp_amber = guider.merge_check(
        &[
            "mistralai/Mistral-7B-v0.1".to_string(),
            "mlx-community/Meta-Llama-3-8B-Instruct-4bit".to_string(),
        ],
        MergeMethod::Moe,
    );
    assert_eq!(comp_amber.verdict, Verdict::Amber);

    // Test ResourceGuard VRAM training calculations
    let guard = ResourceGuard::new(15000, 32768, 102400);

    // Construct dummy TrainSpec
    let train_spec = TrainSpec {
        config: RunConfig {
            version: 1,
            run_id: None,
            train_mode: TrainMode::Sft,
            model: "mistralai/Mistral-7B-v0.1".to_string(),
            backend: BackendConfig {
                kind: BackendKind::Cpu,
                judge_model: None,
            },
            data: DataSpec::Hf {
                jsonl_path: None,
                fingerprint: None,
                hf: HfParams {
                    repo_id: "org/data".to_string(),
                    split: "train".to_string(),
                    revision: None,
                    config: None,
                },
            },
            adapter: AdapterConfig {
                kind: AdapterType::Lora,
                rank: 16,
                alpha: 32,
                dropout: 0.05,
                target_modules: vec![],
                quant_bits: None,
            },
            optim: OptimConfig {
                learning_rate: 2e-4,
                schedule: Schedule::Cosine,
                warmup_steps: 20,
                weight_decay: 0.0,
                grad_accumulation_steps: 8,
            },
            train: TrainParams {
                max_steps: Some(10),
                epochs: None,
                batch_size: 2,
                max_seq_len: 2048,
                save_every: 2,
                packing: false,
            },
            algo: Default::default(),
            output: RunOutput {
                adapter_path: "out/adapter".into(),
                resume_from: None,
            },
        },
        config_path: "run.yaml".into(),
    };

    let check_res = guard.check_train(mistral, &train_spec);
    // Since VRAM estimate of Mistral 7B on 16-bit LoRA is ~16.5GB, it should fail on 16GB VRAM total
    assert!(check_res.is_err(), "should overflow total VRAM limit");
}

#[test]
fn test_run_archive() {
    let temp_dir = std::env::temp_dir().join(format!("sytra-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&temp_dir).unwrap();

    let archive = RunArchive::new(&temp_dir);
    let op_id = Uuid::new_v4();

    let record = OpRecord {
        op_id,
        kind: "train".to_string(),
        config: serde_json::Value::Null,
        artifact_path: "out/adapter".into(),
        status: OpStatus::Done,
        provenance: None,
    };

    archive.store(&record).unwrap();

    let loaded = archive.load(op_id).unwrap();
    assert_eq!(loaded.op_id, op_id);
    assert_eq!(loaded.kind, "train");

    let list = archive.list().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].op_id, op_id);

    archive.delete(op_id).unwrap();
    assert!(archive.load(op_id).is_err());

    std::fs::remove_dir_all(&temp_dir).unwrap();
}

#[test]
fn test_backend_resolver() {
    let resolved = BackendResolver::resolve(BackendKind::Auto);
    // On any host, Auto should resolve to Cuda, Mps, or Cpu
    assert!(
        resolved == BackendKind::Cuda
            || resolved == BackendKind::Mps
            || resolved == BackendKind::Cpu
    );
}
