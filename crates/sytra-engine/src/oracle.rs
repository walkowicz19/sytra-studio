//! Checkpoint-bound correctness gates for architecture runtimes.
//!
//! An oracle file is not a boolean attestation. It contains teacher-forced
//! predictions and selected reference logits which Sytra recomputes through
//! its own runtime before opening a serving socket. The suite is bound to the
//! immutable download revision, file blob identities, config, and runtime
//! index so model metadata alone cannot unlock serving.

use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    KimiOracleOutputs, KimiRuntime, KimiStepMetrics, MixtralRuntime, RuntimeError, RuntimeManifest,
};

pub const ORACLE_FILE: &str = ".sytra-oracle.json";
pub const ORACLE_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogitProbe {
    pub token: u32,
    pub expected: f32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleCase {
    pub name: String,
    pub input_tokens: Vec<u32>,
    pub teacher_forced_predictions: Vec<u32>,
    pub final_logit_probes: Vec<LogitProbe>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleSuite {
    pub schema_version: u32,
    pub adapter: String,
    pub model_fingerprint: String,
    pub reference_implementation: String,
    pub reference_revision: String,
    pub cases: Vec<OracleCase>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OracleCaseReport {
    pub name: String,
    pub positions: usize,
    pub logit_probes: usize,
    pub maximum_logit_error: f32,
    pub metrics: KimiStepMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct OracleReport {
    pub passed: bool,
    pub adapter: String,
    pub model_fingerprint: String,
    pub reference_implementation: String,
    pub reference_revision: String,
    pub cases: Vec<OracleCaseReport>,
}

#[derive(Debug, Error)]
pub enum OracleError {
    #[error("could not read oracle payload {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("oracle payload {path} is invalid JSON: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("oracle contract is invalid: {0}")]
    Contract(String),
    #[error("oracle model identity is invalid: {0}")]
    Identity(String),
    #[error("oracle case {case} failed: {reason}")]
    Mismatch { case: String, reason: String },
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}

impl OracleSuite {
    pub fn load(model_root: impl AsRef<Path>) -> Result<Self, OracleError> {
        let path = model_root.as_ref().join(ORACLE_FILE);
        let bytes = fs::read(&path).map_err(|source| OracleError::Read {
            path: path.clone(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| OracleError::Json { path, source })
    }

    pub fn validate(&self, adapter: &str, vocab_size: usize) -> Result<(), OracleError> {
        if self.schema_version != ORACLE_SCHEMA {
            return Err(OracleError::Contract(format!(
                "schema {} is unsupported",
                self.schema_version
            )));
        }
        if self.adapter != adapter || self.model_fingerprint.len() != 64 {
            return Err(OracleError::Contract(
                "adapter or SHA-256 model fingerprint is invalid".into(),
            ));
        }
        if !(self.reference_implementation.starts_with("transformers@")
            || self.reference_implementation.starts_with("vllm@"))
            || self.reference_revision.trim().len() < 7
        {
            return Err(OracleError::Contract(
                "reference implementation/version and source revision are required".into(),
            ));
        }
        if self.cases.len() < 2 {
            return Err(OracleError::Contract(
                "at least two independent oracle cases are required".into(),
            ));
        }
        let mut has_logits = false;
        let mut has_teacher_forcing = false;
        for case in &self.cases {
            if case.name.trim().is_empty() || case.input_tokens.is_empty() {
                return Err(OracleError::Contract(
                    "oracle cases need names and input tokens".into(),
                ));
            }
            if case.teacher_forced_predictions.len() != case.input_tokens.len() {
                return Err(OracleError::Contract(format!(
                    "case {} has {} inputs but {} teacher-forced predictions",
                    case.name,
                    case.input_tokens.len(),
                    case.teacher_forced_predictions.len()
                )));
            }
            has_teacher_forcing |= case.teacher_forced_predictions.len() >= 2;
            for token in case
                .input_tokens
                .iter()
                .chain(&case.teacher_forced_predictions)
            {
                if *token as usize >= vocab_size {
                    return Err(OracleError::Contract(format!(
                        "case {} contains token {} outside vocabulary {}",
                        case.name, token, vocab_size
                    )));
                }
            }
            for probe in &case.final_logit_probes {
                has_logits = true;
                if probe.token as usize >= vocab_size
                    || !probe.expected.is_finite()
                    || !probe.absolute_tolerance.is_finite()
                    || !probe.relative_tolerance.is_finite()
                    || !(0.0..=0.1).contains(&probe.absolute_tolerance)
                    || !(0.0..=0.05).contains(&probe.relative_tolerance)
                {
                    return Err(OracleError::Contract(format!(
                        "case {} has an invalid or excessively loose logit probe",
                        case.name
                    )));
                }
            }
        }
        if !has_logits || !has_teacher_forcing {
            return Err(OracleError::Contract(
                "suite must cover both reference logits and multi-position teacher forcing".into(),
            ));
        }
        Ok(())
    }
}

pub fn checkpoint_fingerprint(model_root: impl AsRef<Path>) -> Result<String, OracleError> {
    let root = model_root.as_ref();
    let download_path = root.join(".sytra-model.json");
    let download = read_json(&download_path)?;
    let revision = download
        .get("resolved_revision")
        .and_then(serde_json::Value::as_str)
        .filter(|value| value.len() >= 7)
        .ok_or_else(|| OracleError::Identity("immutable resolved_revision is missing".into()))?;
    let files = download
        .get("files")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| OracleError::Identity("download file identities are missing".into()))?;
    if files.is_empty() {
        return Err(OracleError::Identity(
            "download file identity list is empty".into(),
        ));
    }
    let mut identities = Vec::with_capacity(files.len());
    for file in files {
        let path = file
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| OracleError::Identity("download file path is missing".into()))?;
        let relative = Path::new(path);
        if relative.as_os_str().is_empty()
            || !relative
                .components()
                .all(|part| matches!(part, Component::Normal(_) | Component::CurDir))
        {
            return Err(OracleError::Identity(format!(
                "download file path {path:?} is unsafe"
            )));
        }
        let size = file
            .get("size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| OracleError::Identity(format!("download size is missing for {path}")))?;
        let blob = file
            .get("blob_id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                OracleError::Identity(format!("immutable blob identity is missing for {path}"))
            })?;
        let actual = fs::metadata(root.join(relative))
            .map_err(|error| OracleError::Identity(format!("cannot stat {path}: {error}")))?
            .len();
        if actual != size {
            return Err(OracleError::Identity(format!(
                "file {path} has {actual} bytes; expected {size}"
            )));
        }
        identities.push((path.to_owned(), size, blob.to_owned()));
    }
    identities.sort();

    let mut digest = Sha256::new();
    update_digest(&mut digest, revision.as_bytes());
    for (path, size, blob) in identities {
        update_digest(&mut digest, path.as_bytes());
        update_digest(&mut digest, &size.to_le_bytes());
        update_digest(&mut digest, blob.as_bytes());
    }
    for name in ["config.json", ".sytra-runtime.json"] {
        let path = root.join(name);
        let bytes = fs::read(&path).map_err(|source| OracleError::Read { path, source })?;
        update_digest(&mut digest, name.as_bytes());
        update_digest(&mut digest, &bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub trait OracleRuntime {
    fn oracle_manifest(&self) -> &RuntimeManifest;
    fn oracle_vocab_size(&self) -> usize;
    fn compute_oracle_outputs(&self, tokens: &[u32]) -> Result<KimiOracleOutputs, RuntimeError>;
}

impl OracleRuntime for KimiRuntime {
    fn oracle_manifest(&self) -> &RuntimeManifest {
        self.manifest()
    }
    fn oracle_vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    fn compute_oracle_outputs(&self, tokens: &[u32]) -> Result<KimiOracleOutputs, RuntimeError> {
        self.oracle_outputs(tokens)
    }
}

impl OracleRuntime for MixtralRuntime {
    fn oracle_manifest(&self) -> &RuntimeManifest {
        self.manifest()
    }
    fn oracle_vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    fn compute_oracle_outputs(&self, tokens: &[u32]) -> Result<KimiOracleOutputs, RuntimeError> {
        self.oracle_outputs(tokens)
    }
}

pub fn verify_runtime_oracle(
    model_root: impl AsRef<Path>,
    runtime: &impl OracleRuntime,
    suite: &OracleSuite,
) -> Result<OracleReport, OracleError> {
    suite.validate(
        &runtime.oracle_manifest().architecture.adapter,
        runtime.oracle_vocab_size(),
    )?;
    let fingerprint = checkpoint_fingerprint(model_root)?;
    if suite.model_fingerprint != fingerprint {
        return Err(OracleError::Identity(format!(
            "suite targets {}, but checkpoint is {fingerprint}",
            suite.model_fingerprint
        )));
    }
    let mut reports = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let output = runtime.compute_oracle_outputs(&case.input_tokens)?;
        if output.teacher_forced_predictions != case.teacher_forced_predictions {
            let mismatch = output
                .teacher_forced_predictions
                .iter()
                .zip(&case.teacher_forced_predictions)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(0);
            return Err(OracleError::Mismatch {
                case: case.name.clone(),
                reason: format!(
                    "teacher-forced token {mismatch} is {}, expected {}",
                    output.teacher_forced_predictions[mismatch],
                    case.teacher_forced_predictions[mismatch]
                ),
            });
        }
        let mut maximum_logit_error = 0.0_f32;
        for probe in &case.final_logit_probes {
            let actual = output.final_logits[probe.token as usize];
            let error = (actual - probe.expected).abs();
            let allowed =
                probe.absolute_tolerance + probe.relative_tolerance * probe.expected.abs();
            maximum_logit_error = maximum_logit_error.max(error);
            if !actual.is_finite() || error > allowed {
                return Err(OracleError::Mismatch {
                    case: case.name.clone(),
                    reason: format!(
                        "logit for token {} is {actual}, expected {} ± {allowed}",
                        probe.token, probe.expected
                    ),
                });
            }
        }
        reports.push(OracleCaseReport {
            name: case.name.clone(),
            positions: case.input_tokens.len(),
            logit_probes: case.final_logit_probes.len(),
            maximum_logit_error,
            metrics: output.metrics,
        });
    }
    Ok(OracleReport {
        passed: true,
        adapter: suite.adapter.clone(),
        model_fingerprint: fingerprint,
        reference_implementation: suite.reference_implementation.clone(),
        reference_revision: suite.reference_revision.clone(),
        cases: reports,
    })
}

pub fn verify_kimi_oracle(
    model_root: impl AsRef<Path>,
    runtime: &KimiRuntime,
    suite: &OracleSuite,
) -> Result<OracleReport, OracleError> {
    verify_runtime_oracle(model_root, runtime, suite)
}

pub fn verify_mixtral_oracle(
    model_root: impl AsRef<Path>,
    runtime: &MixtralRuntime,
    suite: &OracleSuite,
) -> Result<OracleReport, OracleError> {
    verify_runtime_oracle(model_root, runtime, suite)
}

fn read_json(path: &Path) -> Result<serde_json::Value, OracleError> {
    let bytes = fs::read(path).map_err(|source| OracleError::Read {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| OracleError::Json {
        path: path.to_owned(),
        source,
    })
}

fn update_digest(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn fingerprint_is_bound_to_revision_blob_ids_and_runtime_contract() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-oracle-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model.safetensors"), b"weights").unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join(".sytra-runtime.json"), b"{}").unwrap();
        fs::write(
            root.join(".sytra-model.json"),
            serde_json::to_vec(&serde_json::json!({
                "resolved_revision": "abcdef1234567890",
                "files": [{
                    "path": "model.safetensors",
                    "size": 7,
                    "blob_id": "sha256:deadbeef"
                }]
            }))
            .unwrap(),
        )
        .unwrap();
        let first = checkpoint_fingerprint(&root).unwrap();
        fs::write(root.join("config.json"), b"{\"changed\":true}").unwrap();
        let second = checkpoint_fingerprint(&root).unwrap();
        assert_ne!(first, second);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn suite_rejects_loose_or_incomplete_oracles() {
        let suite = OracleSuite {
            schema_version: ORACLE_SCHEMA,
            adapter: "sytra-kimi-k2.7-code".into(),
            model_fingerprint: "0".repeat(64),
            reference_implementation: "transformers@4.57.0".into(),
            reference_revision: "abcdef12".into(),
            cases: vec![OracleCase {
                name: "only".into(),
                input_tokens: vec![1],
                teacher_forced_predictions: vec![2],
                final_logit_probes: vec![LogitProbe {
                    token: 2,
                    expected: 1.0,
                    absolute_tolerance: 1.0,
                    relative_tolerance: 0.0,
                }],
            }],
        };
        assert!(matches!(
            suite.validate("sytra-kimi-k2.7-code", 3),
            Err(OracleError::Contract(_))
        ));
    }
}
