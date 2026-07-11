use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

use crate::merge_config::MergeConfig;
use crate::run_config::RunConfig;

#[derive(Debug, Clone)]
pub struct TrainSpec {
    pub config: RunConfig,
    pub config_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct MergeSpec {
    pub config: MergeConfig,
    pub config_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishSpec {
    pub op_id: Uuid,
    pub artifact_path: PathBuf,
    pub repo_id: String,
    pub private: bool,
    pub token: String,
}

#[derive(Debug, Clone)]
pub enum Operation {
    Train(TrainSpec),
    Merge(MergeSpec),
    Publish(PublishSpec),
}

#[derive(Debug, Clone)]
pub struct RunnerCmd {
    pub program: String,
    pub args: Vec<String>,
}

impl Operation {
    pub fn kind(&self) -> &'static str {
        match self {
            Operation::Train(_) => "train",
            Operation::Merge(_) => "merge",
            Operation::Publish(_) => "publish",
        }
    }

    pub fn runner_cmd(&self) -> RunnerCmd {
        match self {
            Operation::Train(spec) => RunnerCmd {
                program: "python".to_string(),
                args: vec![
                    "-m".to_string(),
                    "sytra_runner".to_string(),
                    spec.config_path.display().to_string(),
                ],
            },
            Operation::Merge(spec) => RunnerCmd {
                program: "python".to_string(),
                args: vec![
                    "-m".to_string(),
                    "sytra_runner.merge".to_string(),
                    spec.config_path.display().to_string(),
                ],
            },
            Operation::Publish(spec) => RunnerCmd {
                program: "python".to_string(),
                args: vec![
                    "-m".to_string(),
                    "sytra_runner.publish".to_string(),
                    spec.artifact_path.to_string_lossy().to_string(),
                    "--repo-id".to_string(),
                    spec.repo_id.clone(),
                    "--private".to_string(),
                    spec.private.to_string(),
                    "--token".to_string(),
                    spec.token.clone(),
                ],
            },
        }
    }

    /// Falls back to a nil UUID when the config was hand-written (null run_id/op_id).
    pub fn op_id(&self) -> Uuid {
        match self {
            Operation::Train(spec) => spec.config.run_id.unwrap_or(Uuid::nil()),
            Operation::Merge(spec) => spec.config.op_id.unwrap_or(Uuid::nil()),
            Operation::Publish(spec) => spec.op_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpStatus {
    Running,
    Done,
    Error,
    /// Cancelled by the user — distinct from Error so the Runs list can
    /// show it honestly and the run stays eligible for resume.
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpRecord {
    pub op_id: Uuid,
    pub kind: String,
    pub config: serde_json::Value,
    pub artifact_path: PathBuf,
    pub status: OpStatus,
    pub provenance: Option<serde_json::Value>,
}
