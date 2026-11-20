pub mod backend_resolver;
pub mod catalog;
pub mod chat_agents;
pub mod chat_skills;
pub mod training_source;
pub mod commands;
pub mod convert;
pub mod datasource;
pub mod dataset_fabric;
pub mod download;
pub mod env_provisioner;
pub mod inference;
mod helix_guard;
pub mod job_runner;
pub mod materialize;
pub mod mcp_install;
pub mod model_settings;
pub mod models;
pub mod process;
pub mod resource_guard;
pub mod run_archive;
pub mod serve;
pub mod settings;
pub mod transcript;
pub mod validate;
pub mod voice;
pub mod workspace;
pub mod xet_safety;

pub use backend_resolver::BackendResolver;
pub use catalog::{hub_entries, HubCatalogEntry};
pub use datasource::{
    get_datasource, DataSource, DataSourceError, DatasetSpec, Materialized, PreviewRows, SourceKind,
};
pub use download::DownloadService;
pub use env_provisioner::{EnvKind, EnvProvisioner};
pub use xet_safety::{apply_xet_safety, xet_safety_env};
pub use inference::plan_inference;
pub use job_runner::JobRunner;
pub use models::{scan_local_models, LocalModelItem};
pub use process::{apply_desktop_priority, apply_hidden_window, kill_process_tree};
pub use resource_guard::{ResourceError, ResourceGuard};
pub use run_archive::{RunArchive, RunSummary};
pub use serve::ChatServer;
pub use validate::{validate_before_spawn, ValidationError};
pub use voice::{
    VoiceAudioResponse, VoiceRuntimeStatus, VoiceService, VoiceTranscriptResponse,
};
pub use workspace::{
    default_user_workspace, ensure_runner_deployed, find_project_root, python_executable,
    resolve_workspace,
};
