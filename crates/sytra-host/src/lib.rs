pub mod backend_resolver;
pub mod catalog;
pub mod commands;
pub mod convert;
pub mod datasource;
pub mod download;
pub mod env_provisioner;
pub mod inference;
pub mod job_runner;
pub mod materialize;
pub mod process;
pub mod resource_guard;
pub mod run_archive;
pub mod serve;
pub mod settings;
pub mod transcript;
pub mod validate;
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
pub use process::{apply_desktop_priority, kill_process_tree};
pub use resource_guard::{ResourceError, ResourceGuard};
pub use run_archive::{RunArchive, RunSummary};
pub use serve::ChatServer;
pub use validate::{validate_before_spawn, ValidationError};
pub use workspace::{find_project_root, python_executable, resolve_workspace};
