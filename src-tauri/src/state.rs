use std::path::PathBuf;
use std::sync::Mutex;

use sytra_contracts::guider::Guider;
use sytra_host::{ChatServer, DownloadService, JobRunner, ResourceGuard, RunArchive};

pub struct AppState {
    pub archive: Mutex<RunArchive>,
    pub runner: Mutex<JobRunner>,
    pub guard: Mutex<ResourceGuard>,
    pub guider: Mutex<Guider>,
    pub workspace: PathBuf,
    pub downloads: DownloadService,
    pub chat: ChatServer,
}
