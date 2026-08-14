#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod helpers;
mod state;

use sytra_contracts::guider::Guider;
use sytra_host::{
    ChatServer, DownloadService, JobRunner, ResourceGuard, RunArchive, BackendResolver,
    resolve_workspace,
};

use crate::state::AppState;

pub fn run() {
    let workspace = resolve_workspace();
    let runs_dir = workspace.join("runs");
    if let Err(err) = std::fs::create_dir_all(&runs_dir) {
        eprintln!("failed to create runs dir: {err}");
    }

    let env_provisioner = sytra_host::EnvProvisioner::new(&workspace);
    std::thread::spawn(move || {
        if let Err(err) = env_provisioner.provision_merge() {
            eprintln!("merge env provision failed: {err}");
        }
        if let Err(err) = env_provisioner.provision_train() {
            eprintln!("train env provision failed: {err}");
        }
    });

    let detected_vram_mb = BackendResolver::detect_system_vram_mb().unwrap_or(0);
    let detected_ram_mb = BackendResolver::detect_system_ram_mb().unwrap_or(0);
    let memory_limit_mb = if detected_ram_mb == 0 {
        0
    } else {
        sytra_host::settings::AppSettings::load(&workspace).effective_main_memory_mb(detected_ram_mb)
    };
    let downloads = DownloadService::new(&workspace);
    let chat = ChatServer::new(&workspace);
    let state = AppState {
        archive: std::sync::Mutex::new(RunArchive::new(&runs_dir)),
        runner: std::sync::Mutex::new(JobRunner::new(&workspace)),
        guard: std::sync::Mutex::new(ResourceGuard::new(
            detected_vram_mb,
            memory_limit_mb,
            500 * 1024,
        )),
        guider: std::sync::Mutex::new(Guider::new()),
        workspace,
        downloads,
        chat,
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            commands::start_train,
            commands::start_merge,
            commands::stop_op,
            commands::list_runs,
            commands::delete_run,
            commands::estimate_memory,
            commands::get_hardware_info,
            commands::get_settings,
            commands::set_cache_dir,
            commands::set_main_memory_limit,
            commands::guider_recommend,
            commands::merge_check,
            commands::preview_dataset,
            commands::publish_run,
            commands::list_catalog,
            commands::download_model,
            commands::cancel_download,
            commands::convert_model,
            commands::export_model,
            commands::get_download_status,
            commands::list_local_models,
            commands::build_moe_index,
            commands::start_chat_server,
            commands::stop_chat_server,
            commands::plan_inference,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Sytra Studio");
}
