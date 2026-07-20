#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/// Tauri 2 binary entry-point.
/// All orchestration lives in sytra-host; this file only wires Tauri.
fn main() {
    sytra_studio_lib::run();
}
