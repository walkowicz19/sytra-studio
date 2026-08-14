use std::process::{Command, Stdio};

/// Kills the process and its whole descendant tree. `Child::kill` alone is
/// not enough: runners spawn grandchildren that would survive and keep
/// holding GPU memory and file locks.
pub fn kill_process_tree(pid: u32) {
    #[cfg(target_os = "windows")]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if let Err(err) = status {
            eprintln!("failed to kill process tree {pid}: {err}");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let status = Command::new("kill")
            .args(["-TERM", &format!("-{pid}")])
            .status();
        if let Err(err) = status {
            eprintln!("failed to kill process group {pid}: {err}");
        }
    }
}
