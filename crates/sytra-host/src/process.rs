use std::process::{Command, Stdio};

/// Keep Hub/merge/train children below the desktop compositor on every OS.
/// Windows: BELOW_NORMAL. Unix: a separate process group so stop() can
/// signal the tree; Python then `nice(10)`s itself at startup.
pub fn apply_desktop_priority(cmd: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x00004000;
        cmd.creation_flags(BELOW_NORMAL_PRIORITY_CLASS);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
}

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
