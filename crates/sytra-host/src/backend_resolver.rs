use std::process::Command;
use std::sync::OnceLock;
use sytra_contracts::run_config::BackendKind;

static SYSTEM_RAM: OnceLock<u64> = OnceLock::new();
static SYSTEM_VRAM: OnceLock<u64> = OnceLock::new();

pub struct BackendResolver;

impl BackendResolver {
    /// Detects the available accelerator on the current machine.
    pub fn detect_best_backend() -> BackendKind {
        if Self::has_cuda() {
            BackendKind::Cuda
        } else if Self::has_rocm() {
            BackendKind::Rocm
        } else if Self::has_mps() {
            BackendKind::Mps
        } else {
            BackendKind::Cpu
        }
    }

    /// Resolves `BackendKind::Auto` to the best detected accelerator.
    pub fn resolve(requested: BackendKind) -> BackendKind {
        match requested {
            BackendKind::Auto => Self::detect_best_backend(),
            other => other,
        }
    }

    fn has_cuda() -> bool {
        // Quick check if nvidia-smi runs successfully
        let cmd = if cfg!(target_os = "windows") {
            "nvidia-smi.exe"
        } else {
            "nvidia-smi"
        };

        Command::new(cmd)
            .arg("-L")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn has_rocm() -> bool {
        // Quick check if rocm-smi runs successfully
        let cmd = if cfg!(target_os = "windows") {
            "rocm-smi.exe"
        } else {
            "rocm-smi"
        };

        Command::new(cmd)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn has_mps() -> bool {
        // MPS is macOS only
        cfg!(target_os = "macos")
    }

    pub fn detect_system_ram_mb() -> u64 {
        *SYSTEM_RAM.get_or_init(|| {
            if cfg!(target_os = "windows") {
                let output = Command::new("powershell")
                    .args([
                        "-NoProfile",
                        "-Command",
                        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                    ])
                    .output();
                if let Ok(out) = output {
                    if out.status.success() {
                        let s = String::from_utf8_lossy(&out.stdout);
                        if let Ok(bytes) = s.trim().parse::<u64>() {
                            return bytes / (1024 * 1024);
                        }
                    }
                }
            } else if cfg!(target_os = "macos") {
                let output = Command::new("sysctl").args(["-n", "hw.memsize"]).output();
                if let Ok(out) = output {
                    if out.status.success() {
                        let s = String::from_utf8_lossy(&out.stdout);
                        if let Ok(bytes) = s.trim().parse::<u64>() {
                            return bytes / (1024 * 1024);
                        }
                    }
                }
            } else if cfg!(target_os = "linux") {
                if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                    for line in content.lines() {
                        if line.starts_with("MemTotal:") {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 2 {
                                if let Ok(kb) = parts[1].parse::<u64>() {
                                    return kb / 1024;
                                }
                            }
                        }
                    }
                }
            }
            // Fallback default
            65536
        })
    }

    pub fn detect_system_vram_mb() -> u64 {
        *SYSTEM_VRAM.get_or_init(|| {
            // Try nvidia-smi first (works on Windows/Linux)
            let output = if cfg!(target_os = "windows") {
                Command::new("nvidia-smi.exe")
                    .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                    .output()
            } else {
                Command::new("nvidia-smi")
                    .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                    .output()
            };

            if let Ok(out) = output {
                if out.status.success() {
                    let s = String::from_utf8_lossy(&out.stdout);
                    if let Ok(mb) = s.trim().parse::<u64>() {
                        return mb;
                    }
                }
            }

            // Fallback for Windows if nvidia-smi fails
            if cfg!(target_os = "windows") {
                let output = Command::new("powershell")
                    .args(["-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty AdapterRAM | Measure-Object -Maximum).Maximum"])
                    .output();
                if let Ok(out) = output {
                    if out.status.success() {
                        let s = String::from_utf8_lossy(&out.stdout);
                        if let Ok(bytes) = s.trim().parse::<u64>() {
                            return bytes / (1024 * 1024);
                        }
                    }
                }
            } else if cfg!(target_os = "macos") {
                // Apple Silicon unified memory model fallback (approx 75% system memory allocated for VRAM)
                return Self::detect_system_ram_mb() * 3 / 4;
            } else if cfg!(target_os = "linux") {
                // Try AMD ROCm/AMDGPU total VRAM sysfs path
                if let Ok(bytes_str) =
                    std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_total")
                {
                    if let Ok(bytes) = bytes_str.trim().parse::<u64>() {
                        return bytes / (1024 * 1024);
                    }
                }
            }

            // Fallback default
            24576
        })
    }
}
