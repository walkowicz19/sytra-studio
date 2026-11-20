from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareProfile:
    os_name: str = "unknown"
    cpu_logical_cores: int = 1
    cpu_physical_cores: int | None = None
    ram_total_gb: float = 0
    ram_available_gb: float = 0
    cuda_available: bool = False
    gpu_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    vram_total_gb: float = 0
    vram_free_gb: float = 0
    bf16_supported: bool = False
    storage_kind: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware(torch_module=None) -> HardwareProfile:
    values = dict(os_name=platform.system(), cpu_logical_cores=os.cpu_count() or 1)
    try:
        import psutil
        ram = psutil.virtual_memory()
        values.update(cpu_physical_cores=psutil.cpu_count(logical=False),
                      ram_total_gb=ram.total / 1024**3, ram_available_gb=ram.available / 1024**3)
    except ImportError:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in
                    ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                values.update(ram_total_gb=status.total / 1024**3, ram_available_gb=status.available / 1024**3)
        else:
            try:
                from pathlib import Path
                info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                values.update(ram_total_gb=int(info["MemTotal"].split()[0]) / 1024**2,
                              ram_available_gb=int(info["MemAvailable"].split()[0]) / 1024**2)
            except (OSError, KeyError, ValueError):
                pass  # Unknown memory is conservative, never fabricated capacity.
    try:
        if torch_module is None:
            import torch as torch_module
        if torch_module.cuda.is_available():
            device = torch_module.cuda.current_device()
            props = torch_module.cuda.get_device_properties(device)
            free, total = torch_module.cuda.mem_get_info(device)
            values.update(cuda_available=True, gpu_name=props.name,
                          compute_capability=(props.major, props.minor),
                          vram_total_gb=total / 1024**3, vram_free_gb=free / 1024**3,
                          bf16_supported=torch_module.cuda.is_bf16_supported())
    except (ImportError, RuntimeError, OSError):
        pass
    return HardwareProfile(**values)
