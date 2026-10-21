"""Which machine produced a result? Written next to result files so every number can be traced to its machine.

CPU, Apple GPU and NVIDIA GPU can give slightly different numbers (see CONTRIBUTING.md), so every run records
the computer model, the operating system, the versions of the numeric libraries and the git commit.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from segreview.config import REPO_ROOT


def _run(cmd: list[str]) -> str:
    """Run a small command and return its output, or '' if it is not available on this machine."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _mac_hardware() -> dict:
    """Model name, model id, chip and memory of a Mac, read from system_profiler (empty on other systems)."""
    if sys.platform != "darwin":
        return {}
    wanted = {"Model Name": "model_name", "Model Identifier": "model_id", "Chip": "chip", "Memory": "memory"}
    info = {}
    for line in _run(["system_profiler", "SPHardwareDataType"]).splitlines():
        key, _, value = line.strip().partition(": ")
        if key in wanted:
            info[wanted[key]] = value
    info["os"] = f"macOS {_run(['sw_vers', '-productVersion'])} ({_run(['sw_vers', '-buildVersion'])})"
    return info


def parse_cpu_model(cpuinfo: str) -> str:
    """The CPU model name from the text of /proc/cpuinfo ('' if it is not there)."""
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    return ""


def parse_mem_total_gib(meminfo: str) -> float | None:
    """Total memory in GiB (2^30 bytes) from the text of /proc/meminfo (None if it is not there)."""
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 2**20, 1)        # the file counts in kB of 1024 bytes
    return None


def parse_os_release(text: str) -> str:
    """The distribution's name and version from the text of /etc/os-release, e.g. 'Ubuntu 24.04.3 LTS'."""
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return ""


def parse_nvidia_smi(csv_text: str) -> dict:
    """GPU memory, driver and compute capability of the first GPU, from the output of
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits.
    Output: dict (empty if the text does not have the four fields)."""
    lines = csv_text.strip().splitlines()
    parts = [p.strip() for p in lines[0].split(",")] if lines else []
    if len(parts) != 4:
        return {}
    name, memory_mib, driver, capability = parts
    return {"gpu_memory_mib": int(float(memory_mib)), "nvidia_driver": driver, "gpu_compute_capability": capability}


def _read(path: str) -> str:
    """Text of a file, or '' if it cannot be read."""
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _linux_hardware() -> dict:
    """CPU model, memory, distribution and whether this is WSL2, on Linux (empty on other systems).

    In WSL2 (Linux inside Windows) the memory is what Windows gives the Linux machine (by default half of the PC's
    RAM; set in %UserProfile%\\.wslconfig), not the PC's RAM.
    """
    if not sys.platform.startswith("linux"):
        return {}
    release = platform.release()
    info = {"cpu_model": parse_cpu_model(_read("/proc/cpuinfo")),
            "memory_total_gib": parse_mem_total_gib(_read("/proc/meminfo")),
            "wsl": "microsoft" in release.lower()}
    distribution = parse_os_release(_read("/etc/os-release"))
    if distribution:
        info["os"] = f"{distribution} ({'WSL2, ' if info['wsl'] else ''}Linux kernel {release})"
    return info


def _windows_hardware() -> dict:
    """CPU model name, memory and Windows version on Windows (empty on other systems).

    The CPU name is read from the registry (Python itself only gives e.g. 'Intel64 Family 6 Model 183'), the memory
    from the Windows function GlobalMemoryStatusEx (the RAM Windows can use).
    """
    if sys.platform != "win32":
        return {}
    import ctypes
    import winreg

    info = {}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            info["cpu_model"] = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        pass

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page_file", "avail_page_file",
                                                    "total_virtual", "avail_virtual", "avail_extended_virtual")]

    status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        info["memory_total_gib"] = round(status.total_phys / 2**30, 1)
    version = sys.getwindowsversion()
    info["os"] = f"Windows {platform.release()} (version {version.major}.{version.minor}, build {version.build})"
    return info


def _nvidia_gpu() -> dict:
    """GPU memory, NVIDIA driver version and compute capability from nvidia-smi (empty without an NVIDIA GPU)."""
    return parse_nvidia_smi(_run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                                  "--format=csv,noheader,nounits"]))


def machine_info() -> dict:
    """Describe this machine and the code version.

    Output: dict with the computer model (on a Mac), CPU (and its model name on Linux and Windows), OS, memory,
            Python / torch / numpy versions, CUDA and cuDNN versions, GPU name, memory and driver (if any), and the
            git commit plus whether there were uncommitted changes.
    """
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        **_mac_hardware(),
        **_linux_hardware(),
        **_windows_hardware(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **(_nvidia_gpu() if torch.cuda.is_available() else {}),
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        # Uncommitted changes mean the commit alone does not describe the code that ran.
        "git_uncommitted_changes": bool(_run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])),
    }
    info.setdefault("os", platform.platform())
    return info


def write_machine_info(path: Path, info: dict | None = None, **extra) -> dict:
    """Save machine_info() (plus any extra fields, e.g. the config file and run time) as JSON at 'path'.

    info: machine_info() taken earlier, e.g. at the start of a run, before the run writes its own result files
    (otherwise git sees those new files and always reports uncommitted changes). Default: taken now.
    """
    info = {**(machine_info() if info is None else info), **extra}
    path.write_text(json.dumps(info, indent=2, default=str) + "\n")
    return info
