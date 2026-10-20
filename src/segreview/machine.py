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


def machine_info() -> dict:
    """Describe this machine and the code version.

    Output: dict with the computer model (on a Mac), CPU, operating system, Python / torch / numpy versions,
            CUDA version and GPU name (if any), and the git commit plus whether there were uncommitted changes.
    """
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        **_mac_hardware(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        # Uncommitted changes mean the commit alone does not describe the code that ran.
        "git_uncommitted_changes": bool(_run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])),
    }
    info.setdefault("os", platform.platform())
    return info


def write_machine_info(path: Path, **extra) -> dict:
    """Save machine_info() (plus any extra fields, e.g. the config file and run time) as JSON at 'path'."""
    info = {**machine_info(), **extra}
    path.write_text(json.dumps(info, indent=2, default=str) + "\n")
    return info
