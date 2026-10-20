"""Peak memory while part of a run executes: the RAM of this process and its worker processes, and GPU memory.

Used by the G3 smoke test (scripts/g3_smoke_test.py) and meant for the G3 training runs: CONTRIBUTING.md asks for
the peak memory of every long run.

Why not only the RAM of this process: nnU-Net prepares the training batches in separate worker processes
(nnUNet_n_proc_DA), so much of the work, and of the memory, is outside the main process. This module therefore
adds up the memory of the main process and of every process it started, sampled in a background thread:
- Linux (the DTU HPC): each process counts with its PSS ("proportional set size"), from /proc. The workers are
  copies (forks) of the main process and share much of its memory; PSS splits a shared page between the processes
  that share it, so the shared memory is not counted once per worker.
- Windows (Frederik's GPU PC; needs psutil, which only the Windows environment installs): each process counts with
  its private memory (memory that only this process uses). The program libraries that every process loads (e.g.
  torch's DLLs) are shared and left out, so they are not counted once per worker.
- macOS: only the main process's own peak is available (getrusage).
"""

import os
import sys
import threading
from pathlib import Path

PROC = Path("/proc")


def _psutil():
    """The psutil module if it is installed (the Windows environment has it; uv.lock does not), else None."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def parent_pid(pid: int) -> int | None:
    """Linux: parent process id from /proc/<pid>/stat (None if the process ended in the meantime)."""
    try:
        stat = (PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    # The file starts "pid (program name) state ppid ...". The name may contain spaces, so split after its ')'.
    return int(stat.rsplit(")", 1)[1].split()[1])


def process_tree(pid: int) -> list[int]:
    """Linux: a process and all its descendants (its children, their children, ...).

    Found from the parent id of every running process, which every Linux kernel shows in /proc/<pid>/stat
    (a direct list of children, /proc/<pid>/task/*/children, is missing in some kernels).
    """
    children = {}
    for entry in PROC.iterdir():
        if entry.name.isdigit():
            parent = parent_pid(int(entry.name))
            if parent is not None:
                children.setdefault(parent, []).append(int(entry.name))
    tree, todo = [], [pid]
    while todo:
        current = todo.pop()
        tree.append(current)
        todo += children.get(current, [])
    return tree


def pss_bytes(pid: int) -> int:
    """Linux: PSS of one process in bytes, from /proc/<pid>/smaps_rollup (0 if the process ended in the meantime)."""
    try:
        for line in (PROC / str(pid) / "smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024        # the file counts in kB of 1024 bytes
    except (OSError, ValueError):
        pass
    return 0


def tree_pss_gb(pid: int | None = None) -> float:
    """Linux: total PSS of a process (default: this one) and all its descendants, in GB (10^9 bytes)."""
    return sum(pss_bytes(p) for p in process_tree(pid or os.getpid())) / 1e9


def tree_private_gb(pid: int | None = None) -> float:
    """Windows (with psutil): total private memory of a process (default: this one) and all its descendants, in GB."""
    psutil = _psutil()
    try:
        root = psutil.Process(pid or os.getpid())
        processes = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0.0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().private
        except (psutil.NoSuchProcess, psutil.AccessDenied):      # ended in the meantime
            pass
    return total / 1e9


def memory_method() -> str | None:
    """How this system measures the memory of the process tree: 'pss_process_tree' (Linux),
    'private_process_tree' (Windows with psutil), or None (macOS: see own_peak_rss_gb)."""
    if (PROC / str(os.getpid()) / "smaps_rollup").exists():
        return "pss_process_tree"
    if sys.platform == "win32" and _psutil() is not None:
        return "private_process_tree"
    return None


def can_measure_tree() -> bool:
    """True where the memory of the whole process tree can be measured (Linux, or Windows with psutil)."""
    return memory_method() is not None


def tree_memory_gb(pid: int | None = None) -> float:
    """Memory of a process (default: this one) and all its descendants in GB, measured as memory_method() says."""
    return tree_pss_gb(pid) if memory_method() == "pss_process_tree" else tree_private_gb(pid)


def own_peak_rss_gb() -> float | None:
    """Peak resident memory of this process since it started, in GB (getrusage on macOS and Linux, the peak working
    set on Windows with psutil); None where unavailable."""
    try:
        import resource
    except ImportError:                                   # Windows
        psutil = _psutil()
        return None if psutil is None else psutil.Process().memory_info().peak_wset / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9    # bytes on macOS, KiB on Linux


class PeakMemory:
    """Peak memory while a block of code runs:

        with PeakMemory(device) as peak:
            ...                                   # e.g. a few training steps
        peak.as_dict()                            # ram_gb, gpu_allocated_gb, gpu_reserved_gb, ram_method

    ram_gb:           on Linux and Windows the highest total memory of this process and its descendants (see the
                      module docstring), sampled every `interval` seconds (a peak shorter than the interval can be
                      missed); ram_method says how it was measured. On macOS the main process's own peak since it
                      started (ram_method "rss_main_process_since_start").
    gpu_allocated_gb: torch's peak of GPU memory in use by tensors during the block (CUDA only, else None).
    gpu_reserved_gb:  torch's peak of GPU memory it held from the driver during the block. Closer to what nvidia-smi
                      shows, which also counts the CUDA context (a few hundred MB) and other programs.
    Input: the torch device (None = RAM only), the sampling interval in seconds.
    """

    def __init__(self, device=None, interval: float = 0.5):
        self.cuda = device is not None and getattr(device, "type", device) == "cuda"
        self.interval = interval
        self.ram_gb = self.gpu_allocated_gb = self.gpu_reserved_gb = self.ram_method = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self) -> None:
        value = tree_memory_gb()
        self.ram_gb = value if self.ram_gb is None else max(self.ram_gb, value)

    def _watch(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> "PeakMemory":
        if self.cuda:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.ram_method = memory_method()
        if self.ram_method is not None:
            self._sample()
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._sample()
        else:
            self.ram_gb = own_peak_rss_gb()
            self.ram_method = None if self.ram_gb is None else "rss_main_process_since_start"
        if self.cuda:
            import torch
            torch.cuda.synchronize()
            self.gpu_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
            self.gpu_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

    def as_dict(self, prefix: str = "peak_") -> dict:
        """The measured peaks as a flat dict for a results table, rounded to 0.01 GB (None where not measured)."""
        out = {f"{prefix}{name}": None if value is None else round(value, 2)
               for name, value in (("ram_gb", self.ram_gb), ("gpu_allocated_gb", self.gpu_allocated_gb),
                                   ("gpu_reserved_gb", self.gpu_reserved_gb))}
        out[f"{prefix}ram_method"] = self.ram_method
        return out
