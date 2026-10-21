"""Peak memory while part of a run executes: the RAM of this process and its worker processes, and GPU memory.

Used by the G3 smoke test (scripts/g3_smoke_test.py) and meant for the G3 training runs: CONTRIBUTING.md asks for
the peak memory of every long run.

Why not only the RAM of this process: nnU-Net prepares the training batches in separate worker processes
(nnUNet_n_proc_DA), so much of the work, and of the memory, is outside the main process. This module therefore
adds up the RAM of the main process and of every process it started, sampled in a background thread:
- Linux (the DTU HPC): each process counts with its PSS ("proportional set size"), from /proc. The workers are
  copies (forks) of the main process and share much of its memory; PSS splits a shared page between the processes
  that share it, so the shared memory is not counted once per worker.
- Windows (Frederik's GPU PC; needs psutil, which only the Windows environment installs): each process counts with
  its private working set (psutil's "uss": pages in RAM that only this process uses), so the program libraries that
  every process loads (e.g. torch's DLLs) are not counted once per worker. Reading the private working sets is slow
  and pauses the run (in the smoke test every sample made a training step 0.25 s slower), so it is read only once,
  when the block ends (the workers are still running then); what is sampled during the block is cheap.
  Windows also has a second limit, the commit limit (RAM + page file): every process reserves ("commits") memory,
  also memory that never comes into RAM, and on Windows GPU memory and each worker's import of CUDA torch (about
  2 GB) count as well. If the total passes the limit, the run stops with "the paging file is too small" (WinError
  1455). So on Windows the total commit (psutil's "private") is recorded too, as commit_gb.
- macOS: only the main process's own peak is available (getrusage).
In addition, the RAM in use on the whole machine (total minus available) is recorded where it can be read
(system_ram_gb; it includes other programs).
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


def tree_windows_gb(pid: int | None = None, ram: bool = True) -> tuple[float | None, float]:
    """Windows (with psutil): RAM and commit of a process (default: this one) and all its descendants, in GB.

    Output: (sum of the private working sets = RAM only these processes use, sum of the private commit = memory they
    reserved from Windows, which counts toward the commit limit even where it is not in RAM). ram=False skips the
    slow private working sets (the first value is then None) and reads only the commit, which is cheap.
    """
    psutil = _psutil()
    try:
        root = psutil.Process(pid or os.getpid())
        processes = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return (0.0 if ram else None), 0.0
    in_ram = commit = 0
    for process in processes:
        try:
            info = process.memory_full_info() if ram else process.memory_info()
            in_ram, commit = in_ram + (info.uss if ram else 0), commit + info.private
        except (psutil.NoSuchProcess, psutil.AccessDenied):      # ended in the meantime
            pass
    return (in_ram / 1e9 if ram else None), commit / 1e9


def system_ram_used_gb() -> float | None:
    """RAM in use on the whole machine (total minus available), in GB; None where it cannot be read."""
    try:
        values = {line.split(":")[0]: int(line.split()[1]) for line in (PROC / "meminfo").read_text().splitlines()}
        return (values["MemTotal"] - values["MemAvailable"]) * 1024 / 1e9    # Linux: kB of 1024 bytes
    except (OSError, KeyError, ValueError, IndexError):
        pass
    psutil = _psutil()
    if psutil is None:
        return None
    memory = psutil.virtual_memory()
    return (memory.total - memory.available) / 1e9


def memory_method() -> str | None:
    """How this system measures the RAM of the process tree: 'pss_process_tree' (Linux, sampled),
    'uss_process_tree_at_end' (Windows with psutil, read when the block ends), or None (macOS: own_peak_rss_gb)."""
    if (PROC / str(os.getpid()) / "smaps_rollup").exists():
        return "pss_process_tree"
    if sys.platform == "win32" and _psutil() is not None:
        return "uss_process_tree_at_end"
    return None


def can_measure_tree() -> bool:
    """True where the memory of the whole process tree can be measured (Linux, or Windows with psutil)."""
    return memory_method() is not None


def tree_memory_gb(pid: int | None = None, ram: bool = True) -> tuple[float | None, float | None]:
    """RAM (and on Windows the commit) of a process and all its descendants in GB, as memory_method() says.

    Output: (RAM, commit); commit is None on Linux, where it does not limit a run in the same way. ram=False: on
    Windows only the cheap commit (RAM None); on Linux it changes nothing (PSS is read every time).
    """
    if memory_method() == "pss_process_tree":
        return tree_pss_gb(pid), None
    return tree_windows_gb(pid, ram=ram)


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

    ram_gb:           the total RAM of this process and its descendants (see the module docstring); ram_method says
                      how: on Linux the highest PSS, sampled every `interval` seconds (a peak shorter than the
                      interval can be missed); on Windows the private working sets when the block ends; on macOS the
                      main process's own peak since it started (ram_method "rss_main_process_since_start").
    commit_gb:        Windows only: the highest total commit of the process tree, sampled (see the module docstring).
    system_ram_gb:    the highest RAM in use on the whole machine, including other programs (None if unknown).
    gpu_allocated_gb: torch's peak of GPU memory in use by tensors during the block (CUDA only, else None).
    gpu_reserved_gb:  torch's peak of GPU memory it held from the driver during the block. Closer to what nvidia-smi
                      shows, which also counts the CUDA context (a few hundred MB) and other programs.
    Input: the torch device (None = RAM only), the sampling interval in seconds.
    """

    def __init__(self, device=None, interval: float = 0.5):
        self.cuda = device is not None and getattr(device, "type", device) == "cuda"
        self.interval = interval
        self.ram_gb = self.commit_gb = self.system_ram_gb = None
        self.gpu_allocated_gb = self.gpu_reserved_gb = self.ram_method = None
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _higher(old: float | None, new: float | None) -> float | None:
        return new if old is None else old if new is None else max(old, new)

    def _sample(self, full: bool = False) -> None:
        """One measurement. full=False is cheap enough to repeat during the block; full=True also reads the
        Windows private working sets, which pauses the run briefly (only at the start and the end)."""
        if self.ram_method is not None:
            ram, commit = tree_memory_gb(ram=full)
            self.ram_gb, self.commit_gb = self._higher(self.ram_gb, ram), self._higher(self.commit_gb, commit)
        self.system_ram_gb = self._higher(self.system_ram_gb, system_ram_used_gb())

    def _watch(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample(full=False)

    def __enter__(self) -> "PeakMemory":
        if self.cuda:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.ram_method = memory_method()
        self._sample(full=False)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample(full=True)
        if self.ram_method is None:
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
               for name, value in (("ram_gb", self.ram_gb), ("commit_gb", self.commit_gb),
                                   ("system_ram_gb", self.system_ram_gb), ("gpu_allocated_gb", self.gpu_allocated_gb),
                                   ("gpu_reserved_gb", self.gpu_reserved_gb))}
        out[f"{prefix}ram_method"] = self.ram_method
        return out
