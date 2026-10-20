"""Tests of the peak-memory measurement (src/segreview/memory.py) with small synthetic allocations of 300 MB, in this
process and in a child process (as nnU-Net's augmentation workers are). The process-tree tests need Linux (/proc) or
Windows with psutil; the reading of /proc itself is tested on every system with a fake /proc folder."""

import multiprocessing
import os
import sys
import time

import numpy as np
import pytest

from segreview import memory
from segreview.memory import (PeakMemory, can_measure_tree, memory_method, parent_pid, process_tree, pss_bytes,
                              tree_pss_gb)

SIZE = 300_000_000            # bytes
tree_only = pytest.mark.skipif(not can_measure_tree(), reason="needs Linux (/proc) or Windows with psutil")
proc_only = pytest.mark.skipif(memory_method() != "pss_process_tree", reason="needs Linux (/proc)")


def test_process_tree_and_pss_from_a_fake_proc(tmp_path, monkeypatch):
    """The reading of /proc on any system: a fake /proc with {pid: (parent pid, PSS in kB, program name)}."""
    processes = {100: (1, 1000, "python"), 101: (100, 500, "nnUNet worker (1)"), 102: (101, 250, "grand child"),
                 200: (1, 9999, "another program")}
    for pid, (parent, pss_kb, name) in processes.items():
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(f"{pid} ({name}) S {parent} {pid} {pid} 0 -1 4194560\n")
        (tmp_path / str(pid) / "smaps_rollup").write_text(
            f"00400000-7ffd00000000 ---p 00000000 00:00 0 [rollup]\nRss: {2 * pss_kb} kB\nPss: {pss_kb} kB\n")
    (tmp_path / "self").mkdir()                       # entries that are not process ids are ignored
    monkeypatch.setattr(memory, "PROC", tmp_path)
    assert parent_pid(101) == 100                     # a name with spaces and parentheses is read correctly
    assert sorted(process_tree(100)) == [100, 101, 102]
    assert pss_bytes(101) == 500 * 1024
    assert pss_bytes(999) == 0 and parent_pid(999) is None     # a process that ended in the meantime
    assert tree_pss_gb(100) == pytest.approx((1000 + 500 + 250) * 1024 / 1e9)


def _hold_memory(go, ready, done):
    """Child process: when told, allocate SIZE bytes (written, so the pages are really in memory) and hold them."""
    go.wait(30)
    block = np.ones(SIZE // 8)
    ready.set()
    done.wait(30)
    del block


def baseline_gb() -> float:
    with PeakMemory(interval=0.05) as peak:
        pass
    return peak.ram_gb


def start_child():
    """A child process running _hold_memory, and its three events (go, ready, done)."""
    ctx = multiprocessing.get_context("spawn")       # a fresh process, as on Windows; fork would copy our threads
    events = ctx.Event(), ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_memory, args=events)
    child.start()
    return child, events


def stop_child(child, events) -> None:
    for event in events:
        event.set()
    child.join(30)


@proc_only
def test_process_tree_contains_this_process_and_its_child():
    child, events = start_child()
    try:
        assert parent_pid(child.pid) == os.getpid()
        assert {os.getpid(), child.pid} <= set(process_tree(os.getpid()))
    finally:
        stop_child(child, events)


@tree_only
def test_peak_ram_sees_an_allocation_in_this_process():
    before = baseline_gb()
    with PeakMemory(interval=0.05) as peak:
        block = np.ones(SIZE // 8)
        time.sleep(0.3)
        del block
    assert peak.ram_method == memory_method()
    assert peak.ram_gb - before > 0.8 * SIZE / 1e9


@tree_only
def test_peak_ram_includes_child_processes():
    child, (go, ready, done) = start_child()
    try:
        before = baseline_gb()
        with PeakMemory(interval=0.05) as peak:
            go.set()
            assert ready.wait(30)
            time.sleep(0.3)
    finally:
        stop_child(child, (go, ready, done))
    assert peak.ram_gb - before > 0.8 * SIZE / 1e9


def test_no_gpu_numbers_without_cuda():
    with PeakMemory() as peak:
        pass
    assert peak.gpu_allocated_gb is None and peak.gpu_reserved_gb is None
    assert set(peak.as_dict()) == {"peak_ram_gb", "peak_gpu_allocated_gb", "peak_gpu_reserved_gb", "peak_ram_method"}


@pytest.mark.skipif(sys.platform == "win32" and memory._psutil() is None, reason="Windows without psutil")
def test_ram_is_measured_on_this_system():
    with PeakMemory() as peak:
        pass
    assert peak.ram_gb is not None and peak.ram_gb > 0
