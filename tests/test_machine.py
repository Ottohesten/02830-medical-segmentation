"""Tests of the machine description written next to results (src/segreview/machine.py): reading the Linux system
files and nvidia-smi's output (synthetic examples), and the fields CONTRIBUTING.md asks for."""

import sys

import pytest

from segreview.machine import (_windows_hardware, machine_info, parse_cpu_model, parse_mem_total_gib,
                               parse_nvidia_smi, parse_os_release)


def test_cpu_model_from_proc_cpuinfo():
    text = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i9-14900K\nflags\t\t: fpu\n"
    assert parse_cpu_model(text) == "Intel(R) Core(TM) i9-14900K"
    assert parse_cpu_model("") == ""


def test_total_memory_from_proc_meminfo():
    assert parse_mem_total_gib("MemTotal:       16303520 kB\nMemFree:         1000 kB\n") == 15.5
    assert parse_mem_total_gib("") is None


def test_distribution_from_os_release():
    assert parse_os_release('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.3 LTS"\nID=ubuntu\n') == "Ubuntu 24.04.3 LTS"
    assert parse_os_release("") == ""


def test_gpu_from_nvidia_smi():
    assert parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564, 617.14, 8.9\n") == {
        "gpu_memory_mib": 24564, "nvidia_driver": "617.14", "gpu_compute_capability": "8.9"}
    assert parse_nvidia_smi("") == {}
    assert parse_nvidia_smi("No devices were found") == {}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_hardware():
    info = _windows_hardware()
    assert info["cpu_model"] and not info["cpu_model"].startswith("Intel64 Family")   # the model name, not the family
    assert info["memory_total_gib"] > 1
    assert info["os"].startswith("Windows")


def test_machine_info_has_the_fields_contributing_asks_for():
    info = machine_info()
    for key in ("date", "os", "cpu", "cpu_count", "python", "torch", "cuda", "gpu", "git_commit",
                "git_uncommitted_changes"):
        assert key in info
