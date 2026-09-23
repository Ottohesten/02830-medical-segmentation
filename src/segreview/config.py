"""Loading the run configuration and picking the compute device.

Every script takes one argument, --config, that points to a YAML file in configs/.
All settings that change runtime or file locations (device, resolution, number of scans,
TTA passes, folds, organ, paths) come from that file, so switching between a laptop and
the DTU HPC means switching config file, not editing code.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# The repository root is two levels above this file (src/segreview/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


def config_arg_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser that already has the required --config argument.

    Input: a short description of the script (shown by --help).
    Output: an argparse.ArgumentParser that scripts can add more arguments to.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path,
                        help="path to a YAML config, e.g. configs/local.yaml or configs/hpc.yaml")
    return parser


def _absolute(path: str) -> Path:
    """Resolve a path from a config: relative paths are taken from the repository root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path) -> dict:
    """Read a YAML config, merge in its dataset description, and make all paths absolute.

    The environment config (e.g. configs/local.yaml) points to a dataset file with
    dataset.file. The fields of that file are merged into cfg["dataset"]; fields set in the
    environment config (such as n_cases) win. This keeps the dataset description in one place,
    shared by all environments.

    Input: path to the YAML file.
    Output: the config as a nested dict.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = _absolute(value)
    with open(_absolute(cfg["dataset"]["file"])) as f:
        dataset = yaml.safe_load(f)
    cfg["dataset"] = {**dataset, **cfg["dataset"]}
    return cfg


def results_dir(cfg: dict) -> Path:
    """Folder for this config's result files, e.g. results/local_all/. Created if missing."""
    folder = cfg["paths"]["results_dir"] / cfg["name"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def figures_dir(cfg: dict) -> Path:
    """Folder for this config's figures, e.g. results/local_all/figures/. Created if missing."""
    folder = results_dir(cfg) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def pick_device(preference: str) -> torch.device:
    """Choose the torch device.

    'auto' tries CUDA (NVIDIA GPU, e.g. on the HPC) first, then MPS (the GPU in Apple
    Silicon Macs), and falls back to the CPU (e.g. Intel Macs). Any other value
    ('cuda', 'mps', 'cpu') forces that device.

    Input: the compute.device string from the config.
    Output: a torch.device.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_compute(cfg: dict) -> torch.device:
    """Set seeds and CPU threads from the config and return the device to use.

    Input: the loaded config.
    Output: the chosen torch.device.
    """
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg["compute"]["num_threads"] > 0:
        torch.set_num_threads(cfg["compute"]["num_threads"])
    return pick_device(cfg["compute"]["device"])
