"""Planning estimates for G3 from the GPU smoke test: how long one fine-tuning run and the whole plan take, and how
much disk space they need (docs/g3_plan.md, section "Compute"). Used by scripts/g3_smoke_estimate.py.

These are ESTIMATES for planning, computed from measured times and sizes; they are not results.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def run_hours(steps: int, train_wall_s: float, validation_wall_s: float, steps_per_epoch: int = 250,
              validation_steps_per_epoch: int = 50) -> float:
    """Hours for one nnU-Net fine-tuning run of `steps` training steps.

    nnU-Net trains in epochs of steps_per_epoch steps and runs validation_steps_per_epoch validation steps (forward
    pass only) after each epoch. Every step costs its measured wall time (waiting for the batch + the step itself,
    with the augmentation workers of a real run). Not included: starting the trainer and its workers, and saving
    checkpoints (seconds per run or per epoch).
    Input: number of training steps; wall seconds per training step and per validation step; nnU-Net's step counts.
    Output: hours.
    """
    epochs = steps / steps_per_epoch
    return (steps * train_wall_s + epochs * validation_steps_per_epoch * validation_wall_s) / 3600


def evaluation_hours(models: int, scans: int, predict_s: float) -> float:
    """Hours to predict `scans` scans with each of `models` models, at predict_s seconds per scan and model."""
    return models * scans * predict_s / 3600


def remote_size(url: str) -> int:
    """Size in bytes of a file on a web server, without downloading it (an HTTP HEAD request).

    The Hugging Face server answers with a redirect to its storage and gives the file's size in the header
    X-Linked-Size; other servers give Content-Length.
    """
    r = requests.head(url, allow_redirects=False, timeout=60)
    size = r.headers.get("X-Linked-Size")
    if size is None:
        if r.is_redirect:
            r = requests.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        size = r.headers.get("Content-Length")
    if size is None:
        raise ValueError(f"The server gives no size for {url}")
    return int(size)


def remote_sizes(urls: list[str], workers: int = 8) -> list[int]:
    """remote_size of many files, a few requests at a time."""
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(remote_size, urls))


def folder_bytes(path: Path) -> int:
    """Total size of all files in a folder and its subfolders (0 if the folder does not exist)."""
    path = Path(path)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0
