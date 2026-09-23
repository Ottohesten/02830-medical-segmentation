# Where to Look First

Prioritising human review of AI medical image segmentation.
DTU 02830 Advanced Project in Digital Media Engineering, group 2.

A pretrained segmentation model (TotalSegmentator / nnU-Net) segments an organ in many CT scans.
Instead of chasing higher Dice, we rank the scans by the model's own uncertainty, so a human reviewer
checks the most likely failures first, guided by an uncertainty heatmap.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. One lock file (`uv.lock`) covers
Intel Macs, Apple Silicon Macs and Linux (DTU HPC), so everybody gets the same package versions.

```bash
uv sync
```

PyTorch is pinned to 2.2.2 because that is the newest version with wheels for Intel Macs.

## Configurations: local vs. HPC

Every setting that affects runtime or file locations lives in one YAML file per environment:

| File | Meant for | Scans | Model resolution | TTA passes |
|------|-----------|-------|------------------|------------|
| `configs/local.yaml` | Laptop Mac (Intel or Apple Silicon) | 3 | fast (3 mm) | 0 |
| `configs/hpc.yaml` | DTU HPC with an NVIDIA GPU | all 41 | full (1.5 mm) | 8 |

Both files have the same structure: `compute` (device, threads), `paths` (data, weights, predictions,
results), `dataset` (which scans and how many), `model` (resolution, organ, folds), `uncertainty`
and `visualisation`.

Every script takes the config as its only required argument. To switch environment, switch the file:

```bash
uv run python scripts/run_inference.py --config configs/local.yaml   # on a laptop
uv run python scripts/run_inference.py --config configs/hpc.yaml     # on the HPC
```

- `device: auto` picks CUDA if available, otherwise MPS (Apple Silicon GPU), otherwise the CPU.
- Outputs are separated by the config's `name`: predictions go to `data/predictions/<name>/`, and
  result files are called e.g. `results/dice_<name>_<date>.csv`, so local and HPC runs never overwrite each other.
- For your own variant (e.g. more scans on a laptop), copy a config file and edit it. Relative paths are
  resolved from the repository root. On the HPC, `paths` can point to scratch storage.
- The `full` resolution needs more memory and time. On a laptop, stick to `fast`.

## Running the pipeline (phase 1: tracer bullet)

```bash
uv run python scripts/download_data.py  --config configs/local.yaml  # MSD Task09 Spleen, ~1.5 GB
uv run python scripts/download_model.py --config configs/local.yaml  # TotalSegmentator weights
uv run python scripts/run_inference.py  --config configs/local.yaml  # predict the organ in each scan
uv run python scripts/evaluate.py       --config configs/local.yaml  # Dice per scan -> results/
uv run python scripts/show_case.py      --config configs/local.yaml  # figure of one scan -> results/figures/
```

## Data

- **Dataset:** [Medical Segmentation Decathlon](http://medicaldecathlon.com/) Task09 Spleen: 41 CT scans with
  manual spleen masks (Memorial Sloan Kettering). CC-BY-SA 4.0.
- **Model:** TotalSegmentator v2 "total" task, used as released (no training).
- Scans, masks, probabilities and weights live in `data/` and `models/` and are never committed.
  Per scan we store only the organ mask and the organ's probability map, not the probabilities of all 117 classes.

## Layout

```
configs/   one YAML config per environment
src/segreview/  shared code (config, data, weights, inference, metrics, io)
scripts/   runnable pipeline steps
ui/        review interface (G2)
results/   small result files (csv, figures)
data/      scans, ground truth, model output (not in git)
models/    pretrained weights (not in git)
```
