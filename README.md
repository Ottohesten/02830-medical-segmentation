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

Every setting that affects runtime, data or file locations lives in one YAML file per environment:

| File | Meant for | Scans | Model resolution | TTA passes | Perturbations |
|------|-----------|-------|------------------|------------|---------------|
| `configs/local.yaml` | Laptop Mac (Intel or Apple Silicon); tests TTA and perturbations | 3 | fast (3 mm) | 3 | 3 |
| `configs/local_all.yaml` | Laptop Mac, G1 evaluation on all clean MSD Spleen scans | all 41 | fast (3 mm) | 0 | none |
| `configs/kits_pilot.yaml` | Laptop Mac, KiTS23 pilot (random scans, fixed seed) | 20 | fast (3 mm) | 0 | none |
| `configs/hpc.yaml` | DTU HPC with an NVIDIA GPU (not tested yet) | all 41 | full (1.5 mm) | 8 | 9 |

All files have the same structure: `compute` (device, threads), `paths` (data, weights, predictions,
results), `dataset` (which dataset file, how many scans, first or random selection), `model` (resolution, folds),
`uncertainty` (border, TTA, second model), `perturbations`, `evaluation` (what counts as a bad segmentation,
bootstrap) and `visualisation`. All configs also run the 6 mm model as a second model (see below).

Every script takes the config as its only required argument. To switch environment, switch the file:

```bash
uv run python scripts/run_g1.py --config configs/local.yaml   # on a laptop
uv run python scripts/run_g1.py --config configs/hpc.yaml     # on the HPC
```

- `device: auto` picks CUDA if available, otherwise MPS (Apple Silicon GPU), otherwise the CPU.
- Outputs are separated by the config's `name`: model output goes to `data/predictions/<name>/<variant>/`
  (variant = `clean` or a perturbation name) and result files to `results/<name>/`. File names have no date;
  a new run replaces the previous results (older versions are in the git history).
- For your own variant (e.g. more scans on a laptop), copy a config file, give it a new `name` and edit it.
  Relative paths are resolved from the repository root. On the HPC, `paths` can point to scratch storage.
- The `full` resolution, TTA and perturbations multiply the runtime. On an Intel Mac, all 41 scans at 3 mm
  take 1-2 hours; heavier runs belong on the HPC.

## Datasets

A dataset is described once in `configs/datasets/<name>.yaml`: where images and labels are, how to download
them (`tar` archive or `files` = one image + one label per scan), one or more **organ definitions** (which
ground-truth label values form the organ, as a union, and optionally which are left out of Dice), and which
TotalSegmentator classes match the organ (their probabilities are added up). An environment config points to it
with `dataset.file` and chooses `n_cases` and `selection` (`first`, or `random` with the config's seed, drawn from
the dataset's full scan list, so a larger random selection contains a smaller one). Adding a dataset needs a new
dataset file, not changes to inference, uncertainty or evaluation.

- **MSD Task09 Spleen** (`configs/datasets/msd_spleen.yaml`): 41 contrast-enhanced CT scans with manual
  spleen masks (Memorial Sloan Kettering), [Medical Segmentation Decathlon](http://medicaldecathlon.com/), CC-BY-SA 4.0.
- **KiTS23** (`configs/datasets/kits23.yaml`): 489 CT scans of patients with kidney tumours, with kidney (1),
  tumour (2) and cyst (3) masks; scans from many referring hospitals. [kits-challenge.org](https://kits-challenge.org/kits23/),
  CC BY-NC-SA 4.0. Only the selected scans are downloaded (median ~50 MB each). Organ definitions:
  `kidney_tumor_cyst` (used), `kidney_cyst`, `kidney_cyst_ignore_tumor`. Matching TotalSegmentator classes:
  kidney_left + kidney_right + kidney_cyst_left + kidney_cyst_right.
- **Model:** TotalSegmentator v2 "total" task, used as released (no training). Only fold 0 is published.

## Running G1 (uncertainty ranking)

One command runs everything; finished steps are skipped when it is run again:

```bash
uv run python scripts/download_data.py  --config configs/local_all.yaml  # dataset (~1.5 GB MSD Spleen; KiTS pilot 2.6 GB)
uv run python scripts/download_model.py --config configs/local_all.yaml  # TotalSegmentator weights (3 mm + 6 mm)
uv run python scripts/run_g1.py         --config configs/local_all.yaml  # inference -> TTA -> Dice -> scores -> evaluation
```

The steps can also be run one by one: `run_inference.py`, `run_tta.py`, `run_second_model.py`, `evaluate.py` (Dice),
`check_labels.py`, `compute_uncertainty.py`, `evaluate_g1.py`. Extra figures: `plot_dice.py` (Dice histogram) and
`show_case.py --case <id> [--variant <perturbation>]` (one scan with prediction, ground truth and heatmap).

What G1 computes:
- **Uncertainty scores per scan**, from the model's own output only (no ground truth). Higher = more uncertain.
  Voxel entropy summed, averaged in the organ + a border, and per organ volume; the soft-Dice gap; and with
  TTA the disagreement between passes. TTA uses small shifts, intensity changes and noise, not mirroring,
  because the model was trained without mirroring. **Second model:** the 6 mm TotalSegmentator network
  (`uncertainty.second_model_resolution: fastest`) is run next to the 3 mm one; their disagreement
  (1 - Dice between the masks, and mean |p_3mm - p_6mm| near the organ) replaces fold disagreement, which is
  impossible because only one fold is published.
- **Baselines:** random order and small predicted organ volume. **Oracle:** true Dice (upper bound).
- **Evaluation** (uses ground truth): Spearman rho with bootstrap CI, partial rho given volume, the review curve
  (share of bad segmentations found vs. share reviewed) and the quality curve (mean Dice after correcting the
  reviewed scans), each summarised by the area under the curve, all methods in one figure.
- **Heatmaps per scan**: `data/predictions/<name>/<variant>/<case>_entropy.nii.gz` (entropy, 0-1 bits) and
  `<case>_m2_diff.nii.gz` (|p_3mm - p_6mm|), NIfTI on the same grid as the CT, for the review interface.
- **Label check** (datasets with several organ definitions): Dice under each definition and how much of each
  ground-truth label the model calls organ (`label_check.csv`, `figures/label_check.png`).
- Clean and perturbed scans are evaluated and reported separately.

Main outputs in `results/<name>/`: `dice.csv`, `scores.csv`, `g1_metrics_clean.csv`, `g1_scores_clean.csv`,
`figures/g1_curves_clean.png`, `figures/g1_scatter_clean.png` (and `_perturbed` versions if perturbations are configured).

## Layout

```
configs/        one YAML config per environment; configs/datasets/ describes each dataset
src/segreview/  shared code (config, data, weights, inference, augment, uncertainty, evaluation, figures, pipeline)
scripts/        runnable pipeline steps
ui/             review interface (G2)
results/        results/<config name>/: small result files (csv, figures)
data/           scans, ground truth, model output (not in git)
models/         pretrained weights (not in git)
```
