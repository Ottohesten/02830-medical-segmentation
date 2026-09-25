# Where to Look First

Prioritising human review of AI medical image segmentation.
DTU 02830 Advanced Project in Digital Media Engineering, group 2.

A pretrained segmentation model (TotalSegmentator / nnU-Net) segments an organ in many CT scans.
Instead of chasing higher Dice, we rank the scans by the model's own uncertainty, so a human reviewer
checks the most likely failures first, guided by an uncertainty heatmap.

## Quick start

```bash
uv sync                                                                  # install (once, and after each pull)
uv run python scripts/run_ui.py --config configs/study.yaml              # start the review interface
```

The interface opens in Google Chrome at **http://127.0.0.1:8765** (the study only runs in Chrome; set
`server.browser` in `configs/study.yaml` to `default` or `none` to change that). Stop it with Ctrl+C.

- **Participant ids:** real participants are `P01`, `P02`, ... (the number sets the balanced order).
  **Test runs always use `T01`, `T02`, ...**: same order as the P id with the same number, never analysed.
- **Delete test data** (sessions of T ids and all demo-queue edits; P ids are never touched):

  ```bash
  uv run python scripts/clean_test_data.py --config configs/study.yaml --dry-run   # list what would go
  uv run python scripts/clean_test_data.py --config configs/study.yaml             # delete it
  ```

- **Run the tests:** `uv run pytest` (about two minutes; the browser tests need Google Chrome).

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
| `configs/kits100.yaml` | Laptop Mac (overnight), KiTS23: 20 development + 80 test scans (random, fixed seed) | 100 | fast (3 mm) | 0 | none |
| `configs/hpc.yaml` | DTU HPC with an NVIDIA GPU (not tested yet; job scripts in `jobs/`) | all 41 | full (1.5 mm) | 8 | 9 |

All files have the same structure: `compute` (device, threads), `paths` (data, weights, predictions,
results), `dataset` (which dataset file, how many scans, first or random selection), `model` (resolution, folds),
`uncertainty` (border, TTA, second model), `plausibility`, `perturbations`, `evaluation` (which splits may be
evaluated, one or more rules for a bad segmentation, rank combinations, bootstrap) and `visualisation`. All configs also run the 6 mm model as a second model (see below).

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

**Development and test split.** A dataset config can list the development scans (`dataset.split.dev`); all other
selected scans are the test set. Ground truth is only read for the splits in `evaluation.splits`. All choices are
made on `dev`; `test` is added once, after they are locked, so the test result cannot influence them.
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
- **Plausibility check** (paired organs such as the kidneys; GT-free, `src/segreview/plausibility.py`): splits the
  predicted organ at the body's midline and flags a missing side (`plaus_missing_side`) or a large left/right
  difference (`plaus_asymmetry`). It catches confident mistakes, e.g. a missed kidney, that uncertainty cannot see.
  `evaluation.combinations` merges it with an uncertainty score by rank.
- **Label check** (datasets with several organ definitions): Dice under each definition and how much of each
  ground-truth label the model calls organ (`label_check.csv`, `figures/label_check.png`).
- Clean and perturbed scans are evaluated and reported separately.

Main outputs in `results/<name>/`: `dice.csv`, `scores.csv`, `g1_metrics_<set>.csv` (one row per method and
"bad" rule), `g1_scores_<set>.csv`, `figures/g1_curves_<set>_<rule>.png`, `figures/g1_scatter_<set>_<rule>.png`.
`<set>` is `clean` (or `clean_dev` / `clean_test` with a split) and `perturbed` if perturbations are configured.

### Figures and tables for the report

```bash
uv run python scripts/make_report_figures.py --config configs/kits100.yaml
```

Writes `results/kits100/report/`: review and quality curves, AUC minus the volume baseline with paired bootstrap
CIs, three heatmap examples (picked by fixed rules: most certain scan; the most uncertain and the least uncertain of
the bad scans), and the G1 and heatmap tables as CSV and LaTeX (booktabs). Everything is built from the saved G1
results and model output (no new analysis); the script checks that the curve areas equal the evaluated AUCs.
Sizes follow the ACM two-column template (3.33 in / 7 in), PDF with embedded fonts plus a PNG preview.

## Running on the DTU HPC (not tested on the cluster yet)

> **Untested.** The job scripts in `jobs/` follow the DTU HPC (LSF) documentation, and their commands have been run
> on a laptop (`CONFIG=configs/local.yaml bash jobs/hpc_inference.sh`), but they have **never been submitted to the
> cluster**. Watch the first job closely and fix queue names or resources if LSF complains.

One-time setup on a login node (`ssh <user>@login1.hpc.dtu.dk`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # installs uv into ~/.local/bin
git clone <repository url> && cd 02830-medical-segmentation
uv sync                                                  # the lock file includes Linux wheels (torch 2.2.2 + CUDA 12.1)
```

The home folder is small: point `paths.raw_dir`, `paths.models_dir` and `paths.predictions_dir` in
`configs/hpc.yaml` at scratch storage (e.g. `/work3/<user>/segreview/...`). Download data and weights on the login
node, since compute nodes may not have internet access:

```bash
uv run python scripts/download_data.py  --config configs/hpc.yaml
uv run python scripts/download_model.py --config configs/hpc.yaml
```

Submit from the repository root:

```bash
mkdir -p logs
bsub < jobs/hpc_inference.sh                                       # step 2: all scans and perturbations
bsub -w "done(segreview_inference)" < jobs/hpc_g1.sh               # the rest of G1 once inference is done
CONFIG=configs/kits100.yaml bsub < jobs/hpc_g1.sh                  # another config (bsub passes CONFIG on)
bsub -q gpua100 < jobs/hpc_g1.sh                                   # another GPU queue (overrides #BSUB -q)
bjobs            # list your jobs;  bpeek <job id>: live output;  bkill <job id>: stop a job
```

- **Resources** (the `#BSUB` lines): queue `gpuv100`, 1 GPU (exclusive), 4 cores with 8 GB each, 24 h. Keep
  `compute.num_threads` in the config equal to the cores (`-n`).
- **Time limit:** finished scans are skipped, so a job that runs out of time can be submitted again.
- **Output:** `logs/<job>_<id>.out|err`, results in `results/<config name>/` (copy back with `rsync` or git),
  predictions in `paths.predictions_dir`.
- **Known unknowns:** runtime at full resolution (task 291, 1.5 mm) has not been measured; torch 2.2.2 with CUDA 12.1
  runs on V100/A100/H100 but not on the newest (Blackwell) GPUs.

## Review interface (G2)

A browser-based viewer built on [NiiVue](https://github.com/niivue/niivue) (stored in `ui/vendor/niivue`, so no internet
is needed) with a small local Python server. Settings are in `configs/study.yaml` (scan selection, time limit,
heatmap, colours, brush sizes, trackpad behaviour). The study runs in **Google Chrome** on a MacBook with a
**trackpad**; other browsers are not supported (painting does not work in Firefox).

```bash
uv run python scripts/prepare_study.py --config configs/study.yaml   # once: pick scans, write viewer files
uv run python scripts/run_ui.py        --config configs/study.yaml   # start the server and open the browser
```

- **Study mode:** enter a participant id (P01, P02, ...). The participant first sees a short guide (an example image of
  kidneys and a tumour, made from a development scan used nowhere else, and how to use the trackpad), then a
  practice scan, and after it the correct answer (red = marked and kidney, yellow = kidney not marked, blue = marked
  but not kidney). Then 6 scans follow, 3 with and 3 without the uncertainty heatmap, in a balanced order (the number
  in the id sets the order; use a multiple of 4 participants). Each scan has a time limit; the clock starts when the
  scan is ready and stops at "Done" or when the time is up. Then the scan is locked, saved, and a message says so.
  The corrected mask and a log (times, strokes, tools, heatmap use) are saved to
  `data/study/<study name>/sessions/<participant>/`. A session can be resumed: finished scans are skipped.
  Test runs use T ids (T01, ...), which the analysis ignores and `scripts/clean_test_data.py` deletes.
- **On screen:** a short task ("Fix the red outline so it covers the whole kidney, including any tumour. Press Done
  when finished.") and a colour key (red = the AI's outline, blue = where the AI is unsure, only when the heatmap is
  shown).
- **Demo queue:** the test scans ranked by the G1 score, most uncertain first. Heatmap always available, no time
  limit, and the ground truth can be shown.
- **Tools:** Add / Erase with a round brush (a circle shows its size), brush size, Undo (also ⌘Z), Move, two contrast
  presets, uncertainty on/off (only in the "with" condition), Reset view, Done. Slices: arrow keys ↑ ↓, the ▲ ▼
  buttons or the slider on the right (slice number below it; in the "with" condition the most uncertain slices are
  marked). Trackpad: press and drag to paint, two-finger swipe to move the image, pinch to zoom. No right-click or
  mouse wheel is needed.
- **Layers:** the mask is always drawn on top of the heatmap (a see-through fill with a solid outline). NiiVue's own
  slice shader puts overlays over the drawing, so `ui/viewer.js` gives NiiVue its own shader.
- **Heatmap:** voxel entropy, chosen on the KiTS development set because it points best at wrong voxels
  (`scripts/evaluate_heatmaps.py`).
- **Ground truth:** only the practice scan and the demo queue have it (`truth.nii.gz`); the server refuses it for
  study scans.
- **Tests:** `uv run pytest` checks the balancing, the scan selection and the server (files, heatmap and ground-truth
  access, saving, resuming). If Google Chrome is installed, it also drives the page in a headless Chrome like a
  participant (trusted mouse, trackpad and key input through the DevTools protocol, `tests/browser.py`): painting and
  erasing change the saved mask exactly where the brush went, undo works, moving and zooming do not paint, the mask
  is drawn above the heatmap, and the study flow (guide, practice, time limit, answer, scans without heatmap) works.
  On a Mac, headless Chrome uses the graphics card and the browser tests take about a minute.

## Analysing the user study (G2)

```bash
uv run python scripts/analyze_study.py --config configs/study.yaml   # after the study: tables + figure
uv run python scripts/power_study.py   --config configs/study.yaml   # planning: participants needed
```

- **Primary metric per corrected scan:** (Dice after − Dice before) / minutes. Dice before = the model's mask, Dice after
  = the participant's saved mask, both against the ground truth (used only here, never shown in the study scans).
- **Comparison:** per participant, the mean over the 3 scans with and the 3 scans without the heatmap; Wilcoxon
  signed-rank test on the differences (with − without), two-sided, with the rank-biserial correlation as effect size.
  Dice after, Dice gain and time are reported as secondary outcomes.
- **NASA-TLX** (Raw TLX: mean of 6 scales, 0–100) is the last step in the interface. With `study.tlx:
  after_each_block` it is asked after each condition, and the analysis then also compares workload with vs without.
- Only real participant ids (P01, ...) are analysed; test ids (T01, ...) and the practice scan are ignored. Output:
  `results/study_kits/study_{scans,participants,tests,tlx}.csv` and `figures/study_gain_per_min.png`.
- The analysis is tested on **simulated** sessions (`tests/simulated_sessions.py`), which exist only in temporary
  test folders; simulated output is named `SIMULATED_*` and can never be written to `results/`.
- **Power simulation** (`results/study_kits/power_simulation.csv`, `power_needed.csv`): for assumed standardised
  effect sizes d_z (mean / SD of the per-participant differences), the share of simulated studies where the Wilcoxon
  test finds the effect. These are assumptions for planning, not results.

## Layout

```
configs/        one YAML config per environment; configs/datasets/ describes each dataset
src/segreview/  shared code (config, data, weights, inference, augment, uncertainty, evaluation, figures, pipeline)
scripts/        runnable pipeline steps
jobs/           LSF job scripts for the DTU HPC (not tested on the cluster)
ui/             review interface (G2): index.html, app.js (flow), viewer.js (slice viewer), style.css, vendor/niivue
tests/          automatic tests (uv run pytest)
results/        results/<config name>/: small result files (csv, figures)
data/           scans, ground truth, model output (not in git)
models/         pretrained weights (not in git)
```
