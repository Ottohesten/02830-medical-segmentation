# Where to Look First

Prioritizing human review of AI medical image segmentation.
DTU 02830 Advanced Project in Digital Media Engineering, group 2.

**New to the project? Start with [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** (what it is, status, where everything is)
and [PROJECT_LOG.md](PROJECT_LOG.md) (dated log of every result, experiment, idea and decision, including what was dropped and why).
**Before you change code or run experiments, read [CONTRIBUTING.md](CONTRIBUTING.md)** (branches, locked results, logging).

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
  **Test runs always use `T01`, `T02`, ...**: same order as the P id with the same number, never analyzed.
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
evaluated, one or more rules for a bad segmentation, rank combinations, bootstrap) and `visualization`. All configs also run the 6 mm model as a second model (see below).

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
- **KiTS23** (`configs/datasets/kits23.yaml`): 489 CT scans of patients with kidney tumors, with kidney (1),
  tumor (2) and cyst (3) masks; scans from many referring hospitals. [kits-challenge.org](https://kits-challenge.org/kits23/),
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
  reviewed scans), each summarized by the area under the curve, all methods in one figure.
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

### Exploratory follow-up analyses (after the G1 lock)

Questions from the supervisor meeting of 2026-10-07, asked after the G1 choices were locked, so they are
**exploratory** and report the dev and test sets separately. Both scripts only read saved predictions (no inference)
and never change the locked G1 files. Settings: `exploratory:` in `configs/kits100.yaml`.

```bash
uv run python scripts/compare_models.py            --config configs/kits100.yaml   # ~10 min on a MacBook Air M5
uv run python scripts/simulate_voxel_correction.py --config configs/kits100.yaml   # ~10 min
uv run python scripts/explore_disagreement_scores.py --config configs/kits100.yaml  # ~5 min
```

- `compare_models.py` (`src/segreview/model_comparison.py`): **the 6 mm model alone** (Dice of both models per
  scan; stops if the 3 mm Dice differs from `dice.csv`; real finds and false alarms at the top of the locked queue),
  and **where the 3 mm and 6 mm masks disagree**: by ground-truth label, by distance to the 3 mm mask's edge
  (within `grid_band_mm` = the coarser grid; farther = a real difference), and which model is right.
  Outputs: `model_dice_3mm_vs_6mm*.csv`, `queue_top_3mm_vs_6mm.csv`, `model_disagreement_*.csv`,
  `figures/model_disagreement_where.png`.
- `simulate_voxel_correction.py` (`src/segreview/voxel_correction.py`): **does fixing the most uncertain voxels
  help?** In each scan the k % most uncertain voxels of the region (organ + border) are set to ground truth and the
  Dice gain is measured, for entropy, |p_3mm - p_6mm|, distance to the edge, random order and an oracle. Outputs:
  `voxel_correction_*.csv`, `figures/voxel_correction_gain.png`.
- `explore_disagreement_scores.py` (`src/segreview/disagreement_scores.py`): **scores from only the real
  differences** between the 3 mm and 6 mm masks (farther than `grid_band_mm` from the 3 mm edge, and/or only where
  the 6 mm model adds organ), in ml and per 3 mm volume. Candidates and the selection rule are fixed in
  `exploratory.disagreement_scores`; evaluated with the G1 functions per split, with paired bootstrap differences
  to the locked measure and entropy, and false alarms in the top of each queue. kits100 is development data for
  this question. Outputs: `exploratory_disagreement_*.csv`, `exploratory_queue_*.csv`,
  `figures/exploratory_review_curves_test.png`.
- Each script writes the machine, library versions and git commit to `<name>_machine.json` next to its results.

## Robustness timing on an RTX 5060 Ti

The shared environment pins torch 2.2.2 for Intel Macs. It cannot execute kernels on Blackwell GPUs such as
the RTX 5060 Ti. For Otto's robustness experiment, use the separate Linux/Python 3.12 environment below.
`requirements/robustness-gpu.txt` pins the same non-GPU packages plus torch 2.7.1 and torchvision 0.22.1
with CUDA 12.8. It leaves the shared `.venv/`, `pyproject.toml`, and `uv.lock` intact.

```bash
uv venv .venv-gpu --python .venv/bin/python
uv pip sync --python .venv-gpu/bin/python --torch-backend cu128 requirements/robustness-gpu.txt
PYTHONPATH=src .venv-gpu/bin/python scripts/benchmark_robustness.py --config configs/kits_robustness_timing.yaml
```

`PYTHONPATH=src` imports the project code without installing the main project's torch 2.2.2 dependency into
the GPU environment. Run these commands from the repository root, with access to the NVIDIA device.
The CUDA driver must work (`nvidia-smi`); an agent sandbox may hide the GPU even when the host driver works.
The cu128 flag is required when installing from this requirements file.

The benchmark runs the first two development IDs in sorted order with both models, each clean and with
`noise_60`. This is a timing example, not the strength selection for step 3. No ground truth or scores are
read. The benchmark measures both models on the noisy copies; the shared `run_second_model.py` now also handles
every configured perturbation, saving `m2_mask` and `m2_prob` under its variant folder.
Results go to `results/kits_robustness_timing/`: per-prediction `timings.csv`, `timing_summary.csv`, separate
model-load times, the config snapshot, machine metadata, and `runtime_projections.csv`.
Predictions go to `data/predictions/kits_robustness_timing/` and remain outside Git.

The synchronized prediction interval includes resampling, preprocessing, inference, and resampling back;
the total interval also includes reading, perturbing, and saving. First-use CUDA warm-up is included.
Projections estimate clean reference plus 3 or 4 perturbations on 20 or 80 scans from only two measured scans;
they are planning estimates, not confidence intervals or measured full-run times. The config caps the benchmark
at 30 minutes. Existing timings are protected; use `--overwrite` only for a deliberate complete rerun.

## Robustness strength selection on development scans

Use the GPU environment above. Step 3 runs only the 3 mm model on the original 20 development scans,
each clean and with 8 candidate perturbations (180 predictions). It evaluates Dice, not uncertainty scores.

```bash
PYTHONPATH=src .venv-gpu/bin/python scripts/select_robustness_strengths.py --config configs/kits_robustness_dev.yaml --plan
# Ask Otto before --run: the voxel-scaled estimate is about 40 minutes; Otto approved an 8-hour reserve cap.
PYTHONPATH=src .venv-gpu/bin/python scripts/select_robustness_strengths.py --config configs/kits_robustness_dev.yaml --run
```

The config fixes the candidates and an operational selection rule before outcomes are seen: at least 5 cases
lose 0.02 Dice, at least 5 retain Dice >= 0.83, and no more than 2 newly fall below Dice 0.50. The mildest eligible
strength of each type is suggested; a type with no eligible candidate is explicitly excluded from the suggestion.
Contrast strengths 0.8/0.9 remove 80/90% of contrast around the configured 40 HU level.

`--plan` saves the candidate list, rule, case IDs, and runtime proxy using image headers only. `--run` validates
dev-only scope before inference, resumes already saved predictions under an identical config, and writes
per-scan runtime, Dice, paired drops, candidate summaries, suggestions, and machine/config/code provenance to
`results/kits_robustness_dev/`. Predictions stay outside Git in `data/predictions/kits_robustness_dev/`.
`--summarize` repeats development Dice analysis from saved masks without inference. A changed candidate config
requires a new output name. The runtime estimate is a rough size-scaled proxy from two scans and one noise type;
other perturbations and Dice analysis add uncertainty.

Strength suggestions must be reviewed and written into the task, log, and final config before running uncertainty
analysis. Test scans remain outside strength selection. This is a development selection rule, not a change to locked G1 metrics.

The completed 2026-10-21 run took 44.39 minutes. The predeclared rule selects **lowres_8mm only**; tested noise
and contrast strengths have too few clear Dice drops. Lowres_12mm is also eligible, but 8 mm is milder.
`configs/kits_perturbed.yaml` records the locked choice and retains development-only scope (20 scans).
The candidate results remain intact. Per-scan Dice, candidate summary, evidence hashes, prediction manifest,
and the development plot are under `results/kits_robustness_dev/`; `report_source.py` there reproduces the
verification and figure from the repository root. Otto requested step 4 with the locked choice; the question
for Bjørn in task section 6 remains open for report scope. The current test question concerns resolution degradation.

## Robustness inference on the 80 test scans

`configs/kits_robustness_test.yaml` keeps the locked scientific settings and selects the original study's
100 IDs so scan indices remain stable. The runner filters to the 80 test IDs before inference. It runs
both models on clean and `lowres_8mm` images: **320 predictions / 640 mask and probability files**.
The archived `configs/kits_perturbed.yaml` and its strength-lock hash remain unchanged.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python scripts/run_robustness_test.py --config configs/kits_robustness_test.yaml --plan
# The original run was approved by Otto with an 8-hour cap; do not rerun completed inference.
PYTHONPATH=src MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python -u scripts/run_robustness_test.py --config configs/kits_robustness_test.yaml --run
# Check saved outputs without new inference:
PYTHONPATH=src MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python scripts/run_robustness_test.py --config configs/kits_robustness_test.yaml --verify
```

`--plan` reads image headers and existing development timings only. The main-model projection uses all 20 dev
scans for clean/lowres; second-model and I/O estimates use the earlier two-scan benchmark, with noise as the
lowres overhead proxy. Raw-voxel scaling is approximate and final verification is excluded from the estimate.
The plan and exact test IDs are in `results/kits_robustness_test/{run_plan.json,planned_cases.csv}`.

`--run` checks the original strength-lock/config/evidence hashes and the original case split before loading
the models. Both models receive identically seeded perturbations. It writes per-scan/per-variant times,
machine/driver/version/git/source provenance, wall time and peak memory; it resumes completed predictions
only with an identical config and source snapshot. Historical clean-only second-model timing files are
preserved with an explicit `clean` variant. No labels, Dice, uncertainty scores, or G1 evaluation are read or
computed by this runner; those belong to step 5.

Small artifacts go to `results/kits_robustness_test/`; predictions stay outside Git in
`data/predictions/kits_robustness_test/`. Final verification requires every timing pair and all 640 maps,
checks input/output shapes and affines, and records file SHA-256 hashes. It checks headers and compressed
file hashes; it does not re-evaluate segmentation contents. Locked `kits100` outputs remain protected.

The completed 2026-10-21 run took **41.54 minutes including verification** under the approved 8-hour cap.
All **320 predictions / 640 files** passed verification; prediction files total 303,793,357 bytes outside Git.
Peak allocated GPU memory was 3,649.76 MiB; peak process RSS was 8,051.12 MiB. Both final test suites passed:
113 passed, 36 skipped. Machine/config/source/input provenance, per-scan timings, `timing_summary.csv`,
`verification.json`, `post_run_checks.json`, and `prediction_manifest.csv` are in `results/kits_robustness_test/`.
`report_source.py` there regenerates the timing summary and provenance checks (from the repository root):

```bash
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python results/kits_robustness_test/report_source.py --config configs/kits_robustness_test.yaml
```

Step 5 has not started: no test labels, Dice, uncertainty scores, or ranking evaluation were read/computed.
The copied `strength_lock` metadata describes the development-lock moment; actual test-inference status is
recorded in `machine.json`. Development comparisons still require second-model clean/lowres maps for all 20 dev scans.

## Robustness Dice, score changes, and queues (step 5)

`configs/kits_robustness_eval.yaml` fixes the analysis before outcomes: original 20 dev / 80 test patients,
clean + lowres_8mm, locked primary model disagreement and secondary entropy, volume/random/oracle baselines.
The runner copies selected saved maps into its own prediction folder and predicts only the 40 missing
development second-model pairs. Neither completed inference experiment nor locked G1 output is overwritten.

Run the stages separately from the repository root, using `.venv-gpu/` (host GPU access for `--prepare`):

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python scripts/evaluate_robustness.py --config configs/kits_robustness_eval.yaml --plan
PYTHONPATH=src MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python -u scripts/evaluate_robustness.py --config configs/kits_robustness_eval.yaml --prepare
PYTHONPATH=src MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python -u scripts/evaluate_robustness.py --config configs/kits_robustness_eval.yaml --benchmark
# Read runtime_projection.json; ask Otto before any stage expected to exceed 30 minutes.
PYTHONPATH=src MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python -u scripts/evaluate_robustness.py --config configs/kits_robustness_eval.yaml --evaluate
```

`--plan` seals the config, source hashes, case IDs, analysis signs, and bootstrap policy before outcomes.
Each stage has a 30-minute cap. Stage metadata refuses accidental repeated execution; inspect status before
any deliberate recovery. `--benchmark` times scoring on two dev cases without retaining their scores and
times G1 on a synthetic largest queue. It writes a conservative runtime proxy, not a measured full-run result.

`--evaluate` computes scores with the existing G1 `scan_scores` from saved model maps only, before reading
labels. It then saves Dice per view and per model, paired Dice/score changes, degradation and correlation
summaries, and G1 queue metrics/curves. `dice_drop = clean - perturbed`; `score_change = perturbed - clean`,
so positive Spearman change correlation supports detection. Ordinary score-vs-Dice G1 rho retains its
negative-is-good convention. An undefined constant-input correlation is reported as NaN.

Clean, lowres, and mixed queues are reported separately for dev and test. Queue metrics use the same
`evaluate_methods` engine as `scripts/evaluate_g1.py`, the original two bad rules, 2,000 bootstrap samples,
1,000 random orders, 95% intervals, and seed 42. Mixed queues resample patient IDs with both views retained;
their asymptotic independent-row rho p-values are omitted. Default ungrouped G1 behavior is unchanged.
Bad flags are defined in the full queue and retained in bootstrap, as in the original G1 engine. Random
baseline intervals show spread across random orders. Top-10% shares and rank shifts handle ties fairly.

Outputs are under `results/kits_robustness_eval/`: `analysis_plan.json`, `*_machine.json`,
`imported_predictions.csv`, `prediction_manifest.csv`, `scores.csv`, `dice.csv`, `dice_per_model.csv`,
`paired_changes.csv`, `dice_degradation.csv`, `change_correlations.csv`, `g1_metrics.csv`, `g1_curves.csv`,
`queue_rank_changes.csv`, `queue_top_summary.csv`, and `verification.json`. Prediction maps stay outside Git
in `data/predictions/kits_robustness_eval/`. Step 6 will create the single report figure and short summary.

The completed step 5 preparation took **6.74 min** on GPU, followed by a **12.05 min** CPU evaluation
(30-minute caps; no extension). Final suites: **123 passed, 36 skipped** in both environments; eight
expected constant-input warnings come from the tiny synthetic tests. The artifact audit verifies 100 cases,
200 scores, 400 model Dice rows, 100 paired rows, 60 metrics, 800 saved maps, and 720 unchanged archived inputs.

On the 80 test cases, main-model median paired Dice loss is 0.03010 and Dice < 0.83 rises from 11 to 21.
Primary change rho is 0.215 [-0.036, 0.459]; mixed threshold review-AUC is 0.8086 versus volume 0.4648,
random 0.5030, oracle 0.9000. Entropy rises in 79/80 cases, but mixed AUC is similar (0.8068).
Under worst10 the primary lowres-vs-volume interval includes zero; the full results retain both bad rules.
Failure ranking and promoting degraded views are distinct outcomes. No method was changed after test results.

Reproduce the post-run audit and `key_results.json` extraction from saved artifacts:

```bash
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python results/kits_robustness_eval/report_source.py --config configs/kits_robustness_eval.yaml
```

`post_run_checks.json` checks exact cohort/variant/model coverage, source/config/stage hashes, prediction
hashes, and mixed patient grouping. It does not rerun model inference or bootstrap analysis.

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
heatmap, colors, brush sizes, trackpad behavior). The study runs in **Google Chrome** on a MacBook with a
**trackpad**; other browsers are not supported (painting does not work in Firefox).

```bash
uv run python scripts/prepare_study.py --config configs/study.yaml   # once: pick scans, write viewer files
uv run python scripts/run_ui.py        --config configs/study.yaml   # start the server and open the browser
```

- **Study mode:** enter a participant id (P01, P02, ...). The participant first sees a short guide (an example image of
  kidneys and a tumor, made from a development scan used nowhere else, and how to use the trackpad), then a
  practice scan, and after it the correct answer (red = marked and kidney, yellow = kidney not marked, blue = marked
  but not kidney). Then 6 scans follow, 3 with and 3 without the uncertainty heatmap, in a balanced order (the number
  in the id sets the order; use a multiple of 4 participants). Each scan has a time limit; the clock starts when the
  scan is ready and stops at "Done" or when the time is up. Then the scan is locked, saved, and a message says so.
  The corrected mask and a log (times, strokes, tools, heatmap use) are saved to
  `data/study/<study name>/sessions/<participant>/`. A session can be resumed: finished scans are skipped.
  Test runs use T ids (T01, ...), which the analysis ignores and `scripts/clean_test_data.py` deletes.
- **On screen:** a short task ("Fix the red outline so it covers the whole kidney, including any tumor. Press Done
  when finished.") and a color key (red = the AI's outline, blue = where the AI is unsure, only when the heatmap is
  shown).
- **Running the study:** [docs/study/](docs/study/) has the experimenter's [session guide](docs/study/session_guide.md),
  the [consent form](docs/study/consent_form.md), the [pilot test plan](docs/study/pilot_plan.md) and the
  [analysis plan](docs/study/analysis_plan.md) (locked before P01).
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
  (`scripts/evaluate_heatmaps.py`). `study.heatmap_display` sets how it is shown: `full` (default), `top` (only the
  `top_percent` most uncertain voxels near the kidney) or `contour` (a thin line around them). The pilot decides
  (see the pilot plan). The server computes the shown map from the prepared heatmap (no new preparation needed),
  records the setting per session (`sessions/<id>/session.json`) and in every scan log, and refuses to resume a
  session with a different setting.
- **Ground truth:** only the practice scan and the demo queue have it (`truth.nii.gz`); the server refuses it for
  study scans.
- **Tests:** `uv run pytest` checks the balancing, the scan selection and the server (files, heatmap and ground-truth
  access, saving, resuming). If Google Chrome is installed, it also drives the page in a headless Chrome like a
  participant (trusted mouse, trackpad and key input through the DevTools protocol, `tests/browser.py`): painting and
  erasing change the saved mask exactly where the brush went, undo works, moving and zooming do not paint, the mask
  is drawn above the heatmap, and the study flow (guide, practice, time limit, answer, scans without heatmap) works.
  On a Mac, headless Chrome uses the graphics card and the browser tests take about a minute.

## Analyzing the user study (G2)

```bash
uv run python scripts/analyze_study.py --config configs/study.yaml   # after the study: tables + figure
uv run python scripts/power_study.py   --config configs/study.yaml   # planning: participants needed
```

- **Primary metric per corrected scan:** (Dice after − Dice before) / minutes. Dice before = the model's mask, Dice after
  = the participant's saved mask, both against the ground truth (used only here, never shown in the study scans).
- The script carries out the [analysis plan](docs/study/analysis_plan.md), which is locked before the first
  participant; run it once, after the last session. Settings: `analysis:` in `configs/study.yaml`.
- **Comparison:** per participant, the mean over the 3 scans with and the 3 scans without the heatmap; Wilcoxon
  signed-rank test on the differences (with − without), two-sided, α = 0.05, with an exact sign-flip p-value.
  Effect sizes: the rank-biserial correlation and the mean difference with a 95 % bootstrap interval.
  **H1** (primary): gain per minute is higher with the heatmap. **H2**: raw NASA-TLX is lower with it; tested only if
  H1 is supported (fixed order). Dice after, Dice gain, time, timeouts, unchanged scans and order are exploratory.
- **NASA-TLX** (Raw TLX: mean of 6 scales, 0–100) is asked in the interface after each condition block
  (`study.tlx: after_each_block`, the default), so the analysis can compare workload with vs without the heatmap,
  paired per participant. With `after_session` it is asked once, at the end (and H2 cannot be tested).
- Only complete sessions count; incomplete ones are listed in `study_exclusions.csv`. Scans where the time ran out
  and scans the participant left unchanged are kept (and counted).
- All participants must have seen the same heatmap display; the analysis stops otherwise.
- Only real participant ids (P01, ...) are analyzed; test ids (T01, ...) and the practice scan are ignored. Output:
  `results/study_kits/study_{tests,participants,scans,exclusions,conditions,order,tlx}.csv` and
  `figures/study_gain_per_min.png`.
- The analysis is tested on **simulated** sessions (`tests/simulated_sessions.py`), which exist only in temporary
  test folders; simulated output is named `SIMULATED_*` and can never be written to `results/`.
- **Power simulation** (`results/study_kits/power_simulation.csv`, `power_needed.csv`): for assumed standardized
  effect sizes d_z (mean / SD of the per-participant differences), the share of simulated studies where the Wilcoxon
  test finds the effect. These are assumptions for planning, not results.

## G3 (stretch): fine-tuning on corrected scans — plan only

[docs/g3_plan.md](docs/g3_plan.md) describes the experiment (simulated corrections of the top of the review queue,
fine-tuning, evaluation on the `kits100` test set) and what needs the HPC. It is not built. A feasibility smoke test
checks that nnU-Net's own trainer can fine-tune the pretrained model on this setup and measures time and memory per
training step (2 KiTS scans that are not in `kits100`; nothing is saved as a model):

```bash
uv run python scripts/g3_smoke_test.py --config configs/g3_smoke.yaml   # -> results/g3_smoke/
```

## Layout

```
configs/        one YAML config per environment; configs/datasets/ describes each dataset
src/segreview/  shared code (config, data, weights, inference, augment, uncertainty, evaluation, figures, pipeline,
                model_comparison, voxel_correction, machine)
scripts/        runnable pipeline steps
jobs/           LSF job scripts for the DTU HPC (not tested on the cluster)
ui/             review interface (G2): index.html, app.js (flow), viewer.js (slice viewer), style.css, vendor/niivue
tests/          automatic tests (uv run pytest)
results/        results/<config name>/: small result files (csv, figures)
data/           scans, ground truth, model output (not in git)
models/         pretrained weights (not in git)
```
