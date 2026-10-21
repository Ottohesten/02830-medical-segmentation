# Task: Does our ranking still work when the scans get worse?

| | |
|---|---|
| **Owner** | Otto |
| **Contact / lead** | Bjørn |
| **Branch** | `otto/robustness` |
| **Created** | 2026-10-20 |
| **Status** | Step 5 complete; resolution degradation evaluated; report figure/summary next |
| **Target** | Steps 1–4 by 2026-10-30, all steps by 2026-11-13 |

This document is the task description **and** the working notes. Whoever works on the task (with or without an
assistant) updates it after every step: tick the box, write the date and what was done, and update
"Where we are" at the bottom. Then anyone in the group, or a new chat, can see the status and continue.

---

## 1. The idea in plain words

AI models often fail when the data look different from what they were trained on: a noisier scanner, a lower
resolution, a different contrast. This is called *domain shift*. Our ranking (G1) puts the scans the model most
likely got wrong at the top of the queue. So far we have only tested it on the scans as they are.

**Question:** If we make the scans worse on purpose, (a) does the model's segmentation get worse, and (b) does our
uncertainty score notice it, so the damaged scans move up the queue?

If yes, that is strong support for the method: it reacts to exactly the situation where a reviewer is most needed.
If no, that is an important limitation for the report. Both outcomes are useful.

This was part of the original plan (2026-09-23: "natural variation plus controlled perturbations, reported
separately") but was only tried on 3 spleen scans, never on KiTS. See [PROJECT_LOG.md](../../PROJECT_LOG.md).

## 2. Starting points

How the code works is in [README.md](../../README.md) (perturbations: section "Running G1"); the general rules are in
[CONTRIBUTING.md](../../CONTRIBUTING.md). Specific to this task:

- The perturbation code exists (`perturb` in `src/segreview/augment.py`; example config: `configs/local.yaml`) but has
  only been run on 3 spleen scans: `lowres_8mm` clearly lowered Dice, `noise_60` a little, `contrast_60` not at all.
- Evaluate with the locked G1 score (`model_diff_mean_region`) and, as secondary, entropy (`entropy_mean_region`).
- **Own config and folders** (e.g. `configs/kits_perturbed.yaml`, `name: kits_perturbed`). Never write to
  `results/kits100/` or `data/predictions/kits100/`. New model runs are needed, in the new folders only.
- **Choose the strengths on the development set (20 scans) first**, log the choice, then run the test set.

## 3. Steps

Tick each box when done and write the date and a short note (numbers + file paths) under it.

- [x] **1. Get started.** Install the project (`uv sync`), run `uv run pytest`, download the KiTS data and model
  weights (`scripts/download_data.py`, `scripts/download_model.py` with `configs/kits100.yaml`). Read
  [PROJECT_OVERVIEW.md](../../PROJECT_OVERVIEW.md) and be able to explain G1 in two sentences.
  *Notes:*
  - 2026-10-20: Read the overview, contributing rules, and this task. `git pull --ff-only` reported
    "Already up to date" at `eb1cef2`; created `otto/robustness`. No local environment, data, or model folders
    were present at the initial check.
  - 2026-10-20: Otto reported running `uv sync`; verified `.venv/` with Python 3.12.15, torch 2.2.2+cu121,
    TotalSegmentator 2.18.0, and pytest 9.1.1 on Linux x86_64 (kernel 7.2.10-1-cachyos, glibc 2.44).
    PyTorch reports CUDA build 12.1 but `torch.cuda.is_available()` is false in the current environment.
  - 2026-10-20: Ran `UV_OFFLINE=1 UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/otto-robustness-uv-cache uv run pytest -ra`
    against the installed environment, without downloads or dependency changes: **72 passed, 36 skipped,
    0 failed in 3.11 s** (108 collected). Of the skips, 30 require a prepared study and 6 require Google Chrome.
    See `tests/test_study.py`, `tests/test_study_analysis.py`, `tests/test_ui_server.py`, and
    `tests/test_ui_browser.py`.
  - 2026-10-20: Otto approved the data and model downloads. Ran `uv run python -u scripts/download_data.py
    --config configs/kits100.yaml` and `uv run python -u scripts/download_model.py --config configs/kits100.yaml`
    in the existing environment (`UV_OFFLINE=1 UV_NO_SYNC=1`, with the uv cache under `/tmp/`). Both exited 0.
    The scripts needed network access outside the sandbox after sandbox DNS lookups failed.
  - 2026-10-20: Downloaded and verified **100 CT images + 100 labels** in `data/raw/kits23/`:
    20 development and 80 test cases, seed 42. Images total 9,904,809,672 bytes; labels 33,410,151 bytes.
    All pairs have matching shapes and affines; all 200 gzip streams pass CRC and payload-length checks.
    No partial downloads remain. Model tasks **297 (3 mm)** and **298 (6 mm)** are in
    `models/totalsegmentator/nnunet/results/`, each with fold `[0]`, `plans.json`, `dataset.json`, and
    `fold_0/checkpoint_final.pth`; the full model directory is 331,490,973 bytes.
    Machine: AMD Ryzen 5 7500F, Linux, Python 3.12.15, torch 2.2.2+cu121; CUDA unavailable in this environment.
    Verification took 85.78 s. File paths, byte counts, SHA-256 hashes, config hash, source commit, and machine
    details are in `results/robustness_setup/download_verification.json` (setup evidence only).
    No inference or evaluation was run, and no locked G1 files were changed. Data and weights remain outside git.
  - 2026-10-20: Committed setup as `b2269e9` (`Complete Otto robustness setup`). Configured repository-local
    Git identity from Otto's existing commits and authenticated GitHub profile because Git had no identity.
    Push could not authenticate over HTTPS; the connected GitHub integration also rejected write access
    (HTTP 403, "Resource not accessible by integration"). No remote branch or pull request was created.
    After GitHub authentication is configured, run `git push -u origin otto/robustness`, then open a pull request
    into `main` for group review. A follow-up local commit records this publication blocker.
  - 2026-10-20: Otto pushed the branch. Verified with `git ls-remote --heads origin otto/robustness` that
    GitHub has `f388b74`, matching local HEAD and including `b2269e9`; the push blocker is resolved.
    `git pull --ff-only` reported "Already up to date" and offline `uv sync --locked` checked 133 packages.
    Rechecked tests before publishing the status update: **72 passed, 36 skipped, 0 failed in 1.97 s**.
    Creating a draft PR through the connected integration still returns HTTP 403; a PR into `main` remains
    to be opened through GitHub for group review.
    The follow-up status-note commit still needs `git push origin otto/robustness` from Otto's terminal:
    the assistant's noninteractive Git command cannot access his HTTPS credentials.
  - G1 in two sentences: We run the 3 mm and 6 mm TotalSegmentator models on each CT scan and measure their
    disagreement about the kidneys. Scans with more disagreement go earlier in the review queue so a person
    can check likely segmentation errors first, without ground truth being used to rank them.

- [x] **2. Measure the time.** Run the 3 mm and 6 mm models on 2 development scans with one perturbation on your
  machine. Write down seconds per scan per model, and estimate the time for 20 scans × 3–4 perturbations and for
  80 scans. If it is too slow (more than a night), ask Bjørn; Frederik's GPU machine is an option.
  *Notes:*
  - 2026-10-21: PR [#1](https://github.com/Ottohesten/02830-medical-segmentation/pull/1) exists; the step 1
    follow-up commit `366dcf1` is pushed. `git pull --ff-only` reported "Already up to date"; offline
    `uv sync --locked` checked 133 packages before starting step 2.
  - 2026-10-21: Corrected the GPU diagnosis: the sandbox hides the NVIDIA device, while host `nvidia-smi`
    works (RTX 5060 Ti, 16,311 MiB, driver 615.78.08). Host torch 2.2.2+cu121 detects the GPU but cannot execute
    kernels on its `sm_120` architecture. Otto approved a separate GPU environment and downloads.
    `.venv-gpu/` now uses torch 2.7.1+cu128 / torchvision 0.22.1+cu128; tensor and Conv3d checks pass.
    Pinned packages: `requirements/robustness-gpu.in` and `requirements/robustness-gpu.txt`; installation
    requires `--torch-backend cu128`. `uv pip check` verified 136 compatible packages. Shared `.venv/`,
    `pyproject.toml`, and `uv.lock` are preserved. Evidence: `results/kits_robustness_timing/gpu_compatibility.json`.
  - 2026-10-21: Ran `scripts/benchmark_robustness.py` with `configs/kits_robustness_timing.yaml` on CUDA,
    6 CPU threads, seed 42, `nnUNet_compile=false`, with a 30-minute cap. Chose the first two sorted dev IDs
    (`case_00040`, `case_00051`) without inspecting Dice or scores. Ran both models on clean and `noise_60`
    copies (Gaussian noise SD 60 HU, timing example only): **8 predictions in 68.90 s**.
    Per-scan prediction times (including resampling and preprocessing, excluding load/perturb/save):

    | Scan | Clean 3 mm | Noise 3 mm | Clean 6 mm | Noise 6 mm |
    |---|---:|---:|---:|---:|
    | `case_00040` | 23.75 s | 6.64 s | 4.55 s | 4.33 s |
    | `case_00051` | 3.45 s | 2.68 s | 1.61 s | 1.70 s |

    First prediction per model includes CUDA warm-up. Canonical shapes: 512 × 512 × 207 and 512 × 512 × 68.
    Peak allocated GPU memory was 2,503.01 MiB; process peak RSS was 3,907.35 MiB (a cumulative process peak).
    Machine: AMD Ryzen 5 7500F / RTX 5060 Ti, Linux 7.2.10-1-cachyos, Python 3.12.15, torch 2.7.1+cu128,
    CUDA 12.8. Exact rows and machine/config/code hashes: `results/kits_robustness_timing/timings.csv`,
    `machine.json`, `run_config.yaml`, `model_load_times.csv`, and `run_log.txt`.
  - 2026-10-21: **Planning estimates**, not measured full runs: clean reference + 3 / 4 perturbations,
    both models, including loading/perturbing/saving and loading each model once:
    **20 scans: 16.95 / 20.45 min; 80 scans: 67.40 / 81.39 min**.
    Formula: model-load time + N × (mean clean time for both models + P × mean noisy time for both models).
    These use only two scans and one perturbation, include first-use warm-up, and exclude later analysis;
    scan size and other perturbation types can change runtime. Source: `results/kits_robustness_timing/runtime_projections.csv`.
    No overnight/GPU-machine request to Bjørn is needed based on this timing sample. Ask Otto before any
    actual run expected to exceed 30 minutes, including the projected 80-scan runs.
  - 2026-10-21: Verified all 8 timing rows, matching saved config/script hashes, and 16 prediction maps with
    matching grids in `data/predictions/kits_robustness_timing/` (outside Git). No test scans, ground truth,
    Dice, or uncertainty scores were read. Locked G1 files are unchanged.
    Added 9 synthetic tests for dev/test isolation, locked-folder protection, and complete runtime estimates;
    full suites: **81 passed, 36 skipped, 0 failed** in both environments (2.28 s shared; 2.53 s GPU).
    Evidence: `results/kits_robustness_timing/tests_main.txt` and `tests_gpu.txt`; usage is in `README.md`.

- [x] **3. Choose the strengths on the development set (20 scans).** Try a few strengths per type, e.g.
  noise 30/60/120 HU, lowres 6/8/12 mm, contrast 0.8/0.9 (0.6 had no effect on spleen). Goal: perturbations that
  clearly lower Dice for some scans but do not destroy every scan. **Before looking at how the uncertainty scores
  do**, write the chosen list and the reason in the log and in the config. Keep the clean scans as the reference.
  *Notes:*
  - 2026-10-21: Confirmed step 2 commit `6f1bc0a` is pushed on `otto/robustness`; pull was up to date and
    offline `uv sync --locked` checked 133 packages. Prepared `configs/kits_robustness_dev.yaml`
    (`name: kits_robustness_dev`, dev only, seed 42, CUDA / 6 CPU threads) and
    `scripts/select_robustness_strengths.py`. No candidate inference or Dice analysis has started.
  - 2026-10-21: Candidates, fixed before outcomes: noise SD 30/60/120 HU; lowres 6/8/12 mm;
    contrast removal fractions 0.8/0.9 (retain 20/10% around 40 HU). Include clean references:
    **20 dev scans × 9 variants = 180 predictions of the 3 mm model**. The 6 mm model and uncertainty
    scores are not needed for choosing strengths and will not run in this phase.
  - 2026-10-21: Predeclared operational rule: at least **5** scans lose **0.02** Dice or more;
    at least **5** remain at Dice ≥ **0.83**; at most **2** newly fall below Dice **0.50** (counted only
    when clean Dice was ≥ 0.50). Suggest the mildest eligible strength per type; explicitly report a type
    with no eligible candidate. These are working criteria for useful degradation, not new G1 metrics.
    No uncertainty-score performance will be inspected before the final strength list is logged.
  - 2026-10-21: Ran `--plan` only. `results/kits_robustness_dev/run_plan.json` and `planned_cases.csv`
    record candidates, selection rule, all 20 dev IDs/voxel counts, and the timing proxy. Using the two
    measured noisy 3 mm times from step 2 gives an unscaled **18.27 min**, or **39.78 min** after scaling
    by the dev/sample mean raw-voxel ratio **2.1775**. This is a rough planning proxy, not a measured run;
    different perturbations and later Dice analysis add uncertainty. Configured cap: **90 min**.
    The exact config is preserved in `results/kits_robustness_dev/run_config.yaml`.
    **Waiting for Otto's explicit approval before starting this run expected to exceed 30 minutes.**
  - 2026-10-21: Added 11 synthetic tests for exact dev-set isolation, locked output protection, paired Dice
    drops, new failures, weakest-eligible selection, and rejection of missing/invalid/test rows:
    `tests/test_robustness_selection.py` (11 passed in 0.06 s). Implementation: `src/segreview/robustness.py`.
    Both full preflight suites also passed: **92 passed, 36 skipped, 0 failed** (1.92 s shared; 1.99 s GPU).
    Evidence: `results/kits_robustness_dev/preflight_tests_main.txt` and `preflight_tests_gpu.txt`.
  - 2026-10-21: Otto approved the 180-prediction dev run and Dice comparison, explicitly allowing
    an overnight runtime of 8 hours or more. Increased only the runtime cap from 90 to **480 minutes**
    before any candidate inference; candidates and selection criteria are unchanged. Original preapproval
    plan/config are preserved as `results/kits_robustness_dev/preapproval_run_{plan.json,config.yaml}`;
    current `run_plan.json` and `run_config.yaml` reflect the approved cap. This is a reserve limit;
    the rough time estimate remains 39.78 min. Step 4 is not being started.
  - 2026-10-21: **Completed all 180 predictions and development Dice in 2,663.65 s (44.39 min)** with
    `scripts/select_robustness_strengths.py --config configs/kits_robustness_dev.yaml --run` in `.venv-gpu/`.
    Machine: AMD Ryzen 5 7500F, RTX 5060 Ti, driver 615.78.08, Linux 7.2.10-1-cachyos, Python 3.12.15,
    torch 2.7.1+cu128 / CUDA 12.8, 6 CPU threads, seed 42. Peak allocated GPU memory 3,099.21 MiB;
    process peak RSS 7,345.41 MiB. Provenance: `results/kits_robustness_dev/{machine.json,run_attempts.json,
    run_config.yaml,run_log.txt,inference_times.csv}`. Metadata records baseline commit `6f1bc0a`, dirty state,
    and exact source/config SHA-256 hashes; the run code is committed with these results.
  - 2026-10-21: Clean development median Dice **0.9033**, with **4/20** below 0.83. Candidate results:

    | Candidate | Median Dice | Median paired drop | Drop ≥ 0.02 | Still ≥ 0.83 | New < 0.50 | Rule outcome |
    |---|---:|---:|---:|---:|---:|---|
    | noise_30 | 0.90770 | -0.00115 | 0/20 | 17/20 | 0 | Too few clear drops |
    | noise_60 | 0.91300 | -0.00350 | 2/20 | 16/20 | 0 | Too few clear drops |
    | noise_120 | 0.91600 | -0.00485 | 2/20 | 16/20 | 0 | Too few clear drops |
    | lowres_6mm | 0.88495 | 0.00825 | 4/20 | 15/20 | 0 | Too few clear drops |
    | **lowres_8mm** | **0.86700** | **0.02795** | **14/20** | **14/20** | **0** | **Selected: mildest eligible lowres** |
    | lowres_12mm | 0.80480 | 0.10230 | 20/20 | 8/20 | 2 | Eligible; 8 mm is milder |
    | contrast_80 | 0.90620 | -0.00270 | 0/20 | 16/20 | 0 | Too few clear drops |
    | contrast_90 | 0.90860 | -0.00235 | 0/20 | 16/20 | 0 | Too few clear drops |

    Positive paired drop means Dice became worse on that scan; a negative value means improvement.
    Sources: `results/kits_robustness_dev/{dice.csv,dev_dice_drops.csv,dev_degradation_summary.csv}`.
    Under lowres_8mm, 6/20 are below 0.83 (2 newly bad), while 14/20 remain above the threshold.
  - 2026-10-21: **Locked choice: `lowres_8mm` only**, following the unchanged predeclared rule. Noise and
    contrast have no eligible candidate at the tested strengths; do not force them into the test experiment.
    Lowres_12mm is stronger but is not selected because 8 mm already meets the rule. This narrows the current
    test question to resolution degradation; it does not establish general noise/contrast robustness.
    Recorded before any score computation or test inference in `configs/kits_perturbed.yaml` and
    `results/kits_robustness_dev/strength_lock.json`, including evidence hashes, excluded types, and locked
    primary/secondary scores. The final config keeps only 20 dev scans; it does not unlock the test set.
    Candidate config/results remain intact in their own folders. See the scope question for Bjørn in section 6.
  - 2026-10-21: Verified **180 timing rows, 180 dev Dice rows, and 360 prediction files** (252,949,887 bytes)
    with matching mask/probability grids, code/config hashes, and no score files or test Dice rows.
    Evidence: `results/kits_robustness_dev/{verification.json,prediction_manifest.csv}`.
    Predictions stay in `data/predictions/kits_robustness_dev/` outside Git. Figure:
    `results/kits_robustness_dev/figures/dev_dice_degradation.png`; reproducible verification/figure source:
    `results/kits_robustness_dev/report_source.py`. Locked G1 files and shared environment remain unchanged.
  - 2026-10-21: Final config validated with 20 dev cases and only `lowres_8mm`; both full suites pass:
    **92 passed, 36 skipped, 0 failed** (2.23 s shared; 2.26 s GPU). Skips remain 30 study-dependent and
    6 Chrome-dependent tests. Outputs: `results/kits_robustness_dev/tests_main.txt` and `tests_gpu.txt`.
  - 2026-10-21: Step 3 is committed locally. The assistant's noninteractive `git push` still cannot read
    a GitHub username; Otto must run `git push origin otto/robustness` in his authenticated terminal to
    publish the completed step and update PR #1.

- [x] **4. Run the test set (80 scans)** with the chosen perturbations (overnight or on the GPU).
  *Notes:*
  - 2026-10-21: Otto requested step 4. Proceed with the existing locked `lowres_8mm` choice; no further
    strength search or score selection. The experiment concerns resolution degradation only. No answer
    from Bjørn is recorded for the report-scope question in section 6.
  - 2026-10-21: Prepared `configs/kits_robustness_test.yaml` and `scripts/run_robustness_test.py`.
    The config selects the original 100 study IDs to preserve scan indices; the runner restricts inference
    to the original **80 test IDs**, clean and `lowres_8mm`, with both models: **320 predictions / 640 files**.
    Outputs: `results/kits_robustness_test/` and `data/predictions/kits_robustness_test/` (predictions outside Git).
    `configs/kits_perturbed.yaml`, its archived strength lock, and locked G1 files remain unchanged.
  - 2026-10-21: Extended `src/segreview/pipeline.py` so the second model predicts every configured variant,
    with the same full-list perturbation index as the main model. Both steps accept optional split filtering.
    Second-model completion requires both mask and probability; timings use case + variant keys and preserve
    historical clean-only rows. `src/segreview/robustness.py` and the runner reject altered scientific
    settings, case selection/split, archived evidence hashes, or protected output paths before inference.
  - 2026-10-21: Ran `--plan` only (image headers and existing dev timings; no labels or model inference).
    **Planning estimate: 39.66 min**, excluding final verification; **proposed reserve cap: 480 min**.
    The main-model estimate uses all 20 dev scans for the selected variants; the second model and I/O use
    a two-scan clean/noise proxy scaled by raw voxel counts. This is an approximate runtime plan, not a result.
    Exact IDs, sizes, estimate components, config and source/input hashes:
    `results/kits_robustness_test/{run_plan.json,planned_cases.csv,planning_image_sizes.csv,run_config.yaml}`.
  - 2026-10-21: Added **21 synthetic tests** for isolation, locked choices, identical model inputs,
    original seed indices, resuming partial second-model output, legacy timing migration, runtime arithmetic,
    and incomplete/misaligned output rejection. Complete suites: **113 passed, 36 skipped, 0 failed**
    (2.95 s shared, 3.06 s GPU). Evidence: `results/kits_robustness_test/preflight_tests_{main,gpu}.txt`.
    A separate tiny Conv3d smoke test passed on RTX 5060 Ti / driver 615.78.08 / torch 2.7.1+cu128 /
    CUDA 12.8; `results/kits_robustness_test/preflight_gpu.json`. No test images were used by the smoke test.
  - 2026-10-21: Otto explicitly approved the prepared run: "Ja, start med loft på 8 timer".
    Kept the existing **480-minute cap**, strengths, scores, and case split unchanged.
    Approval evidence: `results/kits_robustness_test/run_approval.json`. Executed
    `PYTHONPATH=src MPLCONFIGDIR=/tmp/otto-robustness-mpl .venv-gpu/bin/python -u scripts/run_robustness_test.py --config configs/kits_robustness_test.yaml --run`
    with host GPU access; exit 0. Console output: `results/kits_robustness_test/run_log.txt`.
    Normalized terminal progress line endings and trailing spaces for the text log committed to Git.
  - 2026-10-21: **Completed 320 predictions and verification in 2,492.40 s / 41.54 min** (cap not approached).
    Main-model wall interval: **1,340.81 s**; second-model interval: **1,088.27 s**;
    remaining verification interval: **63.32 s**. Each model predicted all 80 clean and 80 lowres_8mm scans.
    Per-scan prediction-time medians (exclude model setup and load/perturb/save):

    | Model | Clean | Lowres 8 mm | Predictions |
    |---|---:|---:|---:|
    | 3 mm (`fast`) | 4.420 s | 4.565 s | 160 |
    | 6 mm (`fastest`) | 2.890 s | 2.745 s | 160 |

    Exact rows/summary: `results/kits_robustness_test/{inference_times.csv,second_model_times.csv,timing_summary.csv}`.
    Machine: AMD Ryzen 5 7500F, RTX 5060 Ti, driver 615.78.08, Linux 7.2.10-1-cachyos,
    Python 3.12.15, torch 2.7.1+cu128 / CUDA 12.8, 6 CPU threads, seed 42.
    Peak allocated GPU memory **3,649.76 MiB**, process peak RSS **8,051.12 MiB**.
    `results/kits_robustness_test/{machine.json,run_attempts.json,run_config.yaml,run_plan.json}` records
    baseline commit `e608d92`, uncommitted run-code state, and exact config/source/input hashes.
    The run code is committed with these results.
  - 2026-10-21: **Verification passed:** exact 80-case × 2-variant × 2-model timing coverage,
    **640 prediction files / 303,793,357 bytes**, matching input/output shapes and affines, per-file SHA-256,
    and config/source hashes matching the approved plan. Evidence:
    `results/kits_robustness_test/{verification.json,prediction_manifest.csv,post_run_checks.json}`.
    `report_source.py` in that folder reproduces the timing summary and provenance checks with
    `PYTHONPATH=src:. .venv-gpu/bin/python results/kits_robustness_test/report_source.py --config configs/kits_robustness_test.yaml`.
    Predictions remain outside Git in `data/predictions/kits_robustness_test/`.
    No ground truth, Dice, uncertainty scores, or ranking evaluation was read/computed in this step.
    Archived development evidence, shared environment, and locked G1 config/results remain unchanged.
  - 2026-10-21: Final suites: **113 passed, 36 skipped, 0 failed** in both environments
    (2.76 s shared, 2.47 s GPU). Evidence: `results/kits_robustness_test/tests_{main,gpu}.txt`.
    Next: publish this step on `otto/robustness` to update PR #1, then explain step 5.
    Use Otto's authenticated terminal for the push if noninteractive HTTPS authentication is still unavailable.

- [x] **5. Evaluate.** With the existing G1 evaluation (`scripts/evaluate_g1.py`), report clean and perturbed
  separately, for development and test:
  - How much does Dice drop per perturbation (median, and how many scans become "bad", Dice < 0.83)?
  - Does the score rise when Dice falls? Per scan: change in score vs. change in Dice from clean to perturbed
    (Spearman ρ), for the locked score and for entropy.
  - Mixed queue: clean and perturbed scans together. Do the damaged scans move to the top? Review-AUC vs. the
    volume baseline, random and oracle, as in G1.
  *Notes:*
  - 2026-10-21: Otto requested step 5. Prepared `configs/kits_robustness_eval.yaml`,
    `scripts/evaluate_robustness.py`, and `src/segreview/robustness_evaluation.py`; own output folders
    `results/kits_robustness_eval/` and `data/predictions/kits_robustness_eval/`.
    Fixed analysis saved before outcomes in `analysis_plan.json` / `run_config.yaml` in that result folder.
    Use only `lowres_8mm`, primary `model_diff_mean_region`, secondary `entropy_mean_region`, and volume,
    random, oracle baselines. Keep both original G1 bad rules; Dice < 0.83 is the task's fixed bad threshold.
  - 2026-10-21: Sign convention fixed: `dice_drop = clean - perturbed`,
    `score_change = perturbed - clean`; a positive Spearman rho supports sensitivity to deterioration.
    Report median paired drop and clean/perturbed medians separately, bad/new-bad/recovered counts,
    for both models and both splits. Report paired score-change correlations with patient bootstrap CIs.
  - 2026-10-21: Reuse the existing G1 `evaluate_methods` engine (also called by `scripts/evaluate_g1.py`)
    for clean, lowres, and mixed queues, separately for dev/test. Use existing 2,000 bootstrap samples,
    1,000 random orders, 95% intervals, seed 42. Extended the engine with optional patient grouping;
    default G1 behavior is unchanged. In mixed queues, resample case IDs with both views together;
    omit independent-row rho p-values. Bad flags follow the existing full-queue G1 definition through
    bootstrap. Queue top 10% and average-rank shifts use fair ties, fixed before outcomes.
  - 2026-10-21: Preparation copies/hash-checks 720 selected maps (80 test × 2 variants × 4 maps;
    20 dev × 2 variants × 2 main-model maps), then predicts only the missing **40 dev second-model pairs**.
    Their rough size-scaled estimate is **6.61 min**, from step 4's measured second-model wall time and
    recorded dev/test voxel totals. Each stage has a **30-minute cap**; ask Otto before longer runs/extensions.
    Then time GT-free scoring on the two configured dev cases plus a synthetic G1 queue to project the CPU
    stage before reading test scores or labels. No new downloads or test inference are needed.
  - 2026-10-21: Added 10 synthetic tests for paired signs/counts, incomplete/mismatched pairs,
    patient bootstrap, split-specific queues, unchanged G1 point AUCs, GT-free score equivalence, and locked
    score validation. Shared suite: **123 passed, 36 skipped**, 8 expected constant-input warnings from tiny
    synthetic correlations, 2.63 s. `results/kits_robustness_eval/preflight_tests_main.txt`.
    Step remains open pending preparation, timing, evaluation, final verification, and publication.
  - 2026-10-21: Preparation completed in **404.57 s / 6.74 min**: 720 archived input maps copied and
    checksum-verified, 40 missing dev second-model pairs inferred, **800 maps** in the new experiment.
    `results/kits_robustness_eval/{prepare_machine.json,prepare_log.txt,imported_predictions.csv,
    second_model_times.csv,prediction_manifest.csv}`. Peak allocated GPU **496.21 MiB**, RSS **7,241.65 MiB**.
  - 2026-10-21: CPU benchmark completed before test scoring/labels: projected evaluation **15.34 min**
    including a 1.5 runtime safety factor. Size-scaled score proxy 315.49 s; synthetic largest-queue evaluation
    8.16 s × 12 analyses; Dice allowance 2 s × 100 cases. This is a timing estimate, not an accuracy result.
    Evidence: `results/kits_robustness_eval/{benchmark_times.csv,benchmark_machine.json,runtime_projection.json}`.
    Proceed with the fixed evaluation under its existing **30-minute cap**; no longer-run approval is needed
    for this estimate. No choices changed after benchmarking.
  - 2026-10-21: **Evaluation completed in 722.97 s / 12.05 min**, within the 30-minute cap;
    the benchmark took 12.13 s. Scored all model maps without labels first, then read labels for Dice.
    CPU: AMD Ryzen 5 7500F, Linux, Python 3.12.15, torch 2.7.1+cu128 / CUDA build 12.8, 6 threads, seed 42.
    CPU evaluation used no GPU; peak RSS **6,413.74 MiB**. GPU preparation used RTX 5060 Ti /
    driver 615.78.08. Separate stage metadata records baseline commit `6fde679`, dirty run-code state,
    exact config/source hashes, machine/library versions, time, and memory in
    `results/kits_robustness_eval/{prepare,benchmark,evaluate}_machine.json`.
  - 2026-10-21: Dice results (paired drop is the median of each scan's clean minus perturbed Dice,
    not the difference between the two medians):

    | Split | Model | Clean median | Lowres median | Median paired drop | Bad clean → lowres | Newly bad |
    |---|---|---:|---:|---:|---:|---:|
    | Dev (20) | 3 mm | 0.90330 | 0.86700 | 0.02795 | 4 → 6 | 2 |
    | Dev (20) | 6 mm | 0.89510 | 0.89415 | 0.00745 | 7 → 7 | 0 |
    | Test (80) | 3 mm | 0.90955 | 0.87865 | 0.03010 | 11 → 21 | 10 |
    | Test (80) | 6 mm | 0.89430 | 0.88685 | 0.00855 | 16 → 16 | 0 |

    No bad segmentations recovered in these groups. Test main-model Dice fell in **77/80** cases.
    Sources: `results/kits_robustness_eval/{dice.csv,dice_per_model.csv,dice_degradation.csv,paired_changes.csv}`.
  - 2026-10-21: Paired change results, primary first; positive rho means score rises with Dice loss:

    | Split | Score | Change rho [95% bootstrap CI] | Score rises | Median score change |
    |---|---|---|---:|---:|
    | Dev | Model disagreement | -0.074 [-0.529, 0.415] | 9/20 | -0.000463 |
    | Dev | Entropy (secondary) | 0.575 [0.104, 0.894] | 20/20 | 0.037025 |
    | Dev | Negative volume | 0.849 [0.532, 0.980] | 16/20 | 11.296365 |
    | Test | Model disagreement | 0.215 [-0.036, 0.459] | 41/80 | 0.000611 |
    | Test | Entropy (secondary) | 0.243 [0.004, 0.473] | 79/80 | 0.046093 |
    | Test | Negative volume | 0.743 [0.597, 0.845] | 70/80 | 13.950788 |

    Source: `results/kits_robustness_eval/change_correlations.csv`; all CIs use 2,000 finite patient resamples.
    The primary change-correlation interval includes zero: this does not establish a clear response to
    degradation severity. Entropy has a modest positive secondary association; it remains secondary.
    No superiority test between primary and entropy or adjustment for multiple comparisons was added.
  - 2026-10-21: G1 review-AUC under the task's **Dice < 0.83** rule:

    | Queue | Bad views | Primary | Entropy | Volume | Random | Oracle |
    |---|---:|---:|---:|---:|---:|---:|
    | Clean dev | 4/20 | 0.7625 | 0.6875 | 0.5500 | 0.4978 | 0.9000 |
    | Lowres dev | 6/20 | 0.7917 | 0.7167 | 0.5917 | 0.4969 | 0.8500 |
    | Mixed dev | 10/40 | 0.7800 | 0.7250 | 0.5725 | 0.4964 | 0.8750 |
    | Clean test | 11/80 | 0.8313 | 0.8949 | 0.3903 | 0.5045 | 0.9313 |
    | Lowres test | 21/80 | 0.7967 | 0.7455 | 0.4926 | 0.5018 | 0.8688 |
    | Mixed test | 32/160 | 0.8086 | 0.8068 | 0.4648 | 0.5030 | 0.9000 |

    Primary minus volume on lowres test: **0.3042 [0.1490, 0.4709]**; mixed test:
    **0.3438 [0.1795, 0.5052]**. Mixed primary AUC CI **[0.7374, 0.8830]**.
    These support failure ranking beyond volume under the fixed threshold rule.
    Raw AUC ceilings change with bad prevalence; compare random/oracle references as well as raw AUC.
    Under **worst10**, primary test AUC is **0.8547 clean / 0.8047 lowres / 0.8520 mixed**;
    lowres minus volume CI **[-0.1090, 0.6876]** includes zero. Do not claim superiority for every bad rule.
    Source: `results/kits_robustness_eval/g1_metrics.csv`. The original locked G1 headline uses worst10,
    so its 0.855 must not be confused with the Dice-threshold clean AUC here.
  - 2026-10-21: Mixed test queue has **160 views from 80 cases**, not 160 independent patients.
    In the top 16 (10%), **7/16** are lowres for the primary score, **13/16** for entropy, **8/16** for volume.
    Median paired rank gain: **2.0 / 55.5 / 6.0**, respectively. At that point the primary finds **12/32** bad
    segmentations, entropy **13/32**, volume **4/32**. Finding failures and promoting all degraded views
    are different outcomes: the primary still ranks failures usefully but does not consistently increase
    for degradation. Entropy responds more consistently to the image change without a higher mixed AUC
    point estimate. Sources: `results/kits_robustness_eval/{queue_top_summary.csv,queue_rank_changes.csv,g1_curves.csv}`.
  - 2026-10-21: Verification passed: **100 cases, 200 score rows, 400 model Dice rows, 100 paired rows,
    60 metric rows, 800 prediction files / 424,973,659 bytes**; all **720 imported originals unchanged**;
    exact cohort/key coverage and config/source/stage hashes match the pre-outcome plan.
    `results/kits_robustness_eval/{verification.json,post_run_checks.json,prediction_manifest.csv,key_results.json}`.
    Run `PYTHONPATH=src:. .venv-gpu/bin/python results/kits_robustness_eval/report_source.py --config configs/kits_robustness_eval.yaml`
    to reproduce the artifact audit and key-result extraction without rescoring/rebootstrapping.
    Logs have normalized terminal progress line endings/trailing spaces. Prediction maps stay outside Git.
    Completed inference configs/results, development strength evidence, and locked G1 outputs are unchanged.
  - 2026-10-21: Final suites in both environments: **123 passed, 36 skipped**, 8 expected constant-input
    warnings from the tiny synthetic tests (3.03 s shared, 3.04 s GPU).
    `results/kits_robustness_eval/tests_{main,gpu}.txt`. Step 5 complete; publish its commit on
    `otto/robustness` to update PR #1, then explain step 6. No report figure or step 6 summary written yet.

- [ ] **6. One figure and a short summary.** One figure that answers the question at a glance, and a summary of
  5–10 lines in this document (section 4) and in the log. Bjørn uses it for the report.
  *Notes:*

- [ ] **7. Present it to the group** (5 minutes): what you did, what came out, what it means.
  *Notes:*

## 4. Results summary

*(Filled in at step 6.)*

## 5. Decisions made in this task

- 2026-10-20 — Otto owns this task and works on `otto/robustness`, replacing the original Rasmus assignment,
  following Otto's explicit instruction for this work session.
- 2026-10-21 — Keep the group's torch 2.2.2 lock and use a separate pinned Linux GPU environment for Otto's
  RTX 5060 Ti, as approved by Otto. The GPU needs a newer CUDA/PyTorch build; record this environment difference.
- 2026-10-21 — Use `case_00040` and `case_00051` (first two sorted dev IDs) and `noise_60` for timing only.
  Strength selection remains step 3 and must happen before inspecting uncertainty performance.
- 2026-10-21 — Use a separate benchmark for both models and variants. The existing `step_second_model` handles
  clean scans only; production support for perturbed second-model predictions is still needed before large runs.
- 2026-10-21 — Strength selection uses only 3 mm development Dice, with the eight candidates and operational
  rule recorded in step 3 and `configs/kits_robustness_dev.yaml` before candidate inference. Use its own folders;
  keep clean references, and do not compute uncertainty scores until the chosen list is logged.
- 2026-10-21 — Lock `lowres_8mm` under the predeclared rule: 14/20 clear losses, 14/20 still good, no new
  catastrophic failures. Exclude tested noise and contrast because none meets the clear-drop requirement;
  do not select lowres_12mm because a milder eligible choice exists. Source: `configs/kits_perturbed.yaml`
  and `results/kits_robustness_dev/strength_lock.json`. Any later scope/strength change must be logged before test.
- 2026-10-21 — Following Otto's request to continue to step 4, retain the locked resolution-only question.
  Use a new test config/output name, both models, clean plus `lowres_8mm`, and the original 80 held-out cases.
  Separate inference (step 4) from Dice/scoring/evaluation (step 5). Ask before the new long run.
- 2026-10-21 — Step 4 test inference is complete. Preserve its maps and provenance; use saved predictions
  for step 5. The test set has now been used for inference, with no quality or score results inspected.
  Development comparisons still need second-model clean/lowres maps for the original 20 dev cases.

## 6. Questions for Bjørn

- 2026-10-21 — The predeclared rule selects only `lowres_8mm`; none of the tested noise or contrast strengths
  gives ≥ 0.02 Dice loss on five dev scans. Is this single resolution perturbation sufficient for the report,
  or should we plan an additional dev-only strength search / a stronger secondary resolution before test?
  Lowres_12mm is also eligible but was excluded by the mildest-per-type rule. The initial timing plan assumed
  3–4 perturbations; only one is currently locked. At strength lock, no test inference or uncertainty scoring
  had started.
  Follow-up, 2026-10-21: Otto requested step 4; preparation retains the locked resolution-only scope.
  No answer from Bjørn is recorded. Any later expansion must be logged with its timing relative to test use.
  Follow-up after step 4, 2026-10-21: 320 test predictions are now saved for the locked resolution-only
  experiment. No Dice, score results, or ranking evaluation has been inspected; section 6 remains unanswered.
  Follow-up after step 5, 2026-10-21: the fixed resolution-only test evaluation is complete. Primary ranking
  remains useful under Dice < 0.83, but paired score response is inconclusive; entropy is a secondary result.
  Test outcomes have now been inspected. Any expansion must be logged with that timing and must not be
  presented as part of the original untouched test. No answer from Bjørn is recorded.

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action.
A new chat starts by reading this section.)*

- 2026-10-20: Task created. Next action: step 1.
- 2026-10-20: **Step 1 complete** on `otto/robustness`: installation verified; tests 72 passed, 36 skipped,
  no failures; 100 KiTS image/label pairs and both model checkpoints downloaded and verified.
  Evidence: `results/robustness_setup/download_verification.json`. No inference or evaluation has been run.
  **Push confirmed:** GitHub's `otto/robustness` points to `f388b74`, including setup commit `b2269e9`.
  The earlier push blocker is resolved. A PR into `main` is still pending (the connected integration returns
  HTTP 403 when creating one); open it through GitHub for group review.
  The follow-up status-note commit is local until Otto pushes it from his terminal; the assistant's HTTPS
  Git command still lacks credentials. The completed setup itself is already on GitHub.
  Next action: explain step 2 and check compute availability (PyTorch reports no CUDA and `nvidia-smi` cannot
  communicate with the driver in this environment), then prepare a separate `configs/kits_perturbed.yaml`
  with its own output name for a timing run on 2 development scans with 1 perturbation. Do not start a run
  expected to exceed 30 minutes without Otto's approval. Steps 2–7 are untouched.
- 2026-10-21: **Steps 1–2 complete.** PR #1 exists and step 1's follow-up is pushed. The separate GPU environment
  works; both models ran on two dev scans, clean and noisy. Evidence and timing estimates are in
  `results/kits_robustness_timing/`; no accuracy/scoring evaluation or test-set inference was performed.
  Publish the step 2 commit on `otto/robustness` (use Otto's terminal if the assistant still lacks HTTPS
  credentials), which updates PR #1. Next: explain step 3 and choose candidate strengths on the 20 dev scans
  using Dice degradation only. Do not inspect score performance before logging the chosen strengths.
  Before larger runs, add second-model support for perturbed scans; the current shared step handles clean only.
  Ask Otto before any actual run expected to exceed 30 minutes. Steps 3–7 have not started.
- 2026-10-21: **Step 3 prepared, not complete.** Step 2 is pushed (`6f1bc0a`). Candidate config, dev-only
  runner, Dice-selection rule, tests, and run plan are ready. No candidate inference, Dice evaluation, score
  evaluation, or test-set use has started. **Next action: obtain Otto's approval for 180 dev predictions,
  roughly 40 minutes with a 90-minute cap**, then run
  `PYTHONPATH=src .venv-gpu/bin/python scripts/select_robustness_strengths.py --config configs/kits_robustness_dev.yaml --run`
  outside the sandbox for GPU access. Compare dev Dice, explain the outcome, and log the final strengths in
  this task, `PROJECT_LOG.md`, and a final config before inspecting score performance. Steps 4–7 untouched.
- 2026-10-21: **Step 3 run approved and starting.** Otto allows an overnight run; cap raised to 480 min
  before inference, without changing candidates or the predeclared selection rule. The original 90-minute
  plan/config are preserved. Next: finish the 180 dev predictions and Dice summary, document the selected
  strengths and reasons, then commit the completed step. No uncertainty-score analysis or test inference
  is authorized in this run; step 4 waits until Otto returns.
- 2026-10-21: **Steps 1–3 complete.** The approved 180-prediction dev run and Dice comparison finished in
  44.39 min. **`lowres_8mm` is locked** in `configs/kits_perturbed.yaml` under the original rule; tested noise
  and contrast strengths are excluded. Evidence, figure, and exact provenance: `results/kits_robustness_dev/`.
  No uncertainty-score computation or test inference has occurred. Next: explain this single-perturbation
  outcome to Otto and discuss section 6 with Bjørn before deciding the test scope. Then plan step 4, including
  second-model support for perturbed scans and fresh runtime estimates; ask Otto before a new long run.
  The final config still selects 20 dev scans. Steps 4–7 are untouched.
  **Push confirmed, 2026-10-21:** GitHub's `otto/robustness` points to `e608d92`, matching local HEAD and
  publishing step 3 to PR #1. This status confirmation will accompany the next work commit.
- 2026-10-21: **Step 4 prepared, not complete.** Otto requested proceeding with the locked choice;
  `configs/kits_robustness_test.yaml` and `scripts/run_robustness_test.py` are ready for the original 80 test
  scans, clean + `lowres_8mm`, both models (**320 predictions**). Second-model perturbation support and
  split filtering are implemented. Both suites pass (**113 passed, 36 skipped**); GPU smoke test passes.
  Plan/evidence: `results/kits_robustness_test/`; approximate estimate **39.66 min**, proposed cap **480 min**.
  **Next action: ask Otto to approve this new long run**, then execute `--run` with GPU access and save the log.
  No test inference, label reading, or score computation has started. After complete verification, check step 4,
  update the notes/log/overview, and commit/push on `otto/robustness`. Preparation is currently uncommitted.
  Steps 5–7 are untouched; Bjørn's report-scope question remains open, with no answer recorded.
- 2026-10-21: **Step 4 approved and starting.** Otto approved the prepared 320-prediction test run with
  the existing 480-minute cap. Next: finish both models' clean/lowres_8mm predictions, verify all 640 maps
  and timing coverage, then document measured time/memory and commit/push the completed step. Save console
  output to `results/kits_robustness_test/run_log.txt`. No Dice or uncertainty evaluation in this step.
- 2026-10-21: **Steps 1–4 complete.** Both models predicted all 80 test scans, clean + `lowres_8mm`:
  **320 predictions / 640 files**, **41.54 min**, verification passed. Timing, machine/config/source/input
  provenance, memory, file hashes, and passing tests are under `results/kits_robustness_test/`;
  predictions are outside Git in `data/predictions/kits_robustness_test/`.
  No test ground truth, Dice, uncertainty scores, or ranking evaluation was read/computed.
  **Next action:** explain step 5: compare clean/lowres Dice and the locked scores using saved maps.
  For development comparisons, reuse the chosen 3 mm maps and obtain the 6 mm clean/lowres maps for all 20 dev
  scans before scoring. Keep the locked strength and primary/secondary scores. Steps 5–7 have not started.
  Bjørn's report-scope question in section 6 remains unanswered; no message was sent.
  **Push confirmed, 2026-10-21:** GitHub's `otto/robustness` points to `6fde679`, matching local HEAD and
  publishing the completed step 4 to PR #1. This status confirmation will accompany the next work commit.
- 2026-10-21: **Step 5 in progress.** Pre-outcome analysis config/plan, paired summaries, patient-grouped
  G1 bootstrap, and 10 synthetic tests are ready; no new scores or Dice inspected yet.
  Next: copy the selected dev/test maps into `kits_robustness_eval`, infer the missing 40 dev second-model
  pairs (estimate 6.61 min, cap 30 min), then benchmark CPU scoring/evaluation before starting the full
  evaluation. Ask Otto if a stage is projected over 30 min. Fixed choices and analysis details are in step 5
  and `results/kits_robustness_eval/analysis_plan.json`. Steps 6–7 have not started.
- 2026-10-21: **Step 5 evaluation starting.** Preparation complete (800 maps; 6.74 min). CPU benchmark
  projects 15.34 min, within the 30-minute stage cap. Next: compute the fixed scores from model maps first,
  then Dice, paired changes, and split-specific G1 queues with patient bootstrap. No strengths or methods
  will be selected from test outcomes. Evidence remains under `results/kits_robustness_eval/`.
- 2026-10-21: **Steps 1–5 complete.** Evaluation took 12.05 min after 6.74 min dev-map preparation.
  Test 3 mm median paired Dice drop **0.03010**, bad count **11 → 21/80**. Primary change rho **0.215
  [-0.036, 0.459]** is inconclusive; primary mixed threshold review-AUC **0.8086** versus volume **0.4648**,
  random **0.5030**, oracle **0.9000**. Entropy rises in 79/80 cases and promotes more lowres views, but
  mixed AUC is similar (0.8068). Full dev/test results, both bad rules, CIs, hashes, and 123-pass test logs:
  `results/kits_robustness_eval/`. Inputs and locked results remain intact; test outcomes are now inspected.
  **Next action:** commit/push step 5 on `otto/robustness` (Otto's authenticated terminal if needed), then
  explain step 6 and build one report figure plus the requested 5–10-line summary. Keep the fixed methods
  and resolution-only scope; section 6's report-scope question is still unanswered. Steps 6–7 have not started.
  **Git:** step 5 is committed locally. The assistant's push failed because noninteractive HTTPS Git
  cannot read a GitHub username. Otto must run `git push origin otto/robustness` from his authenticated
  terminal to publish the completed step. This status is included in the same unpublished work commit.
