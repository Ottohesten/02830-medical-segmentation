# Task: Does our ranking still work when the scans get worse?

| | |
|---|---|
| **Owner** | Otto |
| **Contact / lead** | Bjørn |
| **Branch** | `otto/robustness` |
| **Created** | 2026-10-20 |
| **Status** | Step 3 complete; lowres_8mm locked on development Dice; test not started |
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

- [ ] **4. Run the test set (80 scans)** with the chosen perturbations (overnight or on the GPU).
  *Notes:*

- [ ] **5. Evaluate.** With the existing G1 evaluation (`scripts/evaluate_g1.py`), report clean and perturbed
  separately, for development and test:
  - How much does Dice drop per perturbation (median, and how many scans become "bad", Dice < 0.83)?
  - Does the score rise when Dice falls? Per scan: change in score vs. change in Dice from clean to perturbed
    (Spearman ρ), for the locked score and for entropy.
  - Mixed queue: clean and perturbed scans together. Do the damaged scans move to the top? Review-AUC vs. the
    volume baseline, random and oracle, as in G1.
  *Notes:*

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

## 6. Questions for Bjørn

- 2026-10-21 — The predeclared rule selects only `lowres_8mm`; none of the tested noise or contrast strengths
  gives ≥ 0.02 Dice loss on five dev scans. Is this single resolution perturbation sufficient for the report,
  or should we plan an additional dev-only strength search / a stronger secondary resolution before test?
  Lowres_12mm is also eligible but was excluded by the mildest-per-type rule. The initial timing plan assumed
  3–4 perturbations; only one is currently locked. No test inference or uncertainty scoring has started.

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
  The final config still selects 20 dev scans. Publish this step on `otto/robustness` to update PR #1; use Otto's
  terminal if the assistant lacks HTTPS credentials. Steps 4–7 are untouched.
  **Git:** step 3's commit is currently local; the assistant's push failed for missing HTTPS credentials.
  Otto should push `otto/robustness` from his terminal before the next step.
