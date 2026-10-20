# Task: Does our ranking still work when the scans get worse?

| | |
|---|---|
| **Owner** | Otto |
| **Contact / lead** | Bjørn |
| **Branch** | `otto/robustness` |
| **Created** | 2026-10-20 |
| **Status** | Step 1 complete; step 2 next |
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
  - G1 in two sentences: We run the 3 mm and 6 mm TotalSegmentator models on each CT scan and measure their
    disagreement about the kidneys. Scans with more disagreement go earlier in the review queue so a person
    can check likely segmentation errors first, without ground truth being used to rank them.

- [ ] **2. Measure the time.** Run the 3 mm and 6 mm models on 2 development scans with one perturbation on your
  machine. Write down seconds per scan per model, and estimate the time for 20 scans × 3–4 perturbations and for
  80 scans. If it is too slow (more than a night), ask Bjørn; Frederik's GPU machine is an option.
  *Notes:*

- [ ] **3. Choose the strengths on the development set (20 scans).** Try a few strengths per type, e.g.
  noise 30/60/120 HU, lowres 6/8/12 mm, contrast 0.8/0.9 (0.6 had no effect on spleen). Goal: perturbations that
  clearly lower Dice for some scans but do not destroy every scan. **Before looking at how the uncertainty scores
  do**, write the chosen list and the reason in the log and in the config. Keep the clean scans as the reference.
  *Notes:*

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

## 6. Questions for Bjørn

*(Write questions here and tell Bjørn.)*

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action.
A new chat starts by reading this section.)*

- 2026-10-20: Task created. Next action: step 1.
- 2026-10-20: **Step 1 complete** on `otto/robustness`: installation verified; tests 72 passed, 36 skipped,
  no failures; 100 KiTS image/label pairs and both model checkpoints downloaded and verified.
  Evidence: `results/robustness_setup/download_verification.json`. No inference or evaluation has been run.
  Next action: explain step 2, check compute availability (PyTorch reports no CUDA and `nvidia-smi` cannot
  communicate with the driver in this environment), then prepare a separate `configs/kits_perturbed.yaml`
  with its own output name for a timing run on 2 development scans with 1 perturbation. Do not start a run
  expected to exceed 30 minutes without Otto's approval. Steps 2–7 are untouched.
