# Task: Catching the errors that uncertainty misses

| | |
|---|---|
| **Owner** | Otto |
| **Contact / lead** | Bjørn |
| **Branch** | `otto/missed-errors` |
| **Created** | 2026-10-21 |
| **Status** | Step 1 complete (2026-11-02); step 2 next |
| **Way of working** | Independent. Explore, try ideas, keep what works. The task is large enough to keep going without waiting for new instructions; questions to Bjørn are welcome. Keep the group posted; coordinate where section 3 says so. |
| **Target** | First results by 2026-11-04; final check on fresh scans and a short summary by 2026-11-18 |

This document is the task description **and** the working notes. Update it after every step and at the end of every
work session (section 7), so the group can follow and a new chat can continue.

---

## 1. The idea in plain words

Our queue (G1) ranks scans by how *unsure* the AI is. That works well, but some of the worst mistakes are mistakes
the AI is *sure* about. Examples from our data:

- **case_00040:** a kidney in the pelvis that the model never found (probability 0, so no uncertainty at all).
- **case_00516:** a child with large tumors in both kidneys; the model found almost nothing (Dice 0.003).
- **Missed tumors:** in bad scans about 85 % of the wrong voxels are tumor the model left out, and about 20 % of all
  errors lie outside the region around the predicted kidney, where an uncertainty heatmap can never point.

**Question: can simple checks that need no ground truth catch these confident errors, and does adding them to the
queue find the bad scans faster?**

A few starting ideas (try them, drop them, or find better ones):

- **Shape:** a kidney with a missed tumor often has a "bite" taken out of it. Compare the predicted kidney with its
  convex hull (how much is missing from a smooth shape?), or measure how compact it is.
- **Size:** kidney volume per side compared with what is normal; a very small or missing kidney is suspicious.
  (A check for a missing side exists already: `src/segreview/plausibility.py`. It caught the pelvic kidney but also
  gave false alarms.)
- **The second model:** where the 6 mm model sees kidney far from the 3 mm outline (see PROJECT_LOG.md, 2026-10-20).
- **What is next to the kidney:** tissue right next to the predicted kidney that looks like kidney or tumor in the CT
  (gray values) but was not included.

## 2. Starting points

- How the code works: [README.md](../../README.md). General rules: [CONTRIBUTING.md](../../CONTRIBUTING.md).
- **Data you need:** the KiTS scans (`scripts/download_data.py --config configs/kits100.yaml`) and the saved model
  output for kits100, `data/predictions/kits100/` (about 340 MB). Get the model output from the group's OneDrive
  (folder "delte store filer", `kits100.zip`) and unzip it so it lands in `data/predictions/kits100/`. **Do not run inference for kits100 yourself.**
- Per-scan Dice, scores and the locked queue: `results/kits100/` (`dice.csv`, `scores.csv`, `g1_scores_clean_*.csv`).
- Earlier related analyses: `results/kits100/model_disagreement_*`, `exploratory_disagreement_*`,
  `voxel_correction_*`, and the robustness task ([g1_robustness.md](g1_robustness.md)).
- Evaluate new scores with the existing G1 evaluation functions (`src/segreview/evaluation.py`): review-AUC for both
  "bad" rules, quality-AUC, Spearman ρ, paired bootstrap against the locked score.

## 3. Rules and coordination

- **Never write to `results/kits100/` or `data/predictions/kits100/`.** Use your own config and folders
  (e.g. `configs/kits_missed_errors.yaml`, `results/kits_missed_errors/`).
- **All 100 kits100 scans are development data for this task** (the G1 test set has already been opened, so
  everything you find on kits100 is exploratory). Explore freely there.
- **The final check is on fresh scans, once.** The G3 pool (the next 100 KiTS scans in the same seeded draw) has not
  been looked at. Before using it: write down the one or two checks you want to test and how you will judge them, in
  this document and in the log, and tell Bjørn. A test of another score on the pool is already locked (PROJECT_LOG.md,
  2026-10-21); both are run together once, so the pool is opened only once (latest 2026-11-18). Frederik runs the models on the pool for G3; coordinate with him to
  reuse his model output instead of running it again.
- **Coordinate** when you want to change shared code (anything in `src/segreview/` that others use) or when you need
  someone else's data. Everything else you decide yourself.
- Seed 42, record the machine, log every result, idea and dropped idea with the date in
  [PROJECT_LOG.md](../../PROJECT_LOG.md). Push to your branch regularly; open a pull request when something is ready.

## 4. Suggested route (adjust as you go)

- [x] **1. Get started.** Install (`uv sync`, `uv run pytest`), get the data (section 2), read
  [PROJECT_OVERVIEW.md](../../PROJECT_OVERVIEW.md). Be able to explain G1 and "confident error" in two sentences.
  *Notes (2026-11-02):*
  - Read the overview, contributing rules, this task and the relevant README instructions. Pulled `main`
    (`0fe9216`, already up to date), then created `otto/missed-errors` before writing files.
  - `uv sync --locked --offline` checked all 133 packages in the shared environment without downloading or
    changing dependencies. `uv run --locked --offline pytest`: **123 passed, 36 skipped, 8 warnings in 3.32 s**.
    Skips require the prepared study or Google Chrome; warnings are constant-input correlations in existing
    synthetic robustness tests. Command/result record: `results/kits_missed_errors/setup_tests.txt`.
  - All selected inputs are already local: **100 CT scans, 100 label maps and 600 prediction/heatmap files**
    (349,419,984 prediction bytes). All **800** canonical header shapes/affines match their CT; the saved
    Dice/score tables cover the expected 100 cases and the original 20/80 split without duplicate IDs.
    Evidence: `results/kits_missed_errors/setup_audit.json` and `setup_inventory.csv`.
    These are presence/header checks, not full voxel-data validation or proof of the archive's origin.
  - Recorded SHA-256 fingerprints for the 600 maps and 61 locked result files; result hashes were unchanged
    across the audit. No downloads, inference, new scores, Dice calculations or ranking evaluation were run;
    label voxels and the G3 pool were not opened. No shared code was changed.
  - Machine: AMD Ryzen 5 7500F, Linux 7.3.1-1-cachyos, Python 3.12.15, torch 2.2.2+cu121 (CUDA build 12.1),
    NumPy 1.26.4; CPU-only audit, seed 42. GPU/driver query was unavailable in this session, and no GPU was used.
    Exact versions, baseline commit and config/lock hashes: `results/kits_missed_errors/setup_machine.json`.
  - **Two-sentence explanation:** G1 puts scans with greater disagreement between the 3 mm and 6 mm models
    earlier in the review queue, using only model output. A confident error occurs when a model is wrong but
    assigns a probability near zero or one; if both models miss the same kidney, their agreement can hide it,
    which motivates independent checks of size, shape or nearby tissue.
- [ ] **2. Look at the misses.** Which bad scans (Dice < 0.83) does the locked queue rank low (outside its top 20 %)?
  Look at each one (`scripts/show_case.py`) and describe in words what the model got wrong.
  *Notes:*
- [ ] **3. Try checks.** Build GT-free checks from section 1 or your own ideas. For each: does it rank the missed bad
  scans higher, and how many false alarms does it give?
  *Notes:*
- [ ] **4. Combine.** Does the locked score + your best check find the bad scans faster than the locked score alone?
  *Notes:*
- [ ] **5. Lock and test once on the fresh scans** (see section 3).
  *Notes:*
- [ ] **6. One figure and a 5–10 line summary** in section 5 and in the log. Present it to the group (5 minutes).
  *Notes:*

## 5. Results summary

*(Filled in at step 6.)*

## 6. Decisions and dropped ideas

- **2026-11-02 — Reuse the existing shared environment and saved kits100 predictions.** Step 1 needs CPU
  file/header checks only. A GPU-environment change is deferred because it would not help this step; no new
  kits100 inference is permitted. No detector ideas have been tested or rejected yet.
- **2026-11-02 — Keep setup evidence separate from locked inputs.** Setup files go to
  `results/kits_missed_errors/`. The audit only reads `configs/kits100.yaml` to reproduce the exact selected
  cohort; an analysis config with its own name/output paths will be added in step 2 before generating figures
  or analysis results. All 100 kits100 cases are development data for this task, with original G1 split labels
  retained for traceability. This does not change the locked G1 analysis.
- **2026-11-02 — Fresh-pool coordination remains required.** The 2026-10-21 log already locks
  `far_disagree_per_volume` for a single joint G3-pool evaluation. Lock our one or two checks and evaluation
  rules before that evaluation, tell Bjørn, and coordinate prediction reuse with Frederik by 2026-11-18.
  No external messages have been sent in this session.

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action,
and ideas for what could make the result better.)*

- **2026-11-02 — Last finished step: 1.** The shared environment and tests pass, and the 100-case input
  inventory/header audit passes. Step 2 has not started; no new scientific results or detector choices exist.
  Working branch: `otto/missed-errors`. Setup evidence is in `results/kits_missed_errors/`.
- **Next concrete action (step 2):** create `configs/kits_missed_errors.yaml` with separate outputs and explicit
  read-only prediction sources. Using saved Dice and `model_diff_mean_region`, list bad cases (Dice < 0.83)
  outside the top 20% of the exploratory 100-case queue; retain original G1 split labels and document ties.
  Inspect each miss and record a plain-language error description. The first-results target is 2026-11-04.
- **Practical improvements for the next session:** (1) save a per-case table of score, rank and error description
  so candidate checks can be judged against the same misses; (2) inspect several axial slices per case, since
  `show_case.py` shows only the slice with the most ground-truth organ and can hide a missed kidney or tumor.
- **Publication:** step 1 was committed as `3f9b975` and pushed to `origin/otto/missed-errors`.
  [Draft PR #2](https://github.com/Ottohesten/02830-medical-segmentation/pull/2) tracks the ongoing task.
  The GitHub app could not create the PR (403); the local `gh` client succeeded. Ask a group member to review
  before merging when the task is ready. No direct message to Bjørn/Frederik has been sent.
