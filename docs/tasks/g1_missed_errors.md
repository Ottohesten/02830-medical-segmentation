# Task: Catching the errors that uncertainty misses

| | |
|---|---|
| **Owner** | Otto |
| **Contact / lead** | Bjørn |
| **Branch** | `otto/missed-errors` |
| **Created** | 2026-10-21 |
| **Status** | Not started |
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
  this document and in the log, and tell Bjørn. Frederik runs the models on the pool for G3; coordinate with him to
  reuse his model output instead of running it again.
- **Coordinate** when you want to change shared code (anything in `src/segreview/` that others use) or when you need
  someone else's data. Everything else you decide yourself.
- Seed 42, record the machine, log every result, idea and dropped idea with the date in
  [PROJECT_LOG.md](../../PROJECT_LOG.md). Push to your branch regularly; open a pull request when something is ready.

## 4. Suggested route (adjust as you go)

- [ ] **1. Get started.** Install (`uv sync`, `uv run pytest`), get the data (section 2), read
  [PROJECT_OVERVIEW.md](../../PROJECT_OVERVIEW.md). Be able to explain G1 and "confident error" in two sentences.
  *Notes:*
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

*(Date — what — why. Also in PROJECT_LOG.md.)*

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action,
and ideas for what could make the result better.)*

- 2026-10-21: Task created. Next action: step 1.
