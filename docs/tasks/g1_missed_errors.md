# Task: Catching the errors that uncertainty misses

| | |
|---|---|
| **Owner** | Otto |
| **Contact / lead** | Bjørn |
| **Branch** | `otto/missed-errors` |
| **Created** | 2026-10-21 |
| **Status** | Steps 1–3 complete (2026-11-02); step 4 next |
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
- [x] **2. Look at the misses.** Which bad scans (Dice < 0.83) does the locked queue rank low (outside its top 20 %)?
  Look at each one (`scripts/show_case.py`) and describe in words what the model got wrong.
  *Notes (2026-11-02):*
  - Independent config `configs/kits_missed_errors.yaml` and isolated `scripts/inspect_missed_errors.py`;
    saved score descending, case ID ascending for exact ties, pooled 100-case exploratory cohort,
    bad = Dice < 0.83, outside the first 20 cases. No tie at the budget boundary. The first 20 contain
    **11/15 bad cases** and 9 cases above/equal to the Dice threshold. **Four bad cases remain outside**:

    | Case | Rank / 100 | Saved Dice | What the reviewed views show |
    |---|---:|---:|---|
    | case_00236 | 23 | 0.8023 | A large GT-labeled tumor is partly excluded, leaving a broad indentation/gap; some kidney tissue is also missing. Both models share much of the omission. |
    | case_00406 | 42 | 0.8004 | An outward-projecting tumor next to the kidney is almost entirely absent from both masks; **99.2%** of GT tumor voxels are missed by 3 mm. |
    | case_00040 | 46 | 0.6607 | The pelvic kidney is absent from both masks in the reviewed pelvic views; the other kidney is segmented higher in the abdomen. |
    | case_00184 | 56 | 0.8176 | Multiple tumor/cyst regions are partly or completely omitted: **42.1%** of GT tumor and **77.2%** of cyst voxels are missed. The 6 mm Dice improves to **0.8547** but omissions remain. |

  - Reviewed all four montages (**19 selected axial/coronal views**) using CT, label-specific GT contours,
    both masks, 3 mm probability and entropy. This is assistant visual review; Otto's own review has not been
    recorded. Full descriptions, images and reading guide:
    [exploratory_review.md](../../results/kits_missed_errors/exploratory_review.md).
    Machine-readable table: `results/kits_missed_errors/exploratory_missed_review.csv`.
    CT-derived montages remain local and are excluded from Git; the script regenerates them from the
    shared inputs. The guide's image links work after local regeneration.
  - Within each case, **98.5–99.9%** of wrong voxels are omissions. Among missed organ voxels,
    **59.6–89.1%** have saved p <= 0.10, **48.8–86.8%** lie outside the predicted organ plus its 5 mm
    uncertainty border, and the 6 mm mask recovers only **2.0–26.8%**. These diagnostics support the
    confident-error mechanism; a low queue rank alone would not prove confidence. A saved zero is quantized
    (1/255 precision), not proof of exact zero before storage. Sources: `exploratory_error_diagnostics.csv`
    and `exploratory_inspection_summary.json` in `results/kits_missed_errors/`.
  - `case_00516` is already rank **3**, Dice **0.0027**, in this pooled queue, so it is not a miss at this
    budget. The pooled queue is exploratory and is not a replacement for the locked 80-case G1 result.
  - CPU processing: **3.92 s** for the first timing case, then **5.43 / 14.71 / 9.63 s** for the remaining
    cases (**33.69 s** summed per-case processing). Machine: AMD Ryzen 5 7500F, Linux 7.3.1-1-cachyos,
    Python 3.12.15, torch 2.2.2+cu121 / CUDA build 12.1, 6 threads, seed 42; no GPU used.
    `exploratory_images_machine.json` and `exploratory_inspection_evidence.json` record provenance/timing.
  - **139 tests passed, 36 skipped, 8 existing constant-input warnings, 2.99 s**, including **16** new
    synthetic checks. `results/kits_missed_errors/exploratory_tests.txt`. Verification passed: 100 queue
    rows, exact four-case diagnostic/review coverage, saved 3 mm/6 mm Dice reproduced within rounding,
    figure/input/config/code hashes, and all **61 locked result files / 600 prediction maps unchanged**.
    `results/kits_missed_errors/exploratory_verification.json`. No inference, new scores, combinations,
    fresh-pool access or shared-code edits.
- [x] **3. Try checks.** Build GT-free checks from section 1 or your own ideas. For each: does it rank the missed bad
  scans higher, and how many false alarms does it give?
  *Notes (2026-11-02):*
  - Separate config `configs/kits_missed_checks.yaml` and output folder `results/kits_missed_checks/`;
    isolated `scripts/check_missed_errors.py`, no shared-code edits. Candidate definitions were written down
    before new outcome evaluation. Shape uses at most two largest 6-connected components >= 20 ml, matching
    the existing plausibility threshold. These are predicted kidney proxies, not anatomical side labels.
  - New calculations use masks and saved GT-free columns; no CT/label maps or Dice during feature extraction.
    Saved side volumes originally came from the existing CT-body-midline check. All 100 feature rows were
    saved before attaching outcomes. Reused unchanged G1 evaluation: both bad rules, review/quality AUC,
    Spearman/partial rho, 2,000 case-bootstrap resamples and paired differences to locked G1, seed 42.
  - At the fixed 20-case budget (bad = Dice < 0.83), standalone results are:

    | Method | Expected bad found | Expected false alarms | Expected original misses included |
    |---|---:|---:|---:|
    | Locked G1 | 11.00 | 9.00 | 0.00 |
    | Entropy (existing) | 13.00 | 7.00 | 2.00 |
    | Small total volume | 3.00 | 17.00 | 2.00 |
    | Missing-side flag | 4.17 | 15.83 | 1.50 |
    | Side asymmetry | 6.00 | 14.00 | 2.00 |
    | Small minimum side | 4.00 | 16.00 | 2.00 |
    | Convex-hull deficit | 9.00 | 11.00 | 1.00 |
    | Voxel-face compactness | 7.00 | 13.00 | 0.00 |
    | Far disagreement (existing) | 12.00 | 8.00 | 1.00 |
    | Far 6 mm additions (existing) | 12.00 | 8.00 | 2.00 |

  - Ties use expected inclusion under random tie breaking, so fractional counts are expectations. The
    missing-side flag identifies only **4 cases: 2 bad and 2 nonbad**. Its other 16 places are random among
    96 unflagged cases; the 1.50 expected original misses is the flagged pelvic case plus 0.50 by chance.
    Actual flag cases: case_00040 (Dice 0.6607), case_00516 (0.0027), case_00293 (0.9475, false alarm),
    case_00563 (0.8779, false alarm). Source: `exploratory_flag_cases.csv`.
  - Hull deficit = 1 - occupied voxel volume / convex hull of voxel boxes. It moves **case_00236 23 → 9**,
    **case_00406 42 → 21** (still outside budget), **case_00184 56 → 36**, but **case_00040 46 → 95**.
    The missing-side flag moves case_00040 to **average tied rank 3**. Shape of the remaining kidney cannot
    detect an absent other kidney; these two checks therefore address different misses.
  - Threshold review-AUC: locked **0.8170**, hull **0.7543**, compactness **0.6623**, minimum-side size
    **0.4497**. Hull minus locked **-0.0627, 95% paired CI [-0.1905, 0.0556]**; worst10 AUC
    **0.7500 vs. 0.8550**, difference **-0.1050 [-0.2857, 0.0543]**. This does not establish a better
    standalone score. The existing far-disagreement threshold AUC is **0.8577**, difference
    **+0.0407 [0.0000, 0.0967]** (lower bound exactly zero); the fresh lock is unchanged.
    All comparisons are exploratory, without multiplicity correction or a claim of generalization.
  - Carry **missing-side + hull deficit** to step 4 as complementary checks, retaining existing alternatives
    as controls. Deprioritize this compactness version (no original miss in top 20) and minimum-side volume
    as the main queue (16/20 false alarms). This does not disprove all shape/size ideas. No combinations
    or fresh-pool checks have run. Readable interpretation:
    [exploratory_checks_summary.md](../../results/kits_missed_checks/exploratory_checks_summary.md).
  - CPU timings: benchmark **5.79 s**, features **84.10 s**, evaluation **50.30 s** (all below 30 min).
    AMD Ryzen 5 7500F, Linux 7.3.1-1-cachyos, Python 3.12.15, torch 2.2.2+cu121 / CUDA build 12.1,
    NumPy 1.26.4, SciPy 1.17.1, 6 threads, seed 42, no GPU. Config snapshot and stage metadata preserve
    baseline `e9e63dc`, exact code/config/source hashes and the dirty run state.
  - **154 tests passed, 36 skipped, 8 existing warnings, 3.08 s**, including 15 new geometry/tie/isolation
    tests. Verification: **100 cases, 195 components, 24 metric rows, 36 paired rows, 2,424 curve rows,
    40 original-miss rank rows**; independent rank-AUC and curve checks pass. All **61 locked results /
    600 source maps** and the completed inspection config remain unchanged.
    Files: `exploratory_{features,components,metrics,paired_vs_locked,budget,miss_ranks}.csv`,
    `exploratory_verification.json` and `exploratory_tests.txt` in `results/kits_missed_checks/`.
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
- **2026-11-02 — Use one pooled exploratory queue for the missed-case inventory.** The 100 cases are all
  development data for this task. Keep their original G1 split labels, attach saved Dice only after sorting
  the locked score, and break score ties by case ID. This defines the four-case target list reproducibly.
- **2026-11-02 — Drop single-slice-only inspection.** A maximal-organ slice can show the correctly segmented
  kidney while hiding the pelvic kidney. Use label-specific error peaks plus a coronal view instead; the
  selected views complement whole-volume counts and do not claim to inspect every slice.
- **2026-11-02 — Start step 3 with cheap size and shape checks.** First assess missing-side/per-side size,
  then shape deficits, on all 100 cases including false alarms. The observations motivate these ideas but
  do not establish that they work: a normal kidney is concave, and a completely absent mass can leave a
  plausible-looking contour. Keep CT-intensity checks as a later option if these simple checks are insufficient.
- **2026-11-02 — Keep CT-derived montages outside Git.** Automatic approval rejected the first step 2 commit
  attempt because its four CT/mask/probability montages conflicted with CONTRIBUTING's data prohibition.
  The commit did not run. Removed the PNGs from staging and added an explicit ignore rule; keep the images
  locally, and commit only code, tables, hashes and notes. Group members can regenerate the figures.
- **2026-11-02 — Keep step 3 separate from completed inspection.** New name/config `kits_missed_checks`
  preserves step 2's settings and hashes. Reuse the existing G1 functions without changing shared code;
  shape uses masks only, with GT outcomes joined only after saving all features.
- **2026-11-02 — Carry missing-side and hull deficit to combination testing.** The flag promotes the pelvic
  case to tied rank 3 with two nonbad alerts; hull promotes case_00236 to 9, but has 11 false alarms at the
  20-case budget and misses the pelvic case. Neither replaces G1; whether their combination helps is step 4.
  Existing asymmetry, entropy and directional disagreement remain controls, not discarded competitors.
- **2026-11-02 — Deprioritize current compactness and minimum-side-volume queues.** Compactness finds
  7 bad / 13 false alarms and no original miss in its first 20. Minimum-side volume finds 4 bad / 16 false
  alarms; its targeted catches do not justify replacing uncertainty with organ size. Save results instead
  of tuning many variants before the deadline. CT-intensity and alternative shape estimators remain later
  options if the simple combinations fail; no claim that these broader ideas cannot work.

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action,
and ideas for what could make the result better.)*

- **2026-11-02 — Last finished step: 3.** Standalone size/shape checks have been evaluated on all 100
  development cases. Hull moves case_00236 23 → 9 but finds only 9/15 bad cases in 20; missing-side flags
  the pelvic case and two nonbad cases. Keep hull + missing-side for step 4; compactness/minimum-side
  volume are deprioritized. No combinations or fresh-pool evaluation yet. All locked inputs are unchanged.
  Branch: `otto/missed-errors`; step 3 results/summary in `results/kits_missed_checks/`; inspection remains
  in `results/kits_missed_errors/`. First detector results are available before the 2026-11-04 target.
- **Next concrete action (step 4):** write a small comparison plan/config before evaluating combinations
  of locked G1 with the missing-side flag and hull deficit. Reuse G1 rank-combination/evaluation functions;
  compare unchanged G1, existing alternatives and fair 20-case budgets. Do not select based only on the
  four misses or change the bad/budget thresholds. Keep the one joint fresh-pool test by 2026-11-18;
  lock one or two checks first, tell Bjørn and reuse Frederik's output as section 3 requires.
- **Practical improvements:** (1) compare targeted combinations before adding more features, since these
  checks find different errors; (2) inspect the two nonbad missing-side alerts (case_00293/case_00563) if
  they still consume useful review slots; (3) keep all 100 cases in false-alarm accounting to avoid fitting
  a detector only to the four examples. No shared-code edits are needed for this isolated experiment.
- **Publication:** [draft PR #2](https://github.com/Ottohesten/02830-medical-segmentation/pull/2) tracks
  the ongoing task. Step 3 was committed as `4f00ca0` and pushed to `origin/otto/missed-errors`.
  PR title/description now cover the standalone screen, its limitations and the planned combination step.
  Step 2's CT-derived montages remain local/ignored. Group review remains required before merging.
  No direct message to Bjørn/Frederik has been sent.
