# Task: Catching the errors that uncertainty misses

| | |
|---|---|
| **Owner** | Otto |
| **Contact / lead** | Bjørn |
| **Branch** | `otto/missed-errors` |
| **Created** | 2026-10-21 |
| **Status** | Steps 1–4 complete; step 5 recipe locked and inputs local (2026-11-03), fresh evaluation pending handoff/preflight |
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
- [x] **4. Combine.** Does the locked score + your best check find the bad scans faster than the locked score alone?
  *Notes (2026-11-02):*
  - `configs/kits_missed_combinations.yaml` fixed six alternatives before outcome evaluation: G1 +
    missing-side, G1 + hull, G1 + both, each by mean/max rank using the unchanged G1 combiner. The isolated
    script reuses step 3 aggregate features and helpers, saves scores before attaching Dice, and preserves
    true rank ties on an exact lattice. No image reads, fitted weights, shared-code edits or fresh scans.
  - Fixed selection rule: highest threshold review-AUC among combinations with higher AUC and no fewer
    top-20 bad cases than G1; ties use quality-AUC then config order. The rule provisionally selects
    **`g1_missing_hull_mean`**. It is a development recommendation, not a prospective lock or proven gain.

    | Method | Review-AUC (Dice < 0.83) | Bad / first 20 | False alarms / first 20 |
    |---|---:|---:|---:|
    | Locked G1 | 0.8170 | 11 | 9 |
    | G1 + missing, mean | 0.8237 | 11 | 9 |
    | G1 + missing, max | 0.8110 | 11 | 9 |
    | G1 + hull, mean | 0.8277 | 11 | 9 |
    | G1 + hull, max | 0.8190 | 10 | 10 |
    | G1 + both, mean (selected) | 0.8470 | 11 | 9 |
    | G1 + both, max | 0.8397 | 11 | 9 |

  - Selected threshold difference to G1: **+0.0300, paired 95% case-bootstrap CI [-0.0320, 0.0864]**.
    Worst10: **0.8690 vs. 0.8550**, difference **+0.0140 [-0.0685, 0.0883]**. All six threshold
    difference CIs include zero. No combination improves expected top-20 bad capture. These are six
    exploratory comparisons on all 100 development cases, without multiplicity correction.
  - Selected combination gains case_00236 (**Dice 0.8023, rank 23 → 9**) but loses case_00582
    (**Dice 0.4605, rank 17 → 45**) at the budget of 20. Seven nonbad cases enter while seven leave.
    Thus there is no net bad-count gain, and the displaced bad case is more severe.
    Simulated mean Dice after 20 perfect corrections is **0.92248 vs. G1 0.92554** (worse).
  - The selected candidate leaves the pelvic case outside budget (**46 → 43**); case_00184 moves
    **56 → 45**, case_00406 **42 → 27**. The max-of-three alternative catches case_00040 and case_00236
    but displaces case_00426 and case_00582: two new finds, two losses, still 11/15 in the first 20.
  - Descriptively, the selected full curve finds **9/15 in the first 10 vs. G1 5/15**, but both find
    11/15 by 20. The 10-case point was not a new selection rule. Full quality-AUC is **0.95398 vs. 0.95242**,
    difference **+0.00157 [-0.00188, 0.00439]**. AUC gains can coexist with worse quality at a specific budget.
    The selected threshold AUC difference vs. entropy is **+0.0060 [-0.0764, 0.0921]** and vs. existing
    far disagreement **-0.0107 [-0.0800, 0.0440]**; superiority over either is not established.
  - Earlier missing-side false alerts case_00293 and case_00563 are ranks **31 / 26** for the selected
    candidate and consume no top-20 slots. No extra image review was needed for this aggregate comparison.
  - Readable result/tradeoffs:
    [exploratory_combination_summary.md](../../results/kits_missed_combinations/exploratory_combination_summary.md).
    Complete outputs in `results/kits_missed_combinations/`: combined scores, both-rule metrics, paired
    comparisons against G1/entropy/far, curves, per-case changes, original-miss ranks and recommendation.
  - CPU benchmark **1.86 s**, fixed evaluation **65.14 s**. AMD Ryzen 5 7500F, Linux 7.3.1-1-cachyos,
    Python 3.12.15, torch 2.2.2+cu121 / CUDA build 12.1, NumPy 1.26.4, SciPy 1.17.1, 6 threads, seed 42.
    No GPU; stage metadata/config snapshot record baseline `29de99a`, dirty state and exact source hashes.
  - **167 passed, 36 skipped, 8 existing warnings, 2.69 s**, including 13 new tests. Verification passed:
    **100 cases, 6 combinations, 32 metric rows, 132 paired rows, 3,232 curve rows, 56 target-rank rows,
    600 case-change rows**. Standalone metrics/CIs reproduce step 3; rank/tie/budget/displacement identities
    pass. All **61 locked results / 600 source maps / 23 earlier-stage output fingerprints** are unchanged.
    Sources: `exploratory_tests.txt` and `exploratory_verification.json` in the combination result folder.
- [ ] **5. Lock and test once on the fresh scans** (see section 3).
  *Notes (2026-11-02; lock complete, actual fresh test pending):*
  - Prospective recipe: [missed_errors_fresh_protocol.md](../missed_errors_fresh_protocol.md) and
    `configs/kits_missed_fresh.yaml`. Only `g1_missing_hull_mean` is added to the already locked far-score
    test. Both compare threshold review-AUC against G1 in one outcome stage: 2,000 case resamples, seed 42,
    95% CI lower bound above numerical zero. Two named per-comparison tests, not familywise 5% evidence.
    Top-20 capture/false alarms, simulated quality, displaced-case Dice and paired CIs are prespecified
    secondary checks. An AUC gain with worse fixed-budget quality must be reported as a tradeoff.
  - Froze **100 IDs at seeded positions 101–200, zero kits100 overlap**, **20 file fingerprints**, including
    **2 original checkpoint hashes**. `results/kits_missed_fresh/{lock.json,cohort.csv,run_config.yaml}`.
    Keep the 2 prior G3 smoke cases in the already agreed pool and disclose their earlier feasibility/GT
    exposure. "No fresh outcomes" here means no new joint ranking outcomes, not that these two prior reports
    never existed. The [official KiTS23 description](https://kits-challenge.org/kits23/) supports case-based
    patient sampling; independence remains a stated inference, not a separate identity audit.
  - The isolated runner has freeze/old-data-benchmark/preflight/GT-free-score/one-time-evaluate stages.
    It refuses missing coordination, nonbaseline checkpoint provenance, changed scientific files, wrong
    case coverage, misaligned grids and nonstandard probabilities. Scores are saved before GT, and an
    exclusive marker prevents silent outcome reopening. No shared `src/segreview/` edits or new inference.
  - **19 new synthetic tests** pass, including the complete synthetic joint outcome stage and its reopening
    refusal. Full suite **186 passed, 36 skipped, 8 existing warnings, 3.15 s**;
    `results/kits_missed_fresh/tests.txt`. Old-data benchmark **43.39 s**, reproducing saved scores and both
    model Dice on case_00574, case_00563 and case_00135 (small/median/large development grids).
    Metrics/secondary-bootstrap projection **67.76 s** with safety factor 2, plus actual Dice I/O scaling
    when fresh header sizes are available; the total fresh stage duration is not yet known.
  - Preparation audit passed: exact frozen cohort, source/checkpoint/config/code hashes, unchanged earlier
    results, and no fresh headers/voxels/scores/outcome marker. `preparation_verification.json` in the result
    folder. CPU AMD Ryzen 5 7500F, Linux 7.3.1-1-cachyos, Python 3.12.15, torch 2.2.2+cu121 / CUDA build 12.1,
    6 threads, seed 42, no GPU. Stage metadata records baseline `a9bf46b`, dirty state and library versions.
  - **Actual test still pending.** No G3 prediction directory exists locally; the expected
    `configs/g3_kits.yaml` was not found on the available Frederik branch. Otto replied that he will check
    coordination and file location and return. Preflight deliberately reports `ready: false` without
    opening fresh headers. No notification or confirmation has been invented, and no direct message sent.
    [Handoff/message draft](../missed_errors_fresh_handoff.md) and `handoff_template.json` are ready.
  *Input handoff update (2026-11-03):*
  - Otto supplied the model maps at `data/predictions/g3_pool/clean`: **600 NIfTI files, 283,184,552 bytes**,
    exactly **100 frozen IDs × 6 map kinds**, with no missing or unexpected map filenames. This is a
    filename/presence audit only; fresh headers, voxel arrays, scores and outcomes remain unopened.
  - All **100 CTs and 100 labels are absent** locally. Otto relayed Bjørn's instruction to reuse Frederik's
    `g3_pool.zip`, retrieve `configs/g3_pool.yaml` from `frederik/g3`, and lock before opening the pool.
    Copied that config unchanged. It declares the original model settings and no pool ranking evaluation.
    Also retrieved G3's `prepare_machine.json` (RTX 4090, Windows 11, torch 2.2.2+cu121); it has no checkpoint
    hashes. `handoff_sources.json` records sources and the limits of this provenance evidence.
  - The upstream config reproduces the frozen IDs, but the **local shared selector ignores its offset**:
    using the generic download command here selects 100 kits100 IDs and 0 pool IDs. Use isolated
    `scripts/prepare_fresh_download.py`; it verifies the frozen cohort and preserves existing files/list.
    No shared code or scientific recipe change. Offline `download_plan.csv` lists the exact 200 missing files.
  - **7 new logistics tests passed; full suite 193 passed, 36 skipped, 8 existing warnings, 4.10 s**.
    `upload_tests.txt`, `upload_audit.json` and `upload_inventory.csv` record this handoff check separately
    from the preserved 2026-11-02 preparation snapshot. All 20 pinned scientific files remain unchanged.
  - Download approval has been requested under Otto's standing rule; no CT/label download has started.
    Complete the source handoff/provenance and data/header/runtime checks before the joint test.
  *Approved download result (2026-11-03):*
  - Otto approved the download. All **100 CTs and 100 labels**, **8,262,473,787 bytes**, are now local
    under `data/raw/kits23/{images,segmentations}/`. Exact plan coverage, nonempty files, **200 SHA-256
    fingerprints** and no remaining planned `.part` files verified. `download_inventory.csv` and
    `download_verification.json` record the result. These are byte/presence checks, not header/geometry checks.
  - A remote connection reset stopped attempt 1 after **195 files**; the identical retry preserved those
    files and fetched the last **5 in 12.85 s**. All files were local **368 s after approval was recorded**,
    within the original 1,800 s budget. Approval, both attempt logs and retry machine/result metadata saved.
  - `handoff_inputs.csv` now lists all 100 cases with existing CT, label and four required model-map paths.
    All **20 scientific fingerprints, 61 locked G1 result files and 600 kits100 maps** are unchanged.
    No fresh headers/voxel arrays, scores, Dice or outcome stage opened. No inference or shared-code edits.
  - Full suite **193 passed, 36 skipped, 8 existing warnings, 3.86 s**, `download_tests.txt`.
    CPU AMD Ryzen 5 7500F, Linux 7.3.1-1-cachyos, Python 3.12.15, torch 2.2.2+cu121, no GPU used;
    machine/code/lock details are saved in the download metadata/verification. Seed remains 42.
    Next: complete baseline provenance/coordination evidence, then header and full-runtime preflight.
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
- **2026-11-02 — Follow the predeclared step 4 selection rule.** Recommend `g1_missing_hull_mean`
  provisionally by full-cohort threshold AUC and the unchanged budget gate. Its +0.0300 AUC difference
  has a CI including zero; top-20 bad capture does not increase and quality at 20 worsens. Step 5 must
  state these limits, lock exact rules and coordinate the one fresh evaluation before opening the pool.
- **2026-11-02 — Deprioritize G1+hull maximum and post-outcome tuning.** G1+hull max finds only 10/15
  bad cases in 20, failing the budget gate. Do not switch the primary rule to first-10 performance or
  worst10, add example-specific weights, or select extra fresh candidates after seeing these outcomes.
  Keep all six results and the severe displaced case visible instead of optimizing away this finding.
- **2026-11-02 — Do not open extra CT montages for the two earlier side false alerts in step 4.** Both
  remain outside the selected candidate's top 20. Aggregate rank/displacement accounting answers their
  immediate budget cost; future image review remains an option if a later question requires it.
- **2026-11-02 — Freeze one candidate and joint execution before fresh ranking outcomes.** Keep the
  selected mean-of-three definition, 5/6 mm borders, 20 ml components, exact rank ties and original weights.
  Pin the next 100 IDs and both checkpoint hashes. Preserve the 2026-10-21 far test and report both named
  per-comparison CIs openly; no claim about a familywise 5% error rate. Do not switch scores after viewing the pool.
- **2026-11-02 — Do not substitute smoke tests or new inference for the pending joint evaluation.** Two
  earlier G3 smoke cases are not a 100-case ranking validation. Keep and disclose them rather than replace
  them after prior GT exposure. Frederik's baseline outputs must be reused once their location/provenance
  is confirmed; no download or >30-minute stage starts without Otto's go-ahead.

- **2026-11-02 — Pin the dataset definition before publication.** Final review found that the dataset YAML
  (organ label definition) also needs a fingerprint. Added it and archived the unpublished 19-file draft in
  `results/kits_missed_fresh/prepublication_draft/`. The final 20-file lock, benchmark and pending preflight
  are regenerated; cohort, scientific settings, code and checkpoints are unchanged. No fresh outcomes opened.

- **2026-11-03 — Preserve the lock while adapting input logistics.** Retrieve Frederik's pool config unchanged,
  but do not cherry-pick changes to the shared data selector: that would change a pinned scientific dependency
  and requires coordination. The isolated download helper implements only the frozen cohort's file transfer.
  No new model inference, pool replacement, score change or outcome access is needed.
- **2026-11-03 — Resume only missing transfers after a connection reset.** Reuse the approved identical plan,
  preserve its 195 complete files and retrieve only the last five within the original total time budget.
  Reject restarting the whole download: it adds transfer time without improving the agreed input set.
  No source-list rewrite or new model inference is needed to recover a transfer error.

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action,
and ideas for what could make the result better.)*

- **2026-11-03 — Last fully finished step: 4. Step 5 scientific lock/preparation is complete; its fresh
  evaluation is not.** Recipe/code, next 100 IDs and baseline checkpoint hashes are pinned in
  `results/kits_missed_fresh/lock.json`; protocol and tested runner are ready. No fresh model/label arrays,
  scores, ranking metrics or outcome marker have been opened/created. Prior G3 smoke exposure is disclosed.
  Development result remains weak: AUC +0.030 [-0.032, 0.086], no top-20 gain and worse quality at 20.
- **Current inputs/dependency:** all 600 expected model maps are local at `data/predictions/g3_pool/clean`;
  all 100 CTs and 100 labels are now local at `data/raw/kits23/{images,segmentations}/` after Otto's approved
  download. Exact plan coverage and 200 byte fingerprints verified; `download_verification.json`.
  `handoff_inputs.csv` lists the 100 cases and existing paths. Bjørn's reuse/config/lock instruction
  was relayed by Otto; source config and G3 preparation-machine metadata are retrieved. Run-specific weight
  hashes are absent from that metadata; do not fabricate confirmation.
- **Next concrete action:** complete the real source provenance/coordination evidence and handoff metadata,
  then verify exact IDs/weights/headers/hashes and project full runtime. The path manifest is ready;
  no more CT/label transfer or inference is needed for the frozen cohort.
  Ask before downloads or any projected >30-minute stage. Then freeze all GT-free scores and run both
  questions together in one outcome stage by 2026-11-18. Keep step 5 unchecked until real results exist.
- **Practical improvements:** (1) finish the existing-output provenance handoff so sharing does not trigger
  duplicate inference; (2) preserve the prespecified quality-at-20/displaced-error
  outputs so an apparent AUC gain cannot hide a worse practical budget. No weights/criteria changes planned.
- **Publication:** [draft PR #2](https://github.com/Ottohesten/02830-medical-segmentation/pull/2) tracks
  the ongoing task. The prospective lock/preparation is committed as **`7703fac`** and confirmed pushed to
  `origin/otto/missed-errors`; the PR title/body now describe the final lock and pending evaluation.
  The 2026-11-03 upload audit, source config and exact download helper/plan are published as **`a67a013`**;
  draft PR #2 includes the received maps, missing CTs/labels and pending download approval.
  The approved download is finished and its verification/notes are ready for commit/push; record publication
  before ending the session.
  Step 4 remains published as `7648905`.
  Step 2's CT-derived montages remain local/ignored. Group review remains required before merging.
  No direct message to Bjørn/Frederik has been sent.
