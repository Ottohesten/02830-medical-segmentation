# Project log — Where to Look First

Everything we have tried, measured, decided, kept or dropped, with dates. Entries are kept even when the idea was
dropped, so the reasons stay on record for the report and the defense.

- **Order:** oldest first, grouped by date. Add new entries at the bottom.
- **Date:** every entry starts under a `## YYYY-MM-DD` heading (use the date the work was done).
- **Tags:** `[result]` a measured number · `[experiment]` something we tried · `[decision]` a choice we made ·
  `[dropped]` an option we rejected, with the reason · `[idea]` not tried yet · `[supervisor]` input from Kristine ·
  `[admin]` setup, tools, course logistics.
- **Numbers** come from code that was actually run. Say where the file is (e.g. `results/kits100/...`).
  Exploratory numbers that are not (yet) in the repo are marked **exploratory**.
- Entries before 2026-10-19 were reconstructed afterwards from the git history and the group's notes.
  Dates marked "approx." are uncertain.

Overview of the project and where everything is: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

---

## approx. 2026-09-02 – 2026-09-15 (weeks 1–3): choosing and framing the project

- `[decision]` Work on Kristine Aavild Sørensen's medical-imaging pitch (`medical_imaging_kajul_E2026.pptx`).
  Reason: the pitch defines its own evaluation axis, "Dice per unit human effort", and public datasets exist.
- `[decision]` The contribution is the human–AI review layer (prioritizing, reviewing, correcting), not a new
  segmentation model. A strong existing model is infrastructure.
- `[supervisor]` Pitch slide 8: "Not about optimizing Dice, but about optimizing accuracy per unit human effort."
  Existing models are "a starting point – not the project contribution".
- `[dropped]` Building or optimizing a segmentation architecture — against the pitch, weeks of training, open-ended risk.
- `[decision]` Four directions were considered: (1) faster correction tools, (2) uncertainty-guided review,
  (3) ranking cases by predicted quality, (4) learning from corrections. We combine them instead of picking one.
- `[dropped]` (1) faster correction tools (brush vs. scribbles vs. click prompts) as the main project.
  Interactive prompt models (SAM2 / MedSAM) were looked at but never adopted.
- `[decision]` Learning from corrections stays in scope only as a side experiment where ground truth plays the reviewer.
  The model stays frozen during user tests.
- `[dropped]` Fine-tuning on corrections from the real study — about 10 participants × 10 scans is far too little.
  Letting the model learn during the study — later participants would get a better model, which mixes up interface
  and model effects.
- `[decision]` Pretrained nnU-Net weights (TotalSegmentator), inference only in G1/G2; training only in G3.
- `[decision]` Staged build order: (1) pretrained model end-to-end with Dice, (2) simple ranking signal,
  (3) review interface tested early, (4) learning loop with a simulated reviewer, (5) trained quality predictor.
  1–3 are core, 4–5 can be dropped.
- `[dropped]` A strictly sequential plan with a trained quality predictor first — GPU-weeks, one delay blocks
  everything, user testing would land in weeks 10–13.
- `[decision]` Dataset criterion: errors must be visible to non-clinicians (organs with recognizable shapes).
- `[dropped]` ImageCAS (coronary arteries) — thin vessels, errors hard to judge for lay people.

## 2026-09-15 / 16: lightning presentation, repository

- `[decision]` Title: "Where to Look First: Prioritizing Human Review of AI Medical Image Segmentation"
  (generic, no organ, since the dataset was not chosen yet).
- `[dropped]` "Uncertainty Segmentation/Quantification of CT-based scans of Brain Tumors" — brain tumors are imaged
  with MR, not CT. "Dice per Minute: …" — saved for the report.
- `[decision]` Motivation written as a need (slow manual review is the bottleneck), not as a list of features.
- `[admin]` Repository created by Otto (`Initial commit`, `uv initial commit`).

## 2026-09-22: premortem, goals, plan

- `[decision]` Goals. **G1 (core):** rank scans by likely failure without ground truth. **G2 (core):** a review
  interface that guides the reviewer to uncertain regions, tested in a user study. **G3 (stretch):** fine-tune the
  model on corrections. Out of scope: new architectures, clinical validation, deployment.
- `[decision]` Plan: tracer bullet weeks 4–5; G1 ("the monkey", hardest part first) weeks 5–7; interface in parallel
  on mocked scores weeks 6–9, pilot week 8; user study weeks 9–10; G3 weeks 10–11; paper weeks 11–13.
  Go/no-go on G1 after week 7; fallback = region-level guidance and report a negative result.
- `[dropped]` Training our own model first (premortem story: six weeks lost). User test only at the end (premortem
  story: a polished UI with nothing tested before week 12).
- `[decision]` G1 evaluation: review curve ("failure detection vs. share of scans sent to review") and quality curve
  ("final quality vs. amount of correction"), each summarized by the area under the curve (AUC). References: random
  order, predicted organ volume, oracle (true Dice). Spearman ρ is secondary.
- `[supervisor]` These outcome measures are the ones on Kristine's pitch slide 8.

## 2026-09-23: tracer bullet, environment, first G1, dataset switch

- `[decision]` G2 outcome per corrected scan: (Dice after − Dice before) / minutes. Participants edit the model's mask.
- `[dropped]` Dice after correction alone — mostly measures how easy the scan was.
- `[decision]` Within-subject study: each participant corrects 3 scans with and 3 without the heatmap, sees each
  scan once; scans and condition order balanced over 4 participant types. Paired Wilcoxon test.
- `[dropped]` The same person seeing a scan both with and without heatmap (learning effect); fixed scan–condition
  pairs (scan difficulty would be mixed up with the heatmap).
- `[decision]` Environment: uv, Python 3.12, torch 2.2.2, numpy 1.26.4, TotalSegmentator 2.18.0, nnunetv2 2.8.1;
  one lock file for Intel Mac, Apple Silicon Mac and Linux (HPC); one YAML config per environment.
- `[dropped]` Python 3.13 — no torch build for Intel macOS.
- `[decision]` Repository moved out of OneDrive (syncing `.venv` and gigabytes of data caused risk of conflicts).
- `[decision]` nnU-Net is called directly with TotalSegmentator's preprocessing, so we get probabilities, not only the
  mask. The 117 classes are reduced at once to one organ probability map (stored as uint8).
- `[dropped]` The TotalSegmentator command line (mask only); storing all 117 class probabilities (gigabytes per scan).
- `[decision]` Low-resolution 3 mm model (`fast`, task 297), fold 0, on CPU locally. Its mask is what reviewers correct.
- `[dropped]` Full resolution 1.5 mm (task 291) locally — too slow; left for the HPC.
- `[experiment]` Tracer bullet on MSD Task09 Spleen (41 CT scans).
- `[result]` Bug found and fixed: the first run gave Dice 0.688 because of a wrong axis order; after the fix 0.959.
  Checked against the official TotalSegmentator with the same model: Dice between the two masks 0.9625.
- `[result]` MSD Spleen, 41 scans: Dice median 0.960 (min 0.866, SD 0.018). Only one real failure: spleen_44 (0.866),
  a scan without contrast. Spearman(Dice, spleen volume) = +0.61: small spleens get lower Dice.
- `[decision]` Add predicted organ volume as a baseline, since a score could look good just by tracking size.
- `[result]` G1 on spleen (41): no score beats the volume baseline (all confidence intervals include 0). Not a G1
  result — there is essentially one failure to find.
- `[decision]` Spleen is too easy; evaluate on data with natural variation (disease, many hospitals) and, as a
  supplement, controlled perturbations (noise, contrast, resolution), reported separately.
- `[result]` Perturbations on 3 spleen scans: low resolution 8 mm clearly lowers Dice (0.935 / 0.938 / 0.904),
  noise slightly, contrast 0.6 not at all.
- `[decision]` Uncertainty scores: voxel entropy (several summaries), test-time augmentation (TTA) and the
  disagreement between TotalSegmentator's 3 mm and 6 mm models.
- `[dropped]` Fold disagreement — only fold 0 of the weights is released. Mirroring as TTA — the model was trained
  without mirroring; we use small shifts, intensity changes and noise instead.
- `[supervisor]` Only publicly available data. The Rigshospitalet research database is ruled out.
- `[decision]` Switch to KiTS23 (kidneys with tumors, CT from many referring hospitals). Pilot: 20 random cases, seed 42.
- `[dropped]` BraTS (MR), ImageCAS (vessels), ACDC (cardiac MR) — no usable TotalSegmentator class or errors not
  visible; TotalSegmentator dataset — it is the model's training data; MSD Pancreas — errors hard to see even for
  clinicians; MSD Liver (28.9 GB) and AMOS (24.2 GB) — too big for the disk; the first 20 KiTS cases instead of a
  random 20.
- `[decision]` Target = kidney + tumor + cyst (KiTS labels 1–3) vs. TotalSegmentator's kidney and kidney-cyst classes.
- `[result]` KiTS pilot: TotalSegmentator labels a median of 81 % of tumor voxels as kidney.
- `[dropped]` Kidney + cyst with tumor as background — punishes the model for a definition difference (median Dice
  0.904 → 0.855). Ignoring tumor voxels — hides the regions with the big errors.
- `[result]` KiTS pilot (20): the worst scans are real model errors, checked visually: case_00582 (Dice 0.46, large
  tumor missed), case_00465 (0.47, large cystic mass missed), case_00040 (0.66, kidney in the pelvis not found at all).
- `[result]` case_00040 is a *confident* error: the model gives probability 0, so neither entropy nor 3 mm/6 mm
  disagreement can catch it. Limitation for the report.
- `[idea]` A stopping threshold on the uncertainty score (e.g. calibrated to Dice < 0.8). Discussed, not adopted.

## 2026-09-24: G1 locked and tested

- `[decision]` kits100: 100 random KiTS cases (seed 42). The 20 pilot cases are the development set, the other 80 the
  test set. Both lists are written into `configs/kits100.yaml`.
- `[dropped]` Staying with 20 cases (only 4 bad cases, very wide intervals); evaluating on all 100 including the pilot
  (choices made on the pilot would inflate the result).
- `[experiment]` Overnight run on the Intel Mac (00:50–04:58): 3 mm and 6 mm on 100 scans. TTA skipped on KiTS
  (15–20 h estimated; left for the HPC).
- `[decision]` All G1 choices locked on the development set at 01:43, before the test set was opened (04:59):
  - Primary score `model_diff_mean_region` = mean |p₃ₘₘ − p₆ₘₘ| in the kidney + 5 mm (most negative Spearman ρ on dev, −0.683).
  - Second locked score `combo_mean_missing` = primary score combined with a ground-truth-free plausibility check
    (is a kidney missing on one side?), motivated by the pelvic kidney.
  - "Bad segmentation" = worst 10 % and, separately, Dice < 0.83 (middle of the gap on dev between 0.802 and 0.855).
  - "Beats volume" = lower end of the 95 % CI of the AUC difference above 0.
- `[dropped]` Left/right volume asymmetry for the plausibility check — noisy, healthy scans reach about 0.5.
- `[decision]` Tied scores use the expected curve over random order within ties (changed 01:35, before the lock).
  It changes no primary test number.
- `[result]` **Final G1 result, test set (80 scans)** (`results/kits100/g1_metrics_clean_test.csv`):
  - Dice of the 3 mm model: median 0.910, min 0.003, max 0.950. 8 bad (worst 10 %), 11 with Dice < 0.83.
  - Primary score: Spearman ρ −0.63 [−0.76; −0.47]. Review-AUC 0.855 vs. volume 0.488 (+0.37 [0.09; 0.65]);
    with Dice < 0.83: 0.831 vs. 0.390 (+0.44 [0.23; 0.66]); quality-AUC 0.954 vs. 0.939 (+0.015 [0.007; 0.025]).
    Beats the volume baseline on all three.
  - Oracle 0.950 / 0.931 / 0.959; random 0.506 / 0.505 / 0.935. About 79 % of the possible gain over random.
  - Plausibility combination: slightly weaker than the primary score alone; adds nothing on test.
- `[decision]` Keep the locked primary score.
- `[result]` Exploratory: entropy (`entropy_mean_region`) did better on test: ρ −0.80, review-AUC 0.894.
- `[dropped]` Switching to entropy after seeing the test result — that would be choosing the method on test data.
  Entropy is reported as a secondary finding.
- `[decision]` G1 is a preliminary "go"; the fallback is not needed (group to confirm).
- `[decision]` Heatmap for reviewers = voxel entropy, chosen on dev by how well it points at wrong voxels.
- `[result]` Voxel-AUC dev / test: entropy 0.875 / 0.864; |p₃ₘₘ − p₆ₘₘ| 0.724 / 0.718; distance to the mask edge
  (reference) 0.849 / 0.841. Entropy beat the edge reference in 76 of 80 test scans.
  (`results/kits100/heatmap_eval_*`)
- `[result]` Known risk: entropy is highest exactly on the mask edge, so the heatmap mostly highlights the outline.
  The effect in G2 may be small.
- `[decision]` Own browser interface built on NiiVue plus a small Python server (logging, saving, conditions).
- `[dropped]` 3D Slicer, ITK-SNAP, MONAI Label — less control over the heatmap condition and timing, many tools that
  confuse non-clinicians.
- `[decision]` The server enforces the conditions: no heatmap in the "without" condition, no ground truth for study scans.
- `[decision]` Study scans: Dice 0.70–0.90, error 50–400 ml, both kidneys found. Blocks A: case_00470/00259/00460,
  B: case_00554/00438/00426; practice case_00184. 5-minute cap per scan.
- `[experiment]` First self-test of the interface by Bjørn. Findings: the heatmap covered the mask, drawing gave no
  visible feedback.
- `[decision]` Fixes: English UI, mask always above the heatmap, see-through fill + outline, blue heatmap, round
  visible brush, full trackpad control, slice slider with uncertain slices marked, guide + practice scan with the
  answer afterwards, demo queue without time cap. Chrome on a MacBook trackpad only.
- `[dropped]` External mouse (everyone uses trackpads); Firefox (drawing does not work); warm heatmap colors
  (hard to tell from the red mask); showing ground truth during study scans.
- `[decision]` Test runs use T-ids (T01 …), never analyzed; a cleanup script deletes only T-id data.

## 2026-09-25: study design details, report figures, checks

- `[decision]` American English everywhere (code, docs, UI, title).
- `[decision]` Raw NASA-TLX after each condition block, so workload with and without heatmap can be compared per person.
- `[dropped]` NASA-TLX once after the session — no paired comparison possible.
- `[result]` Power simulation (exact Wilcoxon, 10,000 runs): for 80 % power, d_z 0.5 → 36 participants, 0.8 → 16,
  1.0 → 12. (`results/study_kits/power_*.csv`; assumed effect sizes, not results.)
- `[decision]` Recruit at least 16 participants (multiple of 4). Group members only as pilots.
- `[decision]` Study materials in `docs/study/`: session guide, consent form, pilot plan.
- `[result]` Report figures and tables for G1 (`results/kits100/report/`).
- `[result]` Reproducibility check in a fresh clone: all G1 result files byte-identical, all heatmaps voxel-identical.
- `[admin]` HPC job scripts written (`jobs/`), never submitted to the cluster.

## 2026-09-30: heatmap display modes, analysis plan, G3 plan

- `[experiment]` Heatmap display modes for the pilot: `full` (default), `top` (10 % most uncertain voxels),
  `contour` (line around them).
- `[result]` On dev: `full` covers 41.7 % of the region and catches 93.4 % of error voxels; `top 10 %` catches 43.5 %
  (edge reference 40.6 %); 98.6 % of the `top` voxels lie within 2 mm of the mask edge.
- `[dropped]` Smoothing the entropy into "regions" — caught fewer error voxels (34 % at 4 mm vs. 41 % for the edge).
  A 1-voxel contour — almost invisible.
- `[decision]` The pilot decides the display mode.
- `[decision]` Analysis plan (`docs/study/analysis_plan.md`, to be locked after the pilot): H1 (gain per minute
  higher with heatmap) primary; H2 (lower NASA-TLX) only if H1 holds. Two-sided Wilcoxon signed-rank, α = 0.05,
  exact p-values, rank-biserial r with bootstrap CI. Complete sessions only, no outlier removal.
- `[dropped]` scipy's Wilcoxon (normal approximation with ties); scipy's permutation method (too slow).
- `[decision]` G3: plan and a feasibility smoke test only (`docs/g3_plan.md`). Ground truth plays the reviewer on the
  top 10/20/40 of the queue, on KiTS cases outside kits100. Compare: original model / queue + correction /
  random + correction / queue without correction. Evaluate on the locked 80 test scans.
- `[result]` Smoke test (`results/g3_smoke/`): fine-tuning can start on a Mac. Fine-tuning must keep *all* pretrained
  weights: loss on one batch with random weights 5.38, nnU-Net's default start (last layer reset) 7.86, all weights
  −0.77. Time: about 16 h for 2,500 steps on the Intel Mac vs. about 20 min on an A100 GPU. Real training needs the HPC.
- `[dropped]` Building G3 fully now — no real correction data, no GPU, G2 is the core result.
- `[admin]` Status e-mail to Kristine sent.

## 2026-10-01 – 2026-10-02: report draft, new laptop

- `[decision]` The G1 method section is drafted in LaTeX (ACM two-column); the group edits.
- `[admin]` Development moved to a MacBook Air (Apple Silicon, 16 GB).
- `[decision]` The locked G1 results must only be reproduced from the saved predictions, never recomputed with new
  inference (small numeric differences would break the locked result).

## 2026-10-06: Kristine's reply, exploratory 6 mm analysis

- `[supervisor]` Kristine (e-mail): we seem to be on the right track. She did not expect the 3 mm and 6 mm models to
  differ enough to give a good estimate and wants to discuss what the disagreement tells us and why they perform so
  differently. Her question for us: are the scans with the biggest errors necessarily the ones where human review
  helps future model training most? Answering it needs a model we train, or can continue training.
- `[experiment]` **Exploratory** (from saved predictions, not yet in the repo): how good is the 6 mm model on its own?
- `[result]` **Exploratory**, test set (80): Dice median 3 mm 0.910 vs. 6 mm 0.894; mean 0.870 vs. 0.848;
  Dice < 0.83 in 11 vs. 16 scans; 3 mm better in 68 of 80. Development set shows the same pattern.
- `[result]` **Exploratory**: of the top 10 in the locked queue, 6 are real finds (3 mm mask is bad:
  case_00192, 00516, 00481, 00107, 00211, 00292) and 4 are false alarms where the 3 mm mask is good (0.88–0.92) but the
  6 mm model fails (case_00280, 00079, 00203, 00526). So part of the queue's noise comes from the "second opinion"
  model being wrong itself, which may explain why plain entropy did a little better on test.
- `[result]` **Exploratory**: in bad scans, about 85 % of the error voxels are missed tumor; only 7 % of the 3 mm
  error voxels lie in the region where the two models disagree.

## 2026-10-07: meeting with Kristine, decision log

- `[supervisor]` Meeting with Kristine. Agreed next steps:
  1. Measure how good the 6 mm model is on its own (done exploratorily on 2026-10-06; to be added to the repo).
  2. Find where the 3 mm and 6 mm models differ: is it only the coarser grid (a thin band along the edge), or real
     differences, e.g. around tumors?
  3. Measure how much overall quality improves when the most uncertain parts are corrected to ground truth.
     At scan level this is our quality curve (quality-AUC 0.954 vs. volume 0.939, oracle 0.959); inside a scan
     (correcting the most uncertain voxels) it has not been measured.
  4. The most uncertain case or region is not necessarily where a correction helps most: it may be an outlier
     unique to one patient that never occurs again. Relevant for which corrections are worth using for training (G3).
- `[admin]` Python 3.14 considered for the new laptop and dropped (torch 2.2.2 needs ≤ 3.12; locked results must be
  reproducible). `.venv` set up with Python 3.12.

## 2026-10-14: work from 2026-09-30 recovered

- `[admin]` The 2026-09-30 work (heatmap display modes, analysis plan, G3 plan and smoke test) existed only on the old
  Intel Mac. Copied to the new laptop together with the TotalSegmentator weights; 86 tests passed; pushed as `812e988`.

## 2026-10-19: project log and overview

- `[decision]` Keep this log in the repository so the whole group can see what was tried and why. Every result,
  experiment, idea and decision is logged with a date, also when it is dropped.
- `[decision]` Add [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) as the starting point for anyone who opens the project.
- `[decision]` Add [CONTRIBUTING.md](CONTRIBUTING.md) with shared working rules: own branch + pull request, tests before
  push, locked kits100 results, one config per experiment, models compared only within one machine, and what to log
  for long runs (config, commit, machine, training log, per-scan predictions and Dice) so questions can be answered
  without training again.
- `[decision]` Runs on different machines are allowed; the machine is recorded for every run and the possible
  differences are discussed in the report. Global seed 42 for everything random (already used in all configs).
- `[decision]` G3 runs in parallel with the user study instead of after it (Kristine's training question), on
  Frederik's PC with an NVIDIA RTX 4090 (24 GB) instead of the DTU HPC. Frederik owns G3. First step: run the smoke
  test on the GPU and measure the time per training step; then build the pipeline with the 6 mm model.
- `[dropped]` Waiting for HPC access — nobody has it set up, and a local GPU is available now.
- `[idea]` Add the 6 mm-alone analysis from 2026-10-06 to the repository as a script and result file, so the numbers
  can be reproduced (from saved predictions only, no new inference).


## 2026-10-19: follow-up on the 2026-10-07 meeting (exploratory)

All three analyses below are **exploratory**: they were asked after the G1 choices were locked, so they cannot change
the locked G1 result. They use only the saved kits100 predictions (no new inference), report the dev (20) and test
(80) sets separately, and leave the locked G1 files unchanged (checked with checksums before and after).
Machine: MacBook Air (Mac17,3, Apple M5, 16 GB), macOS 27.0.1, Python 3.12, torch 2.2.2, CPU only
(`results/kits100/model_comparison_machine.json`, `voxel_correction_machine.json`). Settings: `exploratory:` in
`configs/kits100.yaml`; seed 42 (random voxel order averaged over seeds 42–46).

- `[decision]` The settings go under a new `exploratory:` key in `configs/kits100.yaml` and the results in
  `results/kits100/`, not in a separate config: the analyses only read kits100's saved predictions, and a copy of
  the config could drift from the locked one.
- `[experiment]` 6 mm model alone: Dice of the 3 mm (`mask`) and 6 mm (`m2_mask`) masks against ground truth
  (kidney + tumor + cyst, same Dice function as `dice.csv`) for all 100 scans (`scripts/compare_models.py`).
  The 3 mm Dice is identical to `dice.csv` for all 100 scans.
- `[result]` **Exploratory**, test (80): median Dice 3 mm 0.910 vs. 6 mm 0.894; mean 0.870 vs. 0.848; Dice < 0.83 in
  11 vs. 16 scans; 6 mm better in 12 of 80 (3 mm in 68); Spearman between the two models' Dice 0.89. Dev (20):
  median 0.904 vs. 0.895, mean 0.848 vs. 0.830, < 0.83 in 4 vs. 7, 6 mm better in 5 of 20, Spearman 0.81.
  `results/kits100/model_dice_3mm_vs_6mm.csv`, `model_dice_3mm_vs_6mm_summary.csv`.
- `[result]` **Exploratory**, locked queue (`model_diff_mean_region`), test: top 10 = 6 real finds (3 mm Dice < 0.83)
  and 4 false alarms (3 mm fine, 6 mm < 0.83: case_00280, 00079, 00203, 00526); top 20 = 9 real finds, 4 false alarms,
  7 where both are fine. Dev: top 10 = 4 real finds, 3 false alarms. `results/kits100/queue_top_3mm_vs_6mm.csv`.
  These numbers match the exploratory calculation of 2026-10-06 exactly (medians, means, counts below 0.83,
  "3 mm better in 68 of 80", and the same 6 + 4 scans in the top 10), which is now reproducible from the repository.
- `[experiment]` Where the 3 mm and 6 mm masks disagree: every disagreeing voxel by ground-truth label, distance to the
  3 mm mask's edge (≤ 6 mm ≈ one 6 mm voxel = grid effect; farther = real difference) and which model is right.
  Groups: bad (3 mm Dice < 0.83), false alarm (3 mm fine, 6 mm < 0.83), good (both fine).
- `[result]` **Exploratory**: **about 89 % of the disagreeing voxels lie within 6 mm of the 3 mm edge (grid effect;
  median scan 99 %); the other ~11 % are real differences, and on test 79 % of those are tumor.** Pooled over
  voxels: dev 89 % / test 89 % near the edge. The real differences concentrate in the scans that matter: test bad
  scans 32 % far from the edge (87 % of it tumor), false alarms 26 % (79 % tumor), good scans 3 %. In false alarms
  the 3 mm model is right in 99 % of the far voxels (the 6 mm model leaves out tumor the 3 mm model includes); in
  bad scans only in 49 % (pooled; median scan 22 %), i.e. there the 6 mm model often includes tumor the 3 mm model
  missed. On dev the far part is 45 % tumor and 38 % kidney (one scan, case_00135, dominates). Near the edge the
  3 mm model is right in 61–67 % of the voxels. `results/kits100/model_disagreement_{counts,per_case,summary}.csv`,
  `figures/model_disagreement_where.png`.
- `[experiment]` Simulated correction inside a scan: set the k % most uncertain voxels of the G1 region (predicted
  kidney + 5 mm) to ground truth, k = 1, 2, 5, 10, 20, 50, and measure the Dice gain. Orders: entropy (locked
  heatmap), |p_3mm − p_6mm|, distance to the edge (reference), random (mean of seeds 42–46), oracle (wrong voxels
  first). Ties at the cut-off are shared in proportion, as in the heatmap evaluation (`scripts/simulate_voxel_correction.py`).
- `[result]` **Exploratory**, test, mean Dice gain at k = 10 %: entropy +0.034, edge distance +0.032,
  |p_3mm − p_6mm| +0.017, random +0.008, oracle +0.067. Area under the gain curve for k = 0–10 % (mean gain over that
  range): entropy 0.018 (51 % of the oracle), edge 0.016 (46 %), |Δp| 0.009 (25 %), random 0.004 (12 %); for 0–50 %:
  0.051 / 0.048 / 0.034 / 0.021, oracle 0.069. Entropy beats the edge reference in 77 of 80 test scans (19 of 20 dev)
  on the 0–10 % area, but only by a little. Dev shows the same order and nearly the same numbers.
  `results/kits100/voxel_correction_{summary,auc_summary,auc,per_case}.csv`, `figures/voxel_correction_gain.png`.
- `[result]` **Exploratory**: errors outside the region (e.g. a whole missed kidney) can never be fixed this way. Per
  scan, the share of error voxels outside the region is median 13 % on test (17 % dev), mean 21 % (28 %); pooled over
  all voxels 48 % (37 %), because a few scans with large misses dominate. Share of the Dice gap that cannot be closed
  inside the region: median 12 % (16 %), mean 20 % (26 %). Best reachable mean Dice when the whole region is
  corrected: 0.949 on test (0.927 dev). `results/kits100/voxel_correction_cases.csv`.
- `[decision]` "Oracle" = all wrong voxels first, as asked. It fixes the most wrong voxels, but is not a strict upper
  bound on Dice (fixing a missed voxel and a false voxel change Dice by slightly different amounts).
- `[idea]` The model-disagreement score mixes the grid effect (most of the disagreement) with real tumor
  differences. A score built only from disagreement farther than 6 mm from the edge might separate the two; this
  would be a new, post-lock question and has not been tried.
- `[idea]` The |p_3mm − p_6mm| heatmap points at fixable voxels worse than plain distance to the edge, which supports
  the locked choice of entropy for the interface; entropy's advantage over the edge reference is small, which matches
  the known risk that it mostly lights up the edge.

## 2026-10-20: scores from the real 3 mm vs. 6 mm differences (exploratory)

**Exploratory**, a new question asked after the G1 lock. Saved kits100 predictions only (no new inference); the
locked files are not changed. For this question kits100 is **development data**: dev (20) and test (80) are reported
separately, and a score chosen here must later be tested once on the G3 pool before it can be claimed.

- `[experiment]` Background (2026-10-19 entries): ~89 % of the 3 mm vs. 6 mm disagreement is the grid effect at the
  edge; the real differences (> 6 mm from the 3 mm edge) are mostly tumor. In bad scans the 6 mm model often includes
  tumor the 3 mm model missed; in false alarms it leaves out tumor the 3 mm model included. Hypothesis: counting only
  far disagreement where "6 mm says organ, 3 mm says not organ" ranks better and gives fewer false alarms.
- `[decision]` Written down before anything was computed (config `exploratory.disagreement_scores`). Candidates, all
  GT-free, each in ml and divided by the 3 mm mask volume (8 scores): (a) `far_disagree`: masks disagree and > 6 mm
  from the 3 mm edge, both directions; (b) `far_6mm_adds`: as (a), only 6 mm = organ and 3 mm = not organ;
  (c) `far_6mm_drops`: as (a), only 6 mm = not organ and 3 mm = organ (control, expected poor); (d) `6mm_adds_all`:
  6 mm = organ and 3 mm = not organ, no distance filter. Selection: highest review AUC (bad = Dice < 0.83) on dev;
  ties broken by dev quality AUC, then the listed order. Everything else (worst-10 % review AUC, quality AUC,
  Spearman rho, paired bootstrap differences to `model_diff_mean_region` and `entropy_mean_region`, test results,
  false alarms in the top 10 / 20) is secondary. Dev has only 4 bad scans, so the choice is noisy.
- `[experiment]` `scripts/explore_disagreement_scores.py` (307 s; MacBook Air Mac17,3, Apple M5, 16 GB, macOS 27.0.1,
  Python 3.12, torch 2.2.2, CPU; seed 42; `results/kits100/exploratory_disagreement_machine.json`). Evaluated with the
  G1 `evaluate_methods` (2000 bootstrap resamples, 1000 random orders) plus a new paired bootstrap of the difference
  to the locked measure and to entropy. Checks passed: the new GT-free counts equal those rebuilt from
  `model_disagreement_counts.csv` for all 100 scans; 3 mm volume and Dice equal the locked files; the metric rows of
  the locked measure, entropy, volume, random and oracle equal `g1_metrics_clean_{dev,test}.csv` exactly. The 52
  existing files in `results/kits100/` are unchanged (checksums before and after).
- `[result]` **Exploratory**, dev (selection; review AUC, Dice < 0.83, 4 bad): `far_6mm_drops_ml` 0.788,
  `far_6mm_drops_per_volume` 0.775, `far_disagree_per_volume` 0.763, `far_disagree_ml` 0.750, `far_6mm_adds_ml`
  0.706, `far_6mm_adds_per_volume` 0.694, `6mm_adds_all_per_volume` 0.675, `6mm_adds_all_ml` 0.663; locked measure
  0.763, entropy 0.688. **The pre-registered rule chooses `far_6mm_drops_ml`, the control expected to be poor.** All
  dev CIs overlap widely (e.g. chosen vs. locked +0.025 [−0.050, 0.133]). `results/kits100/exploratory_disagreement_choice.csv`.
- `[result]` **Exploratory**, test (80; review AUC Dice < 0.83 / worst 10 %): `far_disagree_per_volume` 0.887 / 0.895,
  `far_disagree_ml` 0.872 / 0.866, `far_6mm_adds_ml` 0.848 / 0.816, `far_6mm_adds_per_volume` 0.847 / 0.827,
  `6mm_adds_all_per_volume` 0.840 / 0.814, `6mm_adds_all_ml` 0.768 / 0.702, `far_6mm_drops_ml` 0.747 / 0.716,
  `far_6mm_drops_per_volume` 0.745 / 0.713; locked measure 0.831 / 0.855, entropy 0.895 / 0.894, volume 0.390 / 0.487,
  random 0.505 / 0.506, oracle 0.931 / 0.950. Spearman rho: `far_disagree_per_volume` −0.645, locked −0.633,
  `far_6mm_adds_ml` −0.573, chosen `far_6mm_drops_ml` −0.401, entropy −0.795.
  `results/kits100/exploratory_disagreement_metrics.csv`, `figures/exploratory_review_curves_test.png`.
- `[result]` **Exploratory**, paired bootstrap on test (difference, 95 % CI): the chosen `far_6mm_drops_ml` is worse
  than the locked measure (review AUC Dice < 0.83 −0.084 [−0.225, 0.023]; worst 10 % −0.139 [−0.339, −0.007]; rho
  +0.232 [0.061, 0.400]) and than entropy (review AUC −0.148 [−0.292, −0.041]). `far_disagree_per_volume` beats the
  locked measure (review AUC +0.056 [0.008, 0.131] and +0.041 [0.005, 0.088]; quality AUC +0.0014 [−0.0002, 0.0034])
  and ties with entropy on review AUC (−0.008 [−0.044, 0.028]) but has a weaker rho (+0.150 [0.023, 0.291]).
  `far_6mm_adds_ml` vs. locked: +0.017 [−0.159, 0.158]. `results/kits100/exploratory_disagreement_paired.csv`.
- `[result]` **Exploratory**, false alarms (3 mm fine, 6 mm < 0.83) and real finds (3 mm < 0.83) in the queue's top,
  test: top 10 locked 4 false alarms / 6 real finds, entropy 1 / 8, `far_6mm_adds_ml` 0 / 8, `far_6mm_adds_per_volume`
  0 / 9, `6mm_adds_all` 0 / 7 (ml) and 0 / 9 (per volume), `far_disagree` 3 / 7 and 2 / 8, `far_6mm_drops` 3 / 5 and
  4 / 5. Top 20: locked 4 / 9, entropy 3 / 10, `far_6mm_adds` 2 / 9, `6mm_adds_all` 0 / 8 and 1 / 9, `far_disagree`
  4 / 10, `far_6mm_drops` 4 / 8 and 4 / 7. The 4 false alarms of the locked top 10 move to ranks 42, 19, 13 and 66.5
  under `far_6mm_adds_ml` (case_00280, 00079, 00203, 00526), but the real find case_00192 (locked rank 2) drops to 66.5,
  because the 6 mm model adds nothing far from the edge there. Dev top 10: locked 3 / 4, `far_6mm_adds` 1 / 3,
  `far_6mm_drops` 3 / 4 (dev top 20 = all 20 scans). `results/kits100/exploratory_queue_{counts,ranks}.csv`.
- `[result]` **Exploratory**, summary: the direction filter does what the hypothesis said about false alarms on
  test (none in the top 10), but it does not rank better overall, and on dev the control direction ranked best.
  Removing the grid band alone (`far_disagree_per_volume`) gave the best test result of the new scores, about as
  good as entropy on review AUC. These are test-set observations on a question asked after the lock, so they are
  not a result yet.
- `[idea]` Test the chosen score once on the G3 pool (scans not in kits100). Under the pre-registered rule that is
  `far_6mm_drops_ml`. Choosing a different score (e.g. `far_disagree_per_volume` or `far_6mm_adds_*`) would be based
  on the kits100 test results and must be logged with the date and reason before the G3 pool is opened.


## 2026-10-20: task documents, robustness task

- `[decision]` Course weeks skip the autumn break: week 7 is 2026-10-19 – 23, the pilot is in week 8 (by 2026-10-28),
  the user study in weeks 9–10 (2026-11-02 – 11-13).
- `[decision]` Larger tasks get a task document in `docs/tasks/` with steps, notes and status, so the group can follow
  and work can continue across sessions.
- `[decision]` Rasmus owns the robustness task ([docs/tasks/g1_robustness.md](docs/tasks/g1_robustness.md)): does the
  G1 ranking still work when KiTS scans are degraded on purpose (noise, lower resolution, contrast)? Planned on
  2026-09-23, so far only tried on 3 spleen scans.
- `[admin]` Trello board created by Rasmus (date not recorded).

## 2026-10-20: Otto starts the robustness task

- `[decision]` Otto takes ownership of `docs/tasks/g1_robustness.md` on branch `otto/robustness`, replacing the
  original Rasmus assignment, following Otto's explicit instruction. Updated the task and `PROJECT_OVERVIEW.md`.
- `[admin]` Step 1 started: read `PROJECT_OVERVIEW.md`, `CONTRIBUTING.md`, and `docs/tasks/g1_robustness.md`;
  `git pull --ff-only` reported "Already up to date" at `eb1cef2`, then created `otto/robustness`.
  No `.venv/`, `data/`, or `models/` existed at the initial check. No inference or evaluation was run.
- `[admin]` Otto subsequently reported running `uv sync`. Verified `.venv/`: Python 3.12.15,
  torch 2.2.2+cu121 (CUDA build 12.1; CUDA unavailable to PyTorch in the current environment),
  TotalSegmentator 2.18.0, pytest 9.1.1; Linux x86_64, kernel 7.2.10-1-cachyos, glibc 2.44.
- `[result]` Step 1 installation check: `UV_OFFLINE=1 UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/otto-robustness-uv-cache
  uv run pytest -ra` completed with **72 passed, 36 skipped, 0 failed in 3.11 s** (108 collected).
  No downloads or dependency changes. Skips: 30 require a prepared study (`tests/test_study.py`,
  `tests/test_study_analysis.py`, `tests/test_ui_server.py`), 6 require Google Chrome
  (`tests/test_ui_browser.py`). Recorded in `docs/tasks/g1_robustness.md`, step 1.
  At that point, KiTS data and model downloads awaited Otto's approval (completed below).
- `[admin]` Otto approved both downloads. Ran `scripts/download_data.py` and `scripts/download_model.py` with
  `configs/kits100.yaml` through `uv run python -u` in the installed environment (`UV_OFFLINE=1 UV_NO_SYNC=1`,
  uv cache under `/tmp/`). Both exited 0. Sandbox DNS failed for Hugging Face/GitHub; the approved downloads
  succeeded with network access outside the sandbox.
- `[result]` Robustness step 1 complete: **100 images + 100 ground-truth masks** in `data/raw/kits23/`
  (20 dev / 80 test, seed 42). Images 9,904,809,672 bytes; labels 33,410,151 bytes. All pairs have matching
  shapes/affines; all 200 gzip files passed full-stream CRC and payload-length checks; no partial files remain.
  Model tasks 297 (3 mm) and 298 (6 mm) are in `models/totalsegmentator/nnunet/results/`, each with fold `[0]`,
  plans, dataset metadata, and final checkpoint; model directory total 331,490,973 bytes.
  Setup verification took 85.78 s on AMD Ryzen 5 7500F, Linux x86_64, Python 3.12.15, torch 2.2.2+cu121,
  CUDA build 12.1, CUDA unavailable to PyTorch. `results/robustness_setup/download_verification.json` records
  per-file paths, byte counts, SHA-256 hashes, config hash, source commit, machine, and model details.
  Data and weights are excluded from git. No inference/evaluation or changes to locked G1 files.
- `[admin]` Updated `docs/tasks/g1_robustness.md` (step 1 checked and "Where we are") and `PROJECT_OVERVIEW.md`.
  Next: explain step 2, check compute availability (`nvidia-smi` also cannot communicate with the driver in
  this environment), and prepare a separate timing config for 2 dev scans and 1 perturbation. Steps 2–7
  have not started; ask Otto before any run expected to exceed 30 minutes.
- `[admin]` Committed setup on `otto/robustness` as `b2269e9` (`Complete Otto robustness setup`). Git initially
  lacked an author identity; set repository-local identity to Otto's identity from existing commits, with
  the email verified against his authenticated GitHub profile. HTTPS push then failed because command-line
  Git has no credentials; publishing via the connected GitHub integration failed with HTTP 403
  ("Resource not accessible by integration"). No remote branch or PR was created. A follow-up local commit
  records this blocker. Next action: configure GitHub authentication, run `git push -u origin otto/robustness`,
  and open a PR into `main` for group review before proceeding to step 2.

## 2026-10-20: robustness setup push confirmed

- `[admin]` Otto reported pushing `otto/robustness`. Verified via `git ls-remote --heads origin otto/robustness`:
  remote HEAD is `f388b74b52e4aeb7a58b5405e72ad334df2798d7`, matching local HEAD and including setup commit
  `b2269e9`. The earlier push blocker is resolved. Updated `docs/tasks/g1_robustness.md` and `PROJECT_OVERVIEW.md`.
- `[admin]` `git pull --ff-only` reported "Already up to date"; offline `uv sync --locked` checked 133 packages.
  Before publishing the status update, `UV_OFFLINE=1 UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/otto-robustness-uv-cache
  uv run pytest -ra` completed with **72 passed, 36 skipped, 0 failed in 1.97 s**. Skip reasons are unchanged:
  30 need a prepared study, 6 need Chrome (test file paths listed in the step 1 notes above).
- `[admin]` No open PR was found for the branch. Attempting to create a draft PR into `main` via the connected
  GitHub integration returned HTTP 403 ("Resource not accessible by integration"); no PR was created.
  Open a PR through GitHub for group review. Step 2 has not started; next explain the timing run and check
  compute availability. No inference/evaluation or changes to locked G1 files.
- `[admin]` The follow-up status-note commit could not be pushed by the assistant: noninteractive HTTPS Git
  still reports "could not read Username". Otto's confirmed setup push remains on GitHub; only this subsequent
  documentation update needs `git push origin otto/robustness` from his authenticated terminal.

## 2026-10-21: robustness step 2, GPU compatibility and timing

- `[admin]` Otto created PR [#1](https://github.com/Ottohesten/02830-medical-segmentation/pull/1).
  `366dcf1` is pushed; `git pull --ff-only` was up to date and offline `uv sync --locked` checked 133 packages.
- `[result]` GPU diagnosis corrected: the sandbox hides the device. On the host the RTX 5060 Ti works
  (16,311 MiB, NVIDIA driver 615.78.08), but torch 2.2.2+cu121 lacks `sm_120` kernels and a tensor operation fails
  with "no kernel image is available for execution on the device". Source:
  `results/kits_robustness_timing/gpu_compatibility.json`.
- `[decision]` Otto explicitly approved a separate GPU environment and downloads. Installed `.venv-gpu/`
  with torch 2.7.1+cu128 and torchvision 0.22.1+cu128; tensor and Conv3d checks pass. `uv pip check`: 136 compatible
  packages. Non-GPU versions are retained from the installed shared lock. Pins are in
  `requirements/robustness-gpu.{in,txt}`; use `uv pip sync --python .venv-gpu/bin/python --torch-backend cu128
  requirements/robustness-gpu.txt`. Shared `.venv/`, `pyproject.toml`, and `uv.lock` are unchanged.
  Official basis: [PyTorch 2.7 Blackwell support](https://pytorch.org/blog/pytorch-2-7/) and
  [versioned CUDA installation commands](https://pytorch.org/get-started/previous-versions/).
- `[decision]` For timing only, choose the first two sorted development IDs (`case_00040`, `case_00051`) and
  one perturbation, Gaussian noise SD 60 HU (`noise_60`). No Dice or scores were inspected to choose these.
  This is not the perturbation-strength decision for step 3. Config: `configs/kits_robustness_timing.yaml`,
  seed 42, CUDA, 6 CPU threads, no TTA, 30-minute cap, development only.
- `[experiment]` `PYTHONPATH=src nnUNet_compile=false .venv-gpu/bin/python -u scripts/benchmark_robustness.py
  --config configs/kits_robustness_timing.yaml`: both models on clean and noisy copies of both scans,
  **8 predictions in 68.90 s**. AMD Ryzen 5 7500F, RTX 5060 Ti, Linux 7.2.10-1-cachyos, Python 3.12.15,
  torch 2.7.1+cu128, CUDA 12.8. Metadata records source baseline `366dcf1`, uncommitted changes, and exact
  config/script SHA-256 hashes; code and config are committed with this run's evidence.
  `results/kits_robustness_timing/{machine.json,run_config.yaml,run_log.txt,model_load_times.csv}`.
- `[result]` Per-scan prediction times (includes preprocessing/resampling, excludes loading/perturbing/saving),
  `case_00040` / `case_00051`: 3 mm clean **23.75 / 3.45 s**, 3 mm noise **6.64 / 2.68 s**;
  6 mm clean **4.55 / 1.61 s**, 6 mm noise **4.33 / 1.70 s**. First prediction per model includes CUDA warm-up.
  Canonical shapes 512 × 512 × 207 / 512 × 512 × 68. Peak allocated GPU memory 2,503.01 MiB;
  cumulative process peak RSS 3,907.35 MiB. `results/kits_robustness_timing/{timings.csv,timing_summary.csv}`.
- `[result]` **Planning estimates**, not measured full runs: clean reference + 3 / 4 perturbations, both models,
  loading/perturbing/saving and one model load each: **20 scans 16.95 / 20.45 min; 80 scans 67.40 / 81.39 min**.
  Formula: model-load time + N × (mean paired clean time + P × mean paired noisy time). Based on only two scans
  and one perturbation, with first-use warm-up included; different scan sizes/perturbations and later analysis
  add uncertainty. `results/kits_robustness_timing/runtime_projections.csv`. No overnight fallback is indicated
  by this sample. Ask Otto before actual runs expected to exceed 30 minutes.
- `[result]` Verified all 8 timing rows, config/script hashes, and 16 saved prediction maps with matching grids
  in `data/predictions/kits_robustness_timing/` (not in Git). No test scans, ground truth, Dice, or uncertainty
  scores were read; locked G1 files are unchanged. Added 9 synthetic tests covering dev/test isolation,
  locked-folder guards, and estimates that require all 8 measurements. Both complete test suites passed:
  **81 passed, 36 skipped, 0 failed** (2.28 s shared; 2.53 s GPU). Skip reasons unchanged: 30 need a prepared study,
  6 need Chrome. `results/kits_robustness_timing/{tests_main.txt,tests_gpu.txt}`.
- `[admin]` Step 2 checked in `docs/tasks/g1_robustness.md`; overview and README updated. Next: step 3, choose
  strengths from Dice degradation on 20 dev scans before inspecting score performance. The benchmark handles
  perturbed predictions for both models; the shared `step_second_model` still handles clean scans only and
  needs support for variants before larger robustness runs. Steps 3–7 have not started. Publish this step on
  `otto/robustness` to update PR #1; if noninteractive Git still lacks credentials, Otto pushes from his terminal.

## 2026-10-21: robustness step 3 prepared, candidate-run approval pending

- `[admin]` Confirmed `6f1bc0a` pushed on `otto/robustness`; pull up to date, offline `uv sync --locked`
  checked 133 packages. Steps 1–2 complete; no step 3 inference or Dice analysis has started.
- `[decision]` Prepared a separate candidate config `configs/kits_robustness_dev.yaml`, name
  `kits_robustness_dev`, all 20 dev scans, seed 42, CUDA, 6 CPU threads. Candidates fixed before outcomes:
  noise SD 30/60/120 HU, lowres 6/8/12 mm, contrast removal 0.8/0.9 (20/10% retained around 40 HU).
  Clean reference plus 8 candidates = 180 predictions of the 3 mm model. No 6 mm inference, scores, or test use.
- `[decision]` Operational choice rule, recorded before candidate outcomes: at least 5 scans lose ≥ 0.02 Dice,
  at least 5 remain ≥ 0.83, at most 2 newly fall below 0.50 (only clean ≥ 0.50 count as new). Suggest the mildest
  eligible strength per type; explicitly report types without an eligible candidate. This defines useful
  degradation for this task; it does not change locked G1 metrics. Final strengths must be logged before scores.
- `[result]` Planning only: `scripts/select_robustness_strengths.py --config configs/kits_robustness_dev.yaml
  --plan` wrote `results/kits_robustness_dev/{run_plan.json,planned_cases.csv,run_config.yaml}`.
  All 20 dev headers read; no labels read. From step 2's noisy 3 mm times: unscaled estimate 18.27 min;
  dev/sample mean raw-voxel ratio 2.1775 gives **39.78 min**. Rough planning proxy from two scans and one noise
  type; other perturbations and Dice analysis add uncertainty. Configured cap 90 min. **Await Otto's explicit
  approval before this >30-minute run.**
- `[result]` Added `src/segreview/robustness.py`, `scripts/select_robustness_strengths.py`, and 11 synthetic
  tests in `tests/test_robustness_selection.py`: 11 passed in 0.06 s. Tests cover dev/test isolation, locked
  folders, paired median drops, genuinely new failures, mildest eligible choices, and incomplete/invalid data.
  Usage documented in `README.md`. Task, overview, and "Where we are" updated; step 3 remains unchecked.
- `[result]` Complete preflight suites passed in both environments: **92 passed, 36 skipped, 0 failed**
  (1.92 s shared; 1.99 s GPU). Outputs: `results/kits_robustness_dev/preflight_tests_{main,gpu}.txt`.
  Skips remain 30 study-dependent and 6 Chrome-dependent tests. Long-run approval is still pending.

## 2026-10-21: robustness candidate run approved

- `[admin]` Otto approved the 180-prediction step 3 development run and Dice comparison, explicitly allowing
  an overnight runtime of 8 hours or more. No further approval is needed for this candidate run.
- `[decision]` Increased `robustness_selection.max_run_minutes` from 90 to 480 before inference.
  Candidates and selection rule are unchanged. Preserved the preapproval plan/config in
  `results/kits_robustness_dev/preapproval_run_{plan.json,config.yaml}`; refreshed `run_plan.json` and
  `run_config.yaml` with the new cap. The planning proxy remains 39.78 minutes; 480 minutes is a reserve limit.
  Run only development inference and Dice, then document the final strength choice. Do not start step 4.

## 2026-10-21: robustness step 3 complete, lowres_8mm selected before scores/test

- `[experiment]` Completed `scripts/select_robustness_strengths.py --config configs/kits_robustness_dev.yaml
  --run` in `.venv-gpu/` with GPU access outside the sandbox: **180 predictions** (20 dev cases × clean +
  8 candidates, 3 mm model only) and dev Dice in **2,663.65 s / 44.39 min**. The approved 480-minute cap was
  not approached. AMD Ryzen 5 7500F, RTX 5060 Ti, NVIDIA driver 615.78.08, Linux 7.2.10-1-cachyos,
  Python 3.12.15, torch 2.7.1+cu128 / CUDA 12.8, 6 CPU threads, seed 42. Peak allocated GPU memory
  **3,099.21 MiB**, process peak RSS **7,345.41 MiB**. `results/kits_robustness_dev/{machine.json,
  run_attempts.json,run_config.yaml,run_log.txt,inference_times.csv}` records baseline `6f1bc0a`, dirty state,
  and exact source/config hashes; source is committed with the results.
- `[result]` Clean dev median Dice **0.9033**, with **4/20** below 0.83. Lowres 6 / 8 / 12 mm:
  median paired Dice drops **0.00825 / 0.02795 / 0.10230**, clear drops ≥ 0.02 in **4 / 14 / 20** scans;
  bad counts **5 / 6 / 12**, remaining good **15 / 14 / 8**, new catastrophic counts **0 / 0 / 2**.
  Both 8 and 12 mm meet the unchanged predeclared criteria; 8 mm is the milder choice.
  `results/kits_robustness_dev/{dice.csv,dev_dice_drops.csv,dev_degradation_summary.csv}`.
- `[result]` Noise SD 30 / 60 / 120 HU: median paired drops **-0.00115 / -0.00350 / -0.00485**, clear drops
  in **0 / 2 / 2** cases. Contrast removal 0.8 / 0.9: median paired drops **-0.00270 / -0.00235**, clear drops
  in **0 / 0** cases. Negative drops mean improvement. No tested noise/contrast candidate meets the five-case
  clear-drop criterion; these results concern the tested strengths on dev, not general robustness.
  Source: `results/kits_robustness_dev/dev_degradation_summary.csv` (full table in task step 3).
- `[decision]` **Lock `lowres_8mm` only** by the original mildest-eligible-per-type rule: 14 clear losses,
  14 still ≥ 0.83, no new Dice < 0.50. Exclude noise/contrast for insufficient clear losses and lowres_12mm
  because a milder eligible choice exists. Recorded before any uncertainty scoring or test inference in
  `configs/kits_perturbed.yaml` and `results/kits_robustness_dev/strength_lock.json`, with evidence hashes,
  excluded types, primary `model_diff_mean_region`, and secondary `entropy_mean_region`. The final config
  retains 20 dev cases / dev evaluation only; it does not unlock test. Candidate config/results remain intact.
- `[result]` Verification passed: **180 timing rows, 180 dev Dice rows, 360 saved prediction files**
  (**252,949,887 bytes**), matching paired grids and source/config hashes, no test Dice rows or score files.
  `results/kits_robustness_dev/{verification.json,prediction_manifest.csv}`. Predictions remain outside Git in
  `data/predictions/kits_robustness_dev/`. Development figure: `results/kits_robustness_dev/figures/dev_dice_degradation.png`;
  `report_source.py` in the same result folder reproduces the verification and figure. Locked G1 files unchanged.
- `[result]` Final config validated; complete suites pass in both environments: **92 passed, 36 skipped,
  0 failed** (2.23 s shared; 2.26 s GPU), with unchanged study/Chrome skips.
  `results/kits_robustness_dev/{tests_main.txt,tests_gpu.txt}`.
- `[admin]` Checked step 3 and updated task notes, decisions, "Where we are", overview, and README.
  Wrote a question for Bjørn in task section 6: is the single selected resolution perturbation sufficient,
  or should additional dev search / a stronger secondary resolution be planned before test? The earlier timing
  plan assumed 3–4 perturbations; the current rule selects one. No message was sent to Bjørn.
  Step 4 waits for Otto to return; first discuss scope and prepare second-model support for perturbed scans
  and fresh runtime estimates. Publish the step 3 commit on `otto/robustness` to update PR #1; use Otto's terminal
  if the assistant still lacks HTTPS credentials. Steps 4–7 are untouched.
- `[admin]` Saved the completed step 3 in a local commit. The assistant's `git push origin otto/robustness`
  failed because noninteractive Git cannot read a GitHub username. Otto must push from his authenticated
  terminal to publish the step and update PR #1. No test-set work has started.

- 2026-10-21 `[admin]` Otto reported pushing step 3. Verified with `git ls-remote --heads origin
  otto/robustness` that GitHub has `e608d923d212b078214679303cc19a783d547b2a`, matching local HEAD.
  PR #1 now includes the completed development selection. Updated task status; these administrative notes
  will accompany the next work commit. Next: clarify the scope question in task section 6 before test planning.
  No additional experiment or test-set work was started.

- `[decision]` Otto requested robustness step 4. Retain the existing locked `lowres_8mm` choice and original
  primary/secondary scores; no additional strength search. This question concerns resolution degradation only.
  Bjørn's report-scope question remains open in `docs/tasks/g1_robustness.md` section 6; no answer is recorded.
- `[admin]` Before preparation, `git pull --ff-only` was already up to date at `e608d92` on `otto/robustness`;
  offline `uv sync --locked` checked 133 packages. Shared environment, lock file, and locked G1 files unchanged.
- `[experiment]` Prepared `configs/kits_robustness_test.yaml` and `scripts/run_robustness_test.py`: original
  80 test IDs, clean + locked lowres_8mm, both models, **320 predictions / 640 files**. The full 100-ID
  selection is retained for stable indices, with test-only filtering before inference. Archived
  `configs/kits_perturbed.yaml` and development selection evidence remain unchanged. Outputs will be
  `results/kits_robustness_test/` and ignored `data/predictions/kits_robustness_test/`.
- `[admin]` Extended shared second-model inference to all configured variants, preserving full-list seed
  indices and requiring both saved maps. Timing keys include variant, with historical clean rows migrated.
  Added scientific-setting/case-split/output guards, archived lock/evidence verification, exact config/source
  snapshots, per-model timing, machine/memory recording, resumability, and final map-grid/hash verification.
  Files: `src/segreview/{pipeline,robustness}.py`, `scripts/run_robustness_test.py`.
- `[experiment]` Ran the test runner's `--plan` only: **39.66 min estimated**, excluding final verification,
  with a **proposed 480-minute reserve cap**. Main-model times come from all 20 dev scans for clean and lowres;
  second-model/I/O proxies come from two benchmark dev scans (noise proxy for lowres overhead), scaled by raw
  voxel counts. This is approximate planning, not a measured test-run result. IDs/sizes/components/hashes:
  `results/kits_robustness_test/{run_plan.json,planned_cases.csv,planning_image_sizes.csv,run_config.yaml}`.
- `[result]` Added 21 synthetic tests covering test isolation, locked scientific settings, identical inputs,
  stable perturbation indices, partial-output resumption, legacy timing migration, runtime projection arithmetic,
  and output-verification failures. Complete suites: **113 passed, 36 skipped, 0 failed** in **2.95 s** (shared)
  and **3.06 s** (GPU). `results/kits_robustness_test/preflight_tests_{main,gpu}.txt`.
  A tiny Conv3d CUDA smoke test passed on RTX 5060 Ti, driver 615.78.08, torch 2.7.1+cu128 / CUDA 12.8;
  `results/kits_robustness_test/preflight_gpu.json`. The smoke test used no scan or label files.
- `[admin]` Step 4 remains unchecked, prepared but not run; waiting for Otto's approval of this new run over
  30 minutes. No test inference, ground-truth reading, Dice, uncertainty scores, or evaluation was performed.
  Updated README, task decisions/notes/"Where we are", and overview. Preparation is uncommitted; commit and
  push the completed step after inference and verification. Steps 5–7 are untouched.

- `[admin]` Otto explicitly approved the prepared step 4 run: "Ja, start med loft på 8 timer".
  Retain the existing 480-minute cap and unchanged 320-prediction scope (80 test cases, clean + lowres_8mm,
  both models). Approval: `results/kits_robustness_test/run_approval.json`. Starting inference only;
  Dice and uncertainty scoring remain step 5. Console log: `results/kits_robustness_test/run_log.txt`.

- `[experiment]` Completed the approved step 4 command, `scripts/run_robustness_test.py --config
  configs/kits_robustness_test.yaml --run`, with exit 0 and unchanged strengths/scores/split/cap.
  Both models predicted **80 test scans × clean/lowres_8mm = 320 predictions**.
  Total **2,492.40 s / 41.54 min**, including **63.32 s** final verification; main-model wall interval
  **1,340.81 s**, second-model wall interval **1,088.27 s**. Approved cap: 480 min, not approached.
  `results/kits_robustness_test/{machine.json,run_attempts.json,run_log.txt,run_approval.json}`.
  Normalized terminal progress line endings and trailing spaces in the text log committed to Git.
- `[result]` Step 4 prediction-time medians (excluding model setup and load/perturb/save), 80 rows per group:
  3 mm clean **4.420 s**, lowres **4.565 s**; 6 mm clean **2.890 s**, lowres **2.745 s**.
  Per-scan times and summary: `results/kits_robustness_test/{inference_times.csv,second_model_times.csv,timing_summary.csv}`.
  These are measured runtimes, not segmentation-quality results.
- `[result]` Machine: AMD Ryzen 5 7500F, RTX 5060 Ti, driver 615.78.08, Linux 7.2.10-1-cachyos,
  Python 3.12.15, torch 2.7.1+cu128 / CUDA 12.8, 6 CPU threads, seed 42. Peak allocated GPU memory
  **3,649.76 MiB**, peak process RSS **8,051.12 MiB**. `results/kits_robustness_test/machine.json` records
  baseline commit `e608d92`, uncommitted code state, and exact config/source/input hashes. Run code is committed
  with the results; `run_config.yaml` and `run_plan.json` preserve the approved setup.
- `[result]` Step 4 verification passed: exact 80-case × 2-variant × 2-model timing coverage and
  **640 prediction files / 303,793,357 bytes**, matching input/output shapes and affines, file SHA-256 hashes,
  and config/source hashes matching the approved plan. `results/kits_robustness_test/{verification.json,
  prediction_manifest.csv,post_run_checks.json}`. `report_source.py` there reproduces timing/provenance summaries.
  Prediction maps remain outside Git in `data/predictions/kits_robustness_test/`. Archived development evidence,
  `configs/kits_perturbed.yaml`, shared lock/environment, and locked G1 config/results remain unchanged.
- `[result]` Final complete suites: **113 passed, 36 skipped, 0 failed** in **2.76 s** (shared) and
  **2.47 s** (GPU). `results/kits_robustness_test/tests_{main,gpu}.txt`; skips are unchanged study/Chrome requirements.
- `[admin]` Checked step 4 and updated task notes/decisions/"Where we are", overview, and README.
  No test ground truth, Dice, uncertainty scores, or ranking evaluation was read/computed. Next: commit/push
  step 4 on `otto/robustness` to update PR #1, then explain step 5 and prepare the missing second-model maps
  for the original 20 dev scans (clean/lowres) for separate dev reporting. Keep all locked choices.
  Use Otto's authenticated terminal if the assistant's noninteractive Git still lacks HTTPS credentials.
  Steps 5–7 are untouched; Bjørn's report-scope question remains unanswered in task section 6.

- `[admin]` Saved completed step 4 in a local commit. The assistant's `git push origin otto/robustness`
  failed because noninteractive HTTPS Git cannot read a GitHub username. Otto must push the completed step
  from his authenticated terminal to update PR #1. Included this publication status in the same unpublished
  step commit; no extra status-only push is needed. The worktree is clean after committing.

- `[admin]` Otto reported pushing step 4. Verified with `git ls-remote --heads origin otto/robustness`
  that GitHub has `6fde679b767edb196503c33c4f9119c2664a001d`, matching local HEAD and publishing step 4
  to PR #1. Updated "Where we are"; this administrative confirmation will accompany the next work commit.
  Next: explain step 5. No additional inference, Dice, or uncertainty evaluation was started.

- `[decision]` Otto requested robustness step 5. Before inspecting new scores or test Dice, saved the fixed
  analysis in `configs/kits_robustness_eval.yaml` and `results/kits_robustness_eval/{analysis_plan.json,run_config.yaml}`.
  Keep lowres_8mm, primary model_diff_mean_region, secondary entropy_mean_region; compare with volume/random/oracle.
  Dice drop = clean minus perturbed; score rise = perturbed minus clean; positive change-correlation rho supports
  deterioration sensitivity. Clean/lowres/mixed queues and paired summaries stay separate for dev and test.
- `[decision]` Reuse G1's `evaluate_methods` engine with 2,000 bootstrap samples, 1,000 random orders, 95% CIs,
  seed 42, and both original bad rules (task threshold Dice < 0.83). Mixed queues bootstrap case IDs with both
  views retained; no independent-row rho p-value. Full-queue bad flags stay fixed through bootstrap, as in G1.
  Top-10% perturbed share and average-rank shift use fair ties. No new score selection or combinations.
  Files: `src/segreview/{evaluation,robustness_evaluation}.py`, `scripts/evaluate_robustness.py`.
- `[experiment]` Step 5 preparation will copy/hash-check 720 archived maps into the new experiment and predict
  only 40 missing dev second-model pairs. Rough runtime **6.61 min**, derived from step 4's 1,088.27 s
  second-model wall interval × dev/test voxel ratio **0.3641671**. Sources:
  `results/kits_robustness_test/{machine.json,planning_image_sizes.csv}`. Each stage is capped at 30 min;
  ask before any longer run/extension. Benchmark CPU scoring on two dev cases and a synthetic G1 queue before
  starting full evaluation. Original inference configs/results and locked G1 outputs remain unchanged.
- `[result]` Added 10 synthetic tests for the step 5 math and safeguards. Shared suite: **123 passed,
  36 skipped**, 8 expected constant-input warnings from tiny synthetic correlations, **2.63 s**.
  `results/kits_robustness_eval/preflight_tests_main.txt`. Step 5 is in progress; steps 6–7 are untouched.

- `[experiment]` Step 5 preparation completed: **404.57 s / 6.74 min**, 720 copied/hash-verified maps,
  40 missing dev second-model predictions, **800 maps** ready. `results/kits_robustness_eval/{prepare_machine.json,
  prepare_log.txt,imported_predictions.csv,second_model_times.csv,prediction_manifest.csv}`. Peak allocated
  GPU **496.21 MiB**, process peak RSS **7,241.65 MiB**. No scores or labels read in preparation.
- `[experiment]` CPU benchmark on two dev cases and a synthetic largest G1 queue projects **15.34 min**
  for evaluation, including a 1.5 safety factor: voxel-scaled score proxy **315.49 s**, synthetic queue
  **8.16 s × 12**, and Dice allowance **2 s × 100 cases**. `results/kits_robustness_eval/{benchmark_times.csv,
  benchmark_machine.json,runtime_projection.json}`. This is approximate runtime planning, not an accuracy
  result. Starting the fixed evaluation within the existing 30-minute cap; no test-based choices changed.

- `[experiment]` Completed the fixed step 5 CPU evaluation in **722.97 s / 12.05 min**, within its 30-minute
  cap, after 404.57 s GPU preparation and a 12.13 s timing benchmark. Scored all 200 views from model maps
  before loading labels for 400 per-model Dice evaluations. CPU AMD Ryzen 5 7500F, Linux, Python 3.12.15,
  torch 2.7.1+cu128 / CUDA build 12.8, 6 threads, seed 42; CPU peak RSS **6,413.74 MiB**, no GPU used for
  evaluation. GPU preparation used RTX 5060 Ti / driver 615.78.08. Separate metadata records baseline
  `6fde679`, dirty state, exact config/source hashes, machine/library versions, time and memory:
  `results/kits_robustness_eval/{prepare,benchmark,evaluate}_machine.json`. Run code is committed with results.
- `[result]` Main-model clean/lowres medians and median paired drops: dev **0.90330 / 0.86700 / 0.02795**,
  test **0.90955 / 0.87865 / 0.03010**. Dice < 0.83 counts dev **4 → 6/20**, test **11 → 21/80**, with
  **2 / 10** newly bad and no recoveries. Test Dice falls on **77/80** cases.
  Second-model clean/lowres medians and paired drops: dev **0.89510 / 0.89415 / 0.00745**, test
  **0.89430 / 0.88685 / 0.00855**; bad counts **7 → 7/20** and **16 → 16/80**, no newly bad/recovered.
  `results/kits_robustness_eval/{dice.csv,dice_per_model.csv,dice_degradation.csv,paired_changes.csv}`.
- `[result]` Spearman(Dice drop, score rise), dev then test, 95% patient bootstrap CI:
  primary **-0.074 [-0.529, 0.415] / 0.215 [-0.036, 0.459]**;
  secondary entropy **0.575 [0.104, 0.894] / 0.243 [0.004, 0.473]**;
  negative volume **0.849 [0.532, 0.980] / 0.743 [0.597, 0.845]**. All six have 2,000 finite bootstrap samples.
  Primary rises on **9/20 dev, 41/80 test** (median change **-0.000463 / 0.000611**), entropy on
  **20/20 dev, 79/80 test** (**0.037025 / 0.046093**). Primary response to degradation severity is inconclusive;
  entropy's modest positive association remains secondary. No new score selection or head-to-head superiority
  test, and no multiplicity adjustment, was added. `results/kits_robustness_eval/change_correlations.csv`.
- `[result]` Dice < 0.83 review-AUC (primary / entropy / volume / random / oracle):
  clean dev **0.7625 / 0.6875 / 0.5500 / 0.4978 / 0.9000**;
  lowres dev **0.7917 / 0.7167 / 0.5917 / 0.4969 / 0.8500**;
  mixed dev **0.7800 / 0.7250 / 0.5725 / 0.4964 / 0.8750**;
  clean test **0.8313 / 0.8949 / 0.3903 / 0.5045 / 0.9313**;
  lowres test **0.7967 / 0.7455 / 0.4926 / 0.5018 / 0.8688**;
  mixed test **0.8086 / 0.8068 / 0.4648 / 0.5030 / 0.9000**.
  Primary lowres minus volume **0.3042 [0.1490, 0.4709]**; mixed minus volume **0.3438 [0.1795, 0.5052]**.
  Mixed primary AUC CI **[0.7374, 0.8830]**. Raw ceilings vary with bad prevalence; keep oracle/random references.
  `results/kits_robustness_eval/g1_metrics.csv`, with all G1 rho/quality/CI columns retained.
- `[result]` Under worst10, primary test AUC is **0.8547 clean / 0.8047 lowres / 0.8520 mixed**.
  Lowres-vs-volume CI **[-0.1090, 0.6876]** includes zero; superiority is not established under every bad rule.
  Do not compare the task's threshold clean AUC directly with the original G1 worst10 headline 0.855.
  Source: `results/kits_robustness_eval/g1_metrics.csv` (both original bad rules saved).
- `[result]` Mixed test queue: **160 views / 80 case IDs**, **32 bad views**, patient-grouped bootstrap.
  Top 16 lowres views: primary **7**, entropy **13**, volume **8**. Median paired rank gains
  **2.0 / 55.5 / 6.0**, respectively. At top 10%, primary finds **12/32** bad segmentations, entropy
  **13/32**, volume **4/32**. Failure ranking remains useful under the threshold rule while the primary does
  not consistently promote degraded views. Entropy responds more consistently without a higher mixed AUC
  point estimate. `results/kits_robustness_eval/{queue_top_summary.csv,queue_rank_changes.csv,g1_curves.csv}`.
- `[result]` Post-run verification passed: **100 cases, 200 score rows, 400 model Dice rows, 100 paired rows,
  60 metric rows, 800 prediction files / 424,973,659 bytes**, **720 imported inputs unchanged**.
  Exact cohort/key coverage and config/source/stage hashes match the fixed pre-outcome plan; mixed CIs group
  case IDs and independent-row rho p-values are omitted. `results/kits_robustness_eval/{verification.json,
  post_run_checks.json,prediction_manifest.csv,key_results.json}`. `report_source.py` reproduces this audit and
  result extraction without rescoring or bootstrapping. Prediction maps remain outside Git in
  `data/predictions/kits_robustness_eval/`. Completed inference configs/results and locked G1 files unchanged.
- `[result]` Final suites in both environments: **123 passed, 36 skipped**, 8 expected constant-input
  warnings from tiny synthetic tests; **3.03 s shared / 3.04 s GPU**.
  `results/kits_robustness_eval/tests_{main,gpu}.txt`. Normalized progress line endings/trailing spaces in logs.
- `[admin]` Checked step 5 and updated task notes/"Where we are", overview, and README. Test outcomes have
  now been inspected; retain the locked resolution-only scope, primary score, and secondary entropy.
  Next: commit/push step 5 on `otto/robustness` to update PR #1, then explain step 6 (one figure plus the
  requested short report summary). Use Otto's authenticated terminal if noninteractive HTTPS Git still
  lacks credentials. Steps 6–7 have not started; Bjørn's report-scope question remains unanswered.

- `[admin]` Saved completed step 5 in a local commit. `git push origin otto/robustness` failed because
  noninteractive HTTPS Git cannot read a GitHub username. Otto must push from his authenticated terminal.
  Included the publication status in the same unpublished step commit; no extra status-only push is needed.

## 2026-10-21: robustness step 6, report figure and short summary

- `[admin]` Read task status, overview and contributing rules. `git pull --ff-only` was already up to date;
  offline `uv sync --locked` checked 133 packages. Verified GitHub's `otto/robustness` points to `4a0d89c`,
  matching local HEAD and publishing step 5. No extra status-only commit is needed.
- `[result]` Created one **3-panel** figure using `scripts/plot_robustness_summary.py` and the unchanged
  saved step 5 CSVs: paired main-model Dice, primary/secondary change correlations with saved 95% CIs,
  and mixed test review curves with volume/random/oracle references. This is a presentation after test
  evaluation; no scientific choice, score, model, bootstrap, or source result changed.
  Exports: `results/kits_robustness_eval/report/fig_robustness_summary_test.{pdf,png}`.
- `[result]` PDF: **1 page**, **30,465 bytes**, vector output with embedded text. PNG: **2160 × 1620 pixels,
  300 dpi**, **354,912 bytes**; figure size **7.2 × 5.4 inches**. Saved `caption.md` and `summary.md`
  (**8 numbered lines**) in the same report folder. PNG visually inspected; PDF page count/text extraction,
  PNG dimensions, input/output/config/script hashes, and summary length pass `report/verification.json`.
  Rendering used CPU only (AMD Ryzen 5 7500F, Linux, Python 3.12.15, torch 2.7.1+cu128 / CUDA build 12.8,
  Matplotlib 3.11.2). `report/figure_provenance.json` records baseline `4a0d89c`, dirty state and exact hashes.
- `[result]` Eight-line report summary (identical to task section 4 and `report/summary.md`):

  1. An 8 mm resolution perturbation was selected on 20 development scans before test scoring.
  2. On 80 test scans, median 3 mm Dice fell from 0.910 to 0.879; median paired loss was 0.0301.
  3. Segmentations below Dice 0.83 increased from 11/80 to 21/80.
  4. Primary score rise versus Dice loss was inconclusive: ρ = 0.215, 95% CI [-0.036, 0.459].
  5. Entropy rose on 79/80 scans and remains a secondary finding.
  6. Mixed-queue review-AUC was 0.809 for the primary score and 0.807 for entropy, versus volume 0.465, random 0.503, and oracle 0.900.
  7. The primary found 12/32 bad segmentations in the first 16 views; only 7/16 of those views were degraded copies.
  8. Failure ranking remains useful in this resolution stress test, with limited evidence for a response to degradation severity; other domain shifts are outside the tested scope.

- `[admin]` Checked step 6 and updated task notes/"Where we are", overview and README. Original configs,
  strength lock, saved analysis tables and locked G1 outputs remain unchanged; no scans, labels or prediction
  arrays were read. Next: commit/push report material on `otto/robustness` to update PR #1, then guide Otto's
  five-minute presentation in step 7. No presentation or external message to Bjørn has happened.

- `[result]` Final step 6 suites: **123 passed, 36 skipped**, 8 expected constant-input warnings from existing
  synthetic tests; **2.99 s shared / 3.12 s GPU**. `results/kits_robustness_eval/report/tests_{main,gpu}.txt`.
  Verified the eight summary lines match the report source, task section 4 and this log exactly;
  `results/kits_robustness_eval/report/verification.json`. Scientific inputs/configs remain unchanged.

- `[admin]` Saved completed step 6 in a local commit. The assistant's `git push origin otto/robustness`
  failed because noninteractive HTTPS Git cannot read a GitHub username. Otto must push from his authenticated
  terminal. Included publication status in the same unpublished work commit; no extra status-only push is needed.

## 2026-10-21: supervisor advice on G3, task changes

- `[supervisor]` Kristine (meeting): the comparison "most uncertain first" vs. random order is a good way to see whether
  the most uncertain scans are just outliers. Evaluate the test set after each training step (adding 1 or e.g. 10
  corrected scans at a time, whichever is practical). If ranking helps, test Dice should rise faster when training on
  the most uncertain scans first than in random order.
- `[decision]` G3 is run as a learning curve: k = 0, 10, 20, … corrected pool scans, queue order vs. random order
  (several seeds), each point trained from the original model, test Dice after each point. Training settings are still
  chosen on the development set first; the test set is only used to draw the curves.
- `[result]` G3 GPU timing (Frederik, RTX 4090): about 0.04–0.18 s per training step with 16 data workers, so
  2,500 steps take about 7–8 min for both the 6 mm and the 3 mm model; GPU memory at most 17.3 of 24 GB. The PC's RAM
  (34 GB) was almost full with 16 workers.
- `[admin]` Otto's GPU (RTX 5060 Ti) is not supported by the project's torch 2.2.2; the robustness runs used a separate
  environment (`.venv-gpu`, torch 2.7.1 + CUDA 12.8, `requirements/robustness-gpu.txt`). The shared lock file is
  unchanged. To be mentioned in the report together with the other machines.
- `[decision]` Rasmus gets an independent task: catching confident errors that uncertainty misses
  ([docs/tasks/g1_missed_errors.md](docs/tasks/g1_missed_errors.md)). The final check uses the untouched G3 pool once,
  coordinated with Frederik.
- `[admin]` Large shared data (`data/predictions/kits100/`, `data/study/study_kits/`) are shared via the group's
  OneDrive, not git.

---

## Open questions (update when answered, with the date)

- Does G3 become core in response to Kristine's training question? Which cases to correct for training: top of the
  queue, random, or diverse cases? Fine-tune TotalSegmentator or train a smaller model ourselves? Who gets HPC access?
- Is a tumor the model leaves out an error or a label-definition difference? Matters for G3.
- Pilot: which heatmap display mode, and is 5 minutes per scan enough? Then lock the analysis plan.
- Who runs the pilot and sessions; recruit ≥ 16; rooms; fill in `[contact e-mail]` and `[date]` in the consent form;
  where to back up session data.
- Group confirmation of the plan and of G1 "go".
- `configs/hpc.yaml` still points to MSD Spleen.
- Are all KiTS cases separate patients? Check before fixing the G3 pool.
- Report sections and contribution appendix not divided yet.
