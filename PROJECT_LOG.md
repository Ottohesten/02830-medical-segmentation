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

---

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
