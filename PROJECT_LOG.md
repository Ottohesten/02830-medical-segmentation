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
- `[idea]` Add the 6 mm-alone analysis from 2026-10-06 to the repository as a script and result file, so the numbers
  can be reproduced (from saved predictions only, no new inference).

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
- Trello board (mandatory) is missing. Report sections and contribution appendix not divided yet.
