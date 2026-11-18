# Report notes: G1 ranking and heatmap, the review interface, study design

| | |
|---|---|
| **Owner** | Bjørn |
| **Branch** | `main` |
| **Where the details are** | [PROJECT_LOG.md](../../PROJECT_LOG.md), `results/kits100/`, `ui/`, [docs/study/](../study/) |
| **Last updated** | 2026-11-18 |

Notes for the report, not report text. Intervals are the 95% bootstrap intervals stored in the cited file.
A count or a median taken from case rows is marked as such. Higher score means more uncertain, so a useful
score has a negative Spearman correlation with Dice.

## 1. What was done

Rank KiTS23 kidney scans by likely segmentation failure without using ground truth, then point a reviewer at
the uncertain voxels.

One hundred KiTS23 scans, seed 42: 20 development scans, on which every G1 choice was locked on 2026-09-24,
and 80 test scans, opened once after that lock (`configs/kits100.yaml`). The mask a reviewer would edit is
TotalSegmentator's 3 mm model (fold 0 only; a second fold is not published). The ranking score is the mean
absolute difference between the 3 mm and 6 mm organ probabilities inside the predicted kidney plus 5 mm
(`model_diff_mean_region`). It is compared with random order, predicted organ volume, and an oracle that
ranks by true Dice. Ground truth is used only to score the ranking.

Target: kidney + tumor + cyst. On the 20 development rows of `results/kits100/label_check.csv`, the median
share of tumor voxels called kidney or cyst is 0.8079, and median Dice falls from 0.90365 (kidney + tumor +
cyst) to 0.8553 (kidney + cyst). The saved summary over all 100 scans has median tumor coverage 0.8414
(`results/kits100/label_check_summary.csv`).

**Two names for a bad scan, both locked, both reported** (`evaluation.bad_rules`):

- **`worst10`** — the worst 10% by Dice. This is the original G1 headline: the 2026-09-24 log entry leads
  with it, `report.example_bad_rule` is `worst10`, and it is panel (a) of the report figure.
- **`dice_below_083`** — Dice < 0.83. The threshold sits in the development gap between Dice 0.8023 and
  0.8551 (the four lowest development Dice values are 0.4605, 0.4705, 0.6607, 0.8023;
  `results/kits100/g1_scores_clean_dev.csv`).

A score beats the volume baseline when the lower end of the interval for (its AUC − volume AUC) is above 0.
That bar applies to review-AUC under both rules and to quality-AUC. The two review-AUCs are different
numbers (0.8547 and 0.8312 below). Later exploratory score selection uses `dice_below_083`.

The heatmap is a separate choice: binary voxel entropy, locked on the development set because it points at
wrong voxels. The review interface and the study design use that heatmap and this queue. The study has not
produced an outcome in this part.

Otto's fresh-scan test of this score (100 new KiTS cases, `results/kits_missed_fresh/`), Rasmus's user
study and Frederik's fine-tuning all start from this locked score; their results are in their own notes.

## 2. Results

Locked test, 80 scans, primary score `model_diff_mean_region`. Source:
`results/kits100/g1_metrics_clean_test.csv`. Test Dice of the 3 mm mask: median 0.9098, 11 scans below
0.83 (`results/kits100/model_dice_3mm_vs_6mm_summary.csv`); min 0.0027, max 0.9495 over the 80 rows of
`results/kits100/g1_scores_clean_test.csv`. The metrics file has `n_bad` 8 for `worst10` and 11 for
`dice_below_083`.

- Spearman ρ −0.6325 [−0.7564, −0.4694]. Partial ρ given volume −0.6317 [−0.7571, −0.4603]. **Locked.**
- Review-AUC, **`worst10` (headline):** 0.8547 [0.7632, 0.9338]. Volume 0.4875 [0.2544, 0.7259].
  Difference +0.3672 [0.0938, 0.6485]. Random 0.5058 [0.314, 0.6922]. Oracle 0.95 [0.9188, 0.9812].
  Stored share of the random-to-oracle gap (`review_auc_norm`): 0.7854. **Locked.**
- Review-AUC, **`dice_below_083`:** 0.8312 [0.7239, 0.925]. Volume 0.3903 [0.2062, 0.5844].
  Difference +0.4409 [0.2259, 0.6646]. Random 0.5045 [0.3494, 0.6699]. Oracle 0.9312 [0.8938, 0.9688].
  `review_auc_norm` 0.7656. **Locked.**
- Quality-AUC (same for both rules): 0.954 [0.9477, 0.9585]. Volume 0.9392 [0.925, 0.9505].
  Difference +0.0148 [0.0065, 0.0249]. Random 0.9353 [0.9266, 0.9438]. Oracle 0.959 [0.9547, 0.9621].
  **Locked.** Beats volume on this and on both review rules.
- Volume itself vs Dice: ρ −0.051 [−0.2818, 0.1709] (the score is negative volume). **Locked baseline.**

Same file, not the locked primary:

- `entropy_mean_region`, secondary: ρ −0.7951 [−0.8656, −0.6922]; `worst10` review-AUC 0.8938
  [0.8212, 0.9562]; `dice_below_083` review-AUC 0.8949 [0.836, 0.9488]. The marginal intervals overlap
  the primary. No paired comparison of the two scores is in the locked file.
- Second locked score `combo_mean_missing` (primary plus a missing-side check): ρ −0.6114
  [−0.7495, −0.4246]; `worst10` review-AUC 0.8469 [0.7562, 0.9288], difference vs volume +0.3594
  [0.0859, 0.6346]; `dice_below_083` review-AUC 0.8233 [0.7188, 0.915]. Still beats volume; the point
  estimates sit below the primary alone.

Why that primary was locked, development set, 20 scans, 4 below 0.83 and 2 in the worst 10%
(`results/kits100/g1_metrics_clean_dev.csv`): `model_diff_mean_region` had the most negative Spearman
among the entropy and disagreement scores, ρ −0.6827 [−0.8981, −0.2879], against entropy −0.5549
[−0.894, −0.0623]. The combination was chosen separately, by development review-AUC.
Development `dice_below_083` review-AUC 0.7625 [0.5983, 0.9083] vs entropy 0.6875 [0.175, 0.95].
The development intervals are wide.

Heatmap, locked on development voxel-localization, then reported on test. Medians in
`results/kits100/heatmap_eval_dev_summary.csv` and `heatmap_eval_test_summary.csv` (min and max are in
those files; they are not intervals):

- Voxel-AUC, development: entropy 0.8753, distance to the mask edge 0.8489, |p₃ − p₆| 0.7235.
- Voxel-AUC, test: entropy 0.8642, edge 0.8413, |p₃ − p₆| 0.7178. Entropy is above the edge reference
  in 76 of the 80 rows of `heatmap_eval_test.csv`.
- Share of error voxels inside the top 10% of the map, test median: entropy 0.4348, edge 0.4117,
  |p₃ − p₆| 0.2117.

Exploratory, after the lock, saved predictions only. Each file is marked exploratory.

- 6 mm model alone, test (`model_dice_3mm_vs_6mm_summary.csv`): median Dice 0.8944 vs 0.9098 for 3 mm;
  mean 0.8479 vs 0.8703; Dice < 0.83 in 16 vs 11 scans; 3 mm higher in 68 of 80; Spearman between the
  two Dice values 0.8948.
- Locked queue, test (`exploratory_queue_counts.csv`): top 10 has 6 real finds and 4 false alarms
  (3 mm fine, 6 mm below 0.83); top 20 has 9 real finds, 4 false alarms, and 7 where both are fine.
- Where the masks disagree, test, pooled (`model_disagreement_summary.csv`): 0.8929 of disagreeing
  voxels are within 6 mm of the 3 mm edge (median scan 0.9940). Of the voxels farther out, 0.7909 are
  tumor. In the 11 bad scans the near share is 0.6754 and the far tumor share is 0.8663; the 3 mm mask
  is right in 0.4947 of those far voxels (median scan 0.2224). In the false alarms the near share is
  0.7388, the far tumor share is 0.7889, and the 3 mm mask is right in 0.9947 of the far voxels. In the
  good scans the near share is 0.9729.
- Far-only scores. Kits100 is development data for this new question; the rule was written down first
  (`exploratory.disagreement_scores` in `configs/kits100.yaml`): highest development review-AUC under
  `dice_below_083`. It selects `far_6mm_drops_ml`, review-AUC 0.7875
  (`exploratory_disagreement_choice.csv`). Paired difference vs the locked score on that same
  development set: +0.025 [−0.05, 0.1333] (`exploratory_disagreement_paired.csv`). On test
  (`exploratory_disagreement_metrics.csv`), `far_disagree_per_volume` review-AUC is 0.8869
  [0.8219, 0.9463] under `dice_below_083` and 0.8953 [0.8078, 0.9638] under `worst10`. The selected
  `far_6mm_drops_ml` is 0.7472 [0.6062, 0.8864] and 0.7156 [0.525, 0.8922]. Paired test differences,
  `far_disagree_per_volume` minus locked (`exploratory_disagreement_paired.csv`): review-AUC +0.0557
  [0.0075, 0.1313] and +0.0406 [0.0045, 0.0875]; quality-AUC +0.0014 [−0.0002, 0.0034]; vs entropy,
  `dice_below_083` review-AUC −0.008 [−0.0443, 0.0281]. Selected `far_6mm_drops_ml` minus locked:
  −0.0841 [−0.225, 0.0231] and −0.1391 [−0.3393, −0.0071]. Top 10, test
  (`exploratory_queue_counts.csv`): `far_6mm_adds_ml` has 8 real finds and 0 false alarms. One real
  find, case_00192, moves from locked rank 2 to rank 66.5 under that score
  (`exploratory_queue_ranks.csv`). These test observations are not a replacement for the locked score.
- Simulated voxel correction, test, mean Dice gain when the top 10% of region voxels are set to ground
  truth (`voxel_correction_summary.csv`): entropy +0.033894, distance to the edge +0.031736,
  |p₃ − p₆| +0.016684, random +0.008400, oracle +0.067310. Mean area of the gain curve from 0 to 10%
  (`voxel_correction_auc_summary.csv`): entropy 0.018117 (0.508793 of the oracle), edge 0.016215
  (0.455375), |p₃ − p₆| 0.008991 (0.252501), random 0.004211 (0.118262). Entropy's 0–10% area is
  higher than the edge reference in 77 of 80 test rows of `voxel_correction_auc.csv`. Over the 80 test
  rows of `voxel_correction_cases.csv` (medians and means of those columns, 4 decimals): share of
  error voxels outside the region 0.1318 (mean 0.2101); share of the Dice gap that correction inside
  the region cannot close 0.1233 (mean 0.1962); mean Dice if the whole region is corrected 0.9490.

Study planning, assumed effects, not an observed study effect (`results/study_kits/power_needed.csv`,
`power_simulation.csv`, 10,000 simulations): d_z = 0.8 needs 16 participants (simulated power 0.8206);
d_z = 0.5 needs 36 (simulated power 0.8124).

## 3. What it means, and what it does not show

On the locked test, sorting by 3 mm vs 6 mm disagreement finds the bad scans sooner than sorting by
organ size or tossing a coin, under both bad-scan rules, and the partial correlation given volume stays
negative. The headline review curve covers 0.7854 of the gap from random order to a perfect order.
Quality-AUC is a small gain on top of a high baseline: the median scan is already around 0.91, so
correcting the few bad scans can move the dataset mean only a little (oracle quality-AUC 0.959).

The score does not measure a clean second opinion. Most of the disagreement is the 6 mm grid along the
edge. The voxels that are actually far from that edge are mostly tumor, and they show up both in bad
scans and in false alarms where the 6 mm model is the one that is wrong. Four of the ten scans at the
top of the locked test queue are that kind of false alarm.

Entropy would have ranked the test set higher on the point estimates, and it is the better heatmap.
Its advantage over "highlight the outline" is small: the top 10% of the entropy map catches a median
0.4348 of error voxels, against 0.4117 for distance to the edge. A simulated fix of those voxels gains
about the same as fixing by distance to the edge. Errors outside the predicted kidney plus 5 mm,
including a kidney the model never marks, stay out of reach. The report example of a bad scan the queue
does not promote is case_00406: Dice 0.8004, queue position 34 of 80
(`results/kits100/report/heatmap_examples_test.csv`).

The far-disagreement experiment shows the same limit in another way. Removing the grid band can look
better on this test set, and counting only places where the 6 mm model adds organ can clear the four
false alarms from the top 10. The rule that was fixed before those numbers were computed selected the
opposite direction, and that score does not win. Nothing in that experiment replaces `model_diff_mean_region`.

## 4. Limitations

- The test set has 8 or 11 bad scans, depending on the rule. The development set has 4 below 0.83 and
  2 in the worst 10%, so the lock was made on a noisy comparison.
- One public cohort (KiTS23), one organ definition, one published fold, no test-time augmentation.
  The locked numbers were computed once from the saved predictions.
- Quality-AUC compresses differences because most scans are already good.
- Disagreement and entropy both need the model to be unsure. A development scan with one predicted
  side at 0 ml (case_00040, Dice 0.6607) is rank 9 of 20 on the primary score
  (`g1_scores_clean_dev.csv`, `queue_top_3mm_vs_6mm.csv`).
- The voxel-correction numbers set voxels to ground truth. They are not a person, and they have no time.
- The far-score comparison reuses kits100. A score chosen there still needs a new set of scans.
- The user study is designed here and not run in this part. The pilot changed the setup, and the plan
  was locked on 2026-11-18 on the user-study branch (contour heatmap, 5 minutes per scan,
  `study_kits_v2`). The run, the later changes and the result are in
  [rasmus_g2_user_study.md](rasmus_g2_user_study.md).

## 5. Decisions and rejected alternatives

Dates and reasons are in [PROJECT_LOG.md](../../PROJECT_LOG.md). The ones a reader will ask about:

- 2026-09-23. MSD Spleen (41 scans, median Dice 0.9604, minimum 0.8663 on spleen_44;
  `results/local_all/dice_summary.csv`, `dice.csv`) was too easy, and no score beat volume there
  (every `review_auc_minus_volume` interval in `results/local_all/g1_metrics_clean.csv` includes 0
  or sits below it). Switched to KiTS23. Rejected the TotalSegmentator training set, BraTS, ImageCAS,
  ACDC, and the largest CT sets, for leakage, modality, or disk size.
- 2026-09-23. Target is kidney + tumor + cyst. Treating tumor as background punishes the model for
  calling tumor kidney. Fold disagreement was impossible (only fold 0). The second model is the 6 mm
  network. Test-time augmentation was not run on KiTS.
- 2026-09-24. Primary score locked as `model_diff_mean_region` before the test set was opened.
  `entropy_mean_region` stayed a secondary finding after it looked better on test. The missing-side
  combination was the second locked score; left/right volume asymmetry was rejected as noisy.
- 2026-09-24. Both bad-scan rules kept. The fixed threshold is 0.83 because that is the middle of the
  development gap, not a round number chosen after the test.
- 2026-09-24. Heatmap is voxel entropy, not |p₃ − p₆|, because the development voxel-AUC was higher.
  Distance to the edge is a reference, not a model uncertainty.
- 2026-09-24. Own browser viewer on NiiVue, Chrome, MacBook trackpad, English. 3D Slicer, ITK-SNAP,
  and MONAI Label were set aside: less control of the heatmap condition, the timer, and a small tool set.
- 2026-09-24 to 2026-09-30. Study: each person corrects 3 scans with the heatmap and 3 without, order
  balanced in groups of 4, 5 minutes per scan, raw NASA-TLX after each condition. Outcome
  (Dice after − Dice before) / minutes, paired Wilcoxon. H1 is that gain per minute; H2 (workload)
  is tested only if H1 holds (`docs/study/analysis_plan.md`, `configs/study.yaml`). At least 16
  participants for an assumed d_z of 0.8 (the plan; the final number and the stopping rule are in
  [rasmus_g2_user_study.md](rasmus_g2_user_study.md)). Group members are pilots only.
- 2026-10-20. Far-disagreement scores are exploratory. The pre-written rule selects one score on the
  development split. Looking at the test set and then picking `far_disagree_per_volume` would be a
  new choice and would have to be logged before any new scans are scored.

## 6. What went wrong, and surprises

Spleen looked like a success (median Dice 0.9604) and could not test failure detection. The one clear
miss was not enough, and organ size already tracked Dice (negated-volume ρ −0.6093 [−0.773, −0.3662],
`results/local_all/g1_metrics_clean.csv`).

On KiTS the locked score works, and the simpler entropy score has the better test point estimates.
Switching would have meant choosing on the test set. The cost of the lock is that the reported primary
is not the best row in the table.

The 6 mm model is a bit worse than the 3 mm model and creates false alarms at the top of the queue.
Most of their disagreement is the coarse grid. The part that is a real shape difference is mostly
tumor, in both directions: the 6 mm model sometimes includes tumor the 3 mm model missed, and
sometimes drops tumor the 3 mm model had. A filter aimed at one of those directions cleared false
alarms on this test set and also pushed a real find (case_00192) far down the list. The direction the
development rule actually selected was the control, and it ranked worse.

The heatmap mostly lights the outline. Entropy beats the edge reference on voxel-AUC in 76 of 80 test
scans, and the simulated Dice gain at 10% of voxels is only a little larger. That was already the
known risk for the user study.

The missing-side check was added for a kidney the uncertainty score does not promote (case_00040).
On the test set the combination does not improve on the disagreement score alone.

## 7. Figures and tables for the report

- `results/kits100/report/fig_g1_curves_test.pdf` — review curves for both bad-scan rules, and the
  quality curve, with the locked score, the other measures, both baselines, and the oracle.
  **Pick this if there is room for one figure.**
- `results/kits100/report/fig_g1_auc_vs_volume_test.pdf` — paired difference against volume, the
  "beats the baseline" picture. Use if the curves figure does not fit.
- `results/kits100/report/table_g1_test.csv` (and `.tex`) — the locked test table, both review rules.
- `results/kits100/report/fig_heatmap_examples_test.pdf` and `table_heatmaps_test.csv` — three test
  slices (certain and correct, uncertain and wrong, certain and wrong) and the heatmap medians.
- `results/kits100/figures/model_disagreement_where.png` — grid band vs real differences.
  Exploratory.
- `results/kits100/figures/voxel_correction_gain.png` — simulated Dice gain vs share of voxels
  corrected. Exploratory.
- `results/kits100/figures/exploratory_review_curves_test.png` — far-disagreement scores against the
  locked score. Exploratory; easy to over-read.

## 8. Top 3 points for the report

1. Locked result: mean |p₃ − p₆| ranks KiTS failures without ground truth. On 80 test scans the
   headline review-AUC (`worst10`) is 0.8547 [0.7632, 0.9338], against random 0.5058 and oracle 0.95
   (0.7854 of that gap). The same score beats predicted volume under `worst10`, under Dice < 0.83
   (review-AUC 0.8312 [0.7239, 0.925]), and on quality-AUC. The partial correlation given volume
   stays negative, so this is not an organ-size effect.
2. The score and the heatmap both lean on the outline, and both miss confident errors. On the test set, 0.8929 of disagreeing voxels (pooled) lie within
   6 mm of the edge; the rest are mostly tumor, and 4 of the top 10 queue places are 6 mm false
   alarms. Entropy is the better map and, on this test
   set, the better ranker, but it was not the pre-locked score, and it is only a little better than
   highlighting the edge. A bad test scan can sit at queue position 34 (case_00406, Dice 0.8004).
3. What was fixed before the test is the result. Entropy's higher test numbers, and the later
   far-disagreement scores, are secondary or exploratory. The pre-written far-score rule selected the
   control direction, which did not win; the score that looked best on test was not the one that rule
   selected. The interface, the study, and the later fresh-scan check all use the locked score.

## 9. Contribution

- G1 on KiTS: scores from the two released models, the development/test split, the lock, and the
  one-time test evaluation (`results/kits100/g1_metrics_clean_test.csv` and the report table and figures).
- Heatmap comparison and the choice of voxel entropy.
- Browser review interface (`ui/`): slice viewer, brush, heatmap on or off, timer, logging.
- Study design: balancing, time limit, NASA-TLX after each condition, analysis plan, power simulation.
- Exploratory follow-ups on the saved predictions: 6 mm model alone, where the two masks disagree,
  simulated voxel correction, and far-disagreement scores.

## 10. Specific to this part

- **Why KiTS, and why 3 mm vs 6 mm.** Spleen failures were too rare to rank, and volume already
  explained the Dice variation that existed. KiTS has tumors and enough bad scans to measure a
  queue. An ensemble of training folds is not available. The 6 mm model is the second
  published network, so disagreement is the stand-in. It is a coarser grid of a related model, which
  is why so much of the signal sits on the edge.
- **What the lock cost and gave.** It gave a primary score chosen without the test labels:
  `model_diff_mean_region`, on development Spearman. It cost the freedom to report entropy as the
  method. Entropy's test ρ is −0.7951 and its `worst10` review-AUC is 0.8938, against −0.6325 and
  0.8547. Those intervals overlap, and the protocol keeps the locked row. The same pattern returned
  in the far-score experiment: the development rule and the best test row were not the same score.
- **What the exploratory 3 mm vs 6 mm comparison showed.** The 6 mm mask is slightly worse (test
  median Dice 0.8944 vs 0.9098). Pooled, 0.8929 of disagreeing voxels are the grid band; the far
  voxels are 0.7909 tumor. False alarms are places the 6 mm model leaves out tissue the 3 mm model
  has (the 3 mm mask is right in 0.9947 of those far voxels). Bad scans are mixed. Counting only far
  disagreement changes who sits in the top 10, and the honest pre-registered choice did not rank
  better overall.
- **How the interface changed after the first hands-on tests.** The mask stays drawn on top of the
  heatmap, with a see-through fill and an outline. The brush is a circle of the selected size, and
  painting updates at once. Slice changes, painting, and moving the image use the trackpad, arrow
  keys, and on-screen buttons, with no mouse wheel and no right-click. The slice slider shows the
  slice number and, in the heatmap condition, marks high-uncertainty slices. A short guide and one
  practice scan come first; the practice scan then shows the answer. Study scans do not. The toolbar
  is only there while a scan is open. Chrome on a MacBook trackpad is the supported setup (`ui/`).
