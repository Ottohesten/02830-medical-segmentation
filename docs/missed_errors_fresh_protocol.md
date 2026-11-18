# Prospective joint ranking test — 2026-11-02

This freezes Otto's selected missed-errors score before fresh ranking outcomes are opened. It also carries
the **unchanged 2026-10-21 far-disagreement test** into the same evaluation. This document and
`configs/kits_missed_fresh.yaml` define the scientific recipe; separate handoff metadata supplies file paths
and actual coordination evidence. No fresh outcome has been computed by this task at the time of writing.

## Question and development evidence

Does `g1_missing_hull_mean` rank failures better than locked G1 on cases not used for its development?
It was selected among six fixed combinations on all 100 kits100 development cases. Review-AUC was
0.847 versus 0.817, paired difference +0.030, 95% CI [-0.032, 0.086]. It found no additional bad cases
in the first 20 and displaced a more severe error, worsening simulated quality at that budget.
These limitations motivate a fresh check; they are not evidence of established superiority.

## Cohort fixed before outcome access

- Use sorted IDs from the original local KiTS23 `case_list.txt`, permuted by NumPy `default_rng(42)`.
  Positions 1–100 must reproduce the locked kits100 IDs exactly. Positions **101–200** form this pool.
  Freeze the resulting 100 IDs and the source-list hash. Do not replace missing or difficult cases.
- Analyze in case-ID order. Require no overlap with kits100, exact 100-case coverage, and original baseline
  predictions for both released models (3 mm task 297 and 6 mm task 298, fold 0), never G3-fine-tuned weights.
  Reference checkpoint hashes are recorded during freezing and checked against Frederik's provenance.
- G3's earlier smoke test used case_00183 and case_00002, the first two pool cases. Keep them in the
  already specified cohort and disclose that prior feasibility/GT exposure. The pool is held out from
  this ranking-development question; it is not entirely unseen by every other project activity. Simulated
  corrections/training for G3 do not authorize an earlier ranking-metric evaluation of these questions.
- Resample case IDs as independent units, consistent with G1. The official [KiTS23 data description](https://kits-challenge.org/kits23/)
  describes a patient-based cohort and uses the most recent preoperative CT for each case. Treating distinct
  case IDs as distinct patients is an inference from that description, not an independent identity audit.
  Resolve contrary identity/overlap information and log any amendment before outcome access.

## Frozen scores: ground-truth-free

All inputs must share the canonical CT grid. Use kidney + tumor + cyst as the evaluation organ (labels 1, 2, 3).

1. **Locked G1:** `model_diff_mean_region` from the unchanged `scan_scores`: mean absolute difference
   between saved 3 mm and 6 mm organ probabilities in the 3 mm predicted organ plus a 5 mm border.
2. **Missing-side score:** unchanged `paired_organ_check`, body threshold -500 HU, component threshold
   20 ml; score 0/1/2 missing sides. This uses CT and the predicted mask, never labels.
3. **Hull deficit:** unchanged step 3 `mask_features`; maximum `1 - voxel_volume / hull_volume` over
   the two largest 6-connected predicted components of at least 20 ml. Hull uses occupied voxel boxes;
   no eligible component gives deficit 1. No side labels or GT are used to choose components.
4. **One selected combination:** mean of normalized suspicion ranks of scores 1–3, using the shared
   `combine_scores` and step 4 exact rank-lattice canonicalization. Higher means earlier review. Recompute
   these ranks on the complete fresh cohort; do not carry fitted kits100 ranks/weights into it.
5. **Previously locked comparison:** `far_disagree_per_volume`: 3 mm/6 mm mask disagreement strictly
   more than 6 mm from the 3 mm edge, divided by `max(3 mm mask voxels, 1)`, using existing directional
   disagreement/candidate functions. Use full calculation precision, not rounded table values.
6. **Secondary controls:** `entropy_mean_region` and `neg_volume_ml`; random order and oracle from G1.

Use the standard saved probability representation (uint8 with scale 1/255 read as float32), original
weights and the shared organ definition. Freeze all case scores before reading GT voxels. Do not tune
thresholds, component settings, weights, candidate count or tie policy after the pool is opened.

## Primary comparison and success criterion

- Primary outcome for each of the two declared comparisons: **review-AUC, bad = 3 mm Dice < 0.83**.
- Compare selected combination minus locked G1, and previously locked far score minus locked G1.
  Use the unchanged G1 paired bootstrap: **2,000 case resamples, seed 42, two-sided 95% percentile CI**.
  A comparison meets its ranking-improvement criterion only when the lower bound is strictly above zero;
  magnitudes <= 1e-12 are numerical zero. This does not change the earlier far-score scientific criterion.
- These are **two named per-comparison tests**, with unadjusted 95% CIs, not a familywise 5% claim that
  whichever score wins is confirmed. Report both, including null/negative results; no additional fresh candidates.
- Bootstrap the fixed full-cohort score vectors and bad flags, as in G1. Do not refit combinations or
  redefine worst10 inside resamples. Use expected curves over random score-tie breaking.
- If the pool has no bad cases or all cases are bad under the primary rule, label primary discrimination
  uninformative; do not substitute a new threshold. Report finite-bootstrap counts and all secondary outputs.

## Prespecified secondary checks

- Worst-10% review-AUC, quality-AUC, Spearman rho/partial rho and paired differences already produced by G1.
- At **ceil(20% × N)** reviews: expected bad capture, expected false alarms (Dice >= 0.83), and simulated
  mean Dice after perfect correction. Report paired 95% case-bootstrap differences to G1 for capture and
  quality, using 2,000 resamples/seed 42 and the same fixed scores. These do not replace the primary criterion.
- Save each newly included and displaced case with its baseline Dice and inclusion probability. Report
  severe displaced errors by their continuous Dice; do not choose another post-hoc severity threshold.
- An AUC improvement with worse top-20 quality must be described as a tradeoff, not overall workflow superiority.
  These simulated quality curves do not measure actual participant correction time or performance.
- Save per-case 3 mm and 6 mm Dice, raw GT-free scores and score/rank changes. No CT/mask/probability imagery in Git.

## Order, coordination and one-time execution

1. Freeze the config, this protocol, cohort IDs, checkpoint and implementation hashes; commit/push the lock.
2. Tell Bjørn the exact score/test; coordinate the one shared opening with Frederik by 2026-11-18.
   Record actual communication evidence, source inference metadata, matching pool IDs and baseline weight hashes.
   Do not invent notification/confirmation or assume that a branch push is direct coordination.
3. Verify file coverage, headers/affines, hashes and provenance without loading GT voxels. Benchmark on
   old kits100 development inputs and synthetic metrics; ask before downloads or a projected >30-minute stage.
4. Compute and freeze all fresh scores, then evaluate both comparisons in one controlled outcome stage.
   An exclusive `evaluation_started.json` prevents silently opening/replacing results a second time.
   A technical failure requires a logged, identical-analysis recovery; it does not permit new choices.
5. Preserve the lock and all outcomes, update the task/log/overview and publish aggregate results.

## Handoff needed from the group

Supply one CSV with `case_id,ct,label,mask,prob,m2_mask,m2_prob` for the frozen 100 IDs. Paths may be absolute
or relative to that CSV's folder. Reuse Frederik's original maps; do not rerun inference. Supply source
inference metadata identifying machine, model/checkpoint/config versions and the prediction provenance.

Coordination metadata records `bjorn_notified_at`, `frederik_confirmed_at`, `evidence`,
`ranking_metrics_not_previously_evaluated`, `checkpoint_sha256` (model_3mm/model_6mm),
`inference_metadata` (path), and `inputs_manifest` (path). Dates/evidence must describe real coordination.
Missing data or coordination leaves the outcome stage pending; this task does not guess paths or open GT.
