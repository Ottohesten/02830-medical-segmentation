# Report notes: Robustness, missed errors, the locked test on fresh scans

| | |
|---|---|
| **Owner** | Otto |
| **Branch** | `otto/robustness` (merged), `otto/missed-errors` |
| **Where the details are** | [Robustness task](../tasks/g1_robustness.md), [missed-errors task](../tasks/g1_missed_errors.md), `results/kits_robustness_*/`, `results/kits_missed_*/` |
| **Last updated** | 2026-11-18 |
| **Coverage** | Complete for experiments finished by 2026-11-18; update when new results or group feedback arrive |

Working notes for the joint report synthesis; instructions: [README.md](README.md). Intervals below are 95%
bootstrap intervals; counts/ranks and descriptive Dice summaries have no interval unless one is stated.

## 1. What was done

- **Robustness:** does resolution loss reduce segmentation quality, and does the locked queue respond?
  Strengths were selected on 20 development cases without looking at score performance, then tested on the
  original 80 held-out cases, with both released models. Resolution loss resamples CT to 8 mm where the
  original grid is finer, then back to the original grid; labels still align. This simulates acquisition loss.
  Reference: `results/kits_robustness_dev/strength_lock.json`, `configs/kits_robustness_eval.yaml`.
- **Missed errors:** can GT-free size/shape checks catch confident omissions that uncertainty misses?
  All original kits100 cases were exploratory development data for this new question. Compared six fixed
  combinations, then locked mean suspicion ranks of G1 + missing-side + hull deficit.
- **Fresh test:** reused Frederik's original-model maps on the next seeded cohort and froze all scores before
  reading GT. Ran the combination and the previously locked far-disagreement question together once.
  Primary rule: 3 mm Dice < 0.83; improvement requires the lower bound of a two-sided paired review-AUC
  difference interval above zero, treating magnitudes <= 1e-12 as numerical zero. Used 2,000 case resamples,
  seed 42. Both named comparisons are reported without a familywise error claim.
  Recipe/cohort/implementation: `docs/missed_errors_fresh_protocol.md`, `results/kits_missed_fresh/lock.json`.

Score definitions for explaining the work: G1 averages absolute 3 mm/6 mm probability disagreement in the
predicted organ plus 5 mm. Missing-side uses the CT body midline and a 20 ml minimum predicted component;
hull deficit takes the maximum deficit of an individual component relative to its own convex hull, using
at most the two largest eligible predicted components. Their mean suspicion ranks give equal weight
despite different score scales. Far disagreement excludes the 6 mm edge band and divides remaining mask
disagreement by predicted organ size.
Exact connectivity, empty-mask behavior and tie rules are in the frozen protocol. None uses GT.

Evaluation target throughout: kidney + tumor + cyst (KiTS labels 1/2/3), while the model has kidney/cyst
classes and no explicit tumor class. A missed mass is an error under this target; the discrepancy also
reflects a possible difference from the pretrained model's intended label definition.
Source: `configs/datasets/kits23.yaml`; the original label-check discussion belongs to Bjørn's G1 notes.

## 2. Results

- **Robustness, fixed test:** median Dice 0.90955 → 0.87865; median paired loss 0.03010; threshold failures
  11 → 21/80. `results/kits_robustness_eval/dice_degradation.csv` (descriptive, no interval supplied there).
- **Robustness, primary response:** score rise versus Dice loss rho 0.215, 95% CI [-0.036, 0.459]; inconclusive.
  `results/kits_robustness_eval/change_correlations.csv`, test/model_diff_mean_region row.
- **Robustness, secondary response:** entropy rises in 79/80 cases versus primary 41/80; entropy change
  rho 0.243 [0.004, 0.473]. The more consistent response did not establish better failure ranking.
  `results/kits_robustness_eval/change_correlations.csv`, test rows.
- **Robustness, mixed queue:** review-AUC 0.8086 [0.7374, 0.8830], volume 0.4648 [0.3412, 0.5977]; paired
  difference +0.34375 [0.17946, 0.50517]. Clean/degraded copies are resampled together by patient.
  `results/kits_robustness_eval/g1_metrics.csv`, mixed_test/dice_below_083 rows.
- **Inspection, exploratory:** four of 15 threshold-bad cases lie outside G1's first 20 in the pooled
  development queue: case_00236, case_00406, case_00040 and case_00184. Saved organ probability is <= 0.10
  for 59.6–89.1% of their missed voxels; 48.8–86.8% lie outside the score region. This supports the confident
  omission mechanism. `results/kits_missed_errors/exploratory_inspection_summary.json`, `exploratory_review.md`.
- **Standalone checks, exploratory:** hull finds 9 bad/11 false alarms at budget 20, versus G1 11/9.
  Missing-side flags two bad and two nonbad cases, including the pelvic kidney. Neither is a general queue
  replacement. `results/kits_missed_checks/exploratory_budget.csv`, `exploratory_flag_cases.csv`.
- **Combination development, exploratory:** AUC 0.847 [0.768, 0.921] versus G1 0.817 [0.736, 0.897], difference +0.030
  [-0.032, 0.086]; no count gain at the fixed budget and worse simulated correction quality.
  `results/kits_missed_combinations/exploratory_metrics.csv`, `exploratory_paired.csv`.

**Fresh primary, fixed beforehand:** 100 cases, 10 threshold failures; neither candidate meets the criterion.
Sources: `results/kits_missed_fresh/metrics.csv`, `paired.csv`, `evaluation_completed.json`.

| Method | Review-AUC [95% CI] | Paired difference to G1 [95% CI] |
|---|---|---|
| G1 | 0.921 [0.875, 0.965] | Reference |
| Selected combination | 0.907 [0.839, 0.964] | -0.014 [-0.090, 0.043] |
| Far disagreement | 0.936 [0.900, 0.970] | +0.015 [-0.010, 0.050] |

- **Fresh budget, prespecified secondary:** first 20 reviews find 9/9/10 bad cases for G1/combination/far;
  differences in capture 0 [-3, 3] / +1 [-1, 3]. `results/kits_missed_fresh/budget_statistics.csv`.
- **Fresh quality, prespecified secondary:** combination quality-AUC gain +0.001075 [0.000079, 0.001878],
  unadjusted. Simulated mean Dice gain after 20 corrections +0.000983 [-0.002274, 0.005166].
  `results/kits_missed_fresh/paired.csv`, `budget_statistics.csv`.
- **Fresh association, prespecified secondary:** combination-minus-G1 Spearman rho difference -0.182
  [-0.290, -0.090]; more negative means higher suspicion aligns with lower Dice. This unadjusted signal
  concerns continuous quality rather than the primary bad-case rule. `results/kits_missed_fresh/paired.csv`.

## 3. What it means, and what it does not show

- The queue still identifies failures under the chosen resolution stress, but its score does not clearly track
  how much each scan deteriorates. Failure ranking and detecting degradation severity are different questions.
- The original G1 score performs well descriptively on the fresh cohort. Neither added-check recipe is
  established as better; intervals crossing zero also do not establish equivalence.
- Review-AUC summarizes how quickly bad cases are found over all budgets; quality-AUC weights their
  continuous Dice deficits under perfect correction. Equal bad counts can hide different severities.
  A higher whole-curve point estimate need not help at the chosen review budget.
- Broader quality ordering can improve while threshold-failure capture does not. The small secondary
  quality signal cannot replace the primary criterion. Perfect-correction simulations measure neither
  human correction speed nor real achieved quality.

## 4. Limitations

- Resolution stress is one selected perturbation, not general robustness to scanner/domain shifts.
- Fresh cohort has few bad cases; threshold and worst10 select exactly the same cases, not independent support.
- A false alarm means Dice >= 0.83, not error-free segmentation; threshold counts hide continuous severity.
- Mixed robustness has 160 views from 80 cases, with both views retained in each patient bootstrap;
  it measures failure ranking, not identification of every degraded image. Cohorts/bad rules have different
  failure prevalence and oracle AUC ceilings, so raw AUCs across these experiments are not a direct comparison.
- Prior G3 smoke exposure of case_00183 and case_00002 is disclosed; held out from our ranking development
  does not mean never seen by G3. Patient independence follows dataset descriptions, not an identity audit.
- Cohort/bootstraps use seed 42. Runtime and source provenance are machine-specific; reproducibility records
  are in the result folders. Robustness inference used Otto's RTX 5060 Ti with a separate torch 2.7.1+cu128
  environment; follow-up scoring/evaluation ran on CPU with the shared torch 2.2.2 lock. Machine versions:
  `results/kits_robustness_test/machine.json`, `results/kits_missed_fresh/evaluate_machine.json`.
- Visual inspection used selected slices and was performed by the assistant; Otto's own image review is
  not recorded. Stored probabilities have 1/255 precision; stored zero does not establish an exact original
  model probability. `results/kits_missed_errors/exploratory_inspection_summary.json`.

## 5. Decisions and rejected alternatives

- **2026-10-21:** selected the mildest eligible resolution loss by the written development rule. Tested noise
  and contrast strengths had no eligible candidate; did not force them into the final robustness test.
  Eligibility: at least five Dice drops >= 0.02, at least five cases still >= 0.83, at most two newly < 0.50.
  The stronger 12 mm option was eligible, but 8 mm already qualified. Sources:
  `configs/kits_robustness_dev.yaml`, `results/kits_robustness_dev/dev_degradation_summary.csv`.
- **2026-11-02:** carried missing-side/hull checks into combinations; deprioritized compactness and
  minimum-side-volume queues because of poor failure capture/false alarms. Preserved all exploratory results.
- **2026-11-02:** locked the selected combination and joint outcome stage; no example-specific weights or
  switching to a more favorable outcome rule after seeing results.
- **2026-11-04:** retained both primary null findings and rejected further tuning or extending this pool
  to seek a positive result. Preserved G1 as the locked reference.
- **2026-11-18:** merged current main into the task branch without rewriting the old scientific lock;
  historical source remains in commit `402dd91`. No ranking or inference rerun is needed for integration.
- **2026-11-18:** retain concise notes with source paths rather than report prose or a copy of the log.
  Complete the current evidence before the November 20 deadline; leave chapter allocation to joint synthesis.

## 6. What went wrong, and surprises

- A confident omission can leave a plausible outline, so neither uncertainty nor shape alone guarantees
  detection. Normal kidneys are concave, which also gives hull-based false alarms.
- The selected development combination rescued case_00236 (Dice 0.8023, rank 23 → 9), but displaced the
  much worse case_00582 (Dice 0.4605, rank 17 → 45). It did not rescue the pelvic kidney at budget 20.
  `results/kits_missed_combinations/exploratory_selected_changes.csv`, `exploratory_miss_ranks.csv`.
- Equal failure counts can hide different patients: the fresh combination promotes case_00527 but
  displaces case_00560, with similar Dice. `results/kits_missed_fresh/queue_changes.csv`.
- Tested noise/contrast did not reliably lower Dice and sometimes improved it descriptively. We did not
  claim those changes help clinical image quality or general robustness; the selection narrowed the scope.
  `results/kits_robustness_dev/dev_degradation_summary.csv`.
- Original torch could detect Otto's GPU but could not execute its kernels; a separate pinned GPU environment
  resolved robustness inference without changing the shared dependency lock. See robustness task step 2.
- The older data selector ignored pool offset. An isolated exact-cohort download helper avoided changing
  frozen shared source before evaluation. Main now supports offset; preserve the historical execution record.
- False floating-point differences in mathematically equal mean/max ranks were removed on the exact
  rank lattice before fresh locking. Ties at the review boundary use expected inclusion, so fractional
  counts describe random tie breaking. `scripts/combine_missed_errors.py`; fresh rules in the protocol.

## 7. Figures and tables for the report

- **First choice for this follow-up:** `results/kits_missed_fresh/report/fig_missed_fresh_summary.pdf`:
  paired primary intervals alongside failure capture at the fixed budget. A budget-only plot hides uncertainty.
- Robustness: `results/kits_robustness_eval/report/fig_robustness_summary_test.pdf`; the selected stress,
  score response and mixed queue. Captions/short summaries live beside both figures.
- Full numerical sources are linked above; no CT-derived montages are committed.

## 8. Top 3 points for the report

1. G1 performs well descriptively on the fresh cohort; neither locked candidate establishes better primary
   failure ranking. Keep the small unadjusted secondary quality signal separate from this conclusion.
2. G1 still ranks failures under the chosen resolution loss, while its primary response to degradation
   severity is inconclusive. General domain-shift robustness and human-effort benefits remain unestablished.
3. Shared confident omissions can escape disagreement. Plausibility checks recover particular errors but
   introduce false alarms and displacements; equal failure counts can hide a lost, more severe case.

## 9. Contribution

- Otto-owned work, developed with assistant support: robustness setup/benchmark/strength-selection and test
  runners; missed-case inspection and GT-free checks; combination comparison; prospective fresh protocol,
  source/coordination verification and guarded execution; numerical audits, figures, tests and working notes.
- Frederik supplied the reused pool predictions; Bjørn confirmed provenance and the joint lock. Actual group
  presentation remains pending in the task record. Attribution distinguishes the workstream from unaided authorship.

## 10. Specific to this part

- Explain the development rule before showing robustness outcomes; stronger stress was not selected merely
  because it produced a larger effect.
- Explain why averaging ranks lets differently scaled checks share a queue, and why catching one missed
  failure can displace another when the review budget stays fixed.
- In the fresh figure, both primary intervals cross zero. Far's perfect descriptive budget capture cannot
  override that criterion. The good G1 point estimate supports transfer to this cohort, with the limitations above.
