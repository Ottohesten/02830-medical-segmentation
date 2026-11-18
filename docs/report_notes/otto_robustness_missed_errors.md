# Report notes: Robustness, missed errors, the locked test on fresh scans

| | |
|---|---|
| **Owner** | Otto |
| **Branch** | `otto/robustness` (merged), `otto/missed-errors` |
| **Where the details are** | [Robustness task](../tasks/g1_robustness.md), [missed-errors task](../tasks/g1_missed_errors.md), `results/kits_robustness_*/`, `results/kits_missed_*/` |
| **Last updated** | 2026-11-18 |

How to fill this in: [README.md](README.md). Notes, not report text. Every number with its interval and file path.

## 1. What was done

- **Robustness:** does resolution loss reduce segmentation quality, and does the locked queue respond?
  Perturbation strengths were selected on the original development cohort, without score performance; the
  selected 8 mm degradation was then tested once, with both original models and patient-paired bootstrap.
  Reference: `results/kits_robustness_dev/strength_lock.json`, `configs/kits_robustness_eval.yaml`.
- **Missed errors:** can GT-free size/shape checks catch confident omissions that uncertainty misses?
  All original kits100 cases were exploratory development data for this new question. Compared six fixed
  combinations, then locked mean suspicion ranks of G1 + missing-side + hull deficit.
- **Fresh test:** reused Frederik's original-model maps on the next seeded cohort and froze all scores before
  reading GT. Ran the combination and the previously locked far-disagreement question together once.
  Primary rule: 3 mm Dice < 0.83; improvement requires a paired review-AUC difference whose two-sided 95%
  bootstrap interval is above zero. Both named comparisons are reported without a familywise error claim.
  Recipe/cohort/implementation: `docs/missed_errors_fresh_protocol.md`, `results/kits_missed_fresh/lock.json`.

## 2. Results

- **Robustness, fixed test:** median Dice 0.90955 → 0.87865; median paired loss 0.03010; threshold failures
  11 → 21/80. `results/kits_robustness_eval/dice_degradation.csv` (descriptive, no interval supplied there).
- **Robustness, primary response:** score rise versus Dice loss rho 0.215, 95% CI [-0.036, 0.459]; inconclusive.
  `results/kits_robustness_eval/change_correlations.csv`, test/model_diff_mean_region row.
- **Robustness, mixed queue:** review-AUC 0.8086 [0.7374, 0.8830], volume 0.4648 [0.3412, 0.5977]; paired
  difference +0.34375 [0.17946, 0.50517]. Clean/degraded copies are resampled together by patient.
  `results/kits_robustness_eval/g1_metrics.csv`, mixed_test/dice_below_083 rows.
- **Development, exploratory:** selected combination AUC 0.847 versus G1 0.817, difference +0.030
  [-0.032, 0.086]; no count gain at the fixed budget and worse simulated correction quality.
  `results/kits_missed_combinations/exploratory_combination_summary.md`, `exploratory_paired.csv`.
- **Fresh primary, fixed beforehand:** 100 cases, 10 threshold failures. G1 AUC 0.921; combination 0.907,
  difference -0.014 [-0.090, 0.043]; far disagreement 0.936, difference +0.015 [-0.010, 0.050]. Neither
  meets the criterion. `results/kits_missed_fresh/metrics.csv`, `paired.csv`, `evaluation_completed.json`.
- **Fresh budget, prespecified secondary:** first 20 reviews find 9/9/10 bad cases for G1/combination/far;
  differences in capture 0 [-3, 3] / +1 [-1, 3]. `results/kits_missed_fresh/budget_statistics.csv`.
- **Fresh quality, prespecified secondary:** combination quality-AUC gain +0.001075 [0.000079, 0.001878],
  unadjusted. Simulated mean Dice gain after 20 corrections +0.000983 [-0.002274, 0.005166].
  `results/kits_missed_fresh/paired.csv`, `budget_statistics.csv`.

## 3. What it means, and what it does not show

- The queue still identifies failures under the chosen resolution stress, but its score does not clearly track
  how much each scan deteriorates. Failure ranking and detecting degradation severity are different questions.
- The original G1 score performs well descriptively on the fresh cohort. Neither added-check recipe is
  established as better; intervals crossing zero also do not establish equivalence.
- Broader quality ordering can improve while threshold-failure capture does not. The small secondary
  quality signal cannot replace the primary criterion. Perfect-correction simulations measure neither
  human correction speed nor real achieved quality.

## 4. Limitations

- Resolution stress is one selected perturbation, not general robustness to scanner/domain shifts.
- Fresh cohort has few bad cases; threshold and worst10 select exactly the same cases, not independent support.
- Prior G3 smoke exposure of case_00183 and case_00002 is disclosed; held out from our ranking development
  does not mean never seen by G3. Patient independence follows dataset descriptions, not an identity audit.
- Cohort/bootstraps use seed 42. Runtime and source provenance are machine-specific; reproducibility records
  are in the result folders. The newer shared data selector differs from the source used for the fresh test.

## 5. Decisions and rejected alternatives

- **2026-10-21:** selected the mildest eligible resolution loss by the written development rule. Tested noise
  and contrast strengths had no eligible candidate; did not force them into the final robustness test.
- **2026-11-02:** carried missing-side/hull checks into combinations; deprioritized compactness and
  minimum-side-volume queues because of poor failure capture/false alarms. Preserved all exploratory results.
- **2026-11-02:** locked the selected combination and joint outcome stage; no example-specific weights or
  switching to a more favorable outcome rule after seeing results.
- **2026-11-04:** retained both primary null findings and rejected further tuning or extending this pool
  to seek a positive result. Preserved G1 as the locked reference.
- **2026-11-18:** merged current main into the task branch without rewriting the old scientific lock;
  historical source remains in commit `402dd91`. No ranking or inference rerun is needed for integration.

## 6. What went wrong, and surprises

- A confident omission can leave a plausible outline, so neither uncertainty nor shape alone guarantees
  detection. Normal kidneys are concave, which also gives hull-based false alarms.
- Equal failure counts can hide different patients: the fresh combination promotes case_00527 but
  displaces case_00560, with similar Dice. `results/kits_missed_fresh/queue_changes.csv`.
- Original torch could detect Otto's GPU but could not execute its kernels; a separate pinned GPU environment
  resolved robustness inference without changing the shared dependency lock. See robustness task step 2.
- The older data selector ignored pool offset. An isolated exact-cohort download helper avoided changing
  frozen shared source before evaluation. Main now supports offset; preserve the historical execution record.

## 7. Figures and tables for the report

- **First choice for this follow-up:** `results/kits_missed_fresh/report/fig_missed_fresh_summary.pdf`:
  paired primary intervals alongside failure capture at the fixed budget. A budget-only plot hides uncertainty.
- Robustness: `results/kits_robustness_eval/report/fig_robustness_summary_test.pdf`; the selected stress,
  score response and mixed queue. Captions/short summaries live beside both figures.
- Full numerical sources are linked above; no CT-derived montages are committed.

## 8. Top 3 points for the report

1. Confident omissions motivate GT-free plausibility checks, but plausible shapes and false alarms limit them.
2. Neither locked fresh candidate establishes better primary failure ranking than G1; retain the secondary nuance.
3. Ranking failures under resolution loss works better than tracking degradation severity; human-effort benefits
   still require the separate study.

## 9. Contribution

Otto-owned work, developed with assistant support: robustness setup/benchmark/strength-selection and test
runners; missed-case inspection and GT-free checks; combination comparison; prospective fresh protocol,
source/coordination verification and guarded execution; numerical audits, figures, tests and working notes.
Frederik supplied the reused pool predictions; Bjørn confirmed provenance and the joint lock. Actual group
presentation remains pending in the task record. This list describes the workstream, not unaided authorship.

## 10. Specific to this part

- Explain the development rule before showing robustness outcomes; stronger stress was not selected merely
  because it produced a larger effect.
- Explain why averaging ranks lets differently scaled checks share a queue, and why catching one missed
  failure can displace another when the review budget stays fixed.
- In the fresh figure, both primary intervals cross zero. Far's perfect descriptive budget capture cannot
  override that criterion. The good G1 point estimate supports transfer to this cohort, with the limitations above.
