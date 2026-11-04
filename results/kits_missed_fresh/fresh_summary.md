# Joint fresh-pool ranking test — 2026-11-04

Neither locked candidate meets the prespecified primary improvement criterion. Both named comparisons are reported; no method or threshold was changed after opening the pool.

## Cohort and execution

- 100 KiTS23 cases, positions 101–200 of the original seed-42 draw; zero kits100 overlap.
- Original task 297/298 fold-0 models; supplied checkpoint files match the locked SHA-256 values.
- Bjørn confirmed original pool provenance, no earlier joint ranking test and review of the exact locked check, via Otto. Original message timestamp was not supplied; no direct Frederik reply is claimed.
- Prior G3 smoke exposure of case_00183/case_00002 is preserved and disclosed; these cases were not replaced.
- All 100 scores were saved/verified before GT voxel access, then both questions ran in one guarded outcome stage.
- 10 cases have 3 mm Dice < 0.83. These are exactly the worst 10 cases, so the two bad rules are identical here and do not provide separate replication.

## Primary result

Criterion: lower bound of the 95% paired case-bootstrap CI above numerical zero (1e-12); 2,000 resamples, seed 42. These are two named per-comparison intervals, with no familywise claim.

| Method | Review-AUC | Difference to G1 [95% CI] | Criterion met? |
|---|---:|---|---|
| Locked G1 | 0.9210 | Reference | — |
| Selected combination | 0.9070 | -0.0140 [-0.0900, 0.0429] | No |
| Far disagreement | 0.9360 | +0.0150 [-0.0100, 0.0500] | No |

The selected combination has a lower point estimate; far disagreement has a higher one. Both intervals include zero, so the fresh primary analysis establishes neither as an improvement over G1. It does not prove equivalence or rule out smaller effects. G1 is already close to the oracle review-AUC of 0.9500 on this cohort.

## Prespecified 20-case budget

| Method | Bad found / 10 | False alarms | Simulated mean Dice after 20 perfect corrections |
|---|---:|---:|---:|
| Locked G1 | 9 | 11 | 0.931545 |
| Selected combination | 9 | 11 | 0.932528 |
| Far disagreement | 10 | 10 | 0.932524 |
| Entropy control | 9 | 11 | 0.932369 |
| Volume control | 2 | 18 | 0.909336 |

Points average over random breaking of score ties. Two selected-combination cases tie across the budget boundary, so their inclusion probabilities are 0.5. These are simulated perfect corrections, not measured human correction performance/time.

- Selected combination: bad-capture difference +0 [-3, 3]; mean-Dice-after-20 difference +0.000983 [-0.002274, 0.005166].
- Far disagreement: bad-capture difference +1 [-1, 3]; mean-Dice-after-20 difference +0.000978 [-0.001145, 0.004858].

## Queue changes

- Selected combination gains bad case_00527 and loses bad case_00560: one gained/one lost, with four expected nonbad inclusions gained and four removed. All individual changes/probabilities are in queue_changes.csv.
- Far disagreement gains case_00527 without losing a bad case; one nonbad case enters and two leave. All individual changes/probabilities are in far_queue_changes.csv.

| Case | 3 mm Dice | G1 rank | Combination rank | Far rank |
|---|---:|---:|---:|---:|
| case_00527 | 0.778522 | 23 | 11 | 11 |
| case_00560 | 0.777604 | 4 | 39 | 9 |

## Secondary findings and interpretation

- Combination quality-AUC 0.957772 vs G1 0.956697; difference +0.001075 [0.000079, 0.001878].
- Combination Spearman rho -0.706779 vs G1 -0.524740; difference -0.182039 [-0.290077, -0.089880]. More negative means higher suspicion is more associated with lower Dice.
- These prespecified secondary intervals are unadjusted. The positive quality-AUC signal does not change the failed primary criterion. Review-AUC tracks finding threshold-bad cases; quality-AUC also rewards earlier correction of moderate errors above the threshold. They can favor different queues.
- Keep the locked G1 result and report both fresh null primary findings. Do not select a different score, tune weights, replace cases or reopen this pool to obtain a positive result.

## Verification and practical next steps

- Independent rank-AUC identities, curve integrals, fixed-budget points, queue changes, timing/order and hashes passed using saved aggregates only; fresh GT arrays were not reopened. fresh_verification.json.
- All 20 scientific fingerprints, 61 locked G1 result files and 600 kits100 saved maps are unchanged.
- CPU: AMD Ryzen 5 7500F; Linux 7.3.1-1-cachyos; Python 3.12.15; torch 2.2.2+cu121; 6 threads; seed 42. No GPU computation or new inference.
- Score stage 713.23 s, peak resident bytes 9,224,491,008; outcome stage 168.11 s, peak 2,744,385,536. Stage metadata/resource logs preserve baseline d7bea17, dirty state and hashes.
- 193 tests passed, 36 skipped, 8 existing warnings (fresh_tests.txt).
- Next: one figure and a short summary for step 6, explaining the primary/secondary distinction and the two swapped bad cases. Preserve this one-time result rather than begin new tuning before the deadline.
