# Exploratory combinations — step 4

Six alternatives were fixed before evaluation on all 100 kits100 development cases. They average or maximize the normalized suspicion ranks of G1 with the missing-side score, hull deficit, or both. Scores were saved before attaching outcomes. No fitted weights, new thresholds, fresh scans, image/map reads or shared-code changes.

## Result

The predeclared rule recommends **g1_missing_hull_mean** provisionally: highest threshold review-AUC among combinations improving G1 AUC without reducing expected top-20 bad capture. This is development selection, not a prospective lock or established improvement.

Its review-AUC is **0.8470 versus 0.8170**, difference **+0.0300**, paired 95% case-bootstrap CI **[-0.0320, 0.0864]**. All six combinations have intervals including zero for their threshold review-AUC difference to G1. No combination improves the expected number of bad cases in the first 20. Multiple exploratory comparisons and selection make a fresh check necessary.

| Method | Threshold review-AUC | Bad found /20 | False alarms /20 | Original misses in /20 |
|---|---:|---:|---:|---:|
| model_diff_mean_region | 0.8170 | 11.00 | 9.00 | 0.00 |
| g1_missing_mean | 0.8237 | 11.00 | 9.00 | 1.00 |
| g1_missing_max | 0.8110 | 11.00 | 9.00 | 1.00 |
| g1_hull_mean | 0.8277 | 11.00 | 9.00 | 1.00 |
| g1_hull_max | 0.8190 | 10.00 | 10.00 | 1.00 |
| g1_missing_hull_mean | 0.8470 | 11.00 | 9.00 | 1.00 |
| g1_missing_hull_max | 0.8397 | 11.00 | 9.00 | 2.00 |
| entropy_mean_region | 0.8410 | 13.00 | 7.00 | 2.00 |
| far_disagree_per_volume | 0.8577 | 12.00 | 8.00 | 1.00 |

Bad = Dice < 0.83; a false alarm here means Dice >= 0.83, not error-free segmentation. Ties are expected random orderings. Mean/max scores are canonicalized on their exact rank lattice so floating-point noise cannot break true ties.

## What is gained and lost

The selected combination brings **case_00236** (Dice **0.8023**) into the first 20, from rank **23 to 9**. It displaces **case_00582** (Dice **0.4605**) from rank **17 to 45**. It exchanges seven nonbad cases for seven others. Therefore it still finds **11/15 bad cases with 9 false alarms**.

The selected combination does not bring the pelvic case into the first 20: case_00040 moves **46 to 43**. case_00184 moves **56 to 45**; case_00406 moves **42 to 27**. The maximum-of-all-three alternative rescues both case_00040 and case_00236, but displaces case_00426 and case_00582, so its top-20 net gain is also zero.

The two earlier missing-side false alerts stay outside the selected top 20: case_00293 is rank 31 and case_00563 rank 26. No additional image review of them was performed in this aggregate comparison.

## Why AUC and the 20-case budget tell different stories

The full review curve summarizes all budgets. Descriptively, the selected method finds 9/15 bad cases in the first 10 versus G1's 5/15; both find 11/15 after 20. The 10-case point was not a new selection criterion or primary endpoint. Earlier gains can coexist with later losses, so a higher full-curve point estimate does not imply improvement at every budget.

Mean Dice after simulated perfect correction of the first 20 is **0.92248** for the selected combination versus **0.92554** for G1. Thus quality at this fixed budget worsens despite equal bad-case counts. Full quality-AUC is **0.95398 versus 0.95242**, difference **+0.00157 [-0.00188, 0.00439]**; this also does not establish superiority.

Worst-10% review-AUC is **0.8690 versus 0.8550**, paired difference **+0.0140 [-0.0685, 0.0883]**. Against entropy, threshold AUC difference is **+0.0060 [-0.0764, 0.0921]**; against the existing far-disagreement score it is **-0.0107 [-0.0800, 0.0440]**. Neither comparison supports superiority.

## Decisions

Carry only the preselected mean-of-three candidate as a provisional option for step 5 planning, with these limitations stated. Do not claim more top-20 finds or rescue of the pelvic case for this candidate. The six alternatives and their tradeoffs remain saved; none has been tested on the fresh pool.

Deprioritize G1+hull maximum: it loses one net bad case (10/15 found) and fails the predeclared budget gate. Do not switch to the 10-case point, worst10, or an example-specific weight/threshold because it looks better after inspection. If a different rule is later chosen, date and justify that new decision before any fresh-pool access.

Next: prepare the exact prospective lock and coordinate the single joint G3-pool evaluation with Bjørn/Frederik. Fresh results may confirm or reject the small development AUC gain. The previously locked far-disagreement comparison remains unchanged.

## Reproduce

```bash
uv run python scripts/combine_missed_errors.py --config configs/kits_missed_combinations.yaml --stage benchmark
uv run python scripts/combine_missed_errors.py --config configs/kits_missed_combinations.yaml --stage evaluate
```

Input tables and output/source hashes preserve the completed earlier stages. Full metrics, paired comparisons, fair per-case ranks, all gained/lost inclusions, and the selection decision are saved alongside this interpretation. The commands regenerate numerical outputs, not this manual interpretation.
