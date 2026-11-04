# Exploratory size and shape checks — step 3

All 100 kits100 cases are development data for this question. Bad = saved 3 mm Dice < 0.83 (15 cases). The review budget remains 20 cases. False alarms below mean Dice >= 0.83, not a perfect segmentation.

## What changed

GT-free features were saved before attaching outcomes. Size reuses saved left/right volumes. Shape uses the two largest 6-connected predicted components >= 20 ml as kidney proxies, including voxel corners in a physical convex hull. Deficit = 1 - mask volume / hull volume. The second shape measure is 1 - 36*pi*V^2/A^3, using voxel-face surface area. Empty/only-tiny predictions get score 1. A single convex component can look normal despite a missing kidney.

## Standalone comparisons

| Check | Expected bad / 20 | Expected false alarms / 20 | Original misses included (expected) | Review-AUC |
|---|---:|---:|---:|---:|
| Locked G1 | 11.00 | 9.00 | 0.00 | 0.8170 |
| Entropy (existing) | 13.00 | 7.00 | 2.00 | 0.8410 |
| Small total volume | 3.00 | 17.00 | 2.00 | 0.4197 |
| Missing-side flag | 4.17 | 15.83 | 1.50 | 0.5473 |
| Side asymmetry | 6.00 | 14.00 | 2.00 | 0.6477 |
| Small minimum side | 4.00 | 16.00 | 2.00 | 0.4497 |
| Convex-hull deficit | 9.00 | 11.00 | 1.00 | 0.7543 |
| Voxel-face compactness | 7.00 | 13.00 | 0.00 | 0.6623 |
| Far disagreement (existing) | 12.00 | 8.00 | 1.00 | 0.8577 |
| Far 6 mm additions (existing) | 12.00 | 8.00 | 2.00 | 0.8103 |

Tied scores use expected performance over random tie breaking, as in G1. Fractional counts are expectations, not fractional people. The missing-side flag identifies only four scans; the remaining 16 review places are filled randomly among the 96 unflagged cases. Its 1.50 expected original misses comprise the flagged pelvic case plus 0.50 from random inclusion of the other three misses.

## Where each missed case moves

| Method | case_00040 | case_00184 | case_00236 | case_00406 |
|---|---:|---:|---:|---:|
| Locked G1 | 46.0 | 56.0 | 23.0 | 42.0 |
| Entropy (existing) | 90.0 | 20.0 | 9.0 | 32.0 |
| Small total volume | 3.0 | 81.0 | 97.0 | 14.0 |
| Missing-side flag | 3.0 | 52.5 | 52.5 | 52.5 |
| Side asymmetry | 1.5 | 52.0 | 17.0 | 66.0 |
| Small minimum side | 1.5 | 93.0 | 91.0 | 20.0 |
| Convex-hull deficit | 95.0 | 36.0 | 9.0 | 21.0 |
| Voxel-face compactness | 84.0 | 70.0 | 25.0 | 34.0 |
| Far disagreement (existing) | 54.0 | 23.0 | 20.0 | 41.0 |
| Far 6 mm additions (existing) | 82.5 | 15.0 | 14.0 | 31.0 |

Ranks are average ranks within ties; inclusion probabilities are saved separately in exploratory_miss_ranks.csv.

## Interpretation and next action

The hull check finds 9/15 bad cases in its first 20 (11 false alarms), versus 11/15 and 9 false alarms for locked G1. It promotes case_00236 from 23 to 9, case_00406 from 42 to 21 (still outside the unchanged budget), and case_00184 from 56 to 36. It pushes the pelvic case from 46 to 95: shape of the remaining kidney cannot reveal the absent kidney.

The missing-side flag identifies case_00040 (Dice 0.6607), case_00516 (0.0027; already high in G1), and two nonbad cases: case_00293 (0.9475) and case_00563 (0.8779). This is a narrow alert with two false alarms, not an adequate general-purpose queue.

Carry the missing-side flag and hull deficit to step 4 as complementary checks. Compare combinations with the unchanged G1 score; retain existing asymmetry, entropy and far-disagreement methods as controls. No combination or fresh-pool lock has been made here.

Deprioritize this voxel-face compactness version: it rescues no original miss at budget 20 and finds only 7 bad cases with 13 false alarms. Also deprioritize minimum-side volume as the main queue: it finds 4 bad cases with 16 false alarms and mostly follows a size heuristic. The underlying size/shape ideas are not disproved.

For hull versus G1, threshold review-AUC difference is -0.0627, paired 95% case-bootstrap CI [-0.1905, 0.0556]. Its worst-10% difference is -0.1050 [-0.2857, 0.0543]. Neither supports superiority. The existing far-disagreement score has threshold difference +0.0407 [0.0000, 0.0967]; the lower bound is exactly zero, not above zero. It remains reserved for the previously locked fresh test.

All 10 score comparisons are exploratory. No multiplicity correction or out-of-sample claim is made. Both bad rules, quality-AUC, Spearman/partial rho, paired intervals and curves are retained in the CSVs; the pooled queues are not the locked 80-case test result.

## Reproduce

```bash
uv run python scripts/check_missed_errors.py --config configs/kits_missed_checks.yaml --stage benchmark
uv run python scripts/check_missed_errors.py --config configs/kits_missed_checks.yaml --stage features
uv run python scripts/check_missed_errors.py --config configs/kits_missed_checks.yaml --stage evaluate
```

The benchmark estimates runtime; actual feature and evaluation stages took 84.10 s and 50.30 s on the recorded CPU. This summary interprets saved tables; the commands regenerate numerical results, not this manual interpretation. No CT-derived imagery is produced.
