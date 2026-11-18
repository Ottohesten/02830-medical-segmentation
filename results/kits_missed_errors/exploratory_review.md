# Exploratory review of missed cases

CT-derived montages stay local and are excluded from Git. Image links work after regenerating the figures:

```bash
uv run python scripts/inspect_missed_errors.py --config configs/kits_missed_errors.yaml --stage images
```

All 100 kits100 cases are development data for this follow-up. The saved G1 score is sorted descending; exact ties use ascending case ID. Bad means Dice < 0.83. The first 20 cases contain 11 of the 15 bad cases, with four outside the budget. There is no tie at the budget boundary.

These are assistant-reviewed diagnostic views, with ground truth used to select error peaks and crops. They are not user-study screens or clinical assessments. Original G1 split labels are retained in the CSVs; this pooled queue is not the locked 80-case G1 test result.

| Case | Queue rank | Saved Dice | What is missing |
|---|---:|---:|---|
| [case_00236](figures/exploratory_case_00236_inspection.png) | 23 | 0.8023 | Partial tumor and kidney omission |
| [case_00406](figures/exploratory_case_00406_inspection.png) | 42 | 0.8004 | Near-total tumor omission |
| [case_00040](figures/exploratory_case_00040_inspection.png) | 46 | 0.6607 | Missed pelvic kidney |
| [case_00184](figures/exploratory_case_00184_inspection.png) | 56 | 0.8176 | Multiple tumor and cyst regions omitted |

## What the pictures and counts show

The prediction omits tissue rather than adding it: false negatives make up 98.5–99.9% of wrong voxels within each missed case. Low organ probability (saved p <= 0.10) occurs in 59.6–89.1% of their missed voxels, and 48.8–86.8% of missed voxels lie outside the predicted organ plus its 5 mm uncertainty border. These descriptive measurements support the confident-error mechanism; being late in the queue alone would not prove it.

The 6 mm model covers only 2.0–26.8% of the 3 mm false negatives in these cases. It can help, especially in case_00184, but the models share many omissions. No new ranking score or combination has been tested here.

The severe example case_00516 is already rank 3 in this pooled queue (saved Dice 0.0027), so it is not a missed case under this budget. Error severity alone does not determine queue position.

| Case | Missed kidney (%) | Missed tumor (%) | Missed cyst (%) | Missed voxels with p <= 0.10 (%) | Missed voxels outside region (%) |
|---|---:|---:|---:|---:|---:|
| case_00236 | 20.5 | 44.6 | absent | 59.6 | 51.3 |
| case_00406 | 22.4 | 99.2 | absent | 63.9 | 48.8 |
| case_00040 | 51.4 | 14.6 | absent | 89.1 | 86.8 |
| case_00184 | 12.6 | 42.1 | 77.2 | 62.7 | 53.7 |

The first three columns use each GT label as their denominator; the final two use all false-negative organ voxels. Probabilities are saved at 1/255 precision, so a saved zero is not proof of an exact network probability of zero.

## Per-case observations

### case_00236 — rank 23, Dice 0.8023

A large GT-labeled tumor in the left kidney is only partly included, leaving a broad indentation/interior gap; kidney tissue near the hilum is also omitted. The two masks share much of the missing region.

Inspected: Axial z=12 and z=15; coronal y=243. [Full montage](figures/exploratory_case_00236_inspection.png).

### case_00406 — rank 42, Dice 0.8004

An outward-projecting GT-labeled tumor next to the left kidney is almost entirely absent from both masks. Both models trace the remaining kidney while assigning low organ probability to most of the protruding mass.

Inspected: Axial z=52, z=53 and z=60; coronal y=259. [Full montage](figures/exploratory_case_00406_inspection.png).

### case_00040 — rank 46, Dice 0.6607

The GT-labeled kidney in the pelvis has no visible outline from either model in the inspected pelvic views. The other kidney higher in the abdomen is segmented, so uncertainty around that outline cannot cover the distant missing kidney.

Inspected: Axial z=79 versus z=139 and z=141; coronal y=279. [Full montage](figures/exploratory_case_00040_inspection.png).

### case_00184 — rank 56, Dice 0.8176

Both kidneys are present in the prediction, but several GT tumor/cyst regions are partly or completely omitted. A large upper mass and protruding cyst regions remain dark in the probability map. The 6 mm mask improves whole-scan Dice but leaves substantial omissions.

Inspected: Axial z=72, z=75, z=77 and z=81; coronal y=276. [Full montage](figures/exploratory_case_00184_inspection.png).

## Reading the montages

Column 1: CT with the 3 mm mask in red. Column 2: CT with the 6 mm mask in magenta. GT contours are kidney in lime, tumor in orange and cyst in cyan. Columns 3/4 show 3 mm organ probability and entropy on CT, both on a 0–1 scale with dark colors at the low end; white contours show the GT organ union.

Each row is a selected axial error peak or a coronal view. Physical pixel spacing is respected. These selected views do not show every slice; they complement the whole-volume error counts.

## Next checks to try

1. Start with the existing missing-side/size information: the pelvic kidney is absent, so a shape check of the remaining kidney alone cannot find it. Measure false alarms on all 100 cases.
2. Try per-kidney shape/convex-hull deficits for the tumor cases, allowing for the normal kidney indentation and for masses that leave no obvious contour defect. Treat this as a hypothesis, not a demonstrated detector.
3. Keep the second-model directional disagreement as a complementary comparison; quantify shared misses rather than assuming model disagreement reveals all errors.

The fresh G3-pool lock and joint evaluation remain unchanged. Sources: exploratory_queue.csv, exploratory_error_diagnostics.csv, exploratory_inspection_evidence.json and exploratory_missed_review.csv.
