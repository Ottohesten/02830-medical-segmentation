# Analysis plan for the user study (G2)

This plan says how the user study will be analyzed. We write it and lock it **before the first real participant
(P01)**, so the choices below cannot be influenced by the results. `scripts/analyze_study.py` (code in
`src/segreview/study_analysis.py`) carries out exactly this plan, with the settings in `configs/study.yaml`
(`analysis:`). If the plan and the code ever disagree, the plan is what we promised, and the code must be fixed.

## Lock

Fill in when the plan is locked, after the pilot and before P01. Commit this file with the code and configs it refers
to; then write that commit's hash below and commit again (a commit cannot contain its own hash). Nothing in this
plan changes after that, except through [Changes after locking](#changes-after-locking).

| Field | Value |
|-------|-------|
| Locked on (date and time) | ____________________ |
| Locked by | ____________________ |
| Git commit with the locked plan, code and `configs/study.yaml` | ____________________ |
| Heatmap display (`study.heatmap_display`, chosen in the pilot) | ____________________ (now: `full`) |
| Time limit per scan (`study.time_limit_min`) | ____________________ (now: 5 minutes) |
| Study scans, block A / block B (`manifest.json`) | ____________________ (now: case_00470, case_00259, case_00460 / case_00554, case_00438, case_00426) |

## Design in short

- Within-subject design: every participant corrects the AI's kidney outline in 6 CT scans, 3 **with** the
  uncertainty heatmap and 3 **without** it, after one practice scan (with the heatmap) that is not analyzed.
- Balancing: scans 1–3 form block A and 4–6 block B. Half of the participants get the heatmap on block A, half on
  block B, and the order of the conditions is swapped too, giving 4 participant types (`src/segreview/study.py`).
  The number in the participant ID sets the type, so we recruit in groups of 4.
- Time limit per scan (see Lock). When the time is up, the scan is locked and saved as it is.
- Raw NASA-TLX after each condition block (`study.tlx: after_each_block`).
- Participants are fellow students, not clinicians. Group members never take part as participants.

## Hypotheses

- **H1 (primary):** the Dice gain per minute is **higher** with the heatmap than without it.
- **H2 (key secondary):** the perceived workload (raw NASA-TLX) is **lower** with the heatmap than without it.

Both are tested two-sided: a clear effect in the opposite direction is reported as such.

## Outcomes

**Per corrected scan.** Dice is computed against the ground truth with the locked organ definition
`kidney_tumor_cyst` (kidney + tumor + cyst, KiTS23 labels 1–3), on the same cropped voxel grid the participant
edited. The ground truth is used only in this analysis, never shown in the study scans.

- Dice before = the AI's mask against the ground truth; Dice after = the participant's saved mask against it.
- Minutes = the logged time on the scan, from when it is ready to edit until "Done" or until the time limit.
- **Gain per minute = (Dice after − Dice before) / minutes.** Positive = the mask got better.

**Per participant.** The mean over the 3 scans in each condition, and the difference **with − without**. So a
positive difference in gain per minute favors the heatmap, and a negative difference in NASA-TLX does.

**NASA-TLX.** Raw TLX = the mean of the six scales (0–100, higher = more workload), one questionnaire after each
block. The difference is again with − without.

## Statistical tests

- **Test:** Wilcoxon signed-rank test on the per-participant differences, paired per participant, **two-sided**,
  **α = 0.05**. Zero differences are left out of the test (the usual convention); tied absolute differences get
  their average rank.
- **p-value:** exact. We enumerate all 2^n patterns of signs on the ranks (n = number of non-zero differences) and
  count how many give a statistic at least as far from its middle as the observed one. Without ties this is the
  textbook exact Wilcoxon p-value; with ties (likely for NASA-TLX) it is the exact permutation p-value. Above 24
  non-zero differences, 2^20 random sign patterns with the study seed (42) are used instead.
- **Two hypotheses, fixed order:** H1 is tested at α = 0.05. H2 is tested at α = 0.05 **only if H1 is supported**.
  Testing in a fixed order keeps the chance of any false positive among H1 and H2 at 5 % without making the H1 test
  stricter. If H1 is not supported, H2 is reported as exploratory.
- **Decision rule** (`conclusion` in `study_tests.csv`): *supported* if p < 0.05 and the effect goes the expected
  way; *opposite direction* if p < 0.05 the other way; otherwise *not supported*. The direction is the sign of the
  rank-biserial correlation (below), which is the direction the test itself measures.
- **Effect sizes, always reported, whatever the p-value:**
  - matched-pairs rank-biserial correlation r (−1 to +1; +1 = every participant higher with the heatmap);
  - mean difference with a 95 % percentile bootstrap confidence interval (10,000 resamples of participants, seed 42);
  - median difference, and the means per condition.
- **What we can detect:** with 16 participants the test has about 80 % power for a standardized effect
  d_z = 0.8 (mean difference / SD of the differences); d_z = 0.5 would need 36
  (`results/study_kits/power_needed.csv`; these are assumed effect sizes, not results).

## Participants and stopping

- **Target: 16 complete participants** (4 complete groups of 4). If there is time before the end of the study weeks,
  we may add more, but only in complete groups of 4, and the decision is made without looking at any outcome.
- **No interim analysis.** Nobody runs `analyze_study.py` or looks at Dice values of P sessions before the last
  session. The checks after each session (session guide) only look at whether the files exist.
- The analysis is run **once**, after the last session.

## Which data count

| Situation | Rule |
|-----------|------|
| Test runs, demos and the **pilot** | They use T IDs (T01, …). The analysis only reads folders named P + digits, so T IDs are never analyzed. Pilot data are deleted after the pilot (`scripts/clean_test_data.py`), and everything changed after the pilot is changed before the lock. |
| A P ID used for a test by mistake | Move the folder to `data/study/study_kits/excluded/<ID>_test/` before the analysis and note it below. |
| **Dropout** (a participant stops, or a session is incomplete for any other reason) | The participant is left out entirely: the paired comparison needs both conditions in full, and a mean over fewer scans would not be comparable. The session guide moves the folder to `excluded/<ID>_stopped/` (or deletes it if the participant withdraws consent) and gives the same ID to the next participant, so every group of 4 stays complete. If an incomplete session is still in `sessions/`, the analysis leaves it out by itself and lists it in `study_exclusions.csv`. |
| **The time ran out** on a scan | Kept. The gain is measured when the scan was locked, and the minutes are the logged time (the limit). The number of timeouts per condition is reported. |
| **The participant changed nothing** (the saved mask equals the AI's mask) | Kept, with a gain of 0 (so gain per minute 0). This is a real outcome: the participant did not find or fix an error. The number of unchanged scans per condition is reported. A mask made worse (negative gain) is kept too. |
| A scan restarted after a technical problem | Kept as saved (the attempt that was finished); noted in the session log and below. |
| Heatmap switched off by the participant in the "with" condition | Analyzed as assigned ("with"); the time the heatmap was visible is reported. |
| Extreme values (e.g. a large gain in a very short time) | Kept. No outlier removal and no exclusion based on performance. The Wilcoxon test uses ranks, which limits the influence of single extreme values. |
| A NASA-TLX questionnaire missing | The participant is left out of H2 only; n is reported. |
| Different heatmap displays in the logs | The analysis stops: every participant must see the locked display. The server refuses to resume a session with other settings. |
| A logged condition differs from the balancing plan, or real and simulated sessions are mixed | The analysis stops. Find the cause before going on. |

## Secondary and exploratory analyses

Everything below is **exploratory**. We report the numbers and, where a test is run, the unadjusted p-value, but
draw no conclusions from them. Anything not listed here that we look at after seeing the data is called
*post hoc* in the report.

- H2 if H1 is not supported (see above).
- Dice after, Dice gain, and minutes per scan: Wilcoxon as above (`study_tests.csv`, role `exploratory`).
- Per condition (`study_conditions.csv`): number of scans, timeouts and unchanged scans, median minutes, median share
  of the time the heatmap was on (with heatmap), and the median of each NASA-TLX scale and of the raw TLX.
- Order: the primary difference for participants who had the heatmap block first vs second, Mann–Whitney U test,
  two-sided (`study_order.csv`). A large difference would mean the heatmap's effect depends on the order.
- Per scan: Dice before and after for each scan and condition (`study_scans.csv`), e.g. to show which scans
  gained most from the heatmap.

## Output

`uv run python scripts/analyze_study.py --config configs/study.yaml` writes to `results/study_kits/`:

| File | Content |
|------|---------|
| `study_tests.csv` | H1, H2 and exploratory tests: n, means, mean difference with CI, median difference, statistic, p, p-method, r, conclusion, heatmap display |
| `study_participants.csv` | Per participant: order, means per condition and differences |
| `study_scans.csv` | Per saved scan: Dice before/after, minutes, gain per minute, timeout, changed voxels, included |
| `study_exclusions.csv` | Participants left out and why |
| `study_conditions.csv` | Descriptive numbers per condition |
| `study_order.csv` | Order check |
| `study_tlx.csv` | All NASA-TLX answers |
| `figures/study_gain_per_min.png` | One line per participant from without to with the heatmap |

## Changes after locking

Every change after the lock (to this plan, the code, the config or the data rules) is written here with the date,
time and reason, and reported in the paper.

| Date and time | What changed | Why | Who |
|---------------|--------------|-----|-----|
| | | | |
