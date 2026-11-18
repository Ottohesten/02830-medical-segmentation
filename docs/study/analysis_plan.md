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
| Locked on (date and time) | 2026-11-18 11:35 |
| Locked by | Rasmus |
| Git commit with the locked plan, code and `configs/study.yaml` | `17286c8e3953` |
| Heatmap display (`study.heatmap_display`, chosen in the pilot) | `contour` (`top_percent` 10, `contour_margin_mm` 3) |
| Time limit per scan (`study.time_limit_min`) | 5 minutes |
| Study scans, block A / block B (`manifest.json`) | `study_kits_v2`: case_00045, case_00406, case_00260 / case_00470, case_00546, case_00184 (practice: case_00236) |
| Last day of sessions (stopping rule) | 2026-11-23 |

## Design in short

- Within-subject design: every participant corrects the AI's kidney outline in 6 CT scans, 3 **with** the
  uncertainty heatmap and 3 **without** it, after one practice scan (with the heatmap) that is not analyzed.
- Balancing: scans 1–3 form block A and 4–6 block B. Half of the participants get the heatmap on block A, half on
  block B, and the order of the conditions is swapped too, giving 4 participant types (`src/segreview/study.py`).
  The number in the participant ID sets the type, so we recruit in groups of 4.
- Time limit per scan (see Lock). When the time is up, the scan is locked and saved as it is.
- Raw NASA-TLX after each condition block (`study.tlx: after_each_block`).
- Participants are lay people without medical imaging training (e.g. students, friends, family) who have not seen
  the project or the interface; no clinicians or radiographers. Group members never take part as participants.

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
  (`results/study_kits/power_needed.csv`; these are assumed effect sizes, not results). With 8 participants the
  power is about 43 % for d_z = 0.8 and 61 % for d_z = 1.0 (`results/study_kits/power_simulation.csv`), so a result
  that is not significant is weak evidence that the heatmap does not help.

## Participants and stopping

- **Stopping rule (set at the lock):** sessions run until and including the last day in the Lock table. Then the
  analysis uses only **complete ID blocks** (P01–P04, P05–P08, ...: all four participants of the block complete;
  `analysis.complete_groups_only`). Each block holds the 4 participant types once, so the conditions stay balanced.
  Complete participants in an incomplete block are listed in `study_exclusions.csv`. The rule depends only on the
  date and the IDs, never on results. The original target was 16 participants.
- **Several experimenters and computers:** each group member who runs sessions gets whole ID blocks (Rasmus
  P01–P04; the next blocks in the order members ask), so no ID is used twice. Sessions run on the members' own
  computers, all in Google Chrome with the same commit and `study_kits_v2`. The program logs the browser and window
  size; the experimenter writes the computer, operating system and input device (trackpad or mouse) in the session
  log. Because every block runs on one computer and holds all 4 types, and every participant does both conditions
  on the same computer, computer differences do not favor either condition; they add noise. Reported as a
  limitation. After the stopping day, all session folders are copied into one `sessions/` folder for the analysis.
- **At least 8 are needed for a test.** With 4 participants the exact two-sided Wilcoxon test cannot reach
  p < 0.05 (its smallest possible p-value is 2/16 = 0.125); then all numbers are reported descriptively only.
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
- Usability: SUS score (Brooke 1996, [sus.md](sus.md)), asked once at the end of each session, so not split by
  condition. Mean with 95 % percentile bootstrap interval (seed 42) and median over the analyzed participants
  (`study_sus.csv`, `study_sus_summary.csv`).

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
| `study_sus.csv`, `study_sus_summary.csv` | SUS answers and score per analyzed participant; mean, interval, median |
| `figures/study_gain_per_min.png` | One line per participant from without to with the heatmap |

## Changes after locking

Every change after the lock (to this plan, the code, the config or the data rules) is written here with the date,
time and reason, and reported in the paper.

| Date and time | What changed | Why | Who |
|---------------|--------------|-----|-----|
| 2026-11-18, before P01 (no data yet) | Stopping rule: complete ID blocks instead of "the largest multiple of 4 in ID order"; IDs given out in blocks of 4 per experimenter; sessions on several computers, with computer, OS and input device in the session log; session folders collected in the group's OneDrive. Code: `keep_complete_groups`. | The group decided to run sessions at home on their own computers, which the lock did not foresee. With blocks per experimenter, "the first 4 in ID order" could mix blocks and break the balancing. | Rasmus |
| 2026-11-18, after P01 | P01 ran remotely (link to the experimenter's server) in Firefox on Windows 11 with a mouse, not in Chrome. Kept and analyzed as the other participants; the browser is reported. Remote sessions are allowed from now on, but the experimenter checks that the participant uses Google Chrome before the ID is entered. | Decided before anyone looked at P01's results. Firefox is not supported by the interface (painting may not work), so P01's data may understate what he corrected. | Rasmus |
| 2026-11-18, before P01 | **Version for all sessions:** branch `rasmus/user-study`, commit `42d404d165b8`, with `study_kits_v2`. Commits after it change only documents, never code or config. | One version on every computer. | Rasmus |
| 2026-11-18, before P01 | Consent form, SUS and session log online (Google Forms, a file on the Mac) instead of on paper; the participant ID is told, not written on a slip. | Practical: nothing is to be printed. Only the session logistics change; the analysis, data rules, code and config are as locked. | Rasmus |
