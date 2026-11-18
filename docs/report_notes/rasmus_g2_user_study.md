# Report notes: G2 user study and interface improvements

| | |
|---|---|
| **Owner** | Rasmus |
| **Branch** | `rasmus/user-study` |
| **Where the details are** | [docs/tasks/g2_user_study.md](../tasks/g2_user_study.md), [docs/study/](../study/), `results/study_kits/` |
| **Last updated** | 2026-11-18 (design, pilot and lock done; sessions running until 2026-11-23; results after the analysis on 2026-11-24) |

How to fill this in: [README.md](README.md). Notes, not report text. Every number with its interval and file path.

## 1. What was done

Question: does showing the model's uncertainty heatmap help people correct the AI's kidney outline faster?

- Data: 6 study scans and 1 practice scan from kits100 (`study_kits_v2`: case_00045, case_00406, case_00260 /
  case_00470, case_00546, case_00184; practice case_00236), chosen by a fixed rule on Dice and wrong volume
  (`configs/study.yaml`, `selection:`), cropped around the kidneys.
- Method: within-subject. Each participant corrects all 6 scans in our review interface, 3 with the heatmap and 3
  without, after a guide and a practice scan. Scans 1–3 / 4–6 form two blocks; which block gets the heatmap, and
  which condition comes first, is balanced over 4 participant types set by the ID number. 5 minutes per scan.
- Outcomes: Dice gain per minute ((Dice after − Dice before) / minutes, H1, primary), raw NASA-TLX after each block
  (H2, tested only if H1 holds), SUS once at the end (descriptive).
- Test: Wilcoxon signed-rank on the per-participant difference (with − without), two-sided, α = 0.05, exact
  p-value; rank-biserial r and mean difference with 95 % bootstrap CI always reported.
- Fixed beforehand: [docs/study/analysis_plan.md](../study/analysis_plan.md), locked 2026-11-18 11:35 in commit
  `17286c8`, before any participant. Changes after the lock (before P01, no data seen) are listed in the plan under
  "Changes after locking"; sessions run commit `42d404d`.
- Participants: lay people without medical imaging training who have not seen the project; no group members.

## 2. Results

*(After the analysis on 2026-11-24: `results/study_kits/study_tests.csv`, `study_conditions.csv`,
`study_sus_summary.csv`, `figures/study_gain_per_min.png`. Number of participants analyzed, H1, H2, SUS.)*

Pilot (T01, 2026-11-03, not analyzed, old scans `study_kits_v1`; source: task document, step 3):

- All 6 scans ended with Done, none hit the 5-minute limit; times 30–110 s per scan; session about 14 minutes.
- 3 of 6 scans were saved unchanged (0 strokes): case_00259, case_00460, case_00554.
- Scan 1 got slightly worse (Dice 0.847 → 0.835); scans 5 and 6 improved (0.888 → 0.910, 0.723 → 0.783).

## 3. What it means, and what it does not show

*(After 2026-11-24.)*

Already clear from the design and the pilot:

- The heatmap mostly marks the edge of the AI's outline: on the development set the 10 % most uncertain voxels hold
  43 % of the wrong voxels near the kidney, the 10 % closest to the edge hold 41 % (`results/kits100/heatmap_eval_dev_summary.csv`,
  `top_entropy`; pilot plan). So "does it help people" is a real open question, not a formality.
- The pilot is one person and says nothing about the effect; it only shaped the setup.

## 4. Limitations

- Few participants: sessions stop on 2026-11-23, original target was 16. With 8 the test has about 43 % power for
  d_z = 0.8 and 61 % for d_z = 1.0; with 4 it cannot reach p < 0.05 at all (smallest exact p = 0.125)
  (`results/study_kits/power_simulation.csv`). A non-significant result is weak evidence of no effect.
- Lay participants, not clinicians: errors that need anatomy knowledge may go unfixed in both conditions (the pilot
  found one demo scan "too imprecise to judge as a non-expert").
- Several computers and experimenters: each ID block of 4 runs on one computer and holds all 4 types, and every
  participant does both conditions on the same computer, so this adds noise, not bias. Computer, OS and
  trackpad/mouse are logged by hand.
- Only one pilot (a group member, who knew the project); no second pilot with the final scans and display.
- 6 scans, all from one dataset (KiTS23) and one model (TotalSegmentator).
- One round only: no tested version 2 of the interface.

## 5. Decisions and rejected alternatives

- 2026-11-04 — Heatmap display `contour` (thin line about 3 mm around the most uncertain voxels) instead of `full`
  (whole heatmap) or `top` (10 % most uncertain). Pilot: `full` together with the red mask hid the border between
  kidney/tumor and the tissue next to it; `contour` was "the best". Rule fixed before the pilot (pilot plan).
  `top_percent` and margin not tuned on the pilot.
- 2026-11-04 — Mask fill opacity 0.45 → 0.2 (outline unchanged): in all three displays the red fill hid what was
  under the mask; 0.2 judged "much better". Same in both conditions.
- 2026-11-04 — Time limit stays 5 minutes: no timeouts, all scans under 2 minutes (pilot plan rule).
- 2026-11-04 — Guide text: first sentence now says "including any tumor or cyst" (pilot was unsure at first).
- 2026-11-18 — New scans (`study_kits_v2`, `selection.dice_max` 0.90 → 0.88): the unchanged pilot scans had errors
  that were almost only a thin edge along the outline (a teammate's check against G1 numbers); a rule change before
  P01, not hand-picking.
- 2026-11-18 — One round instead of two (Build-Measure-Learn with a version 2): not possible before the
  presentation. Think-aloud dropped with round 2. SUS kept (cheap, standard).
- 2026-11-18 — Stopping rule: sessions until 2026-11-23, analyze only complete ID blocks (P01–P04, P05–P08, ...),
  so the balancing holds; independent of results.
- 2026-11-18 — Participants widened from "fellow students" to lay people without medical imaging training.
- 2026-11-18 — Sessions also on group members' own computers, IDs in blocks of 4 per experimenter; consent and SUS
  as Google Forms, nothing printed.

## 6. What went wrong, and surprises

- The first scan set was unsuitable: in the pilot 3 of 6 scans were not edited at all. Their errors were almost
  only a thin edge, which a lay person can neither see nor fix in 5 minutes, with or without the heatmap. That would
  have given about zero gain in both conditions on half the scans. Found by the pilot, fixed with a stricter rule.
- The pilot thought he had edited every scan; the logs showed no strokes on 3 scans. A browser test was added to
  rule out a saving bug (`tests/test_ui_browser.py::test_painting_is_saved_on_every_study_scan`, passes): nothing
  was lost. Lesson: trust the logs over memory.
- The red mask fill, not the heatmap, was the biggest visibility problem in the pilot.
- The study started later than planned (pilot 2026-11-03 instead of 2026-10-28), so the plan was adapted: fewer
  participants, one round, sessions on several computers.

## 7. Figures and tables for the report

- `results/study_kits/figures/study_gain_per_min.png` (after 2026-11-24): one line per participant from without to
  with the heatmap. **Pick this one if there is room for only one.**
- Table: H1/H2 from `results/study_kits/study_tests.csv` (n, mean difference with CI, r, p).
- Optional: a screenshot of the interface with the `contour` display and the red outline.
- Optional: the pilot table (task document, step 3) to show why the scans were changed.

## 8. Top 3 points for the report

1. *(The H1 result, after 2026-11-24: does the heatmap raise Dice gain per minute, with n, CI and p.)*
2. The pilot changed the study before any data: thin-edge errors were invisible to a lay person (scans replaced),
   and the mask fill and heatmap display were changed so people can see what they correct.
3. The analysis was locked before the first participant; every change after that is dated and explained, and the
   stopping rule depends only on the date and IDs.

## 9. Contribution

- Ran the user study task (owner): pilot participant (T01), display comparison, decisions in section 5.
- Locked the analysis plan; recruiting and running sessions P01–P04; Google Forms for consent and SUS.
- With Claude Code: SUS scoring and the stopping rule in `src/segreview/study_analysis.py` (with tests), the browser
  test for saving on every study scan, test updates for the new settings, session guide, consent form, `sus.md`.

## 10. Specific to this part

- Pilots: one pilot (T01, Rasmus, 2026-11-03/04). Showed: time is enough; old scans had invisible thin-edge errors;
  red fill too strong; `full` heatmap hides the border; tumor instruction unclear at first. Changed: scans, mask
  fill, display, guide text (section 5). A short test run (T02, 3 of 6 scans) with the final scans showed no problems.
- Participants: *(after 2026-11-23: number, how recruited, who ran which block, computers.)* Balancing: 4 types by
  ID number, analyzed only in complete blocks of 4.
- Feedback to changes: each change in section 5 came from a pilot observation; checked by the pilot looking at the
  same scans again (mask fill) or by comparing the three displays on the demo queue. Not checked with new
  participants (no round 2).
- What participants said and did: *(from the session logs and the SUS comments after 2026-11-23, anonymous.)*
  Usability problems with severity 0–4 are logged per session for the report's version-2 list.
