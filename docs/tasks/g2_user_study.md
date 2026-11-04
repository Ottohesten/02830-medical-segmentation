# Task: Run the user study — does the heatmap help people correct faster?

| | |
|---|---|
| **Owner** | Rasmus |
| **Contact / lead** | Bjørn (lead; not part of running the study) |
| **Branch** | `rasmus/user-study` |
| **Created** | 2026-10-20 |
| **Status** | In progress (step 1) |
| **Way of working** | Independent: Rasmus runs the study and makes its decisions. The task is large enough to keep going without waiting for new instructions; questions to Bjørn or the group are welcome, and keep the group posted. |
| **Deadlines** | Pilot run by **2026-10-28** · analysis plan locked before the first participant · study in weeks 9–10 (**2026-11-02 – 11-13**) · analysis done by **2026-11-18** (before the final presentation 2026-11-25) |

This document is the task description **and** the working notes. Whoever works on the task updates it after every
step: tick the box, write the date and what was done, and update "Where we are" at the bottom. Then anyone in the
group, or a new chat, can see the status and continue.

---

## 1. The idea in plain words

G2 is the part of the project with real people. Participants correct the AI's kidney outline in our review
interface, 3 scans **with** the uncertainty heatmap and 3 **without**. We measure how much they improve the outline
per minute (Dice gain per minute) and how demanding it felt (NASA-TLX). **Question: does the heatmap help people
correct faster?**

Why it is a real open question: an analysis on 2026-10-19 showed that the heatmap mostly highlights the edge of the
outline, and that simply "looking at the edge" would work almost as well. It may still help people, or it may not.
Both answers are results for the report. See [PROJECT_LOG.md](../../PROJECT_LOG.md) (2026-10-19).

## 2. Starting points

Everything needed to run the study already exists. Do not rewrite it; use it and improve it where the pilot shows
a problem:

- How to start the interface, test ids (T01 …) and cleanup: [README.md](../../README.md) (Quick start, "Review interface").
- [docs/study/pilot_plan.md](../study/pilot_plan.md) — what the pilot must answer, what to note, and decision rules
  agreed before the pilot (heatmap display mode, time limit).
- [docs/study/session_guide.md](../study/session_guide.md) — how to run a session, the script to read, what to do if
  something goes wrong, the session log.
- [docs/study/consent_form.md](../study/consent_form.md) — still has `[contact e-mail]` and `[date]` to fill in.
- [docs/study/analysis_plan.md](../study/analysis_plan.md) — hypotheses and tests; its "Lock" table is filled in after
  the pilot and **before the first real participant** (P01).
- Analysis: `uv run python scripts/analyze_study.py --config configs/study.yaml` (run once, after the last session).
- General rules: [CONTRIBUTING.md](../../CONTRIBUTING.md). Participant data never goes in git.

Specific to this task:

- The study runs in **Google Chrome on a MacBook with a trackpad**. Use one Mac for all sessions if possible (same
  screen, same speed).
- **Group members can only be pilot participants**, never real participants (they know the design).
- **No peeking:** do not look at results before the last session; do not change settings after P01.
- Recruit at least **16 participants** (a multiple of 4), plus 2–3 spare. Rasmus leads recruiting and the sessions
  and asks the group for help when it is useful.

## 3. Steps

Tick each box when done and write the date and a short note under it.

- [ ] **1. Get started.** Install the project (`uv sync`) and run `uv run pytest`. Get the prepared study
  folder `data/study/study_kits/` (about 280 MB) from the group's OneDrive (folder "delte store filer", `study_kits.zip`) and unzip it so it lands in `data/study/study_kits/`.
  Do **not** run `prepare_study.py` yourself: it needs the locked kits100 model output, and everyone must use exactly
  the same prepared scans. Then try the demo queue and one test session (T01).
  Read the pilot plan and the session guide. Be able to explain G2 in two sentences.
  *Notes:* 2026-10-28: `uv sync` OK. `study_kits.zip` unzipped to `data/study/study_kits/` (282 MB, without the
  `__MACOSX` files): 6 study scans, 1 practice scan (case_00184), 10 queue scans, heatmap `entropy`. `uv run pytest`:
  159 passed with the kits in place (one earlier run had 6 errors that did not come back; probably flaky browser
  tests, watch for it). Server started with `--no-browser` and checked over its API: settings (`full`, 5 min), demo
  queue, T01 plan (scans 1–3 without, 4–6 with heatmap), heatmap refused (403) in the "without" condition and
  ground truth refused for all study scans. The hand click-through in Chrome is skipped on purpose: Rasmus is the
  pilot participant and has not seen the interface, so he must not see it before the pilot.
  G2 in two sentences: participants correct the AI's kidney outline on 6 scans, 3 with and 3 without the
  uncertainty heatmap, and we compare Dice gain per minute and NASA-TLX between the conditions. It tells us whether
  the heatmap helps people correct faster, which is open because the heatmap mostly marks the edge of the outline.

- [x] **2. Plan the pilot.** Agree on a pilot participant from the group who has not used the interface, a time and a
  place. Decide which heatmap display modes the pilot tries (see the pilot plan).
  *Notes:* 2026-11-03: Rasmus is the pilot (has not used the interface), on his own MacBook with Chrome and trackpad.
  No separate experimenter: Rasmus ran it alone, so there are no live observations, only the logs and the interview.
  T01 with `full`; the mode comparison on the demo queue afterwards.

- [x] **3. Run the pilot (by 2026-10-28).** Follow the pilot plan; take notes in its observation form; ask the
  interview questions. Use a T id. Clean up test data afterwards.
  *Notes:* 2026-11-03 (6 days late), T01, `full`, 5 min per scan. Whole session 16:38–16:52 (about 14 min, without
  consent and debriefing). Flow guide → practice → answer → 3 scans → TLX → 3 scans → TLX → end worked; all files saved.

  | # | Scan | Condition | End | Time (s) | Strokes | Dice before → after |
  |---|------|-----------|-----|---------:|--------:|---------------------|
  | 1 | case_00470 | without | done | 110 | 14 | 0.847 → 0.835 |
  | 2 | case_00259 | without | done | 46 | 0 | 0.897 → 0.897 |
  | 3 | case_00460 | without | done | 30 | 0 | 0.890 → 0.890 |
  | 4 | case_00554 | with | done | 44 | 0 | 0.892 → 0.892 |
  | 5 | case_00438 | with | done | 76 | 9 | 0.888 → 0.910 |
  | 6 | case_00426 | with | done | 95 | 13 | 0.723 → 0.783 |

  No timeouts. Only adding, never erasing; on scan 1 the added voxels were mostly outside the kidney.
  Interview: time was enough; hardest was the last scan (several "bubbles" on one tumor); the blue map made it
  easier; at the start unsure whether to mark kidney only or kidney and tumor, the task text ("including any
  tumor") answered it; the trackpad worked very well; TLX questions clear. The pilot believes he edited every scan
  and could see errors, but cannot remember which scans 2 and 4 were.
  Scans 2–4 have no stroke events at all and unchanged masks. To rule out a bug, a new browser test paints on all 6
  study scans across the scan changes and the questionnaire (`test_painting_is_saved_on_every_study_scan`): it
  passes, so most likely nothing was painted there, not lost.
  Mode comparison (2026-11-04, demo queue scans 1–2 `full`, 3–4 `top`, 5–6 `contour`): `full`: the blue together
  with the red made it hard to see the border between kidney/tumor and the tissue next to it (pilot turned the
  heatmap off and used High contrast). `top`: easier to see what lies behind; scan 4 was good, scan 3 hard to judge
  (outline looked imprecise, "not expert enough to correct it"). `contour`: "the best so far". In all modes the red
  fill (`mask_opacity` 0.45) hid the tissue under the mask. Tried 0.2 on the same scans: "MUCH better".
  T01 data and demo-queue edits deleted afterwards (`clean_test_data.py`).

- [ ] **4. Decide from the pilot.** Using the decision rules in the pilot plan, propose: heatmap display mode, time
  limit per scan, and any fixes to the interface or the session script. Write the proposal and the reasons here and in
  the log, and tell the group. Rasmus decides; ask the group if a choice is unclear.
  *Notes:* 2026-11-04, decided (see section 5): time limit 5 min, `heatmap_display.mode: contour`,
  `viewer.mask_opacity` 0.45 → 0.2. Four tests assumed `full` or the old opacity; they now test against the config
  (160 passed). Still open before step 5: (a) a clearer "kidney including any tumor" on the guide screen and in
  the script, (b) a second pilot T03 (heatmap first) with `contour`, if a group member who has not seen the
  interface is available, (c) are the errors on the study scans visible enough (section 6)?

- [ ] **5. Lock the analysis plan.** Fill in the "Lock" table in `docs/study/analysis_plan.md` (date, who, commit,
  display mode, time limit, study scans), commit it, and from now on change nothing in the setup.
  *Notes:*

- [ ] **6. Recruit and schedule.** Fill in the consent form placeholders, print forms, book rooms, make a schedule for
  weeks 9–10 that the group can sign up to run, and agree where session data are backed up (not git). Track:
  confirmed participants per person.
  *Notes:*

- [ ] **7. Run the study (weeks 9–10).** Sessions P01, P02, … in order. After each day: back up the session data and
  write the session log (no results, only whether the session was complete and any problems).
  *Notes:*

- [ ] **8. Analyze (after the last session, by 2026-11-18).** Run the analysis script once, exactly as the locked plan
  says. Write the main numbers and the figure path in the log and in section 4.
  *Notes:*

- [ ] **9. Present to the group** (5 minutes): what we did, what came out, what it means.
  *Notes:*

## 4. Results summary

*(Filled in at step 8.)*

## 5. Decisions made in this task

*(Date — decision — why. Also add each one to PROJECT_LOG.md.)*

- 2026-10-28 — Rasmus is the pilot participant (T01); another group member runs the session — Rasmus has not used
  the interface, and pilots must be group members who have not seen it. So he skipped the hand click-through in step 1.
- 2026-11-04 — Time limit stays 5 min — no scan hit the limit, all ended under 2 min (pilot plan rule).
- 2026-11-04 — Heatmap display `contour` — the pilot found `full` covering the border and preferred `contour`;
  the pilot plan's rule then takes the preferred mode. `top_percent` and `contour_margin_mm` unchanged (not tuned on
  pilots). One pilot only.
- 2026-11-04 — Mask fill `mask_opacity` 0.45 → 0.2, outline (2 px) unchanged — in all three modes the fill hid the
  tissue under the mask; 0.2 tried on the same scans and judged much better. Same in both conditions.

## 6. Questions for Bjørn or the group

*(Write questions here and tell Bjørn or the group.)*

- 2026-11-04 (group): In the pilot, 3 of 6 study scans (case_00259, case_00460, case_00554, Dice 0.89–0.90 before)
  were not edited at all, and the pilot found one queue scan too imprecise to judge as a non-expert. Are the errors
  in the study scans visible enough for lay participants? Changing scans needs a new shared `study_kits.zip`
  (`prepare_study.py` with the locked kits100 output): who would do it, and is it worth it given one pilot?
- 2026-11-04 (group): Is anyone who has not seen the interface available for a second pilot (T03)?

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action.
A new chat starts by reading this section.)*

- 2026-10-20: Task created. Next action: step 1.
- 2026-10-28: Step 1 done (click-through skipped, see notes). Next action: steps 2–3: Rasmus is the pilot
  participant, so another group member who knows the interface runs the session as experimenter (session guide,
  observation form) as T01 with `full` (deadline today).
- 2026-11-03: Pilot T01 run (steps 2–3). Half done: the mode comparison on the demo queue. Next action: mode
  comparison, then step 4 (proposal), then clean up T01. P01 cannot start before step 5 (lock).
- 2026-11-04: Step 3 done (incl. mode comparison), step 4 mostly decided: 5 min, `contour`, mask fill 0.2.
  Next action: clarify "including any tumor" on the guide screen; ask the group the two questions in section 6;
  then second pilot (if possible) and step 5 (lock). The study weeks have started (2026-11-02): P01 is delayed
  until the lock. Note: the pilot plan's rule for "too hard" scans says to rerun `prepare_study.py`, but
  that is no longer allowed locally (shared kits); if that rule fires, ask the group who rebuilds `study_kits.zip`.
