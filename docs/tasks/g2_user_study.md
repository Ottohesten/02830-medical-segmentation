# Task: Run the user study — does the heatmap help people correct faster?

| | |
|---|---|
| **Owner** | Rasmus |
| **Contact / lead** | Bjørn (lead; not part of running the study) |
| **Branch** | `rasmus/user-study` |
| **Created** | 2026-10-20 |
| **Status** | Not started |
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
  *Notes:*

- [ ] **2. Plan the pilot.** Agree on a pilot participant from the group who has not used the interface, a time and a
  place. Decide which heatmap display modes the pilot tries (see the pilot plan).
  *Notes:*

- [ ] **3. Run the pilot (by 2026-10-28).** Follow the pilot plan; take notes in its observation form; ask the
  interview questions. Use a T id. Clean up test data afterwards.
  *Notes:*

- [ ] **4. Decide from the pilot.** Using the decision rules in the pilot plan, propose: heatmap display mode, time
  limit per scan, and any fixes to the interface or the session script. Write the proposal and the reasons here and in
  the log, and tell the group. Rasmus decides; ask the group if a choice is unclear.
  *Notes:*

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

## 6. Questions for Bjørn or the group

*(Write questions here and tell Bjørn or the group.)*

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action.
A new chat starts by reading this section.)*

- 2026-10-20: Task created. Next action: step 1.
