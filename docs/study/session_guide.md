# Session guide for the experimenter

How to run one session of the user study (G2): a participant corrects AI kidney outlines in 6 CT scans, 3 with the
uncertainty heatmap and 3 without, in the balanced order set by their participant ID. One session takes about
30–45 minutes (at most about 50).

## Rules that hold for every session

- **Participants:** students who have not seen the project. Group members never take part (they know the design);
  they only run pilots with test IDs (see [pilot_plan.md](pilot_plan.md)). Nobody takes part twice.
- **IDs:** real participants get `P01`, `P02`, ... in order, with no gaps. The number sets the order of scans and
  conditions, and the design needs complete groups of four (P01–P04, P05–P08, ...). Test runs always use `T01`,
  `T02`, ... Never use a P ID for a test.
- **Same setup for everyone:** the same MacBook, Google Chrome, the built-in trackpad, no external mouse or screen.
- **Say the same thing to everyone.** Read the script below. Do not tell participants that we expect the heatmap to
  help, and do not point at the screen or comment on their work.
- **No names in the data.** The only link between a person and their data is the ID slip they keep themselves.

## Before the study day

1. On the study MacBook, pull the latest version and run `uv sync`. The study must already be prepared on this
   Mac: `data/study/study_kits/` must exist. It is made by `scripts/prepare_study.py`, which needs the KiTS data and
   the model output, so prepare the study on the Mac that has them.
2. Check that `configs/study.yaml` has the settings agreed after the pilot (`time_limit_min`, `tlx: after_each_block`).
   **Do not change the config once P01 has started.**
3. Run `uv run pytest` (about two minutes). All tests should pass.
4. Print for each participant: two copies of the [consent form](consent_form.md). Print one
   [session log](#session-log) sheet for the day.

## Before each session

1. Plug in the charger. Turn on Do Not Disturb, close other apps, and turn off notifications.
2. Start the interface in a terminal:

   ```bash
   uv run python scripts/run_ui.py --config configs/study.yaml
   ```

   Chrome opens at http://127.0.0.1:8765. Leave the terminal open for the whole session.
3. In Chrome: full screen (⌃⌘F), zoom 100 % (⌘0). The start page should show the "Participant ID" field.
4. Look up the next unused P ID in the session log. Write it on the participant's ID slip (the bottom part of their
   copy of the consent form), not on the copy you keep.

## Script

Read the parts in quotes. Short pauses for questions are fine. If the pilot changed the time limit, say the new
limit instead of "5 minutes".

**1. Welcome and consent (about 3 min)**

> "Thanks for coming. This is a study for our course project at DTU. An AI has outlined the kidneys in CT scans,
> and it makes mistakes. Your job is to correct the outlines. You don't need any medical knowledge; the program
> explains what to look for. It takes about 40 minutes. Please read this form, and ask me if anything is unclear."

Give them both copies of the consent form. When they have signed, keep the signed copy (without the ID) and give
them their copy with the ID slip.

**2. The task (1 min)**

> "First there is a short guide and a practice scan that doesn't count. Then come 6 scans, with at most
> 5 minutes each. On some scans the program shows in blue where the AI is uncertain; on others it doesn't.
> Correct each outline as well as you can, and press Done when you think it is right. If the time runs out,
> your work so far is saved and you move on. After every three scans there is a short questionnaire about how
> demanding it was.
> I can help if something technical goes wrong, but not with the scans themselves. I'll sit over here."

Type the ID into the Participant ID field, check it against the session log, and press Start. Sit to the side,
out of their direct view, and stay quiet.

**3. During the session**

The interface leads the way: guide → practice scan (with heatmap) → the correct answer for the practice scan →
scans 1–3 → questionnaire → scans 4–6 → questionnaire → "Thank you!".

Allowed answers to questions:

| Question | Answer |
|----------|--------|
| How does a tool work? (painting, erasing, changing slice, moving the image) | Explain the tool. The guide screen has the same text. |
| Does the tumor/cyst count as kidney? | "Yes, tumors and cysts count as kidney." (It is on the screen.) |
| Is this right? / Where is the mistake? / Should I use the blue? | "Do what you think is right." |
| What do the colors in the practice answer mean? | Red = marked and kidney, yellow = kidney that was not marked, blue = marked but not kidney. |
| How much time is left? | Point them to the timer on the screen. |

Write anything unusual in the session log (questions asked, technical problems, interruptions).

**4. Debriefing (about 3 min)**

> "Thank you, you're done. What we are testing is whether showing where the AI is uncertain helps people correct
> its mistakes faster. Is there anything you would like to tell us about the program or the task?"

Write their comments in the session log (no names). Then:

> "Please don't tell other students about the details, since some of them may take part later. If you want your
> data deleted, send us the ID on your slip."

## After each session

1. Check the saved files in `data/study/study_kits/sessions/<ID>/`: 6 `*_mask.nii.gz`, 6 `*_log.json`,
   `tlx_with_heatmap.json`, `tlx_without_heatmap.json` and a `practice/` folder.
2. Fill in the row in the session log.
3. Copy `data/study/study_kits/sessions/<ID>/` to the group's backup location. **Never put study data in git.**
4. Leave the interface running for the next session, or stop it with Ctrl+C in the terminal.

## If something goes wrong

| Problem | What to do |
|---------|-----------|
| "Could not save" message | Click "Try again". If it fails again, check that the terminal with the server is still running. Do not close the page. |
| The page was closed or reloaded, or Chrome crashed | Restart the server if it stopped, open http://127.0.0.1:8765, and enter the **same ID**. Finished scans are skipped. A scan that was not finished starts again from the AI outline with the full time. Write in the log which scan was restarted. |
| Painting does not work | Check that Add or Erase is selected (not Move) and that the scan is not locked. Press on the trackpad and keep it pressed while moving. |
| The image is zoomed or moved away | Press "Reset view". |
| "Time is up" | Expected: the scan is saved and locked. The participant clicks Continue. |
| Wrong ID typed, and it belongs to an earlier participant | The program skips that person's finished scans. Reload the page and type the right ID. **Delete nothing.** |
| Wrong, unused ID typed, noticed before scan 1 is finished | Delete `data/study/study_kits/sessions/<wrong ID>/`, reload the page, and start again with the right ID (the participant sees the guide and practice again). |
| Wrong, unused ID, noticed later | Let the participant finish with that ID. Give the unused right ID to the next participant, so every group of four stays complete. Write it in the log. |
| The participant wants to stop | Stop at once, no questions asked. Ask if we may keep the data collected so far. If not, delete their folder. If yes, move it to `data/study/study_kits/excluded/<ID>_stopped/`, since the analysis needs both conditions. Either way, give the **same ID** to the next participant, so the groups of four stay complete. Write it in the log. |
| The participant asks to have their data deleted later | Delete `data/study/study_kits/sessions/<ID>/` (and its backup copy). Write the date in the log. |

## Cleaning up

- **After test runs** (T IDs and the demo queue), delete the test data. P IDs are never touched:

  ```bash
  uv run python scripts/clean_test_data.py --config configs/study.yaml --dry-run   # list what would go
  uv run python scripts/clean_test_data.py --config configs/study.yaml             # delete it
  ```

- **End of the study day:** stop the server (Ctrl+C), make sure every session is backed up, and keep the signed
  consent forms in an envelope, apart from the session log.
- **After the study:** `uv run python scripts/analyze_study.py --config configs/study.yaml`.

## Session log

One row per session. No names.

| ID | Date | Start | End | Experimenter | Restarted scans / problems | Comments from the participant |
|----|------|-------|-----|--------------|----------------------------|-------------------------------|
| P01 | | | | | | |
| P02 | | | | | | |
| P03 | | | | | | |
| P04 | | | | | | |
