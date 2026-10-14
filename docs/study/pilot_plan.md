# Pilot test plan

Before the first real participant, one group member goes through a full session as a pilot, run by another
group member with the [session guide](session_guide.md). The pilot's data are never analyzed. Everything we
change after the pilot must be changed **before P01** and written down with the date and the reason.

## Setup

- Same MacBook, Chrome and trackpad as in the real study; run it exactly like a real session, script included.
- Use a **test ID**: `T01` (scans without the heatmap first). If there is time for a second pilot, use `T03`
  (heatmap first), so both orders are tried. A T ID gets the same order as the P ID with the same number.
- The pilot knows the project and may have seen these scans before, so they will probably be **faster** than a
  real participant. Treat the pilot's times as a best case.

## What we want to learn

**1. Is 5 minutes per scan realistic? (main question)**

The time limit is `study.time_limit_min` in `configs/study.yaml`. The session budget leaves little room:
about 15 minutes for consent, guide, questionnaires and debriefing, plus the practice scan and 6 study scans at
up to 5 minutes each, gives about 50 minutes in the worst case. Our goal is about 45 minutes, so the limit cannot
go up.

We agree on these rules **before** the pilot:

| What we see | What we do |
|-------------|-----------|
| Most scans end with Done, or the pilot had stopped editing well before the time ran out | Keep 5 minutes. |
| The time runs out on 3 or more of the 6 scans while the pilot is still editing (strokes in the last 30 seconds) | The scans are too hard for the time. Real participants will be slower still. Choose scans with smaller errors (lower `selection.error_ml_max`, then run `scripts/prepare_study.py` again) instead of raising the limit. |
| All scans are done in under 2 minutes | Keep 5 minutes; the limit is only a safety net. Check that the errors are really visible (see question 4). |

**2. Does the whole flow work on the study laptop?** Guide → practice → answer → scans 1–3 → questionnaire →
scans 4–6 → questionnaire → "Thank you!". All files are saved (see "After each session" in the session guide).

**3. Are the instructions clear?** The guide screen, the task text, the color key, and our script. Where does the
pilot hesitate or ask?

**4. Are the scans suitable?** Can the error be seen and fixed in each scan, or is one scan much harder or easier
than the rest?

**5. How long does the session take**, from welcome to the end of the debriefing?

**6. How should the heatmap be shown?** The entropy heatmap points at wrong voxels only a little better than a map
that just marks the edge of the AI's outline (on the development set, the 10 % most uncertain voxels hold 43 % of
the wrong voxels near the kidney; the 10 % closest to the edge hold 41 %;
`results/kits100/heatmap_eval_dev_summary.csv`). Shown in full, it colors a wide band along the whole edge. So
`study.heatmap_display` in `configs/study.yaml` has three modes:

| Mode | What the participant sees |
|------|---------------------------|
| `full` (default) | The whole heatmap, as in the first user tests. |
| `top` | Only the `top_percent` (10 %) most uncertain voxels near the kidney. They lie right on the edge, so this shows *which parts* of the edge the AI is unsure about. |
| `contour` | A thin blue line `contour_margin_mm` (3 mm) around those voxels. The uncertain part of the edge itself stays uncovered. |

Know this before the pilot: the entropy is highest exactly where the AI's outline is (where its probability
crosses 0.5). So in `top` the blue lies almost only right on the red outline, and the red outline, which is always
drawn on top, hides much of it. In a test screenshot, `top` looked like a thin cyan line along the red outline,
plus a few uncertain lines inside the kidney; `contour` drew a blue line about 3 mm outside the red outline, on
the most uncertain slice all the way round. What `full` shows beyond the edge itself is how *wide* the uncertain
band is.

We agree on this **before** the pilot:

- The pilot session (T01) runs with `full`.
- After the debriefing, show the pilot 2 scans from the demo queue in each mode (the demo queue is not part of the
  study). To switch mode: stop the server (Ctrl+C), change `mode`, and start it again. The terminal prints the
  mode in use. Ask: *In which one could you find the mistake fastest? Did any of them hide what you needed to
  see?*
- A second pilot (T03, heatmap first) runs with the mode the first pilot preferred, if it was not `full`.
- **Rule:** keep `full`, unless the pilots find it distracting or covering the edge **and** prefer `top` or
  `contour`. Then take the one they prefer. Do not tune `top_percent` or `contour_margin_mm` on the pilots.
- Lock the mode before P01 and write the date and the reason in the group's notes. The server saves the setting
  in every session (`session.json`) and scan log, refuses to resume a session with another setting, and the
  analysis stops if participants saw different modes.

## What to note during the pilot

The program logs the time, strokes and tools for each scan by itself. The experimenter notes what the log cannot
show:

| Scan | Condition | Still editing when the time ran out? | Hesitation, questions, tool problems | Other |
|------|-----------|--------------------------------------|--------------------------------------|-------|
| Practice | with heatmap | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |

Also note the clock time at: welcome, start of the guide, start of the practice scan, start of scan 1, end of the
last questionnaire, end of debriefing.

## Questions for the pilot afterwards

1. Was the time per scan enough? On which scans did you run out of time, and what was left to fix?
2. Which scan was the hardest, and why?
3. Was the blue uncertainty map easy to understand? Did you use it, and did it help or distract you? (Then the
   comparison of the display modes under question 6.)
4. Was anything in the guide or the task text unclear?
5. Did the trackpad do what you expected (painting, moving, zooming, changing slice)?
6. Were the questionnaire questions clear?

## After the pilot

1. Print a summary of the pilot's scans, run from the project folder. The analysis only reads P IDs, so the
   script runs it on a temporary copy with the P ID of the same number:

   ```bash
   uv run python - T01 <<'EOF'
   import json, shutil, sys, tempfile
   from pathlib import Path
   from segreview.study import load_study_config, study_dir
   from segreview.study_analysis import scan_table

   study, source = load_study_config(Path("configs/study.yaml"))
   pilot = study_dir(study) / "sessions" / sys.argv[1]
   with tempfile.TemporaryDirectory() as tmp:
       shutil.copytree(pilot, Path(tmp) / ("P" + pilot.name[1:]))
       table = scan_table(study, source, Path(tmp)).sort_values("position")
   last = {}
   for log_path in pilot.glob("*_log.json"):
       strokes = [e["t"] for e in json.loads(log_path.read_text())["events"] if e["type"] == "stroke"]
       last[log_path.name.removesuffix("_log.json")] = strokes[-1] if strokes else None
   table["last_stroke_s"] = table.case_id.map(last)
   print(table[["position", "case_id", "condition", "reason", "duration_s", "strokes", "last_stroke_s",
                "dice_before", "dice_after", "gain_per_min"]].round(3).to_string(index=False))
   EOF
   ```

   `reason` is `done` or `timeout`. For a timeout, `last_stroke_s` close to `duration_s` means the pilot was still
   editing when the time ran out. `dice_after` below `dice_before` means the correction made the mask worse.

2. Go through the rules under questions 1 and 6 and decide on the time limit, the scans and the heatmap display.
   Fix anything that was unclear.
3. Copy the summary and your notes into the group's notes, together with the decisions and the date.
4. Delete the pilot data: `uv run python scripts/clean_test_data.py --config configs/study.yaml` (`--dry-run`
   first). P IDs are never touched.
5. If the config or the scans changed: run `uv run pytest`, and do a short test run with a T ID before P01.
6. Lock the [analysis plan](analysis_plan.md): fill in the "Lock" table (date and time, heatmap display, time limit,
   study scans) and commit it together with the code and `configs/study.yaml`. Then write that commit's hash in the
   table and commit again (a commit cannot contain its own hash). Only then start P01.
