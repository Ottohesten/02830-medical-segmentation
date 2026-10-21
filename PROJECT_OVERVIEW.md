# Project overview — Where to Look First

**Start here.** This file explains what the project is, where it stands, and where to find code, results and the
reasons behind every choice. Keep it up to date: when something changes, update this file and add a dated entry to
[PROJECT_LOG.md](PROJECT_LOG.md).

*Last updated: 2026-10-21 (robustness step 6 complete; report figure/summary ready; group presentation next).*

| | |
|---|---|
| Course | DTU 02830 Advanced Project in Digital Media Engineering, autumn 2026 |
| Group | 2 — Bjørn, Otto, Rasmus, Frederik |
| Supervisor | Kristine Aavild Sørensen, PostDoc, DTU Visual Computing (proposed the project) |
| Course responsible | Jakob Eg Larsen |
| Repository | github.com/Ottohesten/02830-medical-segmentation |

---

## 1. The project in plain words

An AI model can outline organs in CT scans in seconds, but it sometimes gets it wrong and does not say so. A human
therefore has to check its work, and human time is the bottleneck, not the model's accuracy.

Our idea: **sort the scans so the human checks the ones the AI most likely got wrong first**, and when a scan is
opened, **show where in the scan the AI is unsure**. The goal is the most improvement per minute of human work
("Dice per unit human effort"), not the highest possible Dice.

- **Organ:** kidneys (including tumors and cysts) in CT scans from the public dataset KiTS23.
- **Model:** TotalSegmentator (built on nnU-Net), used as released. We do not train it in G1/G2.
- **Dice:** overlap between the AI's outline and the correct outline (ground truth). 1 = perfect, 0 = no overlap.

## 2. Goals and status

| Goal | What | Status (2026-10-19) |
|---|---|---|
| **G1** (core) | Rank scans by likely failure, without ground truth | **Done.** Locked and tested once on 80 test scans |
| **G2** (core) | Review interface with an uncertainty heatmap + user study | Interface and analysis built and tested; **pilot and study not run yet** |
| **G3** (stretch) | Fine-tune the model on corrections | Plan and smoke test only. Now run in parallel by Frederik on an RTX 4090 |

### G1 — how the ranking works and how well it does

We run two versions of TotalSegmentator on every scan: a detailed 3 mm model (its outline is what the human corrects)
and a coarser 6 mm model. **The more they disagree about where the kidney is, the higher the scan goes in the queue.**
The usual trick (ask several copies of the same model) is impossible because only one copy ("fold 0") is published.

- Data: 100 random KiTS23 scans (seed 42). 20 = development set (all choices made here), 80 = test set (opened once,
  after all choices were locked on 2026-09-24).
- Result on the 80 test scans (review-AUC: how quickly the bad scans are found when working down the queue):
  **0.855**, vs. random order 0.506, sorting by organ size 0.488, and a perfect ranking 0.950 → about 79 % of the
  possible gain. Beats the organ-size baseline on all three locked measures.
- A simpler score (entropy) did better on test (0.894), but we kept the method locked in advance. Entropy is reported
  as a secondary finding.
- Limitation: errors the model is *confident* about (e.g. a kidney in the pelvis it never finds) cannot be caught by
  any uncertainty score.
- Exploratory follow-up (2026-10-19, after the lock; `scripts/compare_models.py`): the 6 mm model alone is a bit
  worse than the 3 mm model (test median Dice 0.894 vs. 0.910; below 0.83 in 16 vs. 11 of 80 scans), and 4 of the
  top 10 in the queue are false alarms caused by the 6 mm model being wrong (6 are real finds).
- Where the two models disagree: about 89 % of the disagreeing voxels lie within 6 mm of the 3 mm outline, i.e. the
  coarser grid. The remaining ~11 % are real differences, mostly tumor (79 % on test), and they concentrate in the
  bad scans and the false alarms. So the score mixes a large grid effect with a smaller signal about tumors.
- Exploratory scores from only the real differences (2026-10-20, `scripts/explore_disagreement_scores.py`; kits100
  used as development data for this new question). Counting only disagreement where the 6 mm model adds organ far
  from the 3 mm outline removes all 4 false alarms from the test top 10 (8 real finds instead of 6), but does not
  rank better overall (review-AUC 0.848). Leaving out only the grid band (`far_disagree_per_volume`) gave 0.887 on
  test, better than the locked measure (0.831 with this "bad" rule; paired CI above 0) and level with entropy
  (0.895). The rule fixed in advance picked a different score on the 20 dev scans (the control direction, 0.747 on
  test), so nothing here is a result yet: a chosen score must be tested once on new scans (the G3 pool).

- Robustness (Otto, 2026-10-21; own experiment, locked G1 unchanged): dev-selected lowres_8mm reduces test
  median 3 mm Dice from 0.90955 to 0.87865 (median paired loss 0.03010); Dice < 0.83 rises from 11 to 21/80.
  Primary score change versus Dice loss is inconclusive (rho 0.215, 95% CI [-0.036, 0.459]), while the
  mixed-queue review-AUC under that threshold is 0.8086 versus volume 0.4648 and random 0.5030.
  Entropy rises in 79/80 cases but remains secondary. This supports failure ranking under resolution loss,
  with a limitation in detecting the degradation itself; it does not establish general domain-shift robustness.
  Full dev/test, both-model, both-rule results: `results/kits_robustness_eval/`. The report figure (PDF/PNG),
  caption, and eight-line summary are ready in `results/kits_robustness_eval/report/`; group presentation is next.

### G2 — interface and user study

- Browser interface (NiiVue + small Python server), Chrome on a MacBook trackpad, English UI. The heatmap shows voxel
  entropy (chosen on the development set because it points best at wrong voxels). Known risk: it mostly lights up the
  edge of the outline.
- Exploratory simulation (2026-10-19, `scripts/simulate_voxel_correction.py`): correcting the 10 % most uncertain
  voxels by entropy raises Dice by 0.034 on average (test), slightly more than correcting by distance to the edge
  (0.032), twice as much as by |p_3mm − p_6mm| (0.017), four times random order (0.008); a perfect order gives 0.067.
  About 12 % (median scan) of the Dice gap lies outside the region the heatmap covers and can never be fixed this way.
- Study: each participant corrects 3 scans with and 3 without the heatmap (balanced order, 5 min per scan), then
  fills in NASA-TLX (workload) after each condition. Outcome: (Dice after − Dice before) / minutes, compared with a
  paired Wilcoxon test. At least 16 participants (power simulation); group members only as pilots.

### G3 — learning from corrections

Kristine's question (2026-10-06): are the scans with the biggest errors also the ones where correction helps future
training most? G3 is how we would answer it. The plan and smoke test show fine-tuning can start, but nothing has been
trained, so there is no result yet.

## 3. Where things are

| What | Where |
|---|---|
| **Dated log of everything tried, measured, decided or dropped** | [PROJECT_LOG.md](PROJECT_LOG.md) |
| How to install and run each part | [README.md](README.md) |
| Working rules (git, locked results, logging) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Configurations (one per environment, one per dataset) | `configs/`, `configs/datasets/` |
| Shared code | `src/segreview/` (inference, uncertainty, evaluation, heatmaps, plausibility, UI server, study analysis) |
| Runnable steps | `scripts/` (`run_g1.py` runs all of G1; `run_ui.py` starts the interface; `analyze_study.py` after the study) |
| Review interface | `ui/` (`app.js` flow, `viewer.js` slice viewer) |
| Tests | `tests/` (`uv run pytest`) |
| G1 results (locked) | `results/kits100/` — `g1_metrics_clean_test.csv` is the final G1 result; `report/` has figures and LaTeX tables |
| Exploratory follow-up (3 mm vs. 6 mm, simulated voxel correction) | `results/kits100/model_dice_3mm_vs_6mm*.csv`, `queue_top_3mm_vs_6mm.csv`, `model_disagreement_*.csv`, `voxel_correction_*.csv`, `figures/model_disagreement_where.png`, `figures/voxel_correction_gain.png` |
| Heatmap evaluation | `results/kits100/heatmap_eval_*.csv` |
| Power simulation | `results/study_kits/` |
| User-study documents | `docs/study/` — session guide, consent form, pilot plan, analysis plan (locked after the pilot) |
| G3 plan and smoke test | `docs/g3_plan.md`, `results/g3_smoke/` |
| Task documents (owner, steps, status) | `docs/tasks/` — e.g. [g1_robustness.md](docs/tasks/g1_robustness.md) (Otto; steps 1–6 complete; report figure/summary ready; group presentation next) |
| Robustness report material | [Figure PDF](results/kits_robustness_eval/report/fig_robustness_summary_test.pdf), [PNG](results/kits_robustness_eval/report/fig_robustness_summary_test.png), [caption](results/kits_robustness_eval/report/caption.md), [eight-line summary](results/kits_robustness_eval/report/summary.md) |
| HPC job scripts (never run on the cluster) | `jobs/` |
| Scans, model output, weights (not in git) | `data/`, `models/` — download with `scripts/download_data.py` and `download_model.py` |

## 4. Rules everyone should know

The full working rules (git, branches, locked results, new experiments, what to log) are in
[CONTRIBUTING.md](CONTRIBUTING.md). The most important ones:

- **Ground truth is only used for evaluation**, never in a score or shown in a study scan.
- **The test set is evaluated once.** Choices are made on the development set and locked first. Changes after
  opening the test set must be logged with date and reason.
- **The locked G1 results are reproduced from the saved predictions**, never recomputed with new inference
  (a different machine gives slightly different numbers).
- **Never invent or estimate results.** Every number in the log or report comes from code that was run; say which file.
- **Data and weights never go in git.** Each config writes to `results/<config name>/`, without dates in file names.
- Code, comments, docs and the report in English (American spelling).
- **Log as you go:** every result, experiment, idea and decision gets a dated entry in [PROJECT_LOG.md](PROJECT_LOG.md),
  also when it is dropped (with the reason).

## 5. Timeline

| Date | What |
|---|---|
| Week 8 (by 2026-10-28; course weeks skip the autumn break) | Pilot with a group member; choose heatmap display; lock the analysis plan |
| 2026-10-28 | Lecture on scientific writing |
| Weeks 9–10 (2026-11-02 – 11-13) | User study (≥ 16 participants) |
| 2026-11-25 | Final presentation |
| 2026-12-02 | Report hand-in (ACM two-column, max 5 pages for 4 people, plus contribution appendix and declaration of generative-AI use) |
| 2026-12-18 | Defense |

## 6. Next steps (2026-10-20)

1. Send Kristine the three follow-up results (6 mm alone, where 3 mm and 6 mm differ, simulated voxel correction;
   see the log of 2026-10-19) and decide with the group what goes into the report (as exploratory results).
2. Decide with the group which disagreement score (if any) to test once on the G3 pool: the pre-registered choice
   (`far_6mm_drops_ml`) or a different one, logged with date and reason before the pool is opened (log 2026-10-20).
3. Everyone installs the project and tries the demo (`README.md` → Quick start).
4. Pilot, choose the heatmap display, lock the analysis plan, recruit 16 participants.
5. G3 (Frederik): smoke test on the RTX 4090, then the pipeline with the 6 mm model.
6. Divide the report sections (the Trello board exists).

Open questions are listed at the bottom of [PROJECT_LOG.md](PROJECT_LOG.md).
