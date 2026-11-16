# Report notes: G3 fine-tuning on corrected scans

| | |
|---|---|
| **Owner** | Frederik |
| **Branch** | `frederik/g3` |
| **Where the details are** | [docs/tasks/g3_finetuning.md](../tasks/g3_finetuning.md), [docs/g3_plan.md](../g3_plan.md), `results/g3_*/` (on the branch) |
| **Last updated** | 2026-11-16 (G3 done: phase 2 and task B complete; phase 3 dropped before any run) |

How to fill this in: [README.md](README.md). Notes, not report text. Every number with its interval and file path.

## 1. What was done

Question (Kristine, 2026-10-06 and 2026-10-21): when a reviewer's corrections are used to fine-tune the model, do the
scans the ranking puts first (most uncertain) help more than other scans?

- Data: 100 fresh KiTS23 scans for training (the "pool": positions 101–200 of the seed-42 draw, never in kits100);
  kits100's 20 development and 80 test scans for evaluation.
- Method: TotalSegmentator's 3 mm model fine-tuned on the first *k* = 10, 20, …, 100 pool scans, each point from the
  original weights. Simulated reviewer: the kidney label (with tumor and cyst) is replaced by the ground truth; all
  other structures keep the model's own prediction.
- Compared: the queue order (locked G1 score, most uncertain first) against three random orders (seeds 42–44), and a
  control trained on all 100 scans with the model's own prediction (no correction).
- Fixed beforehand ([docs/g3_plan.md](../g3_plan.md), 2026-10-27): design, primary summary, secondary analyses. The
  training setting was chosen on the development set only (2,500 steps, learning rate 0.001;
  `results/g3_kits/settings_choice.json`) and logged before any fine-tuned model predicted a test scan. The test set
  was used only to draw the curves.
- 40 runs, about 17 h on one RTX 4090 under Windows.

## 2. Results

On the 80 test scans, Dice of kidney + tumor + cyst unless stated. Intervals: 95 % paired bootstrap over the test
scans (2,000 resamples); they do not include training noise.

- **Primary (fixed beforehand):** queue − random, mean over *k* = 10–90: **+0.0019 [−0.0009, +0.0058]**
  (`results/g3_kits/summary.csv`).
- Curves, both orders (descriptive): 0.870 originally, 0.94 at *k* = 10 (queue 0.941, random mean 0.939),
  0.952–0.955 at *k* = 70–100 (`results/g3_kits/curves.csv`).
- Control without correction (secondary, fixed beforehand): 0.877 (`results/g3_kits/models.csv`).
- Training noise: the three random seeds spread 0.001–0.017 per *k* (`results/g3_kits/seed_spread.csv`). At
  *k* = 100 all orders train on the same 100 scans, and queue − random is still +0.0015 (`curves.csv`).
- Secondary, the 11 "bad" test scans (Dice < 0.83 originally; mean 0.608): queue ahead early, +0.071
  [+0.037, +0.114] at *k* = 10, mean over *k* +0.023 [+0.009, +0.048], gone from *k* = 70 (`summary.csv`; nine *k*
  values, not corrected for multiple comparisons).
- Secondary, tumor left out (`kidney_cyst_ignore_tumor`): mean over *k* −0.0010 [−0.0025, +0.0010]; at *k* = 10 the
  queue is behind, −0.0076 [−0.0165, −0.0009]; bad scans, mean over *k* +0.0068 [−0.0009, +0.0185], at *k* = 10
  +0.0164 [−0.0029, +0.0447] (`summary.csv`).
- Secondary, development set (20 scans): +0.0019 [−0.0017, +0.0067] (`summary.csv`).
- Forgetting (secondary): every fine-tuned model relabels 1.7–2.2 % of the non-kidney voxels compared with the
  original model; the control 1.6 % (`models.csv`).
- Task B (exploratory, after the locked test on the pool): the queue's first 10 pool scans have a median tumor of
  527 ml against 20–55 ml in the random orders, and their corrections change a median of 16,242 voxels against
  2,368–3,352 (`results/g3_kits/queue_content.csv`). The 11 bad test scans have a median tumor of 596 ml against 30 ml
  for the other 69 (`results/g3_kits/test_tumor_sizes.csv`).

## 3. What it means, and what it does not show

- Correcting is what helps: 10 corrected scans give over 80 % of the gain at *k* = 100, and the same kind of training
  without correction gives almost nothing (0.877 against 0.870).
- The order of correction did not matter measurably for fine-tuning: the queue's advantage is no larger than the
  training noise (rule fixed beforehand: a difference smaller than the seeds' spread is not read as an effect). This
  says nothing against the queue for reviewers: finding the bad scans is G1's question, and Otto's locked test on
  the pool (2026-11-04).
- The queue's early lead on the bad test scans is most likely the model learning our tumor definition faster (a
  bias foreseen in docs/g3_plan.md): the most uncertain pool scans are the ones with very large tumors, the bad test
  scans are too, and with the tumor left out the lead shrinks: at *k* = 10 from +0.071 to +0.016 [−0.003, +0.045],
  averaged over *k* from +0.023 to +0.007 (on all scans the queue is even behind at *k* = 10). Task B is
  descriptive: it cannot separate the definition from other features of large-tumor scans.
- Not shown: real reviewers (the simulated one is perfect and instant), other organs, datasets or models.
- Not shown either: whether the order matters for the first few corrected scans (*k* < 10), where most of the gain
  is. Phase 3 would have tested this and was dropped (section 5).

## 4. Limitations

- The queue has one training seed, the random orders three; the intervals cover the test scans only. So the queue's
  lead on the bad test scans at *k* = 10 is not checked against training noise (phase 3 would have added two queue
  seeds; dropped).
- No points below *k* = 10, where most of the gain happens (phase 3 would have added *k* = 1, 2, 3, 5; dropped).
- Perfect simulated corrections (the ground truth); real corrections are slower and noisier.
- One setting (chosen at *k* = 50) for every *k*; small *k* means many passes over few scans.
- 80 test scans, of which 11 bad; one dataset, one model, one machine. The Windows torch build has another cuDNN than
  `uv.lock`, so the original model was evaluated again on this PC and nothing is compared with the locked kits100
  numbers from the Mac.
- Forgetting is change, not error: KiTS has no ground truth for other organs.

## 5. Decisions and rejected alternatives

- 2026-10-21/27: a learning curve (queue vs random order) instead of one fine-tuned model (Kristine's advice); every
  point trained from the original weights, so the points are independent.
- 2026-10-28: settings chosen on the development set at *k* = 50 in both orders; within 0.001 Dice the fewer steps
  win (2,500 steps 0.9372 against 6,250 steps 0.9374; `results/g3_kits/settings_grid.csv`). The final checkpoint is
  evaluated, not nnU-Net's "best", which is chosen on the training scans.
- 2026-10-27: the 3 mm model for the curves; the 6 mm model only for a pipeline test.
- 2026-10-20: Windows instead of Linux/WSL on Frederik's PC (owner's choice): a separate environment with the package
  versions of `uv.lock`; `uv.lock` and `pyproject.toml` unchanged.
- 2026-11-16: phase 3 (*k* = 1, 2, 3, 5 and two more training seeds for the queue, about 14 h on the GPU) was designed,
  then dropped before any run. It was optional for the group, and the owner chose not to train more. Hence the two
  limitations in section 4: no points below *k* = 10, and one queue seed.

## 6. What went wrong, and surprises

- Windows Smart App Control blocked torch (2026-10-27); the owner turned it off.
- The PC was switched off twice during runs (power button, 2026-10-27 and 2026-10-28). Since then runs save every
  epoch and continue where they stopped (`frederik/g3_start.ps1`, `frederik/g3_stop.ps1`); nothing was lost.
- The PC's RAM (34 GB), not the GPU, was the limit: with 16 data workers and other programs open it was full.
- Otto's integrity checks failed on Windows only because git checked text files out with CRLF line endings.
- Surprise: over 80 % of the gain comes before *k* = 10, so the planned curve has little resolution where the order
  could matter.
- On the development set the queue was ahead in all four grid settings at *k* = 50; on the test set this did not
  hold up.

## 7. Figures and tables for the report

- **First choice:** `results/g3_kits/figures/learning_curve_test.png` (PDF next to it): test Dice against *k*, queue
  against random (each seed thin, mean dashed), all test scans and the 11 bad ones; the primary summary in the title.
- Table: the per-*k* table in PROJECT_LOG.md, 2026-11-02, or `results/g3_kits/summary.csv`.
- Table for the label-definition point: `results/g3_kits/queue_content.csv` (tumor size and correction size of the
  first 10, 20, 30 scans of each order; a short version is in PROJECT_LOG.md, 2026-11-16).

## 8. Top 3 points for the report

1. Ten corrected scans raise test Dice from 0.870 to about 0.94; training on the model's own predictions instead
   gives 0.877. The corrections give the gain.
2. Correcting in the queue's order (most uncertain first) was not clearly better for fine-tuning than random order:
   +0.0019 [−0.0009, +0.0058], about the size of the training noise.
3. On the 11 worst test scans the queue helps early (+0.07 at *k* = 10), but hardly with the tumor left out (+0.016,
   interval includes 0): the queue starts with the large-tumor scans (median tumor 527 ml against 20–55 ml), so the
   model learns our tumor-counts-as-kidney definition faster, rather than fixing errors better.

## 9. Contribution

- Frederik: the Windows/GPU setup (`frederik/`); the G3 pipeline (`scripts/g3_*.py`, `src/segreview/finetune.py`,
  `src/segreview/g3_analysis.py`, their tests); the settings grid and the 40 curve runs (about 21 h on the GPU); the
  analysis; the task document and log entries; the pool's model output for the group's locked test.
- Made with an AI coding assistant (Claude Code), directed and checked by Frederik.

## 10. Specific to this part

- **Label definition vs fixing errors:** with the tumor counted, Dice rises from 0.870 to 0.953 (random mean,
  *k* = 100); with the tumor left out from 0.893 to 0.945; the control without correction reaches 0.877 and 0.901
  (`curves.csv`, `dice/original.csv`, `dice/nocorr_k100_S2500_L0.001.csv`). So most of the gain remains without the
  tumor; the definition explains part of it, not all (exploratory; the two Dice definitions are not on one scale, so
  no percentage). The order effect, though, is about the definition: the queue front-loads the large-tumor scans
  (task B, section 2).
- **Phase 3 (designed and dropped 2026-11-16, before any run):** *k* = 1, 2, 3, 5 and three training seeds for the
  queue, with an interval that includes training noise (docs/g3_plan.md, "Phase 3"). Nothing was run, so there are
  no results; the design stays in the plan as a record.
- **Training noise vs order:** the random seeds spread 0.001–0.017 per *k*; queue − random is +0.0019 on average and
  +0.0015 at *k* = 100, where all orders use the same scans.
- **One Windows PC with one GPU:** one run at a time (40 runs in about 17 h); RAM was the limit, not GPU memory (at
  most 17.3 of 24 GB); runs that save every epoch and continue after a stop; models compared only within this PC.
- **Forgetting:** 1.7–2.2 % of non-kidney voxels relabeled (the control 1.6 %): fine-tuning itself changes other
  structures a little, with or without corrections. Whether these changes are errors cannot be checked.
