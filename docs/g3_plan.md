# G3 plan: fine-tuning the model on corrected scans (stretch goal)

**Status:** plan and feasibility check only. Nothing has been trained, and the G3 pipeline is not built. G3 is only
started if G1 and G2 are finished (project plan: weeks 10–11, after the user study).

## Question

G1 ranks the scans so that a reviewer looks at the likely failures first, and G2 makes the correction itself faster.
G3 closes the loop: **does the segmentation model get better when it is fine-tuned on the scans that were
corrected, and does it matter which scans were corrected** (the top of the uncertainty queue or random scans)?

Fine-tuning means continuing the training of the pretrained network for a short time, with a low learning rate, on
new data. This is the only place in the project where the model is trained.

## Data

| Set | Scans | Used for |
|-----|-------|----------|
| **G3 pool** | KiTS23 scans that are **not** among the 100 of `kits100`: the next scans of the same seeded draw (seed 42, positions 101, 102, ...), e.g. the first 100 of them | the review queue, the simulated corrections and the training data |
| `kits100` development set | the 20 development scans | choosing the training settings (learning rate, number of epochs) |
| `kits100` test set | the 80 test scans | the final evaluation, **once**, after all choices are locked on the development set |

The pool and `kits100` share no scan; the G3 scripts must check this, as the smoke test already does. As far as we
know each KiTS23 case is a different patient, so no patient is in both; check this in the KiTS23 description before
the pool is fixed. The 2 scans of the smoke test (case_00183, case_00002, positions 101 and
102) are part of the pool; no model from the smoke test is kept.

## Method

1. **Queue.** Run the model on the pool (3 mm and 6 mm model, as in G1) and rank the scans by the locked G1 score
   `model_diff_mean_region` (no ground truth).
2. **Simulated review.** The top *k* scans of the queue are "corrected", and the ground truth plays the reviewer.
   The corrected segmentation is what a reviewer would hand back: the model's own prediction of all 117 structures,
   with the kidney replaced by the ground truth. KiTS kidney and tumor become the model's `kidney_left/right`,
   cysts `kidney_cyst_left/right` (split at the body's midline); kidney voxels the model predicted where the ground
   truth has none become background. Keeping the model's own labels for the other structures stops the training
   from teaching the model to forget them. Budgets *k* = 10, 20 and 40 scans give a learning curve.
   A more realistic reviewer (fixing only part of the wrong voxels, as measured in the user study) is an optional
   extra.
3. **Fine-tuning.** nnU-Net's own trainer, in the variant the model was trained with (`nnUNetTrainerNoMirroring`),
   with nnU-Net's data augmentation and loss (Dice + cross-entropy with deep supervision). Differences from a
   normal nnU-Net training:
   - start from **all** pretrained weights, including the last layer. nnU-Net's standard fine-tuning start
     (`-pretrained_weights`) re-initializes the last layer (`seg_layers`), which throws away the model's knowledge
     of all 117 structures (see the smoke test: the loss starts far higher);
   - learning rate 0.001 instead of 0.01, and far fewer epochs than the 4000 of the original training. The exact
     number of epochs and the learning rate are chosen on the `kits100` development set from a small grid fixed in
     advance (e.g. 10, 25, 50 epochs × learning rate 0.001, 0.0003);
   - foreground oversampling on the corrected kidney.
4. **Which model.** The main experiment fine-tunes the **3 mm model** (task 297), because it makes the masks that
   G1 and G2 use. The **6 mm model** (task 298) is cheaper (patch 64³ instead of 112 × 112 × 128) and is used
   first, to test the pipeline and the settings. Note that fine-tuning changes the G1 uncertainty score as well
   (it uses both models); G3 only evaluates Dice.

## Comparison

All fine-tuned models use the same budget *k*, settings and number of training steps.

| Model | Trained on | Question it answers |
|-------|------------|---------------------|
| A. Original | (not fine-tuned) | baseline |
| **B. Queue + correction** | top *k* of the queue, corrected | the G3 method |
| C. Random + correction | *k* random pool scans, corrected | is the queue a good choice of what to correct? |
| D. Queue, no correction | top *k* of the queue, with the model's own prediction as target | is the gain from the corrections, or only from more training on KiTS images? |

- **Primary:** Dice (organ definition `kidney_tumor_cyst`, as in G1) on the 80 test scans, B vs A, paired per scan:
  Wilcoxon signed-rank test on the per-scan Dice differences and the mean difference with a 95 % bootstrap
  interval. Also reported for the "bad" scans (Dice < 0.83 with the original model), where a gain matters most.
- **Secondary:** B vs C and B vs D, the learning curve over *k*, and the development set on its own.
- Training is random (patch sampling, augmentation). B and C are repeated with 3 seeds if the compute allows,
  and the spread is reported.
- **Forgetting:** KiTS has no ground truth for the other organs, so for each fine-tuned model we report the share of
  non-kidney voxels whose label changed compared with the original model on the test scans.
- **Label definition:** TotalSegmentator calls only part of a tumor "kidney" (median 81–84 % in our label check),
  while our ground truth counts the whole tumor. Fine-tuning will teach the model our definition, and part of any
  gain comes from that alone, not from fixing errors. To see how much, Dice is also reported with the tumor left out
  (`kidney_cyst_ignore_tumor`). None of the controls separates this effect; it is a limitation to report.

## Compute and what needs the HPC

Measured (smoke test on this Mac and the original training logs):

| What | Where | Measured |
|------|-------|----------|
| One training step, 6 mm model, batch 2 | this Mac (Intel i5-8210Y, 2 cores, 8 GB RAM), CPU | 21 s (19–23 s) |
| One training step, 6 mm model, batch 13 (the model's own batch size) | this Mac, CPU | 586 s (511–661 s); the Mac was swapping (8 of 9 GB swap in use), so this is slower per sample than batch 2 |
| Getting one batch ready (loading + nnU-Net's augmentation, 1 process) | this Mac | 1.7 s (batch 2), 11.8 s (batch 13), about 0.8–0.9 s per patch |
| Peak memory while training | this Mac | 2.6 GB (batch 2), 3.5 GB (batch 13) |
| One epoch of the original training (250 training + 50 validation steps; 6 mm batch 13, and 3 mm batch 2 with patch 112 × 112 × 128) | NVIDIA A100 (the TotalSegmentator authors' logs, `progress.png` in the model folders) | about 130–140 s, i.e. at most about 0.5 s per training step, for both models |
| Inference per scan (for the queue and the evaluation), CPU | this Mac (kits100 run) | 3 mm model 138 s, 6 mm model 24 s |

The GPU numbers are the original training's, on another machine; the GPU time on the DTU HPC is not measured yet.
The first HPC job should run the smoke test (`configs/g3_smoke.yaml` with `batch_sizes: [2, 13]` picks CUDA
by itself) to measure it.

What each step needs:

A fine-tuning run is counted in training steps: *epochs* × 250 steps. The grid above gives 2,500 (10 epochs),
6,250 (25) or 12,500 (50) steps.

| Step | This Mac (CPU) | GPU (estimate) | Where |
|------|----------------|----------------|-------|
| Download the pool (100 scans, about 9.6 GB like `kits100`) | fits (17 GB free), but tight | – | HPC (`/work3`), or the Mac |
| Queue: 3 mm + 6 mm inference on 100 pool scans | about 4.5 h (as `kits100`) | not measured | the Mac overnight, or HPC |
| Corrected targets | seconds per scan (10–20 s) | – | anywhere |
| Fine-tuning, **6 mm** model, one run | batch 13: 2,500 steps ≈ 17 days. Batch 2: 2,500 steps ≈ 16 h, but with 6.5 times fewer patches per step | A100: 2,500 steps ≈ 20 min, 12,500 ≈ 1.7 h. V100 perhaps twice that (not measured) | **HPC** |
| Fine-tuning, **3 mm** model, one run | not realistic: a 3 mm patch has 6 times the voxels of a 6 mm patch, so a step at batch 2 carries about as many voxels as a 6 mm step at batch 13, which took about 10 minutes here (not measured for the 3 mm model) | about the same as the 6 mm model (same epoch time in the original training) | **HPC** |
| Evaluation of one fine-tuned model on `kits100` (100 scans) | 6 mm: about 40 min; 3 mm: about 3.8 h | not measured | 6 mm stage on the Mac; 3 mm on **HPC** |

- **CPU cores per GPU job:** nnU-Net's data augmentation for 118 classes is heavy: about 0.8–0.9 s of CPU time per
  64³ patch on this Mac, so 12 s per batch of 13. For the GPU not to wait (about 0.5 s per step), the job needs
  many augmentation processes (`nnUNet_n_proc_DA`): at this Mac's speed about 24, and even if an HPC core is 2–3
  times faster, 8–12. The current job scripts ask for 4 cores (`jobs/`); G3 jobs should ask for 12–16.
- **Total:** B and C at 3 budgets, 3 seeds at *k* = 20, and D: about 11 fine-tuned 3 mm models. At 6,250 steps
  (25 epochs) that is roughly 11 × 1–2 h of training on a V100 plus the evaluation, i.e. one to two days of GPU
  time spread over jobs. The 6 mm stage (a few models, to test the pipeline and choose the settings) comes on top;
  per training step it costs about the same as the 3 mm model.
- **Disk:** the pool as above; a fine-tuned checkpoint is 165 MB (6 mm); the preprocessed data at 3–6 mm are small.

## Feasibility smoke test (30 September)

`uv run python scripts/g3_smoke_test.py --config configs/g3_smoke.yaml` (settings in `configs/g3_smoke.yaml`,
results in `results/g3_smoke/`). It prepares the 2 pool scans exactly as the pipeline would (steps 1–2 above),
starts nnU-Net's own trainer on them, checks the starting loss, and runs a few training steps per batch size. It
does not save a model, and nnU-Net's results folder points to `data/g3_smoke/`, so the pretrained weights are never
touched. On the HPC the same command measures the GPU (the config picks CUDA automatically).

Results (30 September, this Mac; `results/g3_smoke/{cases,sanity,timing}.csv`):

- **It starts.** nnU-Net's own trainer runs on this Mac with the pretrained 6 mm model and the 2 corrected scans:
  data loader with augmentation, loss with deep supervision, and training steps with a stable loss (−0.80 to −0.85).
- **The data line up.** The model's kidney against the ground truth on the 6 mm grid: Dice 0.930 and 0.905. The
  correction changes 267 and 522 voxels (of 176,400 and 281,600).
- **Start from all weights.** Loss on the same batch (batch 2): random weights 5.38; nnU-Net's standard fine-tuning
  start with the last layer re-initialized 7.86 (worse than random); all pretrained weights −0.77, close to the
  original training's final loss (about −0.87 in `progress.png`). So the corrected data match what the model
  expects, and G3 must keep the last layer.
- **Time and memory:** see the table above. A single step at the model's own batch size takes about 10 minutes
  here, so a real fine-tuning run is out of reach on this Mac: the 6 mm model would need weeks (batch 13) or days
  with a smaller batch, and the 3 mm model is not possible at all. The training belongs on a GPU.
- Problems found on the way (fixed in the script): nnU-Net 2.8's trainer needs a `continue_training` entry in the
  plans (its own start script adds it); labels must be passed as a signed type, because nnU-Net marks voxels outside
  the scan with −1; nnU-Net's `load_pretrained_weights` fails without a GPU (it loads without `map_location`); and
  the trainer takes its batch size from the plans when it initializes, so it must be changed there.

## If G3 goes ahead: what to build

1. `configs/g3_kits.yaml`: pool size, budgets *k*, seeds, the grid of training settings, model resolution.
2. `src/segreview/finetune.py`: the corrected targets (from the smoke test), export to nnU-Net's preprocessed
   format, a trainer run that starts from all pretrained weights, and unit tests of the target construction.
3. `scripts/g3_prepare.py` (queue + targets, needs the G1 inference on the pool), `scripts/g3_train.py` (one model
   per job), `scripts/g3_evaluate.py` (Dice on dev/test, paired tests), and an LSF job script in `jobs/`.
4. Lock the settings on the development set, then evaluate the test set once.
