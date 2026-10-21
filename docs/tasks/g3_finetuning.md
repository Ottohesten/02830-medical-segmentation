# Task: G3 — does fine-tuning on corrected scans help, and does it matter which scans are corrected?

| | |
|---|---|
| **Owner** | Frederik |
| **Contact / lead** | Bjørn |
| **Branch** | `frederik/g3` |
| **Created** | 2026-10-20 |
| **Status** | Phase 1 (setup and GPU smoke test) done 2026-10-21; phase 2 waits for the go-ahead |
| **Machine** | Frederik's PC: NVIDIA RTX 4090 (24 GB), Intel i9-14900K (24 cores, 32 threads), 32 GB RAM, Windows 11 Home; the project runs natively on Windows (section 3) |

This document is the task description **and** the working notes. Whoever works on the task (with or without an
assistant) updates it after every step: tick the box, write the date and what was done, and update
"Where we are" at the bottom. Then anyone in the group, or a new chat, can see the status and continue.

---

## 1. The question in plain words

Kristine's question (2026-10-06, meeting 2026-10-07): are the scans with the biggest errors also the ones where a
human correction helps future training most? The most uncertain scan is not necessarily the most useful one to
learn from: it may be an outlier that is unique to one patient.

G3 answers it with a simulated experiment: the ground truth plays the reviewer, the model is fine-tuned on the
corrected scans, and we compare correcting the top of our uncertainty queue with correcting random scans (and with
no correction). The full plan, data and comparison are in [docs/g3_plan.md](../g3_plan.md).

## 2. Rules for this task

The general rules are in [CONTRIBUTING.md](../../CONTRIBUTING.md). Specific to this task:

- **Work in phases.** After each phase: a short summary (what was done, numbers, problems, changed files, a
  proposed commit message), then wait for Frederik's go-ahead before the next phase. Nothing is trained without it.
- **Everything runs in the Windows environment `.venv`** (section 3), started with `.venv\Scripts\python.exe`;
  `uv sync` and `uv run` do not work on Windows because `uv.lock` covers macOS and Linux only.
- **torch 2.2.2 with CUDA 12.1 and `uv.lock` are never changed.** If torch 2.2.2 cannot use the GPU (e.g. an RTX 50
  series card, Blackwell), stop and tell the group. The RTX 4090 (compute capability 8.9) is supported.
- **`torch.compile` is not used** (torch 2.2 does not support it on Python 3.12, and nnU-Net turns it off on
  Windows). Every G3 run sets `nnUNet_compile=false` anyway, as the smoke test does, so a run behaves the same on the HPC.
- **Never touch `results/kits100/`** (locked G1 results) or `data/predictions/kits100/`. G3 has its own configs and
  folders: `results/g3_*/` (small files, in git), `data/` and `models/` (big files, not in git; shared via the
  group's OneDrive; the log says where).
- **Seed 42** for everything random; repeated runs list their seeds in the config.
- **Settings are chosen on the 20 development scans** of kits100, from a grid written in the config **before** the
  runs. The 80 test scans are evaluated once, in phase 3, after everything is locked.
- **Log everything** in [PROJECT_LOG.md](../../PROJECT_LOG.md) (dated, tagged, with numbers and file paths), so new
  questions can be answered without training again (CONTRIBUTING.md, section 4).

## 3. Setup on Frederik's PC (Windows)

The Windows-only tools are in [frederik/](../../frederik/README.md).
`uv.lock` covers only macOS and Linux, so the usual `uv sync` cannot build the environment on Windows. Instead,
`frederik/windows_setup.ps1` builds a separate environment in `.venv` with **the same package versions as
`uv.lock`**, without changing `uv.lock` or `pyproject.toml`:

- every version in `uv.lock` is used as a constraint, and uv resolves the project's dependencies for Windows;
  packages that only Windows needs (e.g. `colorama`) are not in `uv.lock` and get their newest compatible version;
- torch and torchvision come from PyTorch's own server as CUDA 12.1 builds (`torch 2.2.2+cu121`,
  `torchvision 0.17.2+cu121`): the versions of `uv.lock`, but with CUDA (the Windows torch on PyPI has none);
- `psutil` is added (Windows only) to measure the memory of nnU-Net's worker processes.

The script then checks that torch can use the GPU (one small 3D convolution; no training), runs the tests, and
downloads the TotalSegmentator weights and the 2 smoke-test scans. The installed versions are saved in
`results/g3_smoke_gpu/windows_env.txt`, the checks in `results/g3_smoke_gpu/setup_check.txt`.

```powershell
powershell -ExecutionPolicy Bypass -File frederik\windows_setup.ps1    # once (and again after uv.lock changes)
.venv\Scripts\python.exe -m pytest                                     # tests
.venv\Scripts\python.exe scripts\<script>.py --config configs\<config>.yaml
```

- **Git for Windows** must be on PATH, so every run records its git commit in `machine.json` (installed
  2026-10-20 with `winget install --id Git.Git -e`). Commit and push as usual, e.g. with GitHub Desktop.
- **VS Code** finds `.venv` by itself (Python: Select Interpreter → `.venv`).
- **Differences from Linux/macOS to keep in mind:** nnU-Net starts its augmentation workers as new processes
  (spawn) instead of copies (fork), so each run starts more slowly; RAM is measured as the private memory of all
  processes instead of PSS (`src/segreview/memory.py`); GPU numbers can differ slightly between machines and
  operating systems, so `machine.json` records the machine for every run (CONTRIBUTING.md, section 3).

## 4. Steps

Tick each box when done and write the date and a short note (numbers + file paths) under it.

### Phase 1 — setup and smoke test on the GPU

- [x] **1. Machine check.** GPU model and memory, OS, free disk, CPU cores, RAM.
  *Notes:* 2026-10-20: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9 (Ada, not Blackwell), driver
  617.14; Intel Core i9-14900K, 24 cores / 32 threads; 34.1 GB RAM (31.7 GiB); Windows 11 Home 10.0.26200; free
  disk C: 1,035 GB, D: 2,048 GB, E: and F: 4,000 GB each. `results/g3_smoke_gpu/machine_windows.txt`.
- [x] **2. Environment and tests** — `frederik/windows_setup.ps1` (browser tests may be skipped without Chrome; say so).
  *Notes:* 2026-10-20: `.venv` with 123 packages; none differs from the 132 versions pinned by `uv.lock`; torch
  2.2.2+cu121 and torchvision 0.17.2+cu121; Windows-only: colorama 0.4.6, psutil 7.2.2, tzdata 2026.5
  (`results/g3_smoke_gpu/windows_env.txt`). GPU check: torch sees the RTX 4090 (compute capability 8.9, 24.0 GiB,
  CUDA 12.1, cuDNN 8.8.1); one float16 3D convolution runs. Tests: 87 passed, 37 skipped: 30 need a prepared user
  study, 6 browser tests do not find Chrome (the test helper only looks where Chrome is on macOS/Linux), 1 test is
  Linux-only (`results/g3_smoke_gpu/setup_check.txt`). Git for Windows 2.55.0 installed.
- [x] **3. Weights and the 2 smoke-test scans** — `frederik/windows_setup.ps1` (`download_model.py`, and
  `download_data.py --cases case_00183 case_00002`).
  *Notes:* 2026-10-20: TotalSegmentator weights for task 297 (3 mm) and 298 (6 mm), fold 0 each (`models/`,
  0.33 GB); case_00183 and case_00002 with the full KiTS case list (`data/raw/kits23/`, 0.20 GB). The case list
  reproduces the 100 kits100 scans exactly (100 of 100 in `results/kits100/dice.csv`), and the 2 scans are not
  among them.
- [x] **4a. Configs ready.** `configs/g3_smoke_gpu.yaml` (6 mm, batch 2 and 13) and `configs/g3_smoke_gpu_3mm.yaml`
  (3 mm, batch 2): CUDA forced, torch threads 1, augmentation workers 0 and 16, 30 steps (10 warm-up), 10
  validation steps.
  *Notes:* 2026-10-20. The smoke test was extended for the GPU and for Windows (see section 6).
- [x] **4b. Run the smoke tests** (only after the go-ahead; close games and other GPU programs first):

  ```powershell
  .venv\Scripts\python.exe scripts\g3_smoke_test.py --config configs\g3_smoke_gpu.yaml       # -> results/g3_smoke_gpu/
  .venv\Scripts\python.exe scripts\g3_smoke_test.py --config configs\g3_smoke_gpu_3mm.yaml   # -> results/g3_smoke_gpu_3mm/
  ```

  *Notes:* 2026-10-21 (go-ahead from Frederik): both run, about 2.5 min each. Two earlier runs that night were
  replaced because the RAM measurement was wrong (commit instead of RAM; then a measurement that slowed every 3rd
  step); fixed in `src/segreview/memory.py` (log 2026-10-21).
- [x] **5. Report and estimates:** time per training step and per batch preparation, GPU memory and RAM, per model
  and batch size (`summary.csv`); estimate one fine-tuning run of 2,500 and 6,250 steps (6 mm and 3 mm), the whole
  plan (about 11 fine-tuned 3 mm models + evaluation), and the disk space for the pool (100 scans) and kits100.
  *Notes:* 2026-10-21: `scripts/g3_smoke_estimate.py` → `results/g3_smoke_gpu/{estimates,disk,inference_times_gpu}.csv`.
  Numbers in section 5.
- [x] **6. Results and log:** results in `results/g3_smoke_gpu/` and `results/g3_smoke_gpu_3mm/` (with
  `machine.json`), dated `[experiment]`/`[result]` entries in PROJECT_LOG.md. Stop and report.
  *Notes:* 2026-10-21: done (PROJECT_LOG.md, 2026-10-21). Phase 1 finished.

### Phase 2 — G3 pipeline with the 6 mm model (only after the go-ahead)

- [ ] Build what [docs/g3_plan.md](../g3_plan.md) lists under "If G3 goes ahead": `configs/g3_kits.yaml`,
  `src/segreview/finetune.py`, `scripts/g3_prepare.py` (pool = the next 100 scans of the same seed-42 draw after
  kits100; check that none is in kits100; queue by the locked G1 score `model_diff_mean_region`; simulated
  corrections with the ground truth as reviewer), `scripts/g3_train.py`, `scripts/g3_evaluate.py`, with unit tests.
  Start from ALL pretrained weights, including the last layer (smoke test).
- [ ] Write the grid of settings (epochs, learning rate) in the config **before** the runs; choose on the
  development set (20 scans). No test-set evaluation in this phase.
- [ ] Run the chain for the 6 mm model with k = 20: A original, B queue + correction, C random + correction,
  D queue without correction.
- [ ] Log per run: config, commit and machine; loss per epoch (training and validation), learning rate, time per
  epoch; which scans were corrected and how many voxels changed; predictions and Dice per scan and per model; time
  and memory. Small files in `results/g3_*/`; trained models and predictions in `data/` and `models/` (the log says
  where, for the OneDrive).

### Phase 3 — the 3 mm model, more budgets and seeds, the test set

- [ ] Planned when phase 2 is done.

## 5. Results summary

*(Filled in at the end of each phase.)*

**Phase 1 (2026-10-21), RTX 4090 on Windows** (details and files: PROJECT_LOG.md, 2026-10-21):

| | 6 mm, batch 2 | 6 mm, batch 13 | 3 mm, batch 2 |
|---|---|---|---|
| Training step, 16 workers (data wait + step) | 0.040 s | 0.181 s | 0.176 s |
| Validation step | 0.021 s | 0.165 s | 0.208 s |
| Preparing one batch in one process (0 workers) | 0.41 s | 2.15 s | 2.23 s |
| GPU memory, torch peak allocated / reserved | 2.2 / 2.9 GB | 12.4 / 17.3 GB | 11.7 / 16.3 GB |
| RAM of the run, 16 workers (0 workers) | 11.4 (2.3) GB | 12.7 (2.4) GB | 14.2 (2.4) GB |
| PC's RAM in use, 16 workers (of 34.1 GB) | 31.6 GB | 32.3 GB | 34.0 GB |
| Windows commit, 16 workers | 62 GB | 78 GB | 79 GB |
| One fine-tuning run, 2,500 / 6,250 steps (estimate) | 2 / 5 min | 9 / 22 min | 9 / 23 min |

- The start from all pretrained weights is confirmed (loss −0.83 / −0.91 vs. 7.15 / 8.85 with the last layer
  re-initialized), and the corrected targets line up with the model (kidney Dice 0.90–0.93).
- Whole plan (estimate): 11 fine-tuned 3 mm models 4.2 h + predicting kits100 with them and the original 2.5 h + the
  queue 0.3 h ≈ **7 h of GPU time**. Prediction per scan: 3 mm 7.5 s, 6 mm 4.8 s (2 scans).
- Disk: kits100 9.9 GB, the pool 8.3 GB, checkpoints 0.33 GB per fine-tuned model; no problem.
- **RAM is the limit on this PC, not the GPU:** with 16 workers and other programs open (18.7 GB before the test),
  the RAM was full for the 3 mm model. For the real runs: close other programs, or use 12–14 workers (about 12–13
  are needed for the GPU never to wait).

## 6. Decisions made in this task

*(Date — decision — why. Also add each one to PROJECT_LOG.md.)*

- 2026-10-20 — Run natively on Windows in a separate `.venv` with `uv.lock`'s versions (section 3), not in WSL2 —
  Frederik prefers to stay on Windows. WSL2 (Linux inside Windows) was planned first, because `uv.lock` has no
  Windows packages, and dropped the same day before anything was installed.
- 2026-10-21 — RAM on Windows is measured as the private working sets of all processes, read once at the end of each
  run; commit and the PC's RAM in use are sampled — reading the working sets during the run slowed every 3rd step.
- 2026-10-20 — Tools that only this Windows PC needs live in `frederik/` (Frederik's request); the G3 code stays in
  `configs/`, `scripts/` and `src/segreview/` (docs/g3_plan.md), so the group can find and run it elsewhere too.
- 2026-10-20 — Own configs for the GPU smoke test (`g3_smoke_gpu`, `g3_smoke_gpu_3mm`) instead of editing
  `configs/g3_smoke.yaml` — one config per experiment; running the old one would overwrite the Mac's results in
  `results/g3_smoke/`.
- 2026-10-20 — GPU smoke-test settings: device `cuda` forced (no silent fallback to the CPU); torch threads 1, as
  nnU-Net's own training start does on a GPU; augmentation in the main process (0 workers: the CPU time to prepare
  one batch) and with 16 workers (whether the GPU waits for data, as in a real run); 10 warm-up steps of 30; 10
  validation steps; cuDNN benchmark on, as in nnU-Net's training.
- 2026-10-20 — The smoke test checks the kits100 overlap from the dataset's full case list plus the locked kits100
  results, so the 100 kits100 scans need not be downloaded on this PC; it records the RAM of all processes and the
  GPU memory, writes `machine.json`, and runs on Windows (no Unix-only `resource` module).

## 7. Questions for Bjørn

- Before phase 2 reads any ground truth of the pool: the group decides and logs which disagreement score (if any) is
  tested once on the pool (PROJECT_OVERVIEW.md, next steps, item 2), so G3 does not "open" the pool for that test.
- Are all KiTS cases separate patients? Open question in the log; check before the pool is fixed.
- If the smoke test reports that the downloaded case list does not reproduce kits100 (the KiTS file listing may have
  changed since September): can you share `data/raw/kits23/case_list.txt` from your Mac?

## 8. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action.
A new chat starts by reading this section.)*

- 2026-10-20: Task created. Phase 1 steps 1–3 and 4a done: machine check, Windows environment `.venv` (same versions
  as `uv.lock`, torch with CUDA 12.1), tests, weights and the 2 scans, configs. On branch `frederik/g3`, not committed
  yet. Nothing has been trained. **Next action:** Frederik commits the branch (GitHub Desktop) and gives the go-ahead
  for step 4b (the smoke tests). Close games and other GPU programs first: during the setup other programs used
  about 4.7 GB of the GPU's memory and 28–56 % of its time.
- 2026-10-21: **Phase 1 finished** (steps 4b–6): smoke tests for the 6 mm and 3 mm model on the RTX 4090,
  estimates and disk space (section 5, PROJECT_LOG.md 2026-10-21). Not committed yet. **Next action:** Frederik
  commits and decides on phase 2 (the G3 pipeline with the 6 mm model). Before phase 2 reads ground truth of the
  pool, the group logs which disagreement score is tested on the pool (section 7).
