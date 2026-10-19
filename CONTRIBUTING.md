# How we work in this repository

Rules for everyone who changes code, runs experiments or adds results. Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
first to understand the project. If a rule here gets in the way, raise it with the group instead of working around it.

## 1. Git

- **Pull before you start**, then run `uv sync` (the lock file may have changed).
- **Work on your own branch**, never directly on `main`: `git checkout -b <name>/<topic>` (e.g. `otto/g3-training`).
- **Run the tests before you push:** `uv run pytest` (about two minutes). Do not push a branch with failing tests
  without saying so in the pull request.
- **Open a pull request** into `main` and ask someone else in the group to look at it before it is merged.
- Short, plain commit messages in English (e.g. `Add Dice for the 6 mm model`).
- **Never commit:** CT scans, masks, probabilities or model weights (`data/`, `models/`, `*.nii.gz`, `*.npz`),
  participant data from the user study, passwords or tokens. Large files (e.g. trained models) are shared via the
  group's DTU OneDrive/Teams folder; the log says where.

## 2. Results that must not change

- **The G1 results in `results/kits100/` are locked.** Do not overwrite them, and do not rerun model inference for
  kits100. They may only be reproduced from the saved predictions (`scripts/run_g1.py` skips finished steps).
- **The test set (80 scans) is evaluated once per question.** Make choices on the development set (20 scans) first,
  write the choice in the log, then evaluate on test. If you change something after seeing test results, log the
  date, what changed and why.
- **Ground truth is only used for evaluation**, never as input to a score or shown in a study scan.
- **Never invent or estimate results.** Every number in the log, the overview or the report comes from code that
  was run; name the file it is in.

## 3. New experiments

- Every experiment has its **own config** in `configs/` (copy one, give it a new `name`) and therefore its own
  output folders: `results/<name>/` and `data/predictions/<name>/`. No hard-coded paths or numbers in the code.
- **Results from different machines are not mixed.** CPU, Apple GPU (MPS) and NVIDIA GPU (CUDA) give slightly
  different numbers, and GPU training is not exactly repeatable. All models that are compared with each other are
  trained and evaluated on the same machine, including the "original model" reference (do not reuse the kits100
  numbers for it). Record the machine in the log.
- Set seeds explicitly.
- Exploratory analyses are fine, but label them **exploratory** in file names or output and in the log.

## 4. Log everything

Add a dated entry at the bottom of [PROJECT_LOG.md](PROJECT_LOG.md) for every result, experiment, idea and decision,
also when something is dropped (say why). Format: a `## YYYY-MM-DD` heading, then one line per item starting with a tag
(`[result]`, `[experiment]`, `[decision]`, `[dropped]`, `[idea]`, `[supervisor]`, `[admin]`), with numbers and the
path to the result file. Update the status and next steps in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) when they change.

**For long runs (training, large inference), save enough to answer new questions without running again:**

- the exact config, the git commit hash and the machine (CPU/GPU model, OS, torch and CUDA versions);
- a training log: loss per epoch (training and validation), learning rate, time per epoch;
- the trained model after training (and the best checkpoint if it differs), shared outside git;
- which scans were used for training, which were "corrected", and how;
- predictions and Dice **per scan and per model**, not only averages;
- run times and peak memory.

Small files (CSV, figures, logs) go in `results/<name>/` and in git. Big files go outside git, and the log says where.

## 5. Code style

- Code, comments, docs and the report in English (American spelling).
- Every file starts with a short comment on what it does and where it fits in the pipeline; every function has a
  docstring (what it does, input, output). Explain *why* at the non-obvious steps, in plain words, so everyone in the
  group can explain the code at the exam.
- New code comes with tests in `tests/` (small synthetic examples are fine).
- Update [README.md](README.md) when you add a script or change how something is run.
