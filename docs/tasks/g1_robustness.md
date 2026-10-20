# Task: Does our ranking still work when the scans get worse?

| | |
|---|---|
| **Owner** | Rasmus |
| **Contact / lead** | Bjørn |
| **Branch** | `rasmus/robustness` |
| **Created** | 2026-10-20 |
| **Status** | Not started |
| **Target** | Steps 1–4 by 2026-10-30, all steps by 2026-11-13 |

This document is the task description **and** the working notes. Whoever works on the task (with or without an
assistant) updates it after every step: tick the box, write the date and what was done, and update
"Where we are" at the bottom. Then anyone in the group, or a new chat, can see the status and continue.

---

## 1. The idea in plain words

AI models often fail when the data look different from what they were trained on: a noisier scanner, a lower
resolution, a different contrast. This is called *domain shift*. Our ranking (G1) puts the scans the model most
likely got wrong at the top of the queue. So far we have only tested it on the scans as they are.

**Question:** If we make the scans worse on purpose, (a) does the model's segmentation get worse, and (b) does our
uncertainty score notice it, so the damaged scans move up the queue?

If yes, that is strong support for the method: it reacts to exactly the situation where a reviewer is most needed.
If no, that is an important limitation for the report. Both outcomes are useful.

This was part of the original plan (2026-09-23: "natural variation plus controlled perturbations, reported
separately") but was only tried on 3 spleen scans, never on KiTS. See [PROJECT_LOG.md](../../PROJECT_LOG.md).

## 2. Starting points

How the code works is in [README.md](../../README.md) (perturbations: section "Running G1"); the general rules are in
[CONTRIBUTING.md](../../CONTRIBUTING.md). Specific to this task:

- The perturbation code exists (`perturb` in `src/segreview/augment.py`; example config: `configs/local.yaml`) but has
  only been run on 3 spleen scans: `lowres_8mm` clearly lowered Dice, `noise_60` a little, `contrast_60` not at all.
- Evaluate with the locked G1 score (`model_diff_mean_region`) and, as secondary, entropy (`entropy_mean_region`).
- **Own config and folders** (e.g. `configs/kits_perturbed.yaml`, `name: kits_perturbed`). Never write to
  `results/kits100/` or `data/predictions/kits100/`. New model runs are needed, in the new folders only.
- **Choose the strengths on the development set (20 scans) first**, log the choice, then run the test set.

## 3. Steps

Tick each box when done and write the date and a short note (numbers + file paths) under it.

- [ ] **1. Get started.** Install the project (`uv sync`), run `uv run pytest`, download the KiTS data and model
  weights (`scripts/download_data.py`, `scripts/download_model.py` with `configs/kits100.yaml`). Read
  [PROJECT_OVERVIEW.md](../../PROJECT_OVERVIEW.md) and be able to explain G1 in two sentences.
  *Notes:*

- [ ] **2. Measure the time.** Run the 3 mm and 6 mm models on 2 development scans with one perturbation on your
  machine. Write down seconds per scan per model, and estimate the time for 20 scans × 3–4 perturbations and for
  80 scans. If it is too slow (more than a night), ask Bjørn; Frederik's GPU machine is an option.
  *Notes:*

- [ ] **3. Choose the strengths on the development set (20 scans).** Try a few strengths per type, e.g.
  noise 30/60/120 HU, lowres 6/8/12 mm, contrast 0.8/0.9 (0.6 had no effect on spleen). Goal: perturbations that
  clearly lower Dice for some scans but do not destroy every scan. **Before looking at how the uncertainty scores
  do**, write the chosen list and the reason in the log and in the config. Keep the clean scans as the reference.
  *Notes:*

- [ ] **4. Run the test set (80 scans)** with the chosen perturbations (overnight or on the GPU).
  *Notes:*

- [ ] **5. Evaluate.** With the existing G1 evaluation (`scripts/evaluate_g1.py`), report clean and perturbed
  separately, for development and test:
  - How much does Dice drop per perturbation (median, and how many scans become "bad", Dice < 0.83)?
  - Does the score rise when Dice falls? Per scan: change in score vs. change in Dice from clean to perturbed
    (Spearman ρ), for the locked score and for entropy.
  - Mixed queue: clean and perturbed scans together. Do the damaged scans move to the top? Review-AUC vs. the
    volume baseline, random and oracle, as in G1.
  *Notes:*

- [ ] **6. One figure and a short summary.** One figure that answers the question at a glance, and a summary of
  5–10 lines in this document (section 4) and in the log. Bjørn uses it for the report.
  *Notes:*

- [ ] **7. Present it to the group** (5 minutes): what you did, what came out, what it means.
  *Notes:*

## 4. Results summary

*(Filled in at step 6.)*

## 5. Decisions made in this task

*(Date — decision — why. Also add each one to PROJECT_LOG.md.)*

## 6. Questions for Bjørn

*(Write questions here and tell Bjørn.)*

## 7. Where we are

*(Update at the end of every work session: date, last finished step, what is half done, the next concrete action.
A new chat starts by reading this section.)*

- 2026-10-20: Task created. Next action: step 1.
