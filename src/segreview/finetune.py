"""G3: fine-tuning TotalSegmentator on corrected scans, as a learning curve (docs/g3_plan.md, "Design: learning
curve"; task document docs/tasks/g3_finetuning.md). Used by scripts/g3_prepare.py, g3_train.py, g3_evaluate.py,
g3_curves.py and g3_run.py.

The pieces, in the order the G3 scripts use them:
1. The orders of the pool: the queue (most uncertain first, by the locked G1 score; no ground truth) and random
   orders (seeded).
2. The runs of the learning curve: for each order and each k, the first k scans of that order. A run is defined by
   its scans, whether they are corrected, the training seed and the settings; runs that are the same (e.g. k = 100,
   which is the whole pool in every order) are trained once.
3. The training target of a scan: the model's own prediction of all 117 structures with the kidney replaced by the
   ground truth (the ground truth plays the reviewer); for the control, the model's own prediction unchanged. Only
   the kidney counts as "foreground" for nnU-Net's patch sampling, so part of the patches are centred on it.
4. nnU-Net's trainer (the NoMirroring variant the model was trained with), trained on a given list of scans, and
   saving files so that a stopped run can continue.
The nnU-Net parts are imported inside the functions: nnU-Net reads its folders from environment variables when it is
first imported, so the scripts set those first.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

QUEUE = "queue"
NOCORR_SUFFIX = "_nocorr"       # the preprocessed copy of a scan with the model's own prediction as target


# ---- 1. orders of the pool --------------------------------------------------------------------------------------

def queue_order(scores: pd.DataFrame, score: str) -> list[str]:
    """The pool scans, most uncertain first: by the score, highest first; ties by scan id, so the order is fixed.
    Scans without a score (NaN) come last.
    Input: a table with case_id and the score column (results/g3_pool/scores.csv); the score's name.
    """
    df = scores[["case_id", score]].sort_values([score, "case_id"], ascending=[False, True], na_position="last")
    return list(df.case_id)


def random_order(case_ids: list[str], seed: int) -> list[str]:
    """The scans in a random order, drawn with the seed from the sorted ids (the same seed gives the same order)."""
    ids = sorted(case_ids)
    return [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]


def pool_orders(scores: pd.DataFrame, score: str, random_seeds: list[int]) -> dict[str, list[str]]:
    """All orders of the pool: {"queue": [...], "random42": [...], ...}."""
    orders = {QUEUE: queue_order(scores, score)}
    for seed in random_seeds:
        orders[f"random{seed}"] = random_order(list(scores.case_id), seed)
    return orders


def orders_table(orders: dict[str, list[str]]) -> pd.DataFrame:
    """The orders as one table (order, position from 1, case_id), e.g. for results/g3_kits/orders.csv."""
    return pd.DataFrame([{"order": name, "position": i + 1, "case_id": case}
                         for name, cases in orders.items() for i, case in enumerate(cases)])


# ---- 2. runs of the learning curve ------------------------------------------------------------------------------

def setting_name(steps: int, lr: float) -> str:
    """A short name for a training setting, e.g. S2500_L0.001."""
    return f"S{steps}_L{lr:g}"


def order_seed(order: str, queue_seed: int) -> int:
    """The training seed of an order: the queue uses queue_seed, a random order its own seed (random43 -> 43)."""
    return queue_seed if order == QUEUE else int(order.removeprefix("random"))


def plan_runs(orders: dict[str, list[str]], k_values: list[int], setting: dict, queue_seed: int,
              control_k: int | None = None) -> tuple[list[dict], pd.DataFrame]:
    """The training runs for one setting, and which run gives each point of the curves.

    For each order and k: the first k scans of the order, corrected. A run that has the same scans (as a set), the
    same correction and the same seed as an earlier one is not trained again; its point uses the earlier run.
    control_k adds the control: the first control_k scans of the queue with the model's own prediction as target.
    Input: the orders, the k values (k > 0), the setting {"steps", "lr"}, the queue's training seed, control_k.
    Output: (runs, points). A run: run_id, order, k, seed, corrected, steps, lr, cases. points: one row per
            (order, k) with its run_id (the control as order "nocorr").
    """
    runs, points, seen = [], [], {}
    plan = [(order, k, True) for order in orders for k in k_values]
    if control_k:
        plan.append((QUEUE, control_k, False))
    for order, k, corrected in plan:
        if not 0 < k <= len(orders[order]):
            raise ValueError(f"k = {k} is not between 1 and the pool size {len(orders[order])}")
        cases = orders[order][:k]
        seed = order_seed(order, queue_seed)
        key = (frozenset(cases), corrected, seed, setting["steps"], setting["lr"])
        name = order if corrected else "nocorr"
        if key not in seen:
            run_id = f"{name}_k{k:03d}_{setting_name(setting['steps'], setting['lr'])}"
            seen[key] = run_id
            runs.append({"run_id": run_id, "order": name, "k": k, "seed": seed, "corrected": corrected,
                         "steps": setting["steps"], "lr": setting["lr"], "cases": list(cases)})
        points.append({"order": name, "k": k, "run_id": seen[key]})
    return runs, pd.DataFrame(points)


def training_keys(run: dict) -> list[str]:
    """The names of the run's scans in nnU-Net's preprocessed folder: the corrected copy or the uncorrected one."""
    return list(run["cases"]) if run["corrected"] else [c + NOCORR_SUFFIX for c in run["cases"]]


def read_orders(path) -> dict[str, list[str]]:
    """The orders saved by scripts/g3_prepare.py (results/g3_kits/orders.csv), as {order: [case ids in order]}."""
    df = pd.read_csv(path)
    return {name: list(g.sort_values("position").case_id) for name, g in df.groupby("order", sort=False)}


def experiment_runs(g3: dict, orders: dict[str, list[str]], choice: dict | None = None) -> tuple[list[dict],
                                                                                              pd.DataFrame]:
    """All runs of the G3 experiment, each with its stage, and the points of the curves.

    - 'test':  the pipeline test (configs/g3_kits.yaml, pipeline_test): one short run of another model, not a result.
    - 'grid':  every setting of settings_grid at its k, in each of its orders (development set only).
    - 'curve': once a setting is chosen (choice = {"steps", "lr"}): every order at every k, and the control. A curve
               run that is the same as a grid run (the chosen setting at the grid's k) keeps its grid run.
    Output: (runs, points); points has one row per (order, k) of the curves with its run_id (empty before choice).
    """
    seed = g3["orders"]["queue_seed"]
    test = g3["pipeline_test"]
    runs = [{"run_id": f"pipelinetest_{QUEUE}_k{test['k']:03d}_{setting_name(test['steps'], test['lr'])}",
             "stage": "test", "order": QUEUE, "k": test["k"], "seed": seed, "corrected": True, "steps": test["steps"],
             "lr": test["lr"], "cases": orders[QUEUE][:test["k"]], "model_resolution": test["model_resolution"]}]
    grid = g3["settings_grid"]
    for steps in grid["steps"]:
        for lr in grid["lr"]:
            grid_runs, _ = plan_runs({o: orders[o] for o in grid["orders"]}, [grid["k"]], {"steps": steps, "lr": lr},
                                     seed)
            runs += [{**r, "stage": "grid"} for r in grid_runs]
    points = pd.DataFrame(columns=["order", "k", "run_id"])
    if choice is not None:
        curve_runs, points = plan_runs(orders, g3["k_values"], {"steps": choice["steps"], "lr": choice["lr"]}, seed,
                                       g3.get("control_k"))
        known = {r["run_id"] for r in runs}
        runs += [{**r, "stage": "curve"} for r in curve_runs if r["run_id"] not in known]
    return runs, points


def choose_setting(dev_dice: pd.DataFrame, grid_runs: list[dict], n_dev: int, tie: float) -> tuple[dict,
                                                                                                   pd.DataFrame]:
    """The training setting by the rule fixed in docs/g3_plan.md, on the development set only.

    Each setting's score is the mean development Dice over its runs (one per order), each run's Dice being the mean
    over the development scans. Highest score wins; settings within `tie` of the best: the one with fewer steps.
    Input: Dice per scan (columns model = run_id, split, dice), the grid's runs, the number of development scans every
           run must have, the tie margin.
    Output: (the choice {"steps", "lr", "dev_mean_dice"}, the table per run and setting).
    """
    dev = dev_dice[dev_dice.split == "dev"]
    rows = []
    for run in grid_runs:
        dice = dev.loc[dev.model == run["run_id"], "dice"]
        if len(dice) != n_dev:
            raise ValueError(f"{run['run_id']}: {len(dice)} development scans evaluated, {n_dev} needed")
        rows.append({"steps": run["steps"], "lr": run["lr"], "order": run["order"], "run_id": run["run_id"],
                     "dev_mean_dice": dice.mean()})
    table = pd.DataFrame(rows)
    table["setting_mean_dice"] = table.groupby(["steps", "lr"]).dev_mean_dice.transform("mean")
    by_setting = table.groupby(["steps", "lr"], as_index=False).dev_mean_dice.mean()
    near = by_setting[by_setting.dev_mean_dice >= by_setting.dev_mean_dice.max() - tie]
    chosen = near.sort_values(["steps", "dev_mean_dice"], ascending=[True, False]).iloc[0]
    return {"steps": int(chosen.steps), "lr": float(chosen.lr), "dev_mean_dice": float(chosen.dev_mean_dice)}, table


# ---- 3. training targets ----------------------------------------------------------------------------------------

def side_labeled_truth(labels, ct, class_ids: dict, truth_to_classes: dict, body_threshold_hu: float):
    """The ground-truth kidney in the model's own classes, split into patient left and right.

    The midline is the mean left-right position of the body voxels (CT above body_threshold_hu) in the slices that
    contain kidney, as in plausibility.py. The arrays are in canonical RAS order (x, y, z), where a larger x is
    further towards the patient's right.
    Input: ground-truth label map and CT on the same grid, the model's class ids by name, which ground-truth labels
           become which class (e.g. {"kidney": [1, 2], "kidney_cyst": [3]}), body threshold.
    Output: uint8 map with the model's class ids (0 elsewhere).
    """
    organ = np.isin(labels, [v for values in truth_to_classes.values() for v in values])
    out = np.zeros(labels.shape, np.uint8)
    if not organ.any():
        return out
    body = ct[:, :, organ.any(axis=(0, 1))] > body_threshold_hu
    midline = np.nonzero(body)[0].mean()
    right = (np.arange(labels.shape[0]) > midline)[:, None, None]
    for base, truth_labels in truth_to_classes.items():
        part = np.isin(labels, truth_labels)
        out[part & right] = class_ids[f"{base}_right"]
        out[part & ~right] = class_ids[f"{base}_left"]
    return out


def corrected_target(pred: np.ndarray, truth: np.ndarray, kidney_ids: list[int]) -> np.ndarray:
    """The segmentation a reviewer hands back: the model's labels, with the kidney replaced by the ground truth.

    pred: the model's label of every voxel (argmax) on nnU-Net's cropped grid. truth: the ground-truth kidney in the
    model's classes on the same grid (0 elsewhere), with nnU-Net's marker -1 for voxels outside the scan's region.
    Kidney the model predicted where the ground truth has none becomes background; the -1 marker is kept.
    """
    target = np.where(truth > 0, truth, np.where(np.isin(pred, kidney_ids), 0, pred))
    return np.where(truth == -1, -1, target).astype(np.int16)


def own_target(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The control without correction: the model's own labels (only nnU-Net's -1 marker is taken from truth)."""
    return np.where(truth == -1, -1, pred).astype(np.int16)


def kidney_only(target: np.ndarray, kidney_ids: list[int]) -> np.ndarray:
    """The target's kidney classes only, 0 elsewhere: G3's "foreground" for nnU-Net's patch sampling."""
    return np.where(np.isin(target, kidney_ids), target, 0).astype(np.int16)


def other_organs_changed(original: np.ndarray, new: np.ndarray, kidney_ids: list[int]) -> float:
    """Forgetting: the share of the voxels the original model gives another structure (not background, not kidney)
    whose label the fine-tuned model changes. Both label maps on the same grid. NaN if there are no such voxels."""
    if original.shape != new.shape:
        raise ValueError(f"label maps differ in shape: {original.shape} vs {new.shape}")
    other = (original > 0) & ~np.isin(original, kidney_ids)
    return float((new[other] != original[other]).mean()) if other.any() else float("nan")


# ---- 4. nnU-Net's trainer ---------------------------------------------------------------------------------------

def g3_trainer_class():
    """nnU-Net's trainer in the variant the model was trained with (no mirroring), trained on a given list of scans.

    nnU-Net's __init__ cannot take extra arguments (it stores its own arguments for the checkpoint), so the scans are
    set afterwards as trainer.g3_cases. The same scans are used for nnU-Net's per-epoch validation, which here is only a
    training monitor; G3's real evaluation is on kits100 (scripts/g3_evaluate.py).
    """
    from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
        nnUNetTrainerNoMirroring)

    class G3Trainer(nnUNetTrainerNoMirroring):
        g3_cases: list[str] = []

        def do_split(self):
            if not self.g3_cases:
                raise ValueError("Set trainer.g3_cases before training.")
            return list(self.g3_cases), list(self.g3_cases)

        def save_checkpoint(self, filename: str) -> None:
            # A run stopped while saving keeps its previous checkpoint, so it can continue from it (g3_train.py).
            save_atomically(super().save_checkpoint, filename)

    return G3Trainer


def save_atomically(save, path) -> None:
    """Write a file through a temporary file next to it: save(temporary path), then one rename onto `path`. A process
    stopped while writing leaves the previous version of the file intact (and at most a stray .tmp file).
    If save() writes nothing (nnU-Net with checkpointing turned off), `path` is left as it is."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)               # left over from an earlier save that was stopped
    save(tmp)
    if not tmp.exists():
        return
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:               # Windows: another program (e.g. a virus scan) briefly holds the file
            if attempt == 4:
                raise
            time.sleep(0.5)


def training_log(trainer) -> pd.DataFrame:
    """nnU-Net's per-epoch log of a finished training as a table: train and validation loss, learning rate, the
    pseudo Dice nnU-Net computes on the validation batches (here: the training scans) and the epoch's seconds."""
    log = trainer.logger.local_logger.my_fantastic_logging
    n = len(log["train_losses"])
    return pd.DataFrame({
        "epoch": range(n), "train_loss": log["train_losses"], "validation_loss": log["val_losses"][:n],
        "learning_rate": log["lrs"][:n], "pseudo_dice": log["mean_fg_dice"][:n], "ema_pseudo_dice": log["ema_fg_dice"][:n],
        "seconds": np.subtract(log["epoch_end_timestamps"][:n], log["epoch_start_timestamps"][:n])})
