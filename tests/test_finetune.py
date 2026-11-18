"""Tests of G3's learning-curve logic (src/segreview/finetune.py) on small synthetic examples: the orders of the pool,
the plan of runs (with identical runs trained once), the training targets, the forgetting measure and saving files
so that a stopped run can continue."""

import numpy as np
import pandas as pd
import pytest

from segreview.finetune import (choose_setting, corrected_target, experiment_runs, kidney_only, other_organs_changed,
                                own_target, plan_runs, pool_orders, queue_order, random_order, save_atomically,
                                side_labeled_truth, training_keys)

POOL = [f"case_{i:05d}" for i in range(100)]
KIDNEY = [2, 3, 23, 24]                      # the 3 mm model's kidney_right, kidney_left, kidney cysts


def test_queue_order_most_uncertain_first_with_fixed_ties_and_missing_last():
    scores = pd.DataFrame({"case_id": ["a", "b", "c", "d", "e"], "s": [0.1, 0.5, np.nan, 0.5, 0.3]})
    assert queue_order(scores, "s") == ["b", "d", "e", "a", "c"]


def test_random_order_is_a_seeded_permutation():
    first = random_order(POOL[::-1], 42)
    assert sorted(first) == POOL
    assert first == random_order(POOL, 42)               # the input order does not matter
    assert first != random_order(POOL, 43)


def test_plan_runs_counts_and_trains_identical_runs_once():
    scores = pd.DataFrame({"case_id": POOL, "s": np.linspace(0, 1, 100)})
    orders = pool_orders(scores, "s", [42, 43, 44])
    runs, points = plan_runs(orders, list(range(10, 101, 10)), {"steps": 2500, "lr": 0.001}, queue_seed=42,
                             control_k=100)
    assert len(points) == 4 * 10 + 1                     # every order has every point, plus the control
    assert len(runs) == 40                               # queue k=100 and random42 k=100 are the same run
    by_point = points.set_index(["order", "k"]).run_id
    assert by_point[("random42", 100)] == by_point[("queue", 100)] == "queue_k100_S2500_L0.001"
    assert by_point[("random43", 100)] != by_point[("queue", 100)]      # same scans, other seed: its own run
    control = next(r for r in runs if not r["corrected"])
    assert control["run_id"] == "nocorr_k100_S2500_L0.001" and training_keys(control)[0].endswith("_nocorr")
    queue10 = next(r for r in runs if r["run_id"] == "queue_k010_S2500_L0.001")
    assert queue10["cases"] == orders["queue"][:10] and queue10["seed"] == 42
    assert training_keys(queue10) == queue10["cases"]
    with pytest.raises(ValueError):
        plan_runs(orders, [101], {"steps": 2500, "lr": 0.001}, queue_seed=42)


G3 = {"orders": {"queue_seed": 42}, "k_values": list(range(10, 101, 10)), "control_k": 100,
      "pipeline_test": {"model_resolution": "fastest", "k": 10, "steps": 250, "lr": 0.001},
      "settings_grid": {"steps": [2500, 6250], "lr": [0.001, 0.0003], "k": 50, "orders": ["queue", "random42"]}}


def test_experiment_runs_by_stage_and_the_grid_run_is_reused_by_the_curve():
    scores = pd.DataFrame({"case_id": POOL, "s": np.linspace(0, 1, 100)})
    orders = pool_orders(scores, "s", [42, 43, 44])
    runs, points = experiment_runs(G3, orders)                       # before the choice: test and grid only
    assert [r["stage"] for r in runs].count("test") == 1 and [r["stage"] for r in runs].count("grid") == 8
    assert points.empty
    assert runs[0]["model_resolution"] == "fastest" and runs[0]["cases"] == orders["queue"][:10]
    runs, points = experiment_runs(G3, orders, {"steps": 2500, "lr": 0.001})
    assert len(runs) == 1 + 8 + 40 - 2                               # the chosen setting's two k = 50 runs exist
    assert len({r["run_id"] for r in runs}) == len(runs)
    queue50 = points.set_index(["order", "k"]).run_id[("queue", 50)]
    assert next(r for r in runs if r["run_id"] == queue50)["stage"] == "grid"


def test_choose_setting_rule_and_fewer_steps_within_the_tie():
    grid_runs = [{"run_id": f"{o}_{s}_{lr}", "order": o, "steps": s, "lr": lr}
                 for s in (2500, 6250) for lr in (0.001, 0.0003) for o in ("queue", "random42")]
    means = {(2500, 0.001): 0.900, (2500, 0.0003): 0.890, (6250, 0.001): 0.9005, (6250, 0.0003): 0.880}
    dev = pd.DataFrame([{"model": r["run_id"], "split": "dev", "case_id": i, "dice": means[(r["steps"], r["lr"])]}
                        for r in grid_runs for i in range(3)])
    choice, table = choose_setting(dev, grid_runs, n_dev=3, tie=0.001)
    assert (choice["steps"], choice["lr"]) == (2500, 0.001)          # 6250 is better by less than the tie
    choice, _ = choose_setting(dev, grid_runs, n_dev=3, tie=0.0)
    assert (choice["steps"], choice["lr"]) == (6250, 0.001)
    assert len(table) == 8
    with pytest.raises(ValueError):
        choose_setting(dev.iloc[1:], grid_runs, n_dev=3, tie=0.001)     # a run with a missing scan


def test_targets_keep_the_outside_marker_and_replace_only_the_kidney():
    pred = np.array([0, 2, 2, 5, 7, 3, 0])                # model: background, kidney, kidney, liver, ...
    truth = np.array([-1, 2, 0, 0, 3, 0, 2])              # ground-truth kidney in the model's classes; -1 = outside
    assert corrected_target(pred, truth, KIDNEY).tolist() == [-1, 2, 0, 5, 3, 0, 2]
    assert own_target(pred, truth).tolist() == [-1, 2, 2, 5, 7, 3, 0]
    assert kidney_only(np.array([-1, 2, 0, 5, 3, 23]), KIDNEY).tolist() == [0, 2, 0, 0, 3, 23]


def test_forgetting_counts_only_other_structures():
    original = np.array([0, 2, 5, 5, 7, 7])               # background, kidney, liver x2, spleen x2
    new = np.array([5, 0, 5, 6, 7, 0])                    # changes: background (ignored), kidney (ignored), 2 of 4
    assert other_organs_changed(original, new, KIDNEY) == pytest.approx(0.5)
    assert np.isnan(other_organs_changed(np.array([0, 2]), np.array([0, 3]), KIDNEY))
    with pytest.raises(ValueError):
        other_organs_changed(original, new[:3], KIDNEY)


def test_save_atomically_keeps_the_previous_file_when_a_save_is_stopped(tmp_path):
    path = tmp_path / "dice.csv"
    save_atomically(lambda p: p.write_text("first"), path)
    assert path.read_text() == "first"

    def stopped(p):
        p.write_text("half writ")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        save_atomically(stopped, path)
    assert path.read_text() == "first"

    save_atomically(lambda p: None, path)        # nothing written (nnU-Net with checkpointing off): file unchanged
    assert path.read_text() == "first"
    save_atomically(lambda p: p.write_text("second"), path)
    assert path.read_text() == "second" and not (tmp_path / "dice.csv.tmp").exists()


def test_side_labeled_truth_splits_left_and_right_at_the_midline():
    labels = np.zeros((10, 3, 2), np.uint8)
    labels[1, 1, :] = 1                                   # kidney on the patient's left (small x)
    labels[8, 1, :] = 3                                   # cyst on the patient's right (large x)
    ct = np.full(labels.shape, 40.0)                      # the whole width is body
    ids = {"kidney_left": 3, "kidney_right": 2, "kidney_cyst_left": 24, "kidney_cyst_right": 23}
    out = side_labeled_truth(labels, ct, ids, {"kidney": [1, 2], "kidney_cyst": [3]}, -500)
    assert out[1, 1, 0] == 3 and out[8, 1, 0] == 23 and out.sum() == 2 * (3 + 23)
