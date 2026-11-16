"""Tests of the G3 curve summary (src/segreview/g3_analysis.py) on a small hand-made example: 2 orders of random
plus the queue, k = 0, 10, 20, and 4 scans; and of what the first scans of each order contain."""

import numpy as np
import pandas as pd
import pytest

from segreview.g3_analysis import (arm_difference_with_ci, curve_points, curves, dice_matrix, first_k_content,
                                   seed_spread, summary_weights, summary_with_ci)

POINTS = pd.DataFrame({"order": ["queue", "queue", "r1", "r1", "r2", "r2"], "k": [10, 20, 10, 20, 10, 20],
                       "run_id": ["q10", "q20", "a10", "a20", "b10", "q20"]})   # r2 at k = 20 shares the queue's run
DICE = {"original": 0.80, "q10": 0.84, "q20": 0.88, "a10": 0.82, "a20": 0.85, "b10": 0.80}


def dice_rows(split="test", scans=4):
    return pd.DataFrame([{"model": m, "case_id": f"c{i}", "split": split, "dice": d + 0.01 * i}
                         for m, d in DICE.items() for i in range(scans)])


def test_curves_and_the_summary_by_hand():
    points = curve_points(POINTS, ["queue", "r1", "r2"])
    assert set(points[points.k == 0].run_id) == {"original"}
    matrix = dice_matrix(dice_rows(), list(points.run_id), "test", "dice")
    curve = curves(matrix, points).set_index(["order", "k"]).mean_dice
    assert curve[("queue", 0)] == pytest.approx(0.815) and curve[("r2", 20)] == curve[("queue", 20)]
    weights = summary_weights(points, "queue", ["r1", "r2"], [10, 20])
    # queue - mean(random): k = 10: 0.84 - (0.82 + 0.80) / 2 = 0.03; k = 20: 0.88 - (0.85 + 0.88) / 2 = 0.015
    result = summary_with_ci(matrix, weights, n_bootstrap=200, seed=42, ci_level=0.95)
    assert result["estimate"] == pytest.approx((0.03 + 0.015) / 2)
    # every scan has the same difference here, so the interval collapses onto the estimate
    assert result["ci_low"] == pytest.approx(result["estimate"]) and result["ci_high"] == pytest.approx(0.0225)
    spread = seed_spread(curves(matrix, points), ["r1", "r2"]).set_index("k").spread
    assert spread[10] == pytest.approx(0.02) and spread[20] == pytest.approx(0.03)


def test_bootstrap_is_reproducible_and_scans_shared_by_all_models():
    points = curve_points(POINTS, ["queue", "r1", "r2"])
    matrix = dice_matrix(dice_rows(), list(points.run_id), "test", "dice")
    weights = summary_weights(points, "queue", ["r1", "r2"], [10, 20])
    assert summary_with_ci(matrix, weights, 100, 1, 0.9) == summary_with_ci(matrix, weights, 100, 1, 0.9)
    partial = dice_rows()
    partial = partial[~((partial.model == "a10") & (partial.case_id == "c0"))]
    assert len(dice_matrix(partial, list(points.run_id), "test", "dice")) == 3     # c0 dropped for every model
    with pytest.raises(ValueError):
        dice_matrix(dice_rows("dev"), list(points.run_id), "test", "dice")
    assert np.isclose(weights.sum(), 0.0)                                          # a difference of curves


def test_arm_difference_by_hand_and_with_training_noise():
    runs = {"q1": 0.90, "q2": 0.80, "r1": 0.85, "r2": 0.85, "q3": 0.95, "q4": 0.95, "r3": 0.90, "r4": 0.80}
    matrix = pd.DataFrame({m: [d, d, d] for m, d in runs.items()}, index=["c0", "c1", "c2"])
    cells = [(["q1", "q2"], ["r1", "r2"]), (["q3", "q4"], ["r3", "r4"])]
    result = arm_difference_with_ci(matrix, cells, n_bootstrap=500, seed=42, ci_level=0.95)
    assert result["estimate"] == pytest.approx((0.0 + 0.10) / 2)            # k1: 0.85 - 0.85; k2: 0.95 - 0.85
    # every scan is the same, so over the scans alone there is no uncertainty; the runs differ, so two stages widen it
    assert result["ci_low_scans"] == pytest.approx(0.05) and result["ci_high_scans"] == pytest.approx(0.05)
    assert result["ci_low"] < 0.05 < result["ci_high"]
    same = pd.DataFrame({"q1": [0.9, 0.8, 0.7], "q2": [0.9, 0.8, 0.7], "r1": [0.8] * 3, "r2": [0.8] * 3})
    result = arm_difference_with_ci(same, [(["q1", "q2"], ["r1", "r2"])], 500, 1, 0.9)
    # identical runs within each arm: drawing runs changes nothing, so both intervals are the same
    assert result["ci_low"] == result["ci_low_scans"] and result["ci_high"] == result["ci_high_scans"]
    assert result["ci_low"] < result["estimate"] < result["ci_high"]


def test_first_k_content_by_hand():
    per_scan = pd.DataFrame({"tumor_ml": [50.0, 0.0, 10.0, 2.0], "cyst_ml": [0.0, 1.0, 0.0, 0.0],
                             "voxels_changed": [400, 100, 300, 200]}, index=["a", "b", "c", "d"])
    orders = {"queue": ["a", "c", "b", "d"], "r1": ["b", "d", "a", "c"], "r2": ["d", "a", "c", "b"]}
    content = first_k_content(orders, per_scan, [2], ["r1", "r2"]).set_index("order")
    assert content.loc["queue", "n_with_tumor"] == 2 and content.loc["queue", "tumor_ml_mean"] == 30.0
    assert content.loc["r1", "n_with_tumor"] == 1 and content.loc["r1", "n_with_cyst"] == 1
    assert content.loc["queue", "voxels_changed_median"] == 350.0
    pooled = content.loc["random (pooled)"]          # b, d, d, a: d counts once per order it is in
    assert pooled.n_scans == 4 and pooled.n_with_tumor == 3 and pooled.tumor_ml_median == 2.0
    assert pooled.voxels_changed_mean == pytest.approx((100 + 200 + 200 + 400) / 4)
