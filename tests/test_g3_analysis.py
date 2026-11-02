"""Tests of the G3 curve summary (src/segreview/g3_analysis.py) on a small hand-made example: 2 orders of random
plus the queue, k = 0, 10, 20, and 4 scans."""

import numpy as np
import pandas as pd
import pytest

from segreview.g3_analysis import (curve_points, curves, dice_matrix, seed_spread, summary_weights,
                                   summary_with_ci)

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
