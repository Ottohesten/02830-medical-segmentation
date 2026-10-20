"""Unit tests of the exploratory follow-up analyses: the 3 mm vs. 6 mm comparison and the simulated voxel correction.

All examples are tiny synthetic volumes (a single row of voxels), so every expected number can be worked out by hand.
"""

import numpy as np
import pandas as pd
import pytest

from segreview.model_comparison import (case_group, disagreement_counts, disagreement_shares, distance_bin_names,
                                        edge_distance_mm, model_dice_summary, queue_table)
from segreview.voxel_correction import correction_curves, corrected_dice, gain_auc, method_values, top_k_counts

LABEL_NAMES = {0: "background", 1: "kidney", 2: "tumor", 3: "cyst"}


def row(n: int, *ranges) -> np.ndarray:
    """A (1, 1, n) boolean volume that is True at the given x positions (ints or (start, stop) ranges, stop excluded)."""
    v = np.zeros((1, 1, n), dtype=bool)
    for r in ranges:
        if isinstance(r, tuple):
            v[0, 0, r[0]:r[1]] = True
        else:
            v[0, 0, r] = True
    return v


# ---------- part 1: the 6 mm model alone ----------

def test_case_group():
    assert case_group(0.70, 0.95, 0.83) == "bad"           # 3 mm bad, whatever 6 mm does
    assert case_group(0.90, 0.70, 0.83) == "false_alarm"   # 3 mm fine, 6 mm bad
    assert case_group(0.90, 0.85, 0.83) == "good"
    assert case_group(0.83, 0.83, 0.83) == "good"          # the threshold itself is not bad ("Dice < 0.83")


def test_model_dice_summary():
    df = pd.DataFrame({"dice_3mm": [0.9, 0.8, 0.7, 0.95], "dice_6mm": [0.85, 0.85, 0.6, 0.95]})
    s = model_dice_summary(df, 0.83)
    assert s["median_3mm"] == pytest.approx(0.85) and s["median_6mm"] == pytest.approx(0.85)
    assert s["mean_3mm"] == pytest.approx(0.8375) and s["mean_6mm"] == pytest.approx(0.8125)
    assert (s["n_below_3mm"], s["n_below_6mm"]) == (2, 1)
    assert (s["n_3mm_better"], s["n_6mm_better"], s["n_equal"]) == (2, 1, 1)
    # Pearson correlation of the ranks (3, 2, 1, 4) and (2.5, 2.5, 1, 4): 4.5 / sqrt(5 * 4.5) = sqrt(0.9).
    assert s["spearman_rho"] == pytest.approx(np.sqrt(0.9))


def test_queue_table_orders_by_score_and_marks_findings():
    scores = pd.DataFrame({"case_id": ["a", "b", "c", "d"], "s": [0.1, 0.4, 0.3, 0.2]})
    dice = pd.DataFrame({"case_id": ["a", "b", "c", "d"], "dice_3mm": [0.9, 0.6, 0.9, 0.9],
                         "dice_6mm": [0.9, 0.9, 0.5, 0.9]})
    q = queue_table(scores, dice, "s", top_n=3, threshold=0.83)
    assert list(q.case_id) == ["b", "c", "d"] and list(q["rank"]) == [1, 2, 3]
    assert list(q.finding) == ["real find", "false alarm", ""]


# ---------- part 2: where the masks disagree ----------

def test_edge_distance_inside_and_outside():
    mask = row(10, (3, 6))                                   # voxels 3, 4, 5
    d = edge_distance_mm(mask, (1.0, 1.0, 2.0))[0, 0]        # 2 mm between voxel centers along x
    assert d[1] == 4.0 and d[2] == 2.0                       # outside: to the nearest mask voxel (3)
    assert d[3] == 2.0 and d[4] == 4.0 and d[5] == 2.0       # inside: to the nearest voxel outside (2 or 6)
    assert np.isinf(edge_distance_mm(row(5), (1, 1, 1))).all()   # empty mask: no edge


def test_distance_bin_names():
    assert distance_bin_names([3, 6, 12]) == ["0-3 mm", "3-6 mm", "6-12 mm", ">12 mm"]


def _disagreement_example():
    """Labels: x=4 tumor, x=5..12 kidney, x=18 cyst, the rest background. Mask A (3 mm) = 5..14; mask B (6 mm) =
    4..14 without 10, plus 18."""
    labels = np.zeros((1, 1, 20), dtype=np.uint8)
    labels[0, 0, 4] = 2
    labels[0, 0, 5:13] = 1
    labels[0, 0, 18] = 3
    mask_a = row(20, (5, 15))                  # misses the tumor voxel 4 and the cyst 18; includes 13, 14 (wrong)
    mask_b = row(20, (4, 10), (11, 15), 18)    # gets 4 and 18, but misses kidney voxel 10
    return labels, mask_a, mask_b


def test_disagreement_counts_by_label_distance_and_right_model():
    labels, a, b = _disagreement_example()
    # Disagreeing voxels: 4 (tumor, 1 mm outside A, B right), 10 (kidney, 5 mm inside A, A right),
    # 18 (cyst, 4 mm outside A, B right). Voxels 13 and 14 are wrong in both masks, so not a disagreement.
    c = disagreement_counts(a, b, labels, [1, 2, 3], [], LABEL_NAMES, (1.0, 1.0, 1.0), [2, 6])
    got = {(r.label, r.distance_bin, r.right_model): r.n_voxels for r in c.itertuples()}
    assert got == {("tumor", "0-2 mm", "6mm"): 1, ("kidney", "2-6 mm", "3mm"): 1, ("cyst", "2-6 mm", "6mm"): 1}

    # With 2 mm voxels along x every distance doubles: 2 mm (still <= 2), 10 mm and 8 mm (now > 6).
    c2 = disagreement_counts(a, b, labels, [1, 2, 3], [], LABEL_NAMES, (1.0, 1.0, 2.0), [2, 6])
    assert dict(zip(c2.label, c2.distance_bin)) == {"tumor": "0-2 mm", "kidney": ">6 mm", "cyst": ">6 mm"}

    # Ignored labels drop out, as in Dice.
    c3 = disagreement_counts(a, b, labels, [1, 2], [3], LABEL_NAMES, (1.0, 1.0, 1.0), [2, 6])
    assert "cyst" not in set(c3.label) and c3.n_voxels.sum() == 2

    assert disagreement_counts(a, a, labels, [1, 2, 3], [], LABEL_NAMES, (1, 1, 1), [2, 6]).empty


def test_disagreement_shares():
    labels, a, b = _disagreement_example()
    c = disagreement_counts(a, b, labels, [1, 2, 3], [], LABEL_NAMES, (1.0, 1.0, 1.0), [2, 6])
    s = disagreement_shares(c, 2, [2, 6], LABEL_NAMES)
    assert s["n_disagree"] == 3
    assert s["near_share"] == pytest.approx(1 / 3)                 # only the tumor voxel is within 2 mm
    assert s["label_tumor_share"] == pytest.approx(1 / 3)
    assert s["far_label_kidney_share"] == pytest.approx(0.5) and s["far_label_cyst_share"] == pytest.approx(0.5)
    assert s["far_label_tumor_share"] == 0.0
    assert s["right_3mm_share"] == pytest.approx(1 / 3)
    assert s["near_right_3mm_share"] == 0.0 and s["far_right_3mm_share"] == pytest.approx(0.5)
    # A wider grid band (6 mm) puts all three voxels near the edge; then the far shares have nothing to divide by.
    s6 = disagreement_shares(c, 6, [2, 6], LABEL_NAMES)
    assert s6["near_share"] == 1.0 and np.isnan(s6["far_right_3mm_share"])
    with pytest.raises(ValueError):
        disagreement_shares(c, 5, [2, 6], LABEL_NAMES)


# ---------- part 3: simulated voxel correction ----------

def test_corrected_dice():
    assert corrected_dice(8, 2, 4) == pytest.approx(16 / 22)
    assert corrected_dice(8, 2, 4, fixed_fn=1) == pytest.approx(18 / 23)      # a missed voxel becomes a hit
    assert corrected_dice(8, 2, 4, fixed_fp=2) == pytest.approx(16 / 20)      # false voxels removed
    assert corrected_dice(8, 2, 4, 4, 2) == 1.0
    assert corrected_dice(0, 0, 0) == 1.0


def test_top_k_counts_shares_ties():
    values = np.array([3.0, 2.0, 2.0, 1.0])
    flag = np.array([True, False, True, False])
    got = top_k_counts(values, {"f": flag}, [0, 1, 2, 2.5, 3, 4])["f"]
    # k = 2 takes half of the tied pair (values 2), which holds one flagged voxel -> 1 + 0.5.
    assert np.allclose(got, [0, 1, 1.5, 1.75, 2, 2])
    assert np.allclose(top_k_counts(np.array([]), {"f": np.array([], bool)}, [1])["f"], [0])


def test_gain_auc():
    # Trapezoids from k = 0: 0-1 -> 0.05, 1-2 -> 0.15, 2-5 -> 0.6; total 0.8 over a width of 5.
    assert gain_auc([1, 2, 5, 10], [0.1, 0.2, 0.2, 0.3], 5) == pytest.approx(0.16)
    with pytest.raises(ValueError):
        gain_auc([1, 2], [0.1, 0.2], 3)


def _correction_example():
    """Truth = 10..19 plus a far-away blob 35..37; mask = 12..21. TP 8, FP 2 (20, 21), FN 5 (10, 11, 35, 36, 37).
    With a 2 mm border the region is 10..23 (14 voxels); the blob lies outside it and can never be fixed."""
    truth = row(40, (10, 20), (35, 38))
    mask = row(40, (12, 22))
    entropy = np.zeros((1, 1, 40), dtype=np.float32)
    entropy[0, 0, [10, 11]] = 1.0          # the heatmap points exactly at the two missed voxels in the region
    m2_diff = np.zeros((1, 1, 40), dtype=np.float32)
    return truth, mask, {"entropy": entropy, "m2_diff": m2_diff}


def test_correction_curves_by_hand():
    truth, mask, maps = _correction_example()
    k_percent = [100 * 2 / 14, 50, 100]
    info, rows = correction_curves(maps, mask, truth, (1.0, 1.0, 1.0), 2.0, k_percent,
                                   ["entropy", "m2_diff", "boundary", "random", "oracle"], [42, 43, 44])
    assert info["dice_before"] == pytest.approx(16 / 23)
    assert info["region_voxels"] == 14 and info["error_voxels"] == 7
    assert info["error_outside_region_share"] == pytest.approx(3 / 7)
    assert info["dice_ceiling"] == pytest.approx(20 / 23)                      # fix 10, 11, 20, 21
    assert info["unfixable_dice_gap_share"] == pytest.approx((3 / 23) / (7 / 23))
    got = {(r["method"], r["k_percent"]): r["dice_after"] for r in rows}
    assert got[("entropy", k_percent[0])] == pytest.approx(20 / 25)            # the 2 missed voxels fixed
    # m2_diff is 0 everywhere: all region voxels tie, so 2 of 14 voxels fix 2 * 2/14 FN and 2 * 2/14 FP on average.
    a = b = 2 * 2 / 14
    assert got[("m2_diff", k_percent[0])] == pytest.approx(2 * (8 + a) / (16 + 2 + 5 + a - b))
    assert got[("oracle", 50)] == pytest.approx(20 / 23)                       # 7 voxels hold all 4 region errors
    for m in ["entropy", "m2_diff", "boundary", "random", "oracle"]:
        assert got[(m, 100)] == pytest.approx(info["dice_ceiling"])            # whole region = the ceiling
    gains = [r["gain"] for r in rows]
    assert min(gains) >= -1e-12                                                # correcting to the truth never hurts


def test_empty_prediction_cannot_be_corrected():
    truth, _, maps = _correction_example()
    info, rows = correction_curves(maps, np.zeros_like(truth), truth, (1, 1, 1), 2.0, [10], ["entropy"], [42])
    assert info["dice_before"] == 0.0 and info["region_voxels"] == 0
    assert info["error_outside_region_share"] == 1.0 and info["unfixable_dice_gap_share"] == 1.0
    assert rows == [{"method": "entropy", "k_percent": 10, "dice_after": 0.0, "gain": 0.0}]


def test_random_order_is_reproducible_per_seed():
    region = row(30, (0, 30))
    v1 = method_values("random", {}, region, region, (1, 1, 1), region, seed=42)
    v2 = method_values("random", {}, region, region, (1, 1, 1), region, seed=42)
    v3 = method_values("random", {}, region, region, (1, 1, 1), region, seed=43)
    assert np.array_equal(v1, v2) and not np.array_equal(v1, v3)
