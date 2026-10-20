"""Unit tests of the exploratory scores from the real 3 mm vs. 6 mm differences, and of the paired bootstrap difference.

Most examples are a single row of voxels with 3 mm spacing, so every distance and count can be worked out by hand.
"""

import numpy as np
import pandas as pd
import pytest

from segreview.disagreement_scores import (candidate_scores, choose_score, counts_from_breakdown, queue_rank,
                                           score_names, top_counts)
from segreview.evaluation import paired_differences
from segreview.model_comparison import directional_disagreement, disagreement_counts

SPACING = (3.0, 3.0, 3.0)      # one voxel = 3 mm, so 6 mm = two voxels
LABEL_NAMES = {0: "background", 1: "kidney", 2: "tumor", 3: "cyst"}
CANDIDATES = ["far_disagree", "far_6mm_adds", "far_6mm_drops", "6mm_adds_all"]


def row(n: int, *positions) -> np.ndarray:
    """A (1, 1, n) boolean volume that is True at the given x positions."""
    v = np.zeros((1, 1, n), dtype=bool)
    v[0, 0, list(positions)] = True
    return v


def test_directional_disagreement_by_hand():
    a = row(14, 5, 6, 7, 8, 9)
    # 6 mm adds x = 2 (3 voxels = 9 mm from A -> far) and x = 11 (2 voxels = 6 mm -> still near, the band includes
    # 6 mm), and drops x = 7 (inside A, 3 voxels = 9 mm from the nearest outside voxel -> far).
    b = row(14, 2, 5, 6, 8, 9, 11)
    c = directional_disagreement(a, b, SPACING, grid_band_mm=6)
    assert c == {"n_mask_a": 5, "far_disagree": 2, "far_b_adds": 1, "far_b_drops": 1, "b_adds_all": 2}


def test_directional_disagreement_edge_cases():
    same = row(10, 3, 4, 5)
    assert directional_disagreement(same, same, SPACING, 6) == {
        "n_mask_a": 3, "far_disagree": 0, "far_b_adds": 0, "far_b_drops": 0, "b_adds_all": 0}
    # Empty 3 mm mask: there is no edge, so everything the 6 mm model finds is far.
    c = directional_disagreement(row(10), row(10, 1, 2, 3), SPACING, 6)
    assert c == {"n_mask_a": 0, "far_disagree": 3, "far_b_adds": 3, "far_b_drops": 0, "b_adds_all": 3}


def test_candidate_scores():
    counts = {"n_mask_a": 100, "far_disagree": 10, "far_b_adds": 4, "far_b_drops": 6, "b_adds_all": 20}
    s = candidate_scores(counts, 0.027, CANDIDATES, ["ml", "per_volume"])
    assert list(s) == score_names(CANDIDATES, ["ml", "per_volume"])
    assert s["far_disagree_ml"] == pytest.approx(0.27) and s["far_disagree_per_volume"] == pytest.approx(0.10)
    assert s["far_6mm_adds_ml"] == pytest.approx(0.108) and s["far_6mm_adds_per_volume"] == pytest.approx(0.04)
    assert s["far_6mm_drops_per_volume"] == pytest.approx(0.06)
    assert s["6mm_adds_all_ml"] == pytest.approx(0.54) and s["6mm_adds_all_per_volume"] == pytest.approx(0.20)
    # Empty 3 mm mask: divide by 1 voxel instead of 0.
    empty = dict(counts, n_mask_a=0)
    assert candidate_scores(empty, 1.0, ["far_6mm_adds"], ["per_volume"])["far_6mm_adds_per_volume"] == 4
    with pytest.raises(ValueError):
        candidate_scores(counts, 1.0, ["far_disagree"], ["per_liter"])


def test_counts_from_breakdown_by_hand():
    rows = pd.DataFrame({
        "label":        ["tumor", "background", "kidney", "background", "tumor"],
        "distance_bin": [">12 mm", "6-12 mm", "6-12 mm", "0-3 mm", "3-6 mm"],
        "right_model":  ["6mm", "3mm", "3mm", "6mm", "6mm"],
        "n_voxels":     [10, 4, 3, 7, 5]})
    # Far rows: tumor/6mm right (6 mm adds), background/3mm right (6 mm adds), kidney/3mm right (6 mm drops).
    # Near rows: background/6mm right (6 mm drops), tumor/6mm right (6 mm adds).
    c = counts_from_breakdown(rows, 6, [3, 6, 12], {"kidney", "tumor", "cyst"})
    assert c == {"far_disagree": 17, "far_b_adds": 14, "far_b_drops": 3, "b_adds_all": 19}


def test_gt_free_counts_match_the_breakdown_on_a_random_volume():
    """The new GT-free counts and the earlier GT breakdown must agree on any volume (the script checks this too)."""
    rng = np.random.default_rng(42)
    shape = (12, 14, 10)
    a = np.zeros(shape, dtype=bool)
    a[3:9, 4:11, 2:8] = True
    b = a.copy()
    b[rng.random(shape) < 0.08] ^= True    # flip some voxels everywhere, near and far from A's edge
    labels = rng.integers(0, 4, size=shape)
    spacing = (1.5, 2.0, 3.0)
    breakdown = disagreement_counts(a, b, labels, [1, 2, 3], [], LABEL_NAMES, spacing, [3, 6, 12])
    expected = counts_from_breakdown(breakdown, 6, [3, 6, 12], {"kidney", "tumor", "cyst"})
    got = directional_disagreement(a, b, spacing, 6)
    assert expected["far_disagree"] > 0 and expected["far_b_adds"] > 0
    assert {k: got[k] for k in expected} == expected


def test_choose_score_ties():
    m = pd.DataFrame({"method": ["x_ml", "x_per_volume", "y_ml", "y_per_volume", "other"],
                      "review_auc": [0.8, 0.9, 0.9, 0.9, 1.0],
                      "quality_auc": [0.9, 0.95, 0.96, 0.95, 1.0]})
    t = choose_score(m, ["x_ml", "x_per_volume", "y_ml", "y_per_volume"], "review_auc", "quality_auc")
    # 'other' is not a candidate; y_ml wins the review-AUC tie on quality AUC; x_per_volume beats y_per_volume on order.
    assert list(t.method) == ["y_ml", "x_per_volume", "y_per_volume", "x_ml"]
    assert list(t.chosen) == [True, False, False, False]


def test_top_counts_shares_ties():
    score = np.array([3.0, 2.0, 2.0, 0.0])
    groups = np.array(["bad", "false_alarm", "good", "bad"])
    # Top 2: the scan with score 3, plus one of the two tied scans with score 2 (half a chance each).
    assert top_counts(score, groups, 2) == {"real_finds": 1.0, "false_alarms": 0.5, "both_fine": 0.5}
    assert top_counts(score, groups, 10) == {"real_finds": 2.0, "false_alarms": 1.0, "both_fine": 1.0}


def test_queue_rank():
    assert list(queue_rank(np.array([0.5, 0.9, 0.5, 0.1]))) == [2.5, 1.0, 2.5, 4.0]


def test_paired_differences_by_hand():
    dice = np.array([0.5, 0.9, 0.7, 0.95])
    rules = [{"name": "below08", "rule": "threshold", "value": 0.8}]
    scores = {"perfect": -dice, "reverse": dice.copy(), "same": dice.copy()}
    rows = pd.DataFrame(paired_differences(scores, ["reverse"], dice, rules, 200, 0.95, np.random.default_rng(42)))
    p = rows[rows.method == "perfect"].set_index("measure")
    # rho: -1 vs +1. Review curve (bad = scans 1 and 3): perfect 0, .5, 1, 1, 1 -> AUC 0.75; reverse 0, 0, 0, .5, 1
    # -> AUC 0.25.
    assert p.loc["spearman_rho", "diff"] == pytest.approx(-2.0)
    assert p.loc["review_auc", "diff"] == pytest.approx(0.5)
    assert p.loc["review_auc", "bad_rule"] == "below08" and p.loc["quality_auc", "bad_rule"] == ""
    assert p.loc["quality_auc", "diff"] > 0
    assert p.loc["spearman_rho", "share_boot_better"] == 1.0
    # A score identical to the reference: every difference is 0 on every resample.
    s = rows[rows.method == "same"]
    assert (s["diff"] == 0).all() and (s.diff_ci_low == 0).all() and (s.diff_ci_high == 0).all()
    assert (s.share_boot_better == 0).all()
    assert set(rows.reference) == {"reverse"} and "reverse" not in set(rows.method)
