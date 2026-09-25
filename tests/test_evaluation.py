"""Unit tests of the G1 evaluation (src/segreview/evaluation.py): the sign convention of Spearman rho, the review
and quality curves and their areas, ties in the scores, and the edge cases.

Convention: a higher score means MORE uncertain, so a good score has a NEGATIVE Spearman rho with the true Dice,
and the scans with the highest scores are reviewed first.
"""

import itertools

import numpy as np
import pytest

from segreview.evaluation import (area, bad_mask, combine_scores, evaluate_methods, expected_top_k_sums,
                                  partial_spearman, quality_curve, quality_curve_from_score, random_curves,
                                  review_curve, review_curve_from_score, review_order)

CFG = {"seed": 0, "evaluation": {"n_random_orders": 200, "n_bootstrap": 100, "ci_level": 0.95}}


def test_review_curve_by_hand():
    score = np.array([4.0, 3.0, 2.0, 1.0])          # scan 0 is reviewed first
    bad = np.array([True, False, False, True])
    y = review_curve_from_score(score, bad)
    assert np.allclose(y, [0, 0.5, 0.5, 0.5, 1.0])   # share of the 2 bad scans found after k reviews
    # Trapezoid area with x = 0, 1/4, ..., 1: 0.25 * (0.25 + 0.5 + 0.5 + 0.75)
    assert area(y) == pytest.approx(0.5)


def test_quality_curve_by_hand():
    score = np.array([4.0, 3.0, 2.0, 1.0])
    dice = np.array([0.2, 0.9, 0.8, 0.5])
    y = quality_curve_from_score(score, dice)
    # Start: mean Dice 0.6. Correcting a scan sets its Dice to 1: +0.8, +0.1, +0.2, +0.5 (divided by 4).
    assert np.allclose(y, [0.6, 0.8, 0.825, 0.875, 1.0])
    assert area(y) == pytest.approx(0.25 * ((0.6 + 0.8) + (0.8 + 0.825) + (0.825 + 0.875) + (0.875 + 1.0)) / 2)


def test_fixed_order_and_score_versions_agree_without_ties():
    rng = np.random.default_rng(1)
    dice = rng.random(12)
    score = rng.random(12)
    bad = dice < 0.4
    order = review_order(score)
    assert np.allclose(review_curve(order, bad), review_curve_from_score(score, bad))
    assert np.allclose(quality_curve(order, dice), quality_curve_from_score(score, dice))


def test_curve_end_points():
    rng = np.random.default_rng(2)
    dice, score = rng.random(9), rng.random(9)
    bad = bad_mask(dice, "worst_fraction", 0.3)
    r, q = review_curve_from_score(score, bad), quality_curve_from_score(score, dice)
    assert r[0] == 0 and r[-1] == pytest.approx(1)                 # nothing found / everything found
    assert q[0] == pytest.approx(dice.mean()) and q[-1] == pytest.approx(1)   # before / after correcting all
    assert np.all(np.diff(r) >= 0) and np.all(np.diff(q) >= -1e-12)          # never decreasing


def test_ties_give_the_average_over_all_tie_orders():
    """Scans with equal scores have no natural order; the curve must be the mean over all their orders."""
    score = np.array([1.0, 2.0, 2.0, 1.0, 2.0])
    dice = np.array([0.3, 0.9, 0.5, 0.7, 0.2])
    bad = dice < 0.6
    reviews, qualities = [], []
    high, low = [1, 2, 4], [0, 3]                     # score 2 is reviewed before score 1
    for a in itertools.permutations(high):
        for b in itertools.permutations(low):
            order = np.array(a + b)
            reviews.append(review_curve(order, bad))
            qualities.append(quality_curve(order, dice))
    assert np.allclose(review_curve_from_score(score, bad), np.mean(reviews, axis=0))
    assert np.allclose(quality_curve_from_score(score, dice), np.mean(qualities, axis=0))


def test_all_scores_equal_is_like_random():
    dice = np.array([0.1, 0.9, 0.8, 0.95, 0.3, 0.85])
    bad = dice < 0.5
    same = np.zeros(6)
    assert np.allclose(review_curve_from_score(same, bad), np.linspace(0, 1, 7))     # straight line
    assert area(review_curve_from_score(same, bad)) == pytest.approx(0.5)
    assert np.allclose(quality_curve_from_score(same, dice), np.linspace(dice.mean(), 1, 7))
    rand_review, rand_quality, _ = random_curves(dice, bad, 3000, np.random.default_rng(0))
    assert np.allclose(rand_review, np.linspace(0, 1, 7), atol=0.03)                 # random orders agree
    assert np.allclose(rand_quality, np.linspace(dice.mean(), 1, 7), atol=0.01)


def test_no_bad_scans_gives_a_flat_zero_review_curve():
    score = np.array([3.0, 2.0, 1.0])
    y = review_curve_from_score(score, np.zeros(3, bool))
    assert np.array_equal(y, np.zeros(4)) and area(y) == 0.0       # defined (not NaN), nothing to find


def test_oracle_has_the_best_quality_curve():
    rng = np.random.default_rng(3)
    dice = rng.random(10)
    best = area(quality_curve_from_score(-dice, dice))              # oracle: worst Dice first
    for _ in range(200):
        assert area(quality_curve_from_score(rng.random(10), dice)) <= best + 1e-12


def test_area_trapezoid():
    assert area(np.ones(5)) == pytest.approx(1.0)
    assert area(np.linspace(0, 1, 11)) == pytest.approx(0.5)
    assert area(np.array([0.0, 1.0])) == pytest.approx(0.5)


def test_expected_top_k_sums_simple():
    assert np.allclose(expected_top_k_sums(np.array([2.0, 1.0]), np.array([5.0, 7.0])), [0, 5, 12])
    assert np.allclose(expected_top_k_sums(np.array([1.0, 1.0]), np.array([5.0, 7.0])), [0, 6, 12])


def test_bad_rules():
    dice = np.array([0.9, 0.5, 0.83, 0.2, 0.82])
    assert bad_mask(dice, "threshold", 0.83).tolist() == [False, True, False, True, True]   # strictly below
    assert bad_mask(dice, "worst_fraction", 0.4).tolist() == [False, True, False, True, False]
    assert bad_mask(dice, "worst_fraction", 0.01).sum() == 1                  # always at least one
    tied = np.array([0.5, 0.5, 0.5, 0.9])
    assert bad_mask(tied, "worst_fraction", 0.5).sum() == 2                   # exactly round(0.5 * 4), ties or not
    with pytest.raises(ValueError):
        bad_mask(dice, "median", 0.5)


def test_spearman_sign_convention():
    """A score that is high where Dice is low (a good uncertainty measure) must get a NEGATIVE rho and a high
    review AUC; the same score turned around gets a positive rho and a low review AUC."""
    rng = np.random.default_rng(4)
    dice = np.clip(rng.normal(0.85, 0.1, 40), 0, 1)
    volume = rng.uniform(100, 400, 40)
    good = -dice + rng.normal(0, 0.02, 40)
    scores = {"neg_volume_ml": -volume, "good": good, "backwards": -good}
    rows, curves = evaluate_methods(scores, dice, CFG, {"name": "w", "rule": "worst_fraction", "value": 0.2})
    by = {r["method"]: r for r in rows}
    assert by["good"]["spearman_rho"] < -0.8 and by["backwards"]["spearman_rho"] > 0.8
    assert by["good"]["review_auc"] > 0.8 > 0.2 > by["backwards"]["review_auc"]
    assert by["oracle"]["spearman_rho"] == pytest.approx(-1.0)                  # oracle = true Dice, worst first
    assert by["oracle"]["quality_auc"] >= by["good"]["quality_auc"] >= by["random"]["quality_auc"]
    assert by["oracle"]["review_auc_norm"] == pytest.approx(1.0) and by["random"]["review_auc_norm"] == 0.0
    assert by["good"]["rho_ci_low"] <= by["good"]["spearman_rho"] <= by["good"]["rho_ci_high"]
    assert by["neg_volume_ml"]["review_auc_minus_volume"] == 0.0               # the baseline against itself
    assert by["good"]["quality_auc_minus_volume"] > 0


def test_partial_spearman():
    rng = np.random.default_rng(5)
    volume = rng.uniform(100, 400, 50)
    dice = 0.7 + 0.2 * (volume - 100) / 300 + rng.normal(0, 0.02, 50)          # bigger organ, higher Dice
    assert np.isnan(partial_spearman(-volume, dice, volume))                    # the score IS the volume
    beyond = -(dice - 0.2 * (volume - 100) / 300)                               # knows Dice beyond volume
    assert partial_spearman(beyond, dice, volume) < -0.5


def test_combined_score_keeps_the_direction():
    a = np.array([0.1, 0.5, 0.9])
    b = np.array([0.0, 0.0, 1.0])
    mean = combine_scores({"a": a, "b": b}, {"scores": ["a", "b"], "rule": "mean_rank"})
    mx = combine_scores({"a": a, "b": b}, {"scores": ["a", "b"], "rule": "max_rank"})
    assert np.argmax(mean) == 2 and np.argmax(mx) == 2                           # suspicious on both = top
    assert mean[0] <= mean[1] and mx[0] <= mx[1]
