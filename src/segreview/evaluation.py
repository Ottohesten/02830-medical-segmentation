"""G1 evaluation: how well does each score rank the scans? Uses ground truth (Dice) ONLY here.

For every ranking method (uncertainty scores, baselines, oracle) we compute:

- Spearman rho between score and true Dice, with a bootstrap confidence interval (CI).
  Spearman compares ranks, not values: rho = -1 means "the higher the score, the lower the Dice"
  in perfect order. Because of the sign convention (higher score = more uncertain), a good
  method has rho close to -1. The bootstrap CI: draw the scans again with replacement many times,
  recompute rho each time, and take the middle 95 % of the results.

- Review curve (Kristine's "failure detection vs. percentage of cases sent to manual review"):
  sort scans by score (most uncertain first) and send the top x % to review.
  y = share of the bad segmentations that are among those reviewed.

- Quality curve (Kristine's "final quality vs. amount of human correction"):
  same order; the reviewed scans are corrected to ground truth (Dice = 1).
  y = mean Dice over the whole dataset after that correction.
  This is "Dice per unit of human effort" at dataset level, without a user study.

- Area under each curve (AUC), x from 0 to 1. Higher is better for both curves.
  Also "normalized" AUC: (method - random) / (oracle - random). 0 = no better than random,
  1 = as good as the perfect ranking.

- The difference in AUC to the volume baseline, with a paired bootstrap CI (same resampled scans
  for both methods). A method only shows it measures more than organ size if this is above 0.

- Partial Spearman rho "given volume": the rank correlation between score and Dice that is left
  after removing what both have in common with the predicted organ volume. Small organs get lower
  Dice, and many scores also depend on size, so a plain rho can look good just because of size.
  Formula (with Spearman correlations r): r_sd.v = (r_sd - r_sv * r_dv) / sqrt((1 - r_sv^2)(1 - r_dv^2)),
  s = score, d = Dice, v = predicted volume. Negative = the score finds bad scans beyond organ size.

Methods:
- random: random review order, averaged over many random orders.
- oracle: ranks by true Dice (worst first). Uses ground truth, so it is only an upper bound.
- combinations (evaluation.combinations): two GT-free scores merged by their ranks, see combine_scores.

"Bad" segmentations can be defined by several rules at once (evaluation.bad_rules); every rule gets
its own review curve. Spearman rho and the quality curve do not depend on the rule.
"""

import numpy as np
from scipy.stats import rankdata, spearmanr


def bad_mask(dice: np.ndarray, rule: str, value: float) -> np.ndarray:
    """Which scans count as 'bad' (evaluation.bad in the config).

    - threshold:      Dice below 'value'.
    - worst_fraction: the round(value * n) scans with the lowest Dice (at least one).
    """
    if rule == "threshold":
        return dice < value
    if rule == "worst_fraction":
        k = max(1, int(round(value * len(dice))))
        bad = np.zeros(len(dice), dtype=bool)
        bad[np.argsort(dice, kind="stable")[:k]] = True
        return bad
    raise ValueError(f"Unknown 'bad' rule '{rule}'")


def combine_scores(scores: dict[str, np.ndarray], spec: dict) -> np.ndarray:
    """Merge several GT-free scores into one by their ranks (no fitted weights, so nothing to overfit).

    Each score is turned into its rank among the scans being reviewed, scaled to 0-1 (1 = most suspicious).
    - mean_rank: average of the ranks. A scan must be suspicious on both scores to reach the top.
    - max_rank:  the highest of the ranks. A scan reaches the top if EITHER score finds it suspicious,
                 e.g. a confident miss that only the plausibility check notices.

    Input: {score name: values}, and one entry of evaluation.combinations (rule, scores).
    Output: the combined score per scan (higher = review first).
    """
    ranks = np.array([rankdata(scores[name]) / len(scores[name]) for name in spec["scores"]])
    if spec["rule"] == "mean_rank":
        return ranks.mean(axis=0)
    if spec["rule"] == "max_rank":
        return ranks.max(axis=0)
    raise ValueError(f"Unknown combination rule '{spec['rule']}'")


def review_order(score: np.ndarray) -> np.ndarray:
    """Indices of the scans from most to least uncertain (highest score first). Ties keep list order."""
    return np.argsort(-score, kind="stable")


def expected_top_k_sums(score: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum of 'values' over the k highest-scoring scans, for k = 0..n, averaged over random tie-breaking.

    If several scans have the same score (e.g. a yes/no plausibility flag), their order in the queue is
    arbitrary. Instead of letting the file order decide, we use the expected result over all orders:
    reviewing j of the m tied scans gives, on average, j/m of their summed value. For scores without
    ties this is simply the running sum in score order.
    """
    order = np.argsort(-score, kind="stable")
    s_sorted, v_sorted = score[order], values[order].astype(float)
    out = np.zeros(len(score) + 1)
    start = 0
    while start < len(score):
        end = start
        while end < len(score) and s_sorted[end] == s_sorted[start]:
            end += 1
        group_total = v_sorted[start:end].sum()
        size = end - start
        for j in range(1, size + 1):
            out[start + j] = out[start] + group_total * j / size
        start = end
    return out


def review_curve_from_score(score: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """y-values of the review curve for x = 0, 1/n, ..., 1 (share of bad scans found), ties handled fairly."""
    return expected_top_k_sums(score, bad) / max(bad.sum(), 1)


def quality_curve_from_score(score: np.ndarray, dice: np.ndarray) -> np.ndarray:
    """y-values of the quality curve (mean Dice after correcting the top k to Dice 1), ties handled fairly."""
    return (dice.sum() + expected_top_k_sums(score, 1.0 - dice)) / len(dice)


def review_curve(order: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Review curve for a fixed review order (used for the random baseline)."""
    found = np.concatenate([[0], np.cumsum(bad[order])])
    return found / max(bad.sum(), 1)


def quality_curve(order: np.ndarray, dice: np.ndarray) -> np.ndarray:
    """Quality curve for a fixed review order (used for the random baseline)."""
    n = len(dice)
    gain = np.concatenate([[0.0], np.cumsum(1.0 - dice[order])])  # correcting a scan raises its Dice to 1
    return (dice.sum() + gain) / n


def area(y: np.ndarray) -> float:
    """Area under a curve sampled at x = 0, 1/n, ..., 1 (trapezoid rule)."""
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.trapz(y, x))


def curves_for(score: np.ndarray, dice: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Review and quality curve for one ranking score (expected curves if the score has ties)."""
    return review_curve_from_score(score, bad), quality_curve_from_score(score, dice)


def random_curves(dice: np.ndarray, bad: np.ndarray, n_orders: int, rng: np.random.Generator):
    """Average review and quality curves over many random review orders, plus the AUC of each order."""
    reviews, qualities = [], []
    for _ in range(n_orders):
        order = rng.permutation(len(dice))
        reviews.append(review_curve(order, bad))
        qualities.append(quality_curve(order, dice))
    reviews, qualities = np.array(reviews), np.array(qualities)
    aucs = (np.array([area(r) for r in reviews]), np.array([area(q) for q in qualities]))
    return reviews.mean(axis=0), qualities.mean(axis=0), aucs


def partial_spearman(score: np.ndarray, dice: np.ndarray, volume: np.ndarray) -> float:
    """Spearman correlation of score and Dice with the effect of predicted volume removed (see module docstring)."""
    r_sd = spearmanr(score, dice).statistic
    r_sv = spearmanr(score, volume).statistic
    r_dv = spearmanr(dice, volume).statistic
    # If the score is (almost) the volume itself, nothing is left after removing volume: undefined.
    if abs(r_sv) > 1.0 - 1e-9 or abs(r_dv) > 1.0 - 1e-9:
        return float("nan")
    return float((r_sd - r_sv * r_dv) / np.sqrt((1.0 - r_sv ** 2) * (1.0 - r_dv ** 2)))


def _ci(values: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile interval of the finite values (e.g. level 0.95 -> 2.5 % and 97.5 %)."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))


def evaluate_methods(scores: dict[str, np.ndarray], dice: np.ndarray, cfg: dict,
                     bad_rule: dict) -> tuple[list[dict], dict]:
    """Evaluate every ranking method on one set of scans, for one definition of "bad".

    Input: {method name: score per scan} (GT-free scores; must include 'neg_volume_ml', whose
           negative is the predicted volume), true Dice per scan, the config (evaluation, seed) and one
           entry of evaluation.bad_rules.
    Output: (one row of metrics per method, {method: (review curve, quality curve)} for plotting).
            The rows include 'random' and 'oracle'.
    """
    ev = cfg["evaluation"]
    rng = np.random.default_rng(cfg["seed"])
    n = len(dice)
    volume = -scores["neg_volume_ml"]
    bad = bad_mask(dice, bad_rule["rule"], bad_rule["value"])

    # Random baseline: average curve; its CI is the spread over random orders.
    rand_review, rand_quality, (rand_r_aucs, rand_q_aucs) = random_curves(dice, bad, ev["n_random_orders"], rng)
    curves = {"random": (rand_review, rand_quality)}

    # The oracle ranks by true Dice: lowest Dice = highest "score".
    all_scores = dict(scores)
    all_scores["oracle"] = -dice
    for name, s in all_scores.items():
        curves[name] = curves_for(s, dice, bad)

    # Bootstrap: the same resampled scan sets are used for every method, so differences are paired.
    boot_idx = [rng.integers(0, n, size=n) for _ in range(ev["n_bootstrap"])]

    def boot_stats(s: np.ndarray) -> dict[str, np.ndarray]:
        rho, prho, r_auc, q_auc = [], [], [], []
        for idx in boot_idx:
            d, b, sc = dice[idx], bad[idx], s[idx]
            ok = np.ptp(sc) > 0 and np.ptp(d) > 0
            rho.append(spearmanr(sc, d).statistic if ok else np.nan)
            prho.append(partial_spearman(sc, d, volume[idx]) if ok and np.ptp(volume[idx]) > 0 else np.nan)
            r, q = curves_for(sc, d, b) if b.any() else (np.full(n + 1, np.nan), quality_curve_from_score(sc, d))
            r_auc.append(area(r))
            q_auc.append(area(q))
        return {"rho": np.array(rho), "partial_rho": np.array(prho),
                "review": np.array(r_auc), "quality": np.array(q_auc)}

    boots = {name: boot_stats(s) for name, s in all_scores.items()}
    rand_r, rand_q = area(rand_review), area(rand_quality)
    orac_r, orac_q = area(curves["oracle"][0]), area(curves["oracle"][1])

    rows = []
    level = ev["ci_level"]
    for name, s in all_scores.items():
        rev, qual = curves[name]
        rho = spearmanr(s, dice)
        b = boots[name]
        row = {
            "method": name,
            "spearman_rho": rho.statistic, "rho_ci_low": _ci(b["rho"], level)[0], "rho_ci_high": _ci(b["rho"], level)[1],
            "rho_p_value": rho.pvalue,
            "partial_rho_given_volume": partial_spearman(s, dice, volume),
            "partial_rho_ci_low": _ci(b["partial_rho"], level)[0], "partial_rho_ci_high": _ci(b["partial_rho"], level)[1],
            "review_auc": area(rev), "review_auc_ci_low": _ci(b["review"], level)[0],
            "review_auc_ci_high": _ci(b["review"], level)[1],
            "review_auc_norm": (area(rev) - rand_r) / (orac_r - rand_r) if orac_r > rand_r else np.nan,
            "quality_auc": area(qual), "quality_auc_ci_low": _ci(b["quality"], level)[0],
            "quality_auc_ci_high": _ci(b["quality"], level)[1],
            "quality_auc_norm": (area(qual) - rand_q) / (orac_q - rand_q) if orac_q > rand_q else np.nan,
        }
        # Paired difference to the volume baseline (positive = better than volume).
        for curve in ("review", "quality"):
            diff = b[curve] - boots["neg_volume_ml"][curve]
            row[f"{curve}_auc_minus_volume"] = row[f"{curve}_auc"] - area(curves["neg_volume_ml"][0 if curve == "review" else 1])
            row[f"{curve}_auc_minus_volume_ci_low"], row[f"{curve}_auc_minus_volume_ci_high"] = _ci(diff, level)
        rows.append(row)

    rows.append({
        # Spearman is not defined for an average of random orders (its expected value is 0).
        "method": "random", "spearman_rho": np.nan, "rho_ci_low": np.nan, "rho_ci_high": np.nan, "rho_p_value": np.nan,
        "review_auc": rand_r, "review_auc_ci_low": _ci(rand_r_aucs, level)[0], "review_auc_ci_high": _ci(rand_r_aucs, level)[1],
        "review_auc_norm": 0.0,
        "quality_auc": rand_q, "quality_auc_ci_low": _ci(rand_q_aucs, level)[0], "quality_auc_ci_high": _ci(rand_q_aucs, level)[1],
        "quality_auc_norm": 0.0,
    })
    return rows, curves
