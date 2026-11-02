"""G3: the learning curves and their summary, as fixed in docs/g3_plan.md ("Design: learning curve", "What is
reported") before any G3 model was trained. Used by scripts/g3_curves.py.

- Curve: for each order (queue, random seeds) and each k, the mean Dice over the scans of one split. k = 0 is the
  original model in every order.
- Primary summary: the mean over the summary's k values of (queue curve - mean of the random curves), on the test set.
  It is a weighted sum of the models' Dice, so it can be written as a mean over scans of one number per scan; the
  bootstrap resamples the scans (the same scans for every model) and recomputes that mean.
- Training noise: the spread of the random seeds' curves at each k (their max - min).
"""

import numpy as np
import pandas as pd

ORIGINAL = "original"


def curve_points(points: pd.DataFrame, orders: list[str]) -> pd.DataFrame:
    """The points of the curves including k = 0 (the original model) for every order. Columns: order, k, run_id."""
    zero = pd.DataFrame({"order": orders, "k": 0, "run_id": ORIGINAL})
    return pd.concat([zero, points], ignore_index=True).sort_values(["order", "k"], ignore_index=True)


def dice_matrix(dice: pd.DataFrame, models: list[str], split: str, column: str) -> pd.DataFrame:
    """Dice per scan (rows) and model (columns) for one split, for the scans every listed model has."""
    rows = dice[(dice.split == split) & dice.model.isin(models)]
    matrix = rows.pivot_table(index="case_id", columns="model", values=column, aggfunc="first")
    missing = [m for m in models if m not in matrix.columns]
    if missing:
        raise ValueError(f"No {split} Dice for {missing}")
    return matrix[list(dict.fromkeys(models))].dropna()


def curves(matrix: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Mean Dice per point (order, k) over the matrix's scans, with the number of scans."""
    out = points.copy()
    out["mean_dice"] = [matrix[m].mean() for m in out.run_id]
    out["n_scans"] = len(matrix)
    return out


def summary_weights(points: pd.DataFrame, queue: str, random_orders: list[str], ks: list[int]) -> pd.Series:
    """Weight per model so that summary = sum over models of weight x mean Dice: +1/K for the queue's model at each of
    the K values of k, -1/(K x number of random orders) for each random order's model (models shared by several
    points add up their weights)."""
    by_point = points.set_index(["order", "k"]).run_id
    weights = pd.Series(0.0, index=pd.Index(points.run_id.unique()))
    for k in ks:
        weights[by_point[(queue, k)]] += 1 / len(ks)
        for order in random_orders:
            weights[by_point[(order, k)]] -= 1 / (len(ks) * len(random_orders))
    return weights


def summary_with_ci(matrix: pd.DataFrame, weights: pd.Series, n_bootstrap: int, seed: int,
                    ci_level: float) -> dict:
    """The summary (mean over scans of each scan's weighted Dice) with a percentile bootstrap interval over scans."""
    per_scan = (matrix[weights.index] * weights).sum(axis=1).to_numpy()
    rng = np.random.default_rng(seed)
    means = per_scan[rng.integers(0, len(per_scan), (n_bootstrap, len(per_scan)))].mean(axis=1)
    alpha = (1 - ci_level) / 2
    return {"estimate": float(per_scan.mean()), "ci_low": float(np.quantile(means, alpha)),
            "ci_high": float(np.quantile(means, 1 - alpha)), "n_scans": len(per_scan), "n_bootstrap": n_bootstrap}


def seed_spread(curve: pd.DataFrame, random_orders: list[str]) -> pd.DataFrame:
    """Training and order noise: at each k, the min, max and spread of the random orders' mean Dice."""
    rand = curve[curve.order.isin(random_orders)]
    out = rand.groupby("k").mean_dice.agg(["min", "max", "mean"]).reset_index()
    out["spread"] = out["max"] - out["min"]
    return out
