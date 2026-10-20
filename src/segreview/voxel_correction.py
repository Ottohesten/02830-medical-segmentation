"""Simulated correction inside a scan: does fixing the most uncertain voxels first help? (EXPLORATORY, after G1)

G1 measured the quality curve at scan level: review whole scans in queue order and fix them completely. This module
asks the same question one level down, inside a single scan, and so tests whether the heatmap points a reviewer to
the voxels that are worth fixing. It is a simulation with ground truth as a perfect reviewer, so evaluation only.

For every scan:
1. Take the region the heatmap is shown in: the predicted organ plus uncertainty.border_mm around it (the same
   region as in G1 and the heatmap evaluation).
2. Order the region's voxels by a method (highest first, see below).
3. "Correct" the first k % of them: set the mask to the ground truth there. Voxels that were already right do not
   change; only the wrong ones among them are fixed.
4. Compute Dice of the corrected mask against the ground truth over the whole scan. Gain = Dice after - Dice before.

Methods (voxel orders):
  entropy   the locked G2 heatmap: binary entropy of the 3 mm model's organ probability (GT-free)
  m2_diff   |p_3mm - p_6mm|: where the two models disagree (GT-free)
  boundary  closeness to the edge of the predicted mask (reference: "errors are at the edge", no model uncertainty)
  random    a random order (average over several seeds)
  oracle    the wrong voxels first (uses ground truth: the best any order can do)

Ties: many voxels share a value (the maps are stored with 256 levels, and most of the region has entropy 0). When the
cut-off at k falls inside a group of tied voxels, we take the share of that group that fits and count its wrong
voxels in proportion, i.e. the expected result of breaking the ties at random. This is the same rule as the top-X %
capture in heatmap_eval.py and the tied scores in the G1 curves.

How Dice changes: with TP, FP, FN the true positives, false positives and false negatives of the mask,
    Dice = 2 TP / (2 TP + FP + FN).
Fixing a missed voxel (a FN) turns it into a TP; fixing a false voxel (a FP) removes it. After fixing a FN and b FP:
    Dice_after = 2 (TP + a) / (2 TP + FP + FN + a - b).
So only the counts of fixed FN and FP voxels are needed, not a new mask.

Errors outside the region (e.g. a whole kidney the model missed) can never be fixed this way. Correcting the whole
region (k = 100 %) gives the best reachable Dice; the gap that remains is reported as the unfixable share.
"""

import numpy as np

from segreview.heatmap_eval import boundary_map
from segreview.uncertainty import uncertainty_region

GT_FREE = ["entropy", "m2_diff"]          # methods that a real reviewer could see (no ground truth)


def corrected_dice(tp: float, fp: float, fn: float, fixed_fn: float = 0.0, fixed_fp: float = 0.0) -> float:
    """Dice after fixing 'fixed_fn' missed voxels and 'fixed_fp' false voxels (see module docstring). 1 if all empty."""
    denom = 2 * tp + fp + fn + fixed_fn - fixed_fp
    return float(2 * (tp + fixed_fn) / denom) if denom > 0 else 1.0


def top_k_counts(values: np.ndarray, flags: dict[str, np.ndarray], ks: list[float]) -> dict[str, np.ndarray]:
    """How many flagged voxels lie among the k voxels with the highest values, for several k at once.

    Ties at the cut-off are shared in proportion (the expected count if ties were broken at random).

    Input: one value per voxel, {name: boolean flag per voxel} (e.g. fn, fp), the k values (numbers of voxels,
           may be fractional, 0 <= k <= number of voxels).
    Output: {name: array with one expected count per k}.
    """
    out = {name: np.zeros(len(ks)) for name in flags}
    if len(values) == 0:
        return out
    # Group the voxels by value, highest value first. For each group: its size and how many voxels are flagged.
    uniq, inverse = np.unique(-values, return_inverse=True)     # ascending in -value = descending in value
    sizes = np.bincount(inverse, minlength=len(uniq)).astype(float)
    before = np.concatenate([[0.0], np.cumsum(sizes)])            # voxels in all higher groups
    group_flags = {name: np.bincount(inverse, weights=f.astype(float), minlength=len(uniq)) for name, f in flags.items()}
    cum_flags = {name: np.concatenate([[0.0], np.cumsum(g)]) for name, g in group_flags.items()}
    for i, k in enumerate(ks):
        k = min(max(k, 0.0), before[-1])
        g = min(int(np.searchsorted(before, k, side="right")) - 1, len(uniq) - 1)  # group the cut-off falls in
        part = (k - before[g]) / sizes[g]                          # share of that group that is taken
        for name in flags:
            out[name][i] = cum_flags[name][g] + part * group_flags[name][g]
    return out


def method_values(method: str, maps: dict[str, np.ndarray], mask: np.ndarray, error: np.ndarray,
                  spacing: tuple[float, float, float], region: np.ndarray, seed: int | None = None) -> np.ndarray:
    """The value that orders the region's voxels for one method (higher = corrected earlier).

    Input: method name, saved heatmaps ({'entropy': ..., 'm2_diff': ...}), predicted mask, error mask (only for the
           oracle), voxel size, region, and the seed for 'random'.
    Output: 1-D array, one value per region voxel (in the order of region's True voxels).
    """
    if method in maps:
        return maps[method][region].astype(np.float64)
    if method == "boundary":
        return boundary_map(mask, region, spacing)[region].astype(np.float64)
    if method == "random":
        return np.random.default_rng(seed).random(int(region.sum()))
    if method == "oracle":
        return error[region].astype(np.float64)
    raise ValueError(f"Unknown method '{method}'")


def correction_curves(maps: dict[str, np.ndarray], mask: np.ndarray, truth: np.ndarray,
                      spacing: tuple[float, float, float], border_mm: float, k_percent: list[float],
                      methods: list[str], random_seeds: list[int]) -> tuple[dict, list[dict]]:
    """Simulate correcting the top k % of the region for every method and k (see module docstring).

    Input: heatmaps, predicted mask, ground-truth organ mask (same grid), voxel size (mm), border of the region (mm),
           the k values in %, the methods, and the seeds for the random order.
    Output: (case info, rows). Case info: Dice before, error voxels, the share of them outside the region, the best
            Dice reachable inside the region and the unfixable share of the Dice gap. Rows: one per method and k with
            dice_after and gain (Dice after - Dice before); for 'random' the mean over the seeds.
    """
    mask, truth = mask.astype(bool), truth.astype(bool)
    fn, fp = truth & ~mask, mask & ~truth
    tp, n_fp, n_fn = int((mask & truth).sum()), int(fp.sum()), int(fn.sum())
    dice_before = corrected_dice(tp, n_fp, n_fn)

    # Work in a box around the mask (plus the border) so the maps of large scans stay small in memory.
    region_full = uncertainty_region(mask, spacing, border_mm)
    in_fn, in_fp = int(fn[region_full].sum()), int(fp[region_full].sum())
    dice_ceiling = corrected_dice(tp, n_fp, n_fn, in_fn, in_fp)   # every region voxel corrected
    gap = 1.0 - dice_before
    info = {"dice_before": dice_before, "error_voxels": n_fn + n_fp, "fn_voxels": n_fn, "fp_voxels": n_fp,
            "region_voxels": int(region_full.sum()),
            "error_outside_region_share": (n_fn + n_fp - in_fn - in_fp) / (n_fn + n_fp) if n_fn + n_fp else np.nan,
            "dice_ceiling": dice_ceiling,
            "unfixable_dice_gap_share": (1.0 - dice_ceiling) / gap if gap > 0 else np.nan}

    rows = []
    n_region = info["region_voxels"]
    ks = [p / 100.0 * n_region for p in k_percent]
    if n_region == 0:    # empty prediction: nothing to show a heatmap on, nothing can be corrected
        return info, [{"method": m, "k_percent": p, "dice_after": dice_before, "gain": 0.0}
                      for m in methods for p in k_percent]
    idx = np.nonzero(region_full)
    box = tuple(slice(i.min(), i.max() + 1) for i in idx)
    region, m_box = region_full[box], mask[box]
    flags = {"fn": fn[box][region], "fp": fp[box][region]}
    error = (fn | fp)[box]
    maps_box = {name: m[box] for name, m in maps.items()}
    for method in methods:
        seeds = random_seeds if method == "random" else [None]
        dice_after = np.zeros(len(ks))
        for seed in seeds:
            values = method_values(method, maps_box, m_box, error, spacing, region, seed)
            counts = top_k_counts(values, flags, ks)
            dice_after += [corrected_dice(tp, n_fp, n_fn, a, b) for a, b in zip(counts["fn"], counts["fp"])]
        dice_after /= len(seeds)
        rows += [{"method": method, "k_percent": p, "dice_after": d, "gain": d - dice_before}
                 for p, d in zip(k_percent, dice_after)]
    return info, rows


def gain_auc(k_percent: list[float], gains: list[float], up_to: float) -> float:
    """Area under the gain-vs-k curve from k = 0 (gain 0) up to 'up_to' %, divided by that width.

    Dividing by the width makes it the average Dice gain over that range of k, so areas over different ranges can
    be compared. Uses the trapezoid rule on the given k values; 'up_to' must be one of them.
    """
    if up_to not in k_percent:
        raise ValueError(f"auc_up_to_percent {up_to} must be one of k_percent {k_percent}")
    k = np.array([0.0] + [p for p in k_percent if p <= up_to])
    g = np.array([0.0] + [gn for p, gn in zip(k_percent, gains) if p <= up_to])
    return float(np.sum((k[1:] - k[:-1]) * (g[1:] + g[:-1]) / 2) / up_to)
