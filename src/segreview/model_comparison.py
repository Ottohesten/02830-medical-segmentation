"""How good is the 6 mm model on its own, and where do the 3 mm and 6 mm masks disagree? (EXPLORATORY, after G1)

The locked G1 score ranks scans by how much the detailed 3 mm model and the coarse 6 mm model disagree. This
module asks two follow-up questions from the supervisor meeting of 2026-10-07. Both use ground truth, so they are
evaluation only, and both were asked after the G1 choices were locked, so they are exploratory.

1. The 6 mm model alone. Dice of each model against ground truth, per scan. If the "second opinion" model is
   often wrong itself, some scans at the top of the queue are there because the 6 mm model failed, not the
   3 mm model the reviewer corrects (a "false alarm").

2. Where do the two masks disagree? For every voxel where one model says organ and the other does not:
   - its ground-truth label (kidney, tumor, cyst or background);
   - its distance to the edge of the 3 mm mask. The 6 mm model works on a grid of 6 mm voxels, so even if it
     agreed perfectly with the 3 mm model about the organ, its edge would be off by up to about one 6 mm voxel.
     Disagreement within that band (grid_band_mm) is called the grid effect; disagreement farther away is a real
     difference in shape or content (e.g. a tumor one model includes and the other leaves out);
   - which model is right there (exactly one of them is, because they disagree and the truth is 0 or 1).

Scan groups (with the "bad" threshold of the locked G1 evaluation, e.g. Dice < 0.83):
   bad          the 3 mm mask is bad (a real error the queue should find)
   false_alarm  the 3 mm mask is fine, but the 6 mm mask is bad
   good         both masks are fine
"""

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

GROUPS = ["bad", "false_alarm", "good"]
MODEL_A, MODEL_B = "3mm", "6mm"   # A = the main model whose mask the reviewer corrects; B = the second model


def case_group(dice_a: float, dice_b: float, threshold: float) -> str:
    """Put a scan in one of the GROUPS (see module docstring) from the two models' Dice and the 'bad' threshold."""
    if dice_a < threshold:
        return "bad"
    if dice_b < threshold:
        return "false_alarm"
    return "good"


def model_dice_summary(df: pd.DataFrame, threshold: float) -> dict:
    """Compare the two models' Dice over a set of scans.

    Input: DataFrame with columns dice_3mm and dice_6mm (one row per scan), the 'bad' threshold.
    Output: dict with median and mean Dice per model, how many scans are below the threshold per model, in how
            many scans each model is better (equal Dice counted separately), and Spearman's rho between the two
            models' Dice (do they find the same scans hard?).
    """
    a, b = df["dice_3mm"].to_numpy(), df["dice_6mm"].to_numpy()
    rho, p = spearmanr(a, b) if len(df) > 2 else (np.nan, np.nan)
    return {"n": len(df),
            "median_3mm": np.median(a), "median_6mm": np.median(b),
            "mean_3mm": a.mean(), "mean_6mm": b.mean(),
            "n_below_3mm": int((a < threshold).sum()), "n_below_6mm": int((b < threshold).sum()),
            "n_3mm_better": int((a > b).sum()), "n_6mm_better": int((b > a).sum()), "n_equal": int((a == b).sum()),
            "spearman_rho": float(rho), "spearman_p": float(p)}


def queue_table(scores: pd.DataFrame, dice: pd.DataFrame, score_col: str, top_n: int, threshold: float) -> pd.DataFrame:
    """The top of the review queue with both models' Dice.

    The queue is the order the locked score gives (highest = most uncertain = reviewed first).
    Input: scores (case_id + score_col), Dice per scan (case_id, dice_3mm, dice_6mm), how many rows, the threshold.
    Output: one row per queue position: rank (1 = top), case_id, score, dice_3mm, dice_6mm, group, and
            finding = "real find" (3 mm bad), "false alarm" (3 mm fine, 6 mm bad) or "" (both fine).
    """
    q = scores[["case_id", score_col]].merge(dice[["case_id", "dice_3mm", "dice_6mm"]], on="case_id")
    q = q.sort_values(score_col, ascending=False, kind="stable").head(top_n).reset_index(drop=True)
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    q["group"] = [case_group(a, b, threshold) for a, b in zip(q.dice_3mm, q.dice_6mm)]
    q["finding"] = q["group"].map({"bad": "real find", "false_alarm": "false alarm", "good": ""})
    return q.rename(columns={score_col: "score"})


def edge_distance_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """For every voxel, the distance (mm) to the edge of the mask.

    Outside the mask: the distance to the nearest mask voxel. Inside: the distance to the nearest voxel outside.
    Distances run between voxel centers, so a voxel right next to the edge has distance = one voxel size (> 0).
    If the mask is empty there is no edge and every distance is infinite.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return np.full(mask.shape, np.inf)
    outside = distance_transform_edt(~mask, sampling=spacing)
    inside = distance_transform_edt(mask, sampling=spacing) if (~mask).any() else np.full(mask.shape, np.inf)
    return np.where(mask, inside, outside)


def distance_bin_names(edges_mm: list[float]) -> list[str]:
    """Readable names of the distance bins, e.g. [3, 6] -> ['0-3 mm', '3-6 mm', '>6 mm']."""
    lows = [0] + list(edges_mm)
    names = [f"{lo:g}-{hi:g} mm" for lo, hi in zip(lows[:-1], edges_mm)]
    return names + [f">{edges_mm[-1]:g} mm"]


def _bounding_box(arrays: list[np.ndarray], pad: int) -> tuple[slice, ...] | None:
    """Slices of the smallest box holding every True voxel of the arrays, plus 'pad' voxels on each side."""
    union = np.logical_or.reduce(arrays)
    if not union.any():
        return None
    idx = np.nonzero(union)
    return tuple(slice(max(0, i.min() - pad), min(n, i.max() + 1 + pad)) for i, n in zip(idx, union.shape))


def disagreement_counts(mask_a: np.ndarray, mask_b: np.ndarray, labels: np.ndarray, organ_labels: list[int],
                        ignore_labels: list[int], label_names: dict[int, str], spacing: tuple[float, float, float],
                        edges_mm: list[float]) -> pd.DataFrame:
    """Count the voxels where the two masks disagree, broken down three ways (see module docstring).

    Input: mask of model A (3 mm) and model B (6 mm), the ground-truth label map (all on the same voxel grid),
           which label values form the organ and which are ignored, names of the label values, voxel size (mm),
           and the upper edges of the distance bins (mm, distance to the edge of mask A).
    Output: DataFrame with one row per combination that occurs: label, distance_bin, right_model (which model
            agrees with the ground truth there), n_voxels. Empty if the masks agree everywhere.
    Ignored voxels are left out, as in the Dice computation.
    """
    mask_a, mask_b = mask_a.astype(bool), mask_b.astype(bool)
    disagree = mask_a != mask_b
    if ignore_labels:
        disagree &= ~np.isin(labels, ignore_labels)
    columns = ["label", "distance_bin", "right_model", "n_voxels"]
    if not disagree.any():
        return pd.DataFrame(columns=columns)

    # The distance transform only runs on a box around mask A and the disagreement, to save time and memory.
    # Two extra voxels of padding make sure the box border is outside mask A, so inside distances are correct.
    box = _bounding_box([mask_a, disagree], pad=2)
    dist = edge_distance_mm(mask_a[box], spacing)[disagree[box]]
    lab = labels[box][disagree[box]]
    truth = np.isin(lab, organ_labels)
    a_right = mask_a[box][disagree[box]] == truth     # where they disagree, exactly one model is right

    names = distance_bin_names(edges_mm)
    bin_idx = np.digitize(dist, edges_mm, right=True)  # d <= edge goes into that edge's bin; inf into the last
    df = pd.DataFrame({"label": [label_names.get(int(v), f"label_{int(v)}") for v in lab],
                       "distance_bin": np.array(names)[bin_idx],
                       "right_model": np.where(a_right, MODEL_A, MODEL_B)})
    return df.value_counts().rename("n_voxels").reset_index()[columns]


def disagreement_shares(counts: pd.DataFrame, grid_band_mm: float, edges_mm: list[float],
                        label_names: dict[int, str]) -> dict:
    """Summarize disagreement counts (one scan, or several pooled) as shares.

    Input: rows of disagreement_counts (columns label, distance_bin, right_model, n_voxels), the grid band (mm,
           one of edges_mm), the bin edges and the label names.
    Output: dict with
      n_disagree                   number of disagreeing voxels
      near_share                   share within grid_band_mm of the 3 mm edge (the grid effect)
      dist_<bin>_share             share in each distance bin
      label_<name>_share           share per ground-truth label
      far_label_<name>_share       share per label among the voxels farther than grid_band_mm
      right_3mm_share              share where the 3 mm model is right (near_/far_ for the two distance groups)
    Shares are NaN when there are no voxels to divide by.
    """
    if grid_band_mm not in edges_mm:
        raise ValueError(f"grid_band_mm ({grid_band_mm}) must be one of distance_bins_mm {edges_mm}")
    names = distance_bin_names(edges_mm)
    near_bins = set(names[:edges_mm.index(grid_band_mm) + 1])
    n = counts["n_voxels"].to_numpy(dtype=float)
    near = counts["distance_bin"].isin(near_bins).to_numpy()
    a_right = (counts["right_model"] == MODEL_A).to_numpy()
    total = n.sum()

    def share(selected: np.ndarray, within: np.ndarray | None = None) -> float:
        base = n.sum() if within is None else n[within].sum()
        keep = selected if within is None else selected & within
        return float(n[keep].sum() / base) if base > 0 else np.nan

    row = {"n_disagree": int(total), "near_share": share(near)}
    for b in names:
        row[f"dist_{b}_share"] = share((counts["distance_bin"] == b).to_numpy())
    for name in label_names.values():
        is_label = (counts["label"] == name).to_numpy()
        row[f"label_{name}_share"] = share(is_label)
        row[f"far_label_{name}_share"] = share(is_label, ~near)
    row["right_3mm_share"] = share(a_right)
    row["near_right_3mm_share"] = share(a_right, near)
    row["far_right_3mm_share"] = share(a_right, ~near)
    return row
