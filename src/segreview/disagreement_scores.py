"""Scan scores from only the REAL differences between the 3 mm and 6 mm masks (EXPLORATORY, after G1).

The locked G1 score (model_diff_mean_region) averages |p_3mm - p_6mm| over the region. The follow-up of 2026-10-19
showed that about 89 % of the voxels where the two masks disagree lie within 6 mm of the 3 mm mask's edge: that is
the grid effect (the 6 mm model works on 6 mm voxels, so its edge is off by up to one voxel even when it agrees).
The rest are real differences, mostly tumor. In bad scans the 6 mm model often includes tumor the 3 mm model missed;
in false alarms it leaves out tumor the 3 mm model included.

So here we count only part of the disagreement, and ask if that ranks scans better. Candidates (config
exploratory.disagreement_scores.candidates), all GT-free (only the two saved masks are used):
  far_disagree    masks disagree, farther than grid_band_mm from the 3 mm edge (both directions)
  far_6mm_adds    as far_disagree, only where 6 mm = organ and 3 mm = not organ  (the hypothesis)
  far_6mm_drops   as far_disagree, only where 6 mm = not organ and 3 mm = organ  (control, expected poor)
  6mm_adds_all    6 mm = organ and 3 mm = not organ, at any distance             (is the distance filter needed?)
Each comes in two forms (normalizations):
  <name>_ml           volume of those voxels in ml
  <name>_per_volume   their number divided by the number of voxels in the 3 mm mask (a share of the organ, so a
                      big kidney does not get a high score just for being big)
Sign convention as in G1: higher score = more suspicious = reviewed first, so a good score has NEGATIVE Spearman rho
with Dice. Ground truth is only used in the evaluation, never in a score.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from segreview.evaluation import expected_top_k_sums
from segreview.model_comparison import distance_bin_names

# Candidate name -> the count of directional_disagreement it is built from.
CANDIDATE_COUNTS = {"far_disagree": "far_disagree", "far_6mm_adds": "far_b_adds",
                    "far_6mm_drops": "far_b_drops", "6mm_adds_all": "b_adds_all"}


def score_names(candidates: list[str], normalizations: list[str]) -> list[str]:
    """Column names of the scores, in the order of the config (candidate first, then normalization)."""
    return [f"{c}_{n}" for c in candidates for n in normalizations]


def candidate_scores(counts: dict[str, int], voxel_ml: float, candidates: list[str],
                     normalizations: list[str]) -> dict[str, float]:
    """Turn the voxel counts of one scan into the candidate scores.

    Input: output of model_comparison.directional_disagreement, the volume of one voxel (ml), the candidate names
           and the normalizations (ml and/or per_volume).
    Output: {score name: value}. per_volume divides by max(n_mask_a, 1), so an empty 3 mm mask does not divide by 0
            (as entropy_per_volume in G1).
    """
    out = {}
    for c in candidates:
        n = counts[CANDIDATE_COUNTS[c]]
        for norm in normalizations:
            if norm == "ml":
                out[f"{c}_ml"] = n * voxel_ml
            elif norm == "per_volume":
                out[f"{c}_per_volume"] = n / max(counts["n_mask_a"], 1)
            else:
                raise ValueError(f"Unknown normalization '{norm}'")
    return out


def counts_from_breakdown(rows: pd.DataFrame, grid_band_mm: float, edges_mm: list[float],
                          organ_names: set[str]) -> dict[str, int]:
    """Rebuild the directional counts of one scan from the earlier GT breakdown (model_disagreement_counts.csv).

    Used ONLY as a consistency check of the new GT-free counts against the saved result of 2026-10-19. The
    direction follows from the label and which model is right: "6 mm says organ, 3 mm says not" happens where
    the truth is organ and 6 mm is right, or the truth is background and 3 mm is right.
    Input: the scan's rows (label, distance_bin, right_model, n_voxels), grid band, bin edges, organ label names.
    Output: dict with far_disagree, far_b_adds, far_b_drops, b_adds_all (voxel counts).
    """
    names = distance_bin_names(edges_mm)
    near_bins = set(names[:edges_mm.index(grid_band_mm) + 1])
    far = ~rows["distance_bin"].isin(near_bins)
    is_organ = rows["label"].isin(organ_names)
    b_adds = (is_organ & (rows["right_model"] == "6mm")) | (~is_organ & (rows["right_model"] == "3mm"))
    n = rows["n_voxels"]
    return {"far_disagree": int(n[far].sum()), "far_b_adds": int(n[far & b_adds].sum()),
            "far_b_drops": int(n[far & ~b_adds].sum()), "b_adds_all": int(n[b_adds].sum())}


def choose_score(metrics: pd.DataFrame, names: list[str], metric: str, tie_break: str) -> pd.DataFrame:
    """Rank the candidate scores by the pre-registered criterion and mark the one chosen.

    Input: metric rows of ONE split and ONE bad rule (columns method, metric, tie_break), the candidate names in
           config order, the criterion column (e.g. review_auc) and the tie-break column (e.g. quality_auc).
    Output: one row per candidate, best first: method, metric, tie_break, config_order, chosen (True for one row).
    Ties: higher tie_break wins, then the config order. Values are compared after rounding to 10 decimals, so
    two scores giving the same curve count as tied even with floating-point noise.
    """
    table = metrics[metrics.method.isin(names)][["method", metric, tie_break]].copy()
    table["config_order"] = table.method.map({n: i for i, n in enumerate(names)})
    table["_m"], table["_t"] = table[metric].round(10), table[tie_break].round(10)
    table = table.sort_values(["_m", "_t", "config_order"], ascending=[False, False, True]).drop(columns=["_m", "_t"])
    table["chosen"] = np.arange(len(table)) == 0
    return table.reset_index(drop=True)


def top_counts(score: np.ndarray, groups: np.ndarray, top_n: int) -> dict[str, float]:
    """Expected number of scans per group among the top_n of the queue a score gives.

    Groups are model_comparison.GROUPS (bad = real find, false_alarm, good). If scans tie at the cut-off (e.g.
    several scans with no far disagreement at all), the tied scans share the remaining places in proportion, i.e.
    the expected result of breaking the tie at random (the same rule as the G1 curves).
    Output: {"real_finds": ..., "false_alarms": ..., "both_fine": ...}.
    """
    k = min(top_n, len(score))
    return {key: float(expected_top_k_sums(score, groups == g)[k])
            for key, g in [("real_finds", "bad"), ("false_alarms", "false_alarm"), ("both_fine", "good")]}


def queue_rank(score: np.ndarray) -> np.ndarray:
    """Position of every scan in the queue a score gives (1 = reviewed first). Tied scans get their average rank."""
    return rankdata(-score, method="average")
