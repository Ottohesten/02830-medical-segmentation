"""Which heatmap points best at the voxels that are actually wrong? (G2: choosing the heatmap)

This is a different question from G1. G1 ranks whole scans; here we ask, inside one scan, whether the
bright parts of a heatmap are where the model's mask disagrees with the ground truth. A reviewer who
is guided by the heatmap will look there first, so it should light up the errors.

For every scan (evaluation only, uses ground truth):
- region:  the predicted organ plus uncertainty.border_mm around it (the same region as in G1).
           Errors far outside it (e.g. a whole kidney the model missed) cannot be shown by any
           heatmap near the mask; the share of error voxels inside the region is reported as context.
- error:   voxels where the model's mask differs from the ground truth (false positive or false negative).
- voxel AUC: the chance that a random error voxel gets a higher heatmap value than a random correct
           voxel in the region (0.5 = no better than chance, 1 = perfect). Ties count half.
- top-X capture: the share of error voxels that lie in the X % of region voxels with the highest heatmap
           values (heatmap_eval.top_fraction). A useless heatmap captures about X %.

Candidates: 'entropy' (binary entropy of the organ probability) and 'm2_diff' (|p_3mm - p_6mm|).
Reference (not a candidate): 'boundary' = closeness to the edge of the predicted mask, i.e. a heatmap
that only says "errors happen at the edge", with no model uncertainty in it.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

from segreview.uncertainty import uncertainty_region


def voxel_auc(values: np.ndarray, is_error: np.ndarray) -> float:
    """Area under the ROC curve for 'values' separating error from correct voxels (Mann-Whitney U).

    Computed from ranks, so ties (the maps are stored with 256 levels) count half. NaN if the region
    has no error voxels or no correct voxels.
    """
    n_pos, n_neg = int(is_error.sum()), int((~is_error).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(values)
    return float((ranks[is_error].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top_fraction_capture(values: np.ndarray, is_error: np.ndarray, fraction: float) -> float:
    """Share of error voxels among the 'fraction' of voxels with the highest values (ties shared fairly).

    With many equal values (e.g. 0 in most of the region) the voxels at the cut-off value are counted
    in proportion, the same idea as the expected curve for tied scores in G1.
    """
    n_err = int(is_error.sum())
    if n_err == 0:
        return float("nan")
    k = fraction * len(values)
    order = np.argsort(-values, kind="stable")
    v, e = values[order], is_error[order]
    cut = v[min(int(np.ceil(k)) - 1, len(v) - 1)] if k >= 1 else v[0]
    above = v > cut
    at = v == cut
    taken_at = k - above.sum()                      # how many of the tied voxels fit in the top k
    errors = e[above].sum() + (e[at].sum() * taken_at / at.sum() if at.any() else 0.0)
    return float(errors / n_err)


def boundary_map(mask: np.ndarray, region: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Reference heatmap: minus the distance (mm) to the edge of the predicted mask, so the edge is highest.

    Only computed in a box around the region to save time; outside the region the value does not matter.
    """
    out = np.zeros(mask.shape, dtype=np.float32)
    idx = np.nonzero(region)
    box = tuple(slice(i.min(), i.max() + 1) for i in idx)
    m = mask[box].astype(bool)
    dist = np.where(m, distance_transform_edt(m, sampling=spacing), distance_transform_edt(~m, sampling=spacing))
    out[box] = -dist
    return out


def evaluate_scan(maps: dict[str, np.ndarray], mask: np.ndarray, truth: np.ndarray,
                  spacing: tuple[float, float, float], border_mm: float, top_fraction: float) -> dict:
    """Voxel-level error localization of every heatmap for one scan (see module docstring).

    Input: {heatmap name: map}, predicted mask, ground-truth organ mask, voxel size, border (mm),
           top fraction for the capture measure.
    Output: dict with error counts and, per heatmap, auc_<name> and top_<name>.
    """
    mask, truth = mask.astype(bool), truth.astype(bool)
    region = uncertainty_region(mask, spacing, border_mm)
    error = mask != truth
    row = {"error_voxels": int(error.sum()), "error_in_region_share": float(error[region].sum() / max(error.sum(), 1)),
           "region_voxels": int(region.sum())}
    if not region.any():
        return row
    all_maps = {**maps, "boundary": boundary_map(mask, region, spacing)}
    err_r = error[region]
    for name, m in all_maps.items():
        vals = m[region].astype(np.float64)
        row[f"auc_{name}"] = voxel_auc(vals, err_r)
        row[f"top_{name}"] = top_fraction_capture(vals, err_r, top_fraction)
    return row
