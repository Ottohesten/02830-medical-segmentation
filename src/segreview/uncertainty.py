"""Uncertainty scores per scan, computed ONLY from the model's own output (pipeline step 3, G1).

No ground truth is used here. In the real use case (thousands of hospital scans) there is no
ground truth, so a score that needed it would be useless. Ground truth is only used later, in
evaluation.py, to check how well the scores rank the scans.

SIGN CONVENTION: every score is "higher = more uncertain = more likely a bad segmentation".
So when we correlate a score with Dice (higher = better), we expect a NEGATIVE Spearman rho.
The volume baseline follows the same convention: its score is minus the predicted volume,
because small organs tend to get lower Dice.

Voxel-wise uncertainty (the heatmap): binary entropy of the organ probability p,
    H(p) = -p log2(p) - (1 - p) log2(1 - p)   (in bits).
H is 0 when the model is sure (p = 0 or p = 1) and at most 1 bit when p = 0.5.
This is the entropy of "organ vs. everything else", so it only needs the saved organ
probability, not all 117 classes.

Scan-level scores from the saved maps (all GT-free):
- entropy_sum_ml       total entropy summed over the scan, in bit * ml. Grows with the size of the
                       organ's surface, so it is NOT normalised for organ size.
- entropy_mean_region  mean entropy inside the "uncertainty region": the predicted organ plus a
                       border of uncertainty.border_mm around it. Normalised for size.
- entropy_per_volume   total entropy divided by the predicted organ volume. Normalised for size.
- soft_dice_gap        1 - soft Dice between the probability map and the model's own mask. It is the
                       model's own guess of how much Dice it loses at uncertain voxels. Normalised.
With TTA (only if TTA maps exist):
- tta_disagreement     1 - (voxels all passes call organ) / (voxels any pass calls organ), counting the
                       clean prediction as one more pass. 0 = all passes agree. Normalised.
- tta_std_mean_region  mean standard deviation of p over the passes, inside the uncertainty region.
With a second model (only if its maps exist; e.g. the 6 mm model next to the 3 mm model):
- model_disagreement   1 - Dice(mask of main model, mask of second model). The public weights have
                       only one fold, so we cannot compare folds; the 3 mm and 6 mm networks are two
                       separately trained models and play that role. Normalised for size.
- model_diff_mean_region  mean |p_main - p_second| inside the uncertainty region. The voxel-wise
                       |p_main - p_second| is also saved as a heatmap (m2_diff).
                       Caveat: the 6 mm model is coarser, so part of the disagreement is just resolution.
Baseline:
- neg_volume_ml        minus the predicted organ volume (ml).
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# Smallest probability used inside log2, to avoid log(0). Has no visible effect on the result.
EPS = 1e-6


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Voxel-wise binary entropy in bits (0 = certain, 1 = completely unsure). See module docstring."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def uncertainty_region(mask: np.ndarray, spacing: tuple[float, float, float], border_mm: float) -> np.ndarray:
    """The predicted organ plus every voxel within border_mm (in real millimetres) of it.

    Errors happen at and just outside the organ's edge, so this is where uncertainty matters.
    Using millimetres (not voxels) makes the border the same physical size in scans with
    thin and thick slices. The distance transform gives, for every voxel, the distance to the
    nearest organ voxel; to save time it only runs on a box around the organ.

    Input: predicted mask, voxel size in mm, border width in mm.
    Output: boolean array of the same shape. Empty if the mask is empty.
    """
    region = np.zeros(mask.shape, dtype=bool)
    if not mask.any():
        return region
    pad = [int(np.ceil(border_mm / s)) + 1 for s in spacing]
    idx = np.nonzero(mask)
    box = tuple(slice(max(0, i.min() - p), min(n, i.max() + 1 + p)) for i, p, n in zip(idx, pad, mask.shape))
    dist = distance_transform_edt(~mask[box].astype(bool), sampling=spacing)
    region[box] = dist <= border_mm
    return region


def scan_scores(mask: np.ndarray, prob: np.ndarray, spacing: tuple[float, float, float], border_mm: float,
                tta_votes: np.ndarray | None = None, tta_std: np.ndarray | None = None,
                n_tta: int = 0, m2_mask: np.ndarray | None = None,
                m2_prob: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Compute all scan-level scores of one scan (see module docstring).

    Input: predicted mask (0/1), organ probability, voxel size (mm), border (mm), and optionally
           the TTA vote count and standard deviation maps with the number of TTA passes, and the
           second model's mask and probability.
    Output: (dict of scores, voxel-wise entropy map for the heatmap).
    """
    mask = mask.astype(bool)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    entropy = binary_entropy(prob)
    region = uncertainty_region(mask, spacing, border_mm)
    n_mask = int(mask.sum())

    scores = {
        "pred_volume_ml": n_mask * voxel_ml,
        "neg_volume_ml": -n_mask * voxel_ml,
        "entropy_sum_ml": float(entropy.sum()) * voxel_ml,
        # An empty prediction has no region; then there is nothing to average and the score is 0.
        "entropy_mean_region": float(entropy[region].mean()) if region.any() else 0.0,
        "entropy_per_volume": float(entropy.sum()) / max(n_mask, 1),
    }
    # Soft Dice between probability and mask: 2 * sum(p * m) / (sum(p) + sum(m)).
    denom = float(prob.sum()) + n_mask
    scores["soft_dice_gap"] = 1.0 - 2.0 * float(prob[mask].sum()) / denom if denom > 0 else 0.0

    if tta_votes is not None and n_tta > 0:
        votes = tta_votes.astype(np.int32) + mask   # the clean prediction counts as one more pass
        n_total = n_tta + 1
        any_organ = votes > 0
        all_organ = votes == n_total
        scores["tta_disagreement"] = 1.0 - all_organ.sum() / any_organ.sum() if any_organ.any() else 0.0
        scores["tta_std_mean_region"] = float(tta_std[region].mean()) if region.any() else 0.0

    if m2_mask is not None and m2_prob is not None:
        from segreview.metrics import dice
        scores["model_disagreement"] = 1.0 - dice(mask, m2_mask)
        diff = model_difference(prob, m2_prob)
        scores["model_diff_mean_region"] = float(diff[region].mean()) if region.any() else 0.0
    return scores, entropy


def model_difference(prob: np.ndarray, prob2: np.ndarray) -> np.ndarray:
    """Voxel-wise disagreement between two models: |p_main - p_second| (0 = agree, 1 = opposite)."""
    return np.abs(prob - prob2)
