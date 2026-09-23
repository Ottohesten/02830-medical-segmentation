"""Segmentation quality metrics. Used ONLY for evaluation, never inside an uncertainty score.

Ground truth is what we compare against to find out how good the model really was.
The uncertainty scores (phase 2) must be computed from the model's own output alone,
because in the real use case (13,000 hospital scans) there is no ground truth.
"""

import numpy as np


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks.

    Dice = 2 * |pred AND truth| / (|pred| + |truth|). It is 1 for a perfect match and 0 when
    the masks do not overlap at all. If both masks are empty (organ absent and correctly not
    predicted), we return 1.0, since the model did exactly the right thing.

    Input: two boolean or 0/1 arrays of the same shape.
    Output: Dice as a float in [0, 1].
    """
    pred, truth = pred.astype(bool), truth.astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs ground truth {truth.shape}")
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, truth).sum() / total)
