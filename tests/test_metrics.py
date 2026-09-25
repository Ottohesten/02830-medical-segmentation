"""Unit tests of the Dice coefficient (src/segreview/metrics.py), including the edge cases."""

import numpy as np
import pytest

from segreview.metrics import dice


def masks(shape=(4, 4, 2)):
    return np.zeros(shape, bool), np.zeros(shape, bool)


def test_known_value():
    pred, truth = masks()
    pred.flat[[0, 1, 2, 3]] = True                 # 4 predicted voxels
    truth.flat[[1, 2, 3, 4, 5, 6]] = True          # 6 true voxels, 3 of them overlap
    assert dice(pred, truth) == pytest.approx(2 * 3 / (4 + 6))
    assert dice(pred, truth) == dice(truth, pred)  # symmetric


def test_perfect_and_disjoint():
    pred, truth = masks()
    pred[0] = truth[0] = True
    assert dice(pred, truth) == 1.0
    other = np.zeros_like(pred)
    other[1] = True
    assert dice(other, truth) == 0.0


def test_empty_masks():
    pred, truth = masks()
    assert dice(pred, truth) == 1.0                # organ absent and correctly not predicted
    truth[0, 0, 0] = True
    assert dice(pred, truth) == 0.0                # organ missed completely
    assert dice(truth, np.zeros_like(truth)) == 0.0   # organ predicted where there is none


def test_ignore_mask_leaves_voxels_out():
    pred, truth = masks()
    truth[0] = True
    pred[0] = True
    pred[1, 0, 0] = True                           # one false positive ...
    ignore = np.zeros_like(pred)
    ignore[1, 0, 0] = True                         # ... in an ignored voxel: it does not count
    assert dice(pred, truth, ignore) == 1.0
    assert dice(pred, truth) < 1.0
    assert dice(pred, truth, np.ones_like(pred)) == 1.0          # everything ignored = both empty
    assert dice(pred, truth, np.zeros_like(pred)) == dice(pred, truth)


def test_accepts_0_1_integers_and_rejects_shape_mismatch():
    pred, truth = masks()
    pred[0] = truth[0] = True
    assert dice(pred.astype(np.uint8), truth.astype(np.uint8)) == 1.0
    with pytest.raises(ValueError):
        dice(pred, truth[:, :, :1])
