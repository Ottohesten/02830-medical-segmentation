"""Tests of the heatmap display modes (src/segreview/heatmap_display.py) on small made-up volumes.

The volumes are tiny and every expected answer can be worked out by hand: which voxels are "top", that nothing
outside the region is shown, that ties and zeros are handled as documented, and that the contour is a ring around
the kept voxels in their own slice only.
"""

import gzip

import nibabel as nib
import numpy as np
import pytest

from segreview.heatmap_display import (check_display, display_label, display_map, display_nifti_bytes,
                                       grow_in_slices, outline, top_voxels)
from segreview.io import QUANT_LEVELS
from segreview.uncertainty import uncertainty_region

SPACING = (1.0, 1.0, 1.0)
BORDER = 2.0


def cube_scene(seed=0):
    """A 30 x 30 x 6 volume with a 10 x 10 x 4 'kidney' mask and a heatmap with distinct values everywhere."""
    mask = np.zeros((30, 30, 6), bool)
    mask[10:20, 10:20, 1:5] = True
    heat = np.random.default_rng(seed).permutation(mask.size).reshape(mask.shape).astype(np.float32) / mask.size
    return heat, mask


def test_full_is_the_heatmap_itself():
    heat, mask = cube_scene()
    shown = display_map(heat, mask, SPACING, BORDER, check_display({"mode": "full"}))
    assert np.array_equal(shown, heat)


@pytest.mark.parametrize("percent", [5, 10, 37.5, 100])
def test_top_keeps_exactly_the_highest_share_of_the_region(percent):
    heat, mask = cube_scene()
    heat[0, 0, 0] = 1.0                      # far outside the region: must never be shown, however high
    region = uncertainty_region(mask, SPACING, BORDER)
    shown = display_map(heat, mask, SPACING, BORDER, check_display({"mode": "top", "top_percent": percent}))
    kept = shown > 0
    k = int(np.ceil(percent / 100 * region.sum()))
    assert kept.sum() == k                   # no ties in this heatmap, so exactly k voxels
    assert not (kept & ~region).any()
    assert heat[kept].min() >= heat[region & ~kept].max() if percent < 100 else True   # the highest ones
    assert np.array_equal(shown[kept], heat[kept])      # kept voxels keep their own value


def test_top_keeps_all_ties_and_never_zeros():
    _, mask = cube_scene()
    region = uncertainty_region(mask, SPACING, BORDER)
    heat = np.zeros(mask.shape, np.float32)
    heat[region] = 0.5                       # every region voxel ties
    assert top_voxels(heat, mask, SPACING, BORDER, 10).sum() == region.sum()
    heat[:] = 0
    heat[12, 12, 2] = heat[13, 12, 2] = heat[14, 12, 2] = 0.8   # only 3 uncertain voxels
    assert top_voxels(heat, mask, SPACING, BORDER, 50).sum() == 3


def test_empty_mask_shows_nothing_in_top_and_contour():
    heat, mask = cube_scene()
    mask[:] = False
    for mode in ("top", "contour"):
        assert not display_map(heat, mask, SPACING, BORDER, check_display({"mode": mode})).any()


def test_contour_is_a_ring_around_the_kept_voxels_in_their_own_slice():
    kept = np.zeros((25, 25, 5), bool)
    kept[12, 12, 2] = True
    ring = outline(grow_in_slices(kept, (1.0, 1.0, 1.0), 3.0))
    xs, ys = np.nonzero(ring[:, :, 2])
    dist = np.hypot(xs - 12, ys - 12)
    assert ring.sum() > 0 and not ring[:, :, [0, 1, 3, 4]].any()      # nothing in the other slices
    assert dist.min() > 1 and dist.max() <= 3                         # the 2-voxel rim of the 3 mm disc
    assert ring[:, :, 2].sum() == (grow_in_slices(kept, (1.0, 1.0, 1.0), 3.0)[:, :, 2]).sum() - 5   # disc minus its
    # inner diamond of 5 voxels (the center and its 4 neighbors, what is left after peeling off 2 layers)
    assert not (ring & kept).any()                                    # the uncertain voxel itself stays clear
    # Thin in-plane voxels (0.5 mm in x): the same 3 mm are twice as many voxels in x as in y.
    wide = grow_in_slices(kept, (0.5, 1.0, 1.0), 3.0)[:, :, 2]
    assert np.ptp(np.nonzero(wide)[0]) == 12 and np.ptp(np.nonzero(wide)[1]) == 6


def test_contour_mode_draws_full_strength_lines_around_the_top_voxels():
    heat, mask = cube_scene()
    display = check_display({"mode": "contour", "top_percent": 10, "contour_margin_mm": 2})
    shown = display_map(heat, mask, SPACING, BORDER, display)
    kept = top_voxels(heat, mask, SPACING, BORDER, 10)
    assert set(np.unique(shown)) == {0.0, 1.0}
    assert not (shown.astype(bool) & kept).any()
    # With margin 0 the line is the edge of the kept voxels themselves.
    zero = display_map(heat, mask, SPACING, BORDER, {**display, "contour_margin_mm": 0})
    assert np.array_equal(zero.astype(bool), outline(kept))


def test_nifti_bytes_keep_the_grid_and_storage():
    heat, mask = cube_scene()
    affine = np.diag([0.8, 0.8, 2.5, 1.0])
    affine[:3, 3] = [-10, 5, 30]
    heat_img = nib.Nifti1Image(np.round(heat * QUANT_LEVELS).astype(np.uint8), affine)
    heat_img.header.set_slope_inter(1.0 / QUANT_LEVELS, 0.0)
    heat_img = nib.Nifti1Image.from_bytes(heat_img.to_bytes())      # read back like a file, so the scale applies
    assert heat_img.get_fdata().max() == pytest.approx(1.0)
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    display = check_display({"mode": "top", "top_percent": 20})
    data = display_nifti_bytes(heat_img, mask_img, BORDER, display)
    assert data[:2] == b"\x1f\x8b"                                    # gzipped, like the prepared files
    img = nib.Nifti1Image.from_bytes(gzip.decompress(data))
    assert img.shape == mask.shape and np.allclose(img.affine, affine)
    expected = display_map(heat_img.get_fdata(dtype=np.float32), mask, (0.8, 0.8, 2.5), BORDER, display)
    assert np.allclose(img.get_fdata(), expected, atol=0.5 / QUANT_LEVELS)


def test_display_settings_are_checked():
    assert check_display(None) == {"mode": "full", "top_percent": 10, "contour_margin_mm": 3}
    assert check_display({"mode": "top", "top_percent": 5})["top_percent"] == 5
    for bad in ({"mode": "best"}, {"mode": "top", "top_percent": 0}, {"mode": "top", "top_percent": 101},
                {"top_percent": "10"}, {"top_percent": True}, {"contour_margin_mm": -1}, {"smooth_mm": 2}):
        with pytest.raises(ValueError):
            check_display(bad)


def test_display_labels():
    assert display_label(check_display({"mode": "full", "top_percent": 20})) == "full"
    assert display_label(check_display({"mode": "top", "top_percent": 10})) == "top 10 %"
    assert display_label(check_display({"mode": "contour", "top_percent": 7.5})) == "contour 7.5 %, 3 mm margin"
