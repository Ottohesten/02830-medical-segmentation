"""How the uncertainty heatmap is shown in the review interface (G2): all of it, or only its most uncertain part.

Why this exists: the entropy heatmap points at the wrong voxels only a little better than a map that just says
"errors are at the edge of the mask" (voxel AUC on the test set 0.864 vs 0.841, results/kits100/heatmap_eval_*).
Shown in full, it covers a wide band along the whole edge. The pilot compares display modes
(study.heatmap_display in configs/study.yaml) before the real study:

- full:    the heatmap as saved (the viewer hides values below viewer.heatmap_min). This is the default.
- top:     only the top_percent most uncertain voxels of the region keep their value; all other voxels are 0.
           These voxels lie right on the edge of the mask, so "top" shows WHICH PARTS of the edge are uncertain.
- contour: a thin line (2 voxels) around the voxels that "top" keeps, contour_margin_mm away from them, drawn in
           every axial slice (the slices the participant looks at) at full strength. The uncertain part of the edge
           itself stays clear, so the participant can see the CT and the mask there.

A limitation to know before the pilot: the entropy is highest exactly where the model's probability crosses 0.5,
and that is where the mask's edge is. So the top voxels lie almost all ON the edge of the mask, and the mask's
solid outline, which is always drawn on top, covers much of them. What the entropy adds to "errors are at the edge" is mainly how WIDE
the uncertain band is, and uncertain lines inside the mask; "full" shows that, "top" mostly does not.

The region is the predicted kidney plus uncertainty.border_mm around it (5 mm), the same region that
heatmap_eval.py uses. So "top 10 %" shows exactly the voxels its top-10 % capture measure counted.

Only the model's own output is used (heatmap and predicted mask), never the ground truth. The review server
computes the shown map from the prepared files when the browser asks for it, so changing the mode needs no new
preparation of the study scans.
"""

import gzip

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from segreview.io import QUANT_LEVELS
from segreview.uncertainty import uncertainty_region

DISPLAY_MODES = ("full", "top", "contour")
LINE_WIDTH_VOXELS = 2       # contour line width; a 1-voxel line (about 1 screen pixel) was hard to see
DEFAULTS = {"mode": "full", "top_percent": 10, "contour_margin_mm": 3}


def check_display(display: dict | None) -> dict:
    """Validate study.heatmap_display and fill in defaults.

    Input: the heatmap_display part of the study config (or None = all defaults).
    Output: {"mode", "top_percent", "contour_margin_mm"}. Raises ValueError for an unknown mode or a bad number.
    """
    out = {**DEFAULTS, **(display or {})}
    unknown = set(out) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"study.heatmap_display has unknown keys {sorted(unknown)}")
    if out["mode"] not in DISPLAY_MODES:
        raise ValueError(f"study.heatmap_display.mode must be one of {DISPLAY_MODES}, not '{out['mode']}'")
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if not number(out["top_percent"]) or not 0 < out["top_percent"] <= 100:
        raise ValueError(f"study.heatmap_display.top_percent must be in (0, 100], not {out['top_percent']!r}")
    if not number(out["contour_margin_mm"]) or out["contour_margin_mm"] < 0:
        raise ValueError(f"study.heatmap_display.contour_margin_mm must be 0 or more, not {out['contour_margin_mm']!r}")
    return out


def display_label(display: dict) -> str:
    """Short text for tables and figures, e.g. 'full', 'top 10 %' or 'contour 10 %, 3 mm margin'."""
    if display["mode"] == "full":
        return "full"
    label = f"{display['mode']} {display['top_percent']:g} %"
    return label + (f", {display['contour_margin_mm']:g} mm margin" if display["mode"] == "contour" else "")


def top_voxels(heat: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float], border_mm: float,
               top_percent: float) -> np.ndarray:
    """The most uncertain voxels of the region: those with the top_percent highest heatmap values.

    Voxels that tie with the cut-off value are all kept (the map has only 256 levels, and which of two equal
    voxels to drop would be arbitrary), so slightly more than top_percent can be kept. A voxel with value 0 is
    never kept: there is nothing uncertain to show.

    Input: heatmap (0-1), predicted mask, voxel size (mm), region border (mm), share to keep (%).
    Output: boolean array, True = kept.
    """
    region = uncertainty_region(mask.astype(bool), spacing, border_mm)
    if not region.any():
        return region
    values = heat[region]
    k = max(1, int(np.ceil(top_percent / 100 * values.size)))
    cut = np.partition(values, values.size - k)[values.size - k]      # the k-th highest value
    return region & (heat >= cut) & (heat > 0)


def grow_in_slices(kept: np.ndarray, spacing: tuple[float, float, float], margin_mm: float) -> np.ndarray:
    """Every voxel within margin_mm of a kept voxel IN THE SAME AXIAL SLICE (the last axis).

    Growing per slice matches what the participant sees: an area around the uncertain voxels of that slice.
    """
    grown = kept.copy()
    if margin_mm <= 0:
        return grown
    for z in np.flatnonzero(kept.any(axis=(0, 1))):
        dist = distance_transform_edt(~kept[:, :, z], sampling=spacing[:2])   # mm to the nearest kept voxel
        grown[:, :, z] = dist <= margin_mm
    return grown


def outline(area: np.ndarray, width: int = LINE_WIDTH_VOXELS) -> np.ndarray:
    """The edge of an area in each axial slice, 'width' voxels thick (the inner rim of the area).

    Erosion peels one layer of voxels off the area per step (a voxel stays only if its 4 in-plane neighbors are in
    the area); what the peeling removes is the rim. Nothing crosses between slices.
    """
    cross = np.zeros((3, 3, 1), bool)
    cross[1, :, 0] = cross[:, 1, 0] = True          # the 4 in-plane neighbors, nothing across slices
    return area & ~binary_erosion(area, structure=cross, iterations=width, border_value=0)


def display_map(heat: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float], border_mm: float,
                display: dict) -> np.ndarray:
    """The heatmap as the participant sees it, for one display setting (see module docstring).

    Input: heatmap (0-1), predicted mask, voxel size (mm), region border (mm), checked display settings.
    Output: float32 map (0-1) on the same grid.
    """
    if display["mode"] == "full":
        return heat.astype(np.float32)
    kept = top_voxels(heat, mask, spacing, border_mm, display["top_percent"])
    if display["mode"] == "top":
        return np.where(kept, heat, 0).astype(np.float32)
    area = grow_in_slices(kept, spacing, display["contour_margin_mm"])
    return outline(area).astype(np.float32)          # contour: full strength, so the thin line is easy to see


def display_nifti_bytes(heat_img: nib.Nifti1Image, mask_img: nib.Nifti1Image, border_mm: float,
                        display: dict) -> bytes:
    """The shown heatmap of one prepared scan as gzipped NIfTI bytes, for the review server.

    Same grid, affine and storage as the prepared heatmap (0-255 with scale 1/255, see io.py).
    """
    heat = heat_img.get_fdata(dtype=np.float32)
    mask = np.asanyarray(mask_img.dataobj) > 0
    spacing = tuple(float(z) for z in heat_img.header.get_zooms()[:3])
    shown = display_map(heat, mask, spacing, border_mm, display)
    img = nib.Nifti1Image(np.round(np.clip(shown, 0, 1) * QUANT_LEVELS).astype(np.uint8), heat_img.affine)
    img.header.set_slope_inter(1.0 / QUANT_LEVELS, 0.0)
    return gzip.compress(img.to_bytes(), compresslevel=1)
