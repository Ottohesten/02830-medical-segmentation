"""Reading and writing the per-scan model output in paths.predictions_dir.

Layout: data/predictions/<config name>/<variant>/<case>_<kind>.nii.gz
- variant: "clean" for the original scans, or the name of a perturbation (e.g. "noise_60").
- kind:
  mask        the model's organ mask (0/1)
  prob        the organ's softmax probability (0-1)
  entropy     voxel-wise uncertainty heatmap: binary entropy of prob, in bits (0-1). This is the
              heatmap the review interface shows.
  tta_votes   in how many TTA passes each voxel was predicted as organ (0..n_passes)
  tta_std     standard deviation of prob over the TTA passes (0-0.5)

All files are NIfTI (.nii.gz) with the scan's own affine, so any viewer (3D Slicer, ITK-SNAP,
a web viewer) can overlay them on the CT. Values in [0, 1] are stored as whole numbers 0..255
with a scale factor of 1/255 in the NIfTI header: about 4x smaller than float32, and nibabel
(and other readers) turn them back into 0-1 values automatically.
We never store the probabilities of all 117 classes.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

QUANT_LEVELS = 255              # values in [0, 1] are stored as integers 0..255
QUANTISED = {"prob", "entropy", "tta_std"}


def prediction_dir(cfg: dict, variant: str = "clean") -> Path:
    """Folder for one variant's predictions, e.g. data/predictions/local/clean."""
    return cfg["paths"]["predictions_dir"] / cfg["name"] / variant


def map_path(cfg: dict, variant: str, case_id: str, kind: str) -> Path:
    """Path of one saved map, e.g. data/predictions/local/clean/spleen_10_prob.nii.gz."""
    return prediction_dir(cfg, variant) / f"{case_id}_{kind}.nii.gz"


def save_map(cfg: dict, variant: str, case_id: str, kind: str, data: np.ndarray, affine: np.ndarray) -> None:
    """Save one per-voxel map as NIfTI. Maps with values in [0, 1] are quantised (see module docstring)."""
    path = map_path(cfg, variant, case_id, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind in QUANTISED:
        img = nib.Nifti1Image(np.round(np.clip(data, 0, 1) * QUANT_LEVELS).astype(np.uint8), affine)
        img.header.set_slope_inter(1.0 / QUANT_LEVELS, 0.0)
    else:
        img = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(img, path)


def load_map(cfg: dict, variant: str, case_id: str, kind: str) -> np.ndarray:
    """Load one saved map. Quantised maps come back as float32 in [0, 1], the others as uint8."""
    img = nib.load(map_path(cfg, variant, case_id, kind))
    if kind in QUANTISED:
        return img.get_fdata(dtype=np.float32)
    return np.asanyarray(img.dataobj).astype(np.uint8)


def has_prediction(cfg: dict, variant: str, case_id: str) -> bool:
    """True if mask and probability of this scan and variant are already saved."""
    return all(map_path(cfg, variant, case_id, k).exists() for k in ("mask", "prob"))


def voxel_spacing(cfg: dict, variant: str, case_id: str) -> tuple[float, float, float]:
    """Voxel size (mm) of a saved map, read from its header."""
    return tuple(float(z) for z in nib.load(map_path(cfg, variant, case_id, "mask")).header.get_zooms()[:3])


def affine_of(cfg: dict, variant: str, case_id: str) -> np.ndarray:
    """Affine of a saved map, so derived maps (e.g. entropy) can be saved on the same grid."""
    return nib.load(map_path(cfg, variant, case_id, "mask")).affine
