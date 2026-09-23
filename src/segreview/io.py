"""Reading and writing the per-scan model output in paths.predictions_dir.

For every scan we store two small files (never the full 117-class probabilities):
- <case>_mask.nii.gz : the model's organ mask (0/1).
- <case>_prob.nii.gz : the organ's softmax probability, stored as 0-255 with a scale factor
                       of 1/255 in the NIfTI header. This makes the file about 4x smaller than
                       float32, and nibabel turns it back into 0-1 values when reading.
Outputs of different configs go to separate subfolders (named after the config), so a
local run and an HPC run never overwrite each other.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from segreview.inference import OrganPrediction

PROB_LEVELS = 255  # probability is stored as an integer 0..255


def prediction_dir(cfg: dict) -> Path:
    """Folder for this config's predictions, e.g. data/predictions/local."""
    return cfg["paths"]["predictions_dir"] / cfg["name"]


def mask_path(cfg: dict, case_id: str) -> Path:
    return prediction_dir(cfg) / f"{case_id}_mask.nii.gz"


def prob_path(cfg: dict, case_id: str) -> Path:
    return prediction_dir(cfg) / f"{case_id}_prob.nii.gz"


def save_prediction(cfg: dict, case_id: str, pred: OrganPrediction) -> None:
    """Save the mask and the (quantised) organ probability of one scan as NIfTI files."""
    prediction_dir(cfg).mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred.mask.astype(np.uint8), pred.affine), mask_path(cfg, case_id))

    prob_img = nib.Nifti1Image(np.round(pred.prob * PROB_LEVELS).astype(np.uint8), pred.affine)
    prob_img.header.set_slope_inter(1.0 / PROB_LEVELS, 0.0)
    nib.save(prob_img, prob_path(cfg, case_id))


def load_mask(cfg: dict, case_id: str) -> np.ndarray:
    """Load the saved organ mask of one scan as a uint8 array."""
    return np.asanyarray(nib.load(mask_path(cfg, case_id)).dataobj).astype(np.uint8)


def load_prob(cfg: dict, case_id: str) -> np.ndarray:
    """Load the saved organ probability of one scan as float32 values in [0, 1]."""
    return nib.load(prob_path(cfg, case_id)).get_fdata(dtype=np.float32)
