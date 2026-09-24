"""Anatomical plausibility check for a paired organ (e.g. the kidneys), WITHOUT ground truth (G1).

Uncertainty scores only notice errors the model is unsure about. Some errors are made with full
confidence: in the KiTS development set the model missed a kidney that sat in the pelvis completely
(probability 0 everywhere), so entropy and model disagreement were low. A simple anatomical check
catches such cases: people normally have two kidneys of roughly similar size.

How it works (only the predicted mask and the CT image are used):
1. Find the body's left-right midline: the average left-right position of all body voxels
   (CT value above plausibility.body_threshold_hu, i.e. not air), in the slices where the organ is
   predicted. Using the body instead of the image centre handles patients lying off-centre.
2. Split the predicted organ at that midline into a patient-left and a patient-right part.
3. On each side, find connected pieces of the mask (connected-component labelling) and count the
   side as "found" if its largest piece is at least plausibility.min_component_ml.
4. Scores:
   - plaus_asymmetry = 1 - (smaller side volume / larger side volume). 0 = equal sizes,
     1 = one side empty. Higher = less plausible (same sign convention as the uncertainty scores).
   - plaus_missing_side = number of sides where no organ was found (0, 1 or 2). Coarser, but it ignores
     the natural left-right difference in size that makes the asymmetry noisy.
   - n_sides_found (0, 1 or 2) and the side volumes are saved as information.

Limits: a patient who really has only one kidney (e.g. after surgery) is flagged too. That is fine for
a review queue: such a scan is worth a look anyway.
"""

import numpy as np
from scipy.ndimage import label


def paired_organ_check(mask: np.ndarray, ct: np.ndarray, spacing: tuple[float, float, float],
                       settings: dict) -> dict:
    """Plausibility scores of a paired organ for one scan (see module docstring).

    Input: predicted mask and CT (both canonical RAS, same grid), voxel size in mm, and the config's
           plausibility settings (body_threshold_hu, min_component_ml).
    Output: dict with plaus_asymmetry, plaus_missing_side, n_sides_found, left_ml, right_ml, n_components.
    """
    mask = mask.astype(bool)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    if not mask.any():
        return {"plaus_asymmetry": 1.0, "plaus_missing_side": 2, "n_sides_found": 0, "left_ml": 0.0,
                "right_ml": 0.0, "n_components": 0}

    # 1. Midline from the body in the slices (last axis = z) where the organ is predicted.
    z = np.flatnonzero(mask.any(axis=(0, 1)))
    body = ct[:, :, z.min():z.max() + 1] > settings["body_threshold_hu"]
    x_positions = np.nonzero(body)[0]
    midline = x_positions.mean() if len(x_positions) else mask.shape[0] / 2.0

    # 2.+3. In canonical RAS order the first axis runs towards the patient's RIGHT (checked on MSD Spleen:
    # the spleen, which is on the patient's left, sits at small x). So x below the midline = patient's left.
    x = np.arange(mask.shape[0])[:, None, None]
    min_voxels = settings["min_component_ml"] / voxel_ml
    volumes, found, n_components = {}, 0, 0
    for side, in_side in (("left", x < midline), ("right", x >= midline)):
        part = mask & in_side
        volumes[side] = float(part.sum()) * voxel_ml
        pieces, n = label(part)
        sizes = np.bincount(pieces.ravel())[1:] if n else np.array([])
        big = int((sizes >= min_voxels).sum())
        n_components += big
        found += int(big > 0)

    # 4. Asymmetry of the two side volumes.
    larger = max(volumes["left"], volumes["right"])
    asymmetry = 1.0 - min(volumes["left"], volumes["right"]) / larger if larger > 0 else 1.0
    return {"plaus_asymmetry": asymmetry, "plaus_missing_side": 2 - found, "n_sides_found": found,
            "left_ml": volumes["left"], "right_ml": volumes["right"], "n_components": n_components}
