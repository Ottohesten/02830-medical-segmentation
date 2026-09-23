"""Running the pretrained TotalSegmentator network on one CT scan (pipeline step 2).

We call nnU-Net directly instead of the TotalSegmentator command line tool, because the
command line tool only returns the final mask. For uncertainty we need the network's raw
scores (logits) and the softmax probabilities computed from them.

The model predicts 117 structures, so the full probability map is 117 values per voxel.
That is far too big to store (and barely fits in memory on a laptop). So right after the
network has run, we reduce the 117 channels to a few per-voxel maps about our organ only,
and throw the rest away. Only those small maps are kept and saved.

Steps for one scan (they copy what TotalSegmentator itself does):
1. Reorient to canonical (RAS) orientation.
2. Resample to the network's voxel size (3 mm in fast mode, 1.5 mm in full mode).
3. nnU-Net preprocessing (CT intensity normalisation) and sliding-window prediction -> logits.
4. Reduce logits to organ maps (see reduce_logits).
5. Resample the organ maps back to the original voxel grid, so they line up with the ground truth.

For test-time augmentation (TTA) the same steps run on a slightly changed copy of the scan
(see augment.py).
"""

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

from segreview.augment import TTAPass
from segreview.weights import ModelSpec, available_folds, model_folder, model_spec

# Value used for the organ "margin" outside the region nnU-Net looked at.
# Any clearly negative number works: it only has to mean "definitely not the organ".
OUTSIDE_MARGIN = -100.0


@dataclass
class OrganPrediction:
    """The model's output for one scan, reduced to the chosen organ, on the canonical voxel grid."""
    mask: np.ndarray        # uint8, 1 where the model predicts the organ
    prob: np.ndarray        # float32 in [0, 1], softmax probability of the organ
    affine: np.ndarray      # voxel -> world coordinates (canonical orientation)


def reduce_logits(logits: torch.Tensor, organ_channels: list[int], chunk: int) -> np.ndarray:
    """Reduce the full logits (one channel per class) to two maps about the organ.

    The organ can consist of several classes (e.g. left and right kidney); call that set S.
    - prob:   softmax probability of the organ. Softmax turns raw scores (logits) into probabilities
              that sum to 1 over all classes: p_k = exp(l_k) / sum_j exp(l_j). The organ's
              probability is the sum over its classes: p_organ = sum_{k in S} p_k.
    - margin: highest logit inside S minus the highest logit outside S. The model's final mask
              takes the class with the highest score (argmax), so margin > 0 exactly where the
              model labels the voxel as (part of) the organ. Unlike the mask, the margin is a smooth
              number, which lets us resample it to another grid without blocky edges.

    The work is done a few slices at a time (chunk) to keep memory use low.

    Input: logits with shape (classes, a, b, c), the organ's channel indices, slices per chunk.
    Output: float32 array with shape (2, a, b, c): [margin, prob].
    """
    inside = torch.tensor(organ_channels)
    outside = torch.tensor([k for k in range(logits.shape[0]) if k not in organ_channels])
    out = np.empty((2, *logits.shape[1:]), dtype=np.float32)
    for start in range(0, logits.shape[-1], chunk):
        block = logits[..., start:start + chunk].float()
        margin = block[inside].max(dim=0).values - block[outside].max(dim=0).values
        out[0, ..., start:start + chunk] = margin.numpy()
        out[1, ..., start:start + chunk] = torch.softmax(block, dim=0)[inside].sum(dim=0).numpy()
    return out


class OrganSegmenter:
    """Loads the pretrained network once and predicts the organ for any number of scans."""

    def __init__(self, cfg: dict, device: torch.device):
        """Set up the nnU-Net predictor from the config.

        Input: the loaded config and the torch device to run on.
        """
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.cfg = cfg
        self.spec: ModelSpec = model_spec(cfg)
        folder = model_folder(cfg, self.spec)
        folds = cfg["model"]["folds"]
        missing = set(folds) - set(available_folds(folder))
        if missing:
            raise ValueError(f"Folds {sorted(missing)} are not in the released weights ({folder}).")

        self.predictor = nnUNetPredictor(
            tile_step_size=cfg["model"]["tile_step_size"],
            use_gaussian=True,
            use_mirroring=False,  # the model was trained without mirroring; our own TTA is in augment.py
            perform_everything_on_device=device.type != "cpu",
            device=device,
            allow_tqdm=True,
        )
        self.predictor.initialize_from_trained_model_folder(str(folder), use_folds=folds,
                                                            checkpoint_name="checkpoint_final.pth")

    def predict(self, image: nib.Nifti1Image, tta: TTAPass | None = None) -> OrganPrediction:
        """Predict the organ for one canonical CT image.

        Input: a nibabel image in canonical orientation, and optionally one test-time augmentation
               (TTA) pass. The augmentation is applied to the resampled scan before the network, and
               any spatial shift is undone on the output maps, so the result lines up with the scan.
        Output: an OrganPrediction on the same voxel grid as the input image.
        """
        from totalsegmentator.resampling import change_spacing

        p = self.predictor
        spacing = self.spec.spacing_mm

        # Step 2: resample to the network's voxel size (cubic interpolation, as TotalSegmentator does).
        image_rs = change_spacing(image, spacing, order=3, dtype=np.int32)

        # Step 3: nnU-Net preprocessing. Because the image already has the network's spacing,
        # nnU-Net's own resampling does nothing here; it only crops and normalises intensities.
        # nnU-Net's NIfTI reader (the one the model was trained with) flips the axis order from
        # (x, y, z) to (z, y, x), so we do the same; otherwise the network sees the scan sideways.
        volume = np.asanyarray(image_rs.dataobj).astype(np.float32)
        if tta is not None:
            volume = tta.augment(volume)
        data = volume.transpose(2, 1, 0)[None]
        props = {"spacing": [spacing] * 3}
        preprocessor = p.configuration_manager.preprocessor_class(verbose=False)
        data_pp, _, props = preprocessor.run_case_npy(data, None, props, p.plans_manager,
                                                      p.configuration_manager, p.dataset_json)
        logits = p.predict_logits_from_preprocessed_data(torch.from_numpy(data_pp))

        # Step 4: keep only organ information, then free the big logits array.
        maps = reduce_logits(logits, self.spec.organ_channels, self.cfg["compute"]["reduce_chunk_slices"])
        del logits

        # Undo nnU-Net's cropping and axis transposition, so the maps match image_rs voxel for voxel.
        full = np.stack([np.full(props["shape_before_cropping"], OUTSIDE_MARGIN, dtype=np.float32),
                         np.zeros(props["shape_before_cropping"], dtype=np.float32)])
        full = insert_crop_into_image(full, maps, props["bbox_used_for_cropping"])
        full = full.transpose([0, *[i + 1 for i in p.plans_manager.transpose_backward]])
        full = full.transpose(0, 3, 2, 1)  # back from nnU-Net's (z, y, x) to nibabel's (x, y, z)
        if tta is not None:
            full = tta.undo_shift(full, fill=[OUTSIDE_MARGIN, 0.0])

        # Step 5: resample each map back to the original grid (linear interpolation).
        # force_affine makes the result share the exact affine of the input image.
        back = [change_spacing(nib.Nifti1Image(m, image_rs.affine), spacing, target_shape=image.shape,
                               order=1, dtype=np.float32, force_affine=image.affine).get_fdata(dtype=np.float32)
                for m in full]
        margin, prob = back
        return OrganPrediction(mask=(margin > 0).astype(np.uint8),
                               prob=np.clip(prob, 0.0, 1.0),
                               affine=image.affine)
