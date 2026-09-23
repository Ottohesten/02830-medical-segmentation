"""Changing scans on purpose: test-time augmentation (TTA) and controlled perturbations.

Two different uses, both configured in the YAML config:

1. TTA (uncertainty.tta): run the network several extra times on slightly changed copies of the
   same scan and see whether the answer changes. If small, harmless changes flip voxels between
   organ and background, the model is unsure there. The model was trained WITHOUT mirroring
   (TotalSegmentator's "NoMirroring" trainer), so we do not flip the scan. Instead each pass uses
   a small random shift, a small intensity scaling/offset and a little noise. The shift is undone
   on the output, so all passes line up with the original scan.

2. Perturbations (perturbations): degrade the scan (noise, lower contrast, lower resolution) to
   create segmentation errors on purpose. This gives the G1 evaluation bad cases to find, as a
   supplement to naturally hard scans. Results on perturbed scans are reported separately.
"""

from dataclasses import dataclass

import nibabel as nib
import numpy as np

# CT value of air in Hounsfield units. Used to fill voxels that are shifted in from outside the scan.
AIR_HU = -1024.0


def shift_array(a: np.ndarray, shift: tuple[int, int, int], fill: float) -> np.ndarray:
    """Move a 3D array by whole voxels along each axis. Voxels that move in from outside get 'fill'.

    Unlike np.roll, nothing wraps around from the other side.

    Input: 3D array, shift per axis (positive = towards higher index), fill value.
    Output: shifted copy with the same shape.
    """
    out = np.full_like(a, fill)
    src, dst = [], []
    for s, n in zip(shift, a.shape):
        src.append(slice(max(0, -s), n - max(0, s)))
        dst.append(slice(max(0, s), n - max(0, -s)))
    out[tuple(dst)] = a[tuple(src)]
    return out


@dataclass
class TTAPass:
    """The random settings of one TTA pass, and how to apply and undo them."""
    shift: tuple[int, int, int]   # voxels, at the network's resolution
    scale: float                  # intensity multiplier, e.g. 1.03
    offset_hu: float              # intensity offset in HU
    noise_std_hu: float           # standard deviation of added Gaussian noise
    seed: int                     # makes the noise reproducible

    def augment(self, volume: np.ndarray) -> np.ndarray:
        """Apply this pass to a CT volume (in HU, on the network's voxel grid)."""
        rng = np.random.default_rng(self.seed)
        out = volume * self.scale + self.offset_hu
        out = out + rng.normal(0.0, self.noise_std_hu, size=out.shape).astype(np.float32)
        return shift_array(out.astype(np.float32), self.shift, AIR_HU)

    def undo_shift(self, maps: np.ndarray, fill: list[float]) -> np.ndarray:
        """Shift output maps (channels, x, y, z) back, so they line up with the unshifted scan.

        'fill' gives, per channel, the value for voxels the shifted scan did not cover
        (e.g. 'certainly not organ').
        """
        back = tuple(-s for s in self.shift)
        return np.stack([shift_array(m, back, f) for m, f in zip(maps, fill)])


def sample_tta_passes(cfg: dict, case_index: int) -> list[TTAPass]:
    """Draw the random settings for all TTA passes of one scan.

    The random generator is seeded with (config seed, scan index), so the same scan always gets
    the same passes, no matter which other scans are processed.

    Input: the loaded config (uncertainty.tta), the scan's position in the case list.
    Output: a list with uncertainty.tta.n_passes TTAPass objects.
    """
    t = cfg["uncertainty"]["tta"]
    rng = np.random.default_rng([cfg["seed"], case_index])
    passes = []
    for _ in range(t["n_passes"]):
        shift = tuple(int(v) for v in rng.integers(-t["max_shift_voxels"], t["max_shift_voxels"] + 1, size=3))
        passes.append(TTAPass(
            shift=shift,
            scale=float(1.0 + rng.uniform(-t["intensity_scale"], t["intensity_scale"])),
            offset_hu=float(rng.uniform(-t["intensity_shift_hu"], t["intensity_shift_hu"])),
            noise_std_hu=float(t["noise_std_hu"]),
            seed=int(rng.integers(2**31)),
        ))
    return passes


def perturb(image: nib.Nifti1Image, spec: dict, cfg: dict, case_index: int) -> nib.Nifti1Image:
    """Make a degraded copy of a CT scan, on exactly the same voxel grid as the original.

    Because the grid does not change, the original ground truth still fits the perturbed scan.

    Types (spec["type"]), with spec["strength"]:
    - noise:    add Gaussian noise with standard deviation = strength (HU).
    - contrast: pull every value towards a fixed grey level (perturbation_settings.contrast_level_hu)
                by the fraction 'strength': new = level + (old - level) * (1 - strength).
                0 = unchanged, 1 = completely flat.
    - lowres:   blur by resampling to voxels of 'strength' mm (only along axes that are finer than
                that) and back again, like a scan taken at lower resolution.

    Input: canonical image, one entry of the config's perturbations list, config, scan index (seed).
    Output: a new nibabel image with the same shape and affine.
    """
    from totalsegmentator.resampling import change_spacing

    data = image.get_fdata(dtype=np.float32)
    kind, strength = spec["type"], spec["strength"]
    if kind == "noise":
        rng = np.random.default_rng([cfg["seed"], case_index])
        data = data + rng.normal(0.0, strength, size=data.shape).astype(np.float32)
    elif kind == "contrast":
        level = cfg["perturbation_settings"]["contrast_level_hu"]
        data = level + (data - level) * (1.0 - strength)
    elif kind == "lowres":
        img = nib.Nifti1Image(data, image.affine)
        coarse = [max(z, strength) for z in image.header.get_zooms()[:3]]
        low = change_spacing(img, coarse, order=1, dtype=np.float32)
        data = change_spacing(low, coarse, target_shape=image.shape, order=1, dtype=np.float32,
                              force_affine=image.affine).get_fdata(dtype=np.float32)
    else:
        raise ValueError(f"Unknown perturbation type '{kind}'")
    return nib.Nifti1Image(data.astype(np.float32), image.affine)
