"""Save a figure of one scan: CT slice with prediction and ground truth, plus the probability and
the entropy heatmap (if computed).

A quick visual check that image, prediction and ground truth line up, and of where the model is unsure.
The axial slice with the most ground-truth organ voxels is shown.
Output: results/<config name>/figures/case_<case>[_<variant>].png

Usage: uv run python scripts/show_case.py --config configs/local.yaml [--case spleen_10] [--variant noise_60]
"""

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

from segreview.augment import perturb
from segreview.config import config_arg_parser, figures_dir, load_config
from segreview.data import list_cases, load_canonical, load_organ_mask
from segreview.io import load_map, map_path
from segreview.metrics import dice


def to_display(slice_2d: np.ndarray) -> np.ndarray:
    """Turn an axial slice from canonical (RAS) array order into the usual radiological view.

    In the array, the first axis runs towards the patient's right and the second towards the front.
    Radiologists view slices from the feet: front at the top, patient's right on the left of the screen.
    """
    return slice_2d.T[::-1, ::-1]


def main():
    parser = config_arg_parser(__doc__)
    parser.add_argument("--case", help="case id, e.g. spleen_10 (default: first scan in the config)")
    parser.add_argument("--variant", default="clean", help="clean or the name of a perturbation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    vis = cfg["visualisation"]

    cases = list_cases(cfg)
    index = next(i for i, c in enumerate(cases) if c.case_id == args.case) if args.case else 0
    case = cases[index]

    image = load_canonical(case.image_path)
    if args.variant != "clean":  # show the scan the model actually saw
        spec = next(p for p in cfg["perturbations"] if p["name"] == args.variant)
        image = perturb(image, spec, cfg, index)
    ct = image.get_fdata(dtype=np.float32)
    truth = load_organ_mask(cfg, case)
    pred = load_map(cfg, args.variant, case.case_id, "mask").astype(bool)
    prob = load_map(cfg, args.variant, case.case_id, "prob")
    has_entropy = map_path(cfg, args.variant, case.case_id, "entropy").exists()
    z = int(truth.sum(axis=(0, 1)).argmax())

    # CT values are in Hounsfield units. A soft-tissue "window" maps [level - width/2, level + width/2]
    # to black..white, so organs are easy to tell apart.
    low = vis["ct_window_level"] - vis["ct_window_width"] / 2
    high = vis["ct_window_level"] + vis["ct_window_width"] / 2
    ct_slice = to_display(ct[:, :, z])

    fig, axes = plt.subplots(1, 3 if has_entropy else 2, figsize=(16 if has_entropy else 11, 5.5))
    axes[0].imshow(ct_slice, cmap="gray", vmin=low, vmax=high)
    axes[0].contour(to_display(truth[:, :, z]), levels=[0.5], colors="lime", linewidths=1.2)
    axes[0].contour(to_display(pred[:, :, z]), levels=[0.5], colors="red", linewidths=1.2)
    axes[0].set_title(f"{case.case_id}, slice {z}: ground truth (green) vs prediction (red)")

    axes[1].imshow(ct_slice, cmap="gray", vmin=low, vmax=high)
    shown = axes[1].imshow(to_display(prob[:, :, z]), cmap="magma", vmin=0, vmax=1, alpha=0.6)
    axes[1].set_title("organ probability (softmax)")
    fig.colorbar(shown, ax=axes[1], fraction=0.046)
    if has_entropy:
        entropy = load_map(cfg, args.variant, case.case_id, "entropy")
        axes[2].imshow(ct_slice, cmap="gray", vmin=low, vmax=high)
        shown = axes[2].imshow(to_display(entropy[:, :, z]), cmap="magma", vmin=0, vmax=1, alpha=0.7)
        axes[2].set_title("uncertainty heatmap (entropy, bits)")
        fig.colorbar(shown, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Dice (whole scan) = {dice(pred, truth):.3f}   |   config: {cfg['name']}, "
                 f"{cfg['model']['resolution']} resolution, variant: {args.variant}")
    fig.tight_layout()

    suffix = "" if args.variant == "clean" else f"_{args.variant}"
    out = figures_dir(cfg) / f"case_{case.case_id}{suffix}.png"
    fig.savefig(out, dpi=vis["dpi"])
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
