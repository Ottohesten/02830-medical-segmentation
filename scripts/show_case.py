"""Save a figure of one scan: CT slice with prediction and ground truth, plus the organ probability.

A quick visual sanity check that image, prediction and ground truth line up.
The axial slice with the most ground-truth organ voxels is shown.
Output: results/figures/case_<case>_<config name>_<date>.png

Usage: uv run python scripts/show_case.py --config configs/local.yaml [--case spleen_10]
"""

from datetime import date

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

from segreview.config import config_arg_parser, load_config
from segreview.data import list_cases, load_canonical
from segreview.io import load_mask, load_prob
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
    args = parser.parse_args()
    cfg = load_config(args.config)
    vis = cfg["visualisation"]

    cases = list_cases(cfg)
    case = next(c for c in cases if c.case_id == args.case) if args.case else cases[0]

    ct = load_canonical(case.image_path).get_fdata(dtype=np.float32)
    truth = np.asanyarray(load_canonical(case.label_path).dataobj) == cfg["dataset"]["gt_label"]
    pred = load_mask(cfg, case.case_id).astype(bool)
    prob = load_prob(cfg, case.case_id)
    z = int(truth.sum(axis=(0, 1)).argmax())

    # CT values are in Hounsfield units. A soft-tissue "window" maps [level - width/2, level + width/2]
    # to black..white, so organs are easy to tell apart.
    low = vis["ct_window_level"] - vis["ct_window_width"] / 2
    high = vis["ct_window_level"] + vis["ct_window_width"] / 2
    ct_slice = to_display(ct[:, :, z])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(ct_slice, cmap="gray", vmin=low, vmax=high)
    axes[0].contour(to_display(truth[:, :, z]), levels=[0.5], colors="lime", linewidths=1.2)
    axes[0].contour(to_display(pred[:, :, z]), levels=[0.5], colors="red", linewidths=1.2)
    axes[0].set_title(f"{case.case_id}, slice {z}: ground truth (green) vs prediction (red)")

    axes[1].imshow(ct_slice, cmap="gray", vmin=low, vmax=high)
    shown = axes[1].imshow(to_display(prob[:, :, z]), cmap="magma", vmin=0, vmax=1, alpha=0.6)
    axes[1].set_title(f"{cfg['model']['organ']} probability (softmax)")
    fig.colorbar(shown, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Dice (whole scan) = {dice(pred, truth):.3f}   |   config: {cfg['name']}, "
                 f"{cfg['model']['resolution']} resolution")
    fig.tight_layout()

    out = cfg["paths"]["results_dir"] / "figures" / f"case_{case.case_id}_{cfg['name']}_{date.today().isoformat()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=vis["dpi"])
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
