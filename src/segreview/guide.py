"""Example image for the instruction screen of the user study (G2): what a kidney and a tumor look like in CT.

The participants are students, not clinicians, so before the practice scan they see one annotated CT slice.
The image is made once by scripts/prepare_study.py from a development-set scan that is not used anywhere else
in the study (not a study, practice or demo scan), so it gives no answers away.

The image shows the same axial slice twice:
- left:  the CT with the kidneys, the tumor (and a cyst, if there is one) outlined and named;
- right: everything that should be marked filled in red. Tumors and cysts count as kidney, as in the
         evaluation (organ definition kidney_tumor_cyst).
The slice is drawn like in the viewer: the patient's right side on the LEFT of the image, the front of
the body at the top (the radiological convention: you look at the patient from the feet).
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

from segreview.data import list_cases, load_canonical, load_label_map


def _to_screen(slice_2d: np.ndarray) -> np.ndarray:
    """Turn an axial slice from the canonical (RAS) array into screen orientation.

    In RAS, +x is the patient's right and +y the front. On screen the patient's right is on the left and the
    front at the top, so the rows are y from front to back and the columns x from right to left.
    """
    return slice_2d.T[::-1, ::-1]


def pick_guide_slice(study: dict, source: dict, exclude: set[str]) -> tuple[str, int]:
    """Choose the scan and axial slice for the example image.

    Candidates: development-set scans that are not used in the study. The slice should show a tumor that
    grows from a clearly visible kidney, and the other kidney as well. So, per slice: split everything
    labeled (kidney, tumor, cyst) into connected pieces; take the piece with the most tumor; its score is
    min(kidney area, tumor area) within that piece. Slices with fewer than two pieces of at least
    guide.min_piece_cm2 (the two kidneys) are skipped. The highest score wins.

    Input: study config, source config, scan ids to leave out.
    Output: (case id, slice index in the canonical array).
    """
    g = study["guide"]
    kidney_value, tumor_value = g["labels"]["kidney"], g["labels"]["tumor"]
    best = (-1.0, None, None)
    for case in list_cases(source, splits=["dev"]):
        if case.case_id in exclude:
            continue
        labels = load_label_map(case)
        zooms = load_canonical(case.label_path).header.get_zooms()[:3]
        pixel_cm2 = zooms[0] * zooms[1] / 100.0
        tumor_per_slice = (labels == tumor_value).sum(axis=(0, 1)) * pixel_cm2
        for z in np.flatnonzero(tumor_per_slice > best[0]):     # a slice with less tumor cannot win
            sl = labels[:, :, z]
            pieces, n = ndimage.label(sl > 0)
            sizes = np.bincount(pieces.ravel(), minlength=n + 1)[1:] * pixel_cm2
            if (sizes >= g["min_piece_cm2"]).sum() < 2:
                continue
            tumor_in = np.bincount(pieces[sl == tumor_value], minlength=n + 1)
            piece = int(np.argmax(tumor_in))
            kidney_in = np.count_nonzero((pieces == piece) & (sl == kidney_value))
            score = min(kidney_in, tumor_in[piece]) * pixel_cm2
            if score > best[0]:
                best = (score, case.case_id, int(z))
    if best[1] is None:
        raise ValueError("No development-set slice with kidney and tumor found for the guide image.")
    return best[1], best[2]


def make_guide_image(study: dict, source: dict, case_id: str, z: int, out_path: Path) -> None:
    """Draw the two-panel example image (see the module docstring) and save it as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = study["guide"]
    case = next(c for c in list_cases(source) if c.case_id == case_id)
    img = load_canonical(case.image_path)
    ct = _to_screen(np.asanyarray(img.dataobj)[:, :, z].astype(np.float32))
    labels = _to_screen(load_label_map(case)[:, :, z])
    sx, sy = img.header.get_zooms()[:2]

    # Cut to the kidneys plus a margin, so they are large enough to see.
    rows, cols = np.nonzero(labels > 0)
    my, mx = int(g["margin_mm"] / sy), int(g["margin_mm"] / sx)
    r0, r1 = max(rows.min() - my, 0), min(rows.max() + my, ct.shape[0])
    c0, c1 = max(cols.min() - mx, 0), min(cols.max() + mx, ct.shape[1])
    ct, labels = ct[r0:r1, c0:c1], labels[r0:r1, c0:c1]

    w = study["viewer"]["window_presets"][0]
    lo, hi = w["level"] - w["width"] / 2, w["level"] + w["width"] / 2
    width_mm, height_mm = ct.shape[1] * sx, ct.shape[0] * sy
    extent = (0, width_mm, height_mm, 0)                     # in mm, so the pixels are not stretched
    xs = (np.arange(ct.shape[1]) + 0.5) * sx                 # pixel centers in mm, for the outlines
    ys = (np.arange(ct.shape[0]) + 0.5) * sy
    panel_in = 5.5
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_in, panel_in * height_mm / width_mm + 0.75), facecolor="white")
    for ax in axes:
        ax.imshow(ct, cmap="gray", vmin=lo, vmax=hi, extent=extent, interpolation="bilinear")
        ax.set_axis_off()
        for x, text in ((0.035, "R"), (0.965, "L")):
            ax.text(x, 0.93, text, transform=ax.transAxes, color="white", fontsize=15, fontweight="bold",
                    ha="center", va="center")

    # Left: outline and name each structure. Outlines follow the outer edge (small holes filled).
    colors = g["colors"]
    left = axes[0]
    for key, value in g["labels"].items():
        mask = ndimage.binary_fill_holes(labels == value)
        if not mask.any():
            continue
        left.contour(xs, ys, mask.astype(float), levels=[0.5], colors=[colors[key]], linewidths=2)
        pieces, n = ndimage.label(mask)
        sizes = np.bincount(pieces.ravel())[1:]
        # Name the largest piece; for the kidney, the largest piece on each side of the image.
        chosen = {}
        for piece in np.argsort(sizes)[::-1]:
            cy, cx = ndimage.center_of_mass(pieces == piece + 1)
            side = "left" if (cx + 0.5) * sx < width_mm / 2 else "right"
            if key == "kidney":
                chosen.setdefault(side, (piece, cx, cy))
            else:
                chosen.setdefault("only", (piece, cx, cy))
        for side, (piece, cx, cy) in chosen.items():
            rows, cols = np.nonzero(pieces == piece + 1)
            px, py = (cx + 0.5) * sx, (cy + 0.5) * sy
            if key == "kidney":          # beside the kidney, towards the edge of the image
                tx = cols.min() * sx - 0.07 * width_mm if side == "left" else (cols.max() + 1) * sx + 0.07 * width_mm
                ty, ha = py - 0.12 * height_mm, "right" if side == "left" else "left"
            else:                        # tumor below, cyst above
                tx, ha = px, "center"
                ty = (rows.max() + 1) * sy + 0.1 * height_mm if key == "tumor" else rows.min() * sy - 0.1 * height_mm
            left.annotate(key, xy=(px, py), xytext=(tx, ty), color=colors[key], fontsize=13, fontweight="bold",
                          ha=ha, va="center", arrowprops={"arrowstyle": "->", "color": colors[key], "lw": 1.8})
    left.set_title("Kidneys and a tumor", fontsize=13)

    # Right: what should be marked, in the viewer's mask color.
    right = axes[1]
    organ = labels > 0
    rgba = np.zeros(organ.shape + (4,))
    red = np.array(study["viewer"]["mask_color"]) / 255
    rgba[organ] = [*red, study["viewer"]["mask_opacity"]]
    right.imshow(rgba, extent=extent, interpolation="nearest")
    right.contour(xs, ys, organ.astype(float), levels=[0.5], colors=[red], linewidths=2)
    right.set_title("What should be marked (red): kidney + tumor + cyst", fontsize=13)

    fig.text(0.5, 0.015, "R = the patient's right side. It is shown on the LEFT, as if you look at the patient "
             "from the feet. The front of the body is at the top.", ha="center", va="bottom", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
