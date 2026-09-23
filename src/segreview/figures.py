"""Figures for the G1 evaluation: review curve, quality curve and score-vs-Dice scatter plots.

Colours follow the method, never its rank, so a method has the same colour in every figure.
Uncertainty measures get distinct colours; the baselines and the oracle are neutral grey/black
with different line styles, so they read as reference lines and can be told apart without colour.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

# Readable names and fixed styles per method: (label, colour, line style).
METHODS = {
    "entropy_sum_ml":      ("Entropy, total (not size-normalised)", "#2a78d6", "-"),
    "entropy_mean_region": ("Entropy, mean in organ + border",      "#eb6834", "-"),
    "entropy_per_volume":  ("Entropy per organ volume",             "#1baf7a", "-"),
    "soft_dice_gap":       ("Soft-Dice gap",                        "#eda100", "-"),
    "tta_disagreement":    ("TTA disagreement",                     "#e87ba4", "-"),
    "tta_std_mean_region": ("TTA std, mean in organ + border",      "#4a3aa7", "-"),
    "neg_volume_ml":       ("Baseline: small predicted volume",     "#52514e", "-."),
    "random":              ("Baseline: random order",               "#9a9994", "--"),
    "oracle":              ("Oracle: true Dice (upper bound)",      "#0b0b0b", ":"),
}


def label(method: str) -> str:
    """Readable name of a method, for legends and tables."""
    return METHODS.get(method, (method,))[0]


def _style(ax) -> None:
    """Neutral, recessive axes so the data stands out."""
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_curves(rows: list[dict], curves: dict, title: str, path: Path, dpi: int) -> None:
    """Review curve and quality curve side by side, all methods in the same figure.

    The legend lists each method with the area under the curve (AUC), best first.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)
    auc = {r["method"]: (r["review_auc"], r["quality_auc"]) for r in rows}
    panels = [("Review curve: bad segmentations found", "share of bad segmentations found", 0),
              ("Quality curve: mean Dice after correcting reviewed scans", "mean Dice of whole dataset", 1)]
    for ax, (panel_title, ylabel, i) in zip(axes, panels):
        for method in sorted(curves, key=lambda m: -auc[m][i]):
            y = curves[method][i]
            x = np.linspace(0, 1, len(y))
            name, colour, ls = METHODS.get(method, (method, TEXT_MUTED, "-"))
            ax.plot(x, y, color=colour, linestyle=ls, linewidth=2, label=f"{name}  (AUC {auc[method][i]:.3f})")
        ax.set_title(panel_title, color=TEXT, loc="left", fontsize=11)
        ax.set_xlabel("share of scans sent to review (most uncertain first)", color=TEXT_MUTED)
        ax.set_ylabel(ylabel, color=TEXT_MUTED)
        ax.set_xlim(0, 1)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    fig.suptitle(title, color=TEXT, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_scatter(scores: dict[str, np.ndarray], dice: np.ndarray, bad: np.ndarray, case_ids: list[str],
                 rows: list[dict], title: str, path: Path, dpi: int) -> None:
    """One small panel per method: score (x) against true Dice (y), with Spearman rho in the title.

    Bad segmentations (by the config's rule) are drawn as open rings and named, so it is easy to
    see whether a method puts them on the right (high uncertainty).
    """
    rho = {r["method"]: (r["spearman_rho"], r["rho_ci_low"], r["rho_ci_high"]) for r in rows}
    methods = list(scores)
    cols = 3
    rows_n = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.3 * cols, 3.6 * rows_n), facecolor=SURFACE, squeeze=False)
    for ax, method in zip(axes.flat, methods):
        s = scores[method]
        ax.scatter(s[~bad], dice[~bad], s=28, color="#2a78d6", edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.scatter(s[bad], dice[bad], s=46, facecolor="none", edgecolor=TEXT, linewidth=1.5, zorder=4)
        for i in np.flatnonzero(bad):
            ax.annotate(case_ids[i], (s[i], dice[i]), xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=TEXT_MUTED)
        r, lo, hi = rho[method]
        ax.set_title(f"{label(method)}\nrho = {r:.2f}  [{lo:.2f}, {hi:.2f}]", color=TEXT, fontsize=9, loc="left")
        ax.set_xlabel("score (higher = more uncertain)", color=TEXT_MUTED, fontsize=8)
        ax.set_ylabel("true Dice", color=TEXT_MUTED, fontsize=8)
        _style(ax)
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)
    fig.suptitle(title, color=TEXT, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
