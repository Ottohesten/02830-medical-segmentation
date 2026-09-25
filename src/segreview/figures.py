"""Figures for the G1 evaluation: review curve, quality curve, score-vs-Dice scatter plots, and the
Dice distribution under different ground-truth definitions of the organ.

Colors follow the method, never its rank, so a method has the same color in every figure.
Uncertainty measures get distinct colors; the baselines and the oracle are neutral gray/black
with different line styles, so they read as reference lines and can be told apart without color.
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

# Readable names and fixed styles per method: (label, color, line style).
METHODS = {
    "entropy_sum_ml":      ("Entropy, total (not size-normalized)", "#2a78d6", "-"),
    "entropy_mean_region": ("Entropy, mean in organ + border",      "#eb6834", "-"),
    "entropy_per_volume":  ("Entropy per organ volume",             "#1baf7a", "-"),
    "soft_dice_gap":       ("Soft-Dice gap",                        "#eda100", "-"),
    "tta_disagreement":    ("TTA disagreement",                     "#e87ba4", "-"),
    "tta_std_mean_region": ("TTA std, mean in organ + border",      "#4a3aa7", "-"),
    "model_disagreement":  ("Main vs second model: 1 - Dice",       "#008300", "-"),
    "model_diff_mean_region": ("Main vs second model: mean |dp| in organ + border", "#e34948", "-"),
    "plaus_asymmetry":     ("Plausibility: left/right asymmetry",   "#e87ba4", "--"),
    "plaus_missing_side":  ("Plausibility: organ side missing",     "#e87ba4", ":"),
    "combo_mean_asym":     ("Combination: mean rank (with asymmetry)", "#4a3aa7", "-."),
    "combo_max_asym":      ("Combination: max rank (with asymmetry)",  "#4a3aa7", "--"),
    "combo_mean_missing":  ("Combination: mean rank (with missing side)", "#4a3aa7", ":"),
    "combo_max_missing":   ("Combination: max rank (with missing side)",  "#4a3aa7", "-"),
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
    # Room below the panels for one legend line per method, so the legend never covers the curves.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0 + 0.2 * len(curves)), facecolor=SURFACE)
    auc = {r["method"]: (r["review_auc"], r["quality_auc"]) for r in rows}
    panels = [("Review curve: bad segmentations found", "share of bad segmentations found", 0),
              ("Quality curve: mean Dice after correcting reviewed scans", "mean Dice of whole dataset", 1)]
    for ax, (panel_title, ylabel, i) in zip(axes, panels):
        for method in sorted(curves, key=lambda m: -auc[m][i]):
            y = curves[method][i]
            x = np.linspace(0, 1, len(y))
            name, color, ls = METHODS.get(method, (method, TEXT_MUTED, "-"))
            ax.plot(x, y, color=color, linestyle=ls, linewidth=2, label=f"{name}  (AUC {auc[method][i]:.3f})")
        ax.set_title(panel_title, color=TEXT, loc="left", fontsize=11)
        ax.set_xlabel("share of scans sent to review (most uncertain first)", color=TEXT_MUTED)
        ax.set_ylabel(ylabel, color=TEXT_MUTED)
        ax.set_xlim(0, 1)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left", bbox_to_anchor=(0.0, -0.13))
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


def plot_label_check(df, definitions: list[str], coverage_cols: list[str], dice_bin_width: float,
                     share_bin_width: float, seed: int, title: str, path: Path, dpi: int) -> None:
    """Top row: Dice distribution for each organ definition. Bottom row: how much of each ground-truth
    label the model calls organ (coverage).

    Every panel is a histogram with every scan drawn as a dot below it, so single scans stay visible.

    Input: table with one row per scan and columns dice_<definition> and coverage_label_<v>.
    """
    from matplotlib.ticker import MaxNLocator

    rows = [[(f"dice_{d}", f"Dice, definition '{d}'", "Dice", dice_bin_width) for d in definitions],
            [(c, f"Share of GT label {c.split('_')[-1]} called organ", "share (0-1)", share_bin_width)
             for c in coverage_cols]]
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(4, n_cols, figsize=(4.4 * n_cols, 8.4), facecolor=SURFACE,
                             height_ratios=[4, 1, 4, 1], squeeze=False)
    jitter = np.random.default_rng(seed).uniform(-0.3, 0.3, len(df))
    for r, panels in enumerate(rows):
        for i in range(n_cols):
            ax, strip = axes[2 * r, i], axes[2 * r + 1, i]
            if i >= len(panels):
                ax.set_visible(False)
                strip.set_visible(False)
                continue
            col, panel_title, xlabel, width = panels[i]
            values = df[col].to_numpy(dtype=float)
            ok = np.isfinite(values)
            lo = np.floor(np.nanmin(values) / width) * width if ok.any() else 0.0
            edges = np.arange(min(lo, 1.0 - width), 1.0 + width / 2, width)
            ax.hist(values[ok], bins=edges, color="#2a78d6", edgecolor=SURFACE, linewidth=1)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            stats = (f"min {np.nanmin(values):.3f}  median {np.nanmedian(values):.3f}  max {np.nanmax(values):.3f}"
                     if ok.any() else "no data")
            ax.set_title(f"{panel_title}\n{stats}  (n = {int(ok.sum())})", color=TEXT, fontsize=9, loc="left")
            ax.set_ylabel("number of scans", color=TEXT_MUTED, fontsize=8)
            strip.scatter(values[ok], jitter[ok], s=22, color="#2a78d6", edgecolor=SURFACE, linewidth=1, zorder=3)
            strip.set_ylim(-1, 1)
            strip.set_yticks([])
            strip.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=8)
            strip.set_xlim(ax.get_xlim())
            for a in (ax, strip):
                _style(a)
            strip.grid(False)
    fig.suptitle(title, color=TEXT, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
