"""Figures of the exploratory follow-up analyses (3 mm vs. 6 mm disagreement, simulated voxel correction).

They use the report's style (report.use_style: ACM sizes, serif text, PDF + PNG) and its method colors, so the
entropy and model-disagreement heatmaps have the same color here as in the G1 figures. The numbers come from the
summary CSV files the scripts write; the figures only draw them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segreview.model_comparison import distance_bin_names
from segreview.report import DOUBLE, GRID, HIGHLIGHT, INK, MUTED, save, use_style

GROUP_TITLES = {"bad": "3 mm bad", "false_alarm": "false alarm (6 mm bad)", "good": "both fine", "all": "all scans"}
# Far-away disagreement by ground-truth label: categorical slots 1-4 of the validated palette (adjacent pairs pass
# the color-blind check); the grid-effect band is neutral gray, so it reads as "nothing special".
LABEL_COLORS = {"background": "#2a78d6", "kidney": "#eb6834", "tumor": "#1baf7a", "cyst": "#eda100"}
NEAR_COLOR = "#cfcec8"
SURFACE = "#fcfcfb"
# Voxel orders in the correction simulation: GT-free heatmaps in their report colors, references in gray/black.
METHOD_STYLE = {   # method -> (label, color, line style, width, marker)
    "entropy": ("Entropy (locked heatmap)", HIGHLIGHT["entropy_mean_region"], "-", 1.4, "o"),
    "m2_diff": ("|p 3 mm − p 6 mm|", HIGHLIGHT["model_diff_mean_region"], "-", 1.4, "s"),
    "boundary": ("Distance to edge (reference)", MUTED, "--", 1.0, None),
    "random": ("Random order (mean of seeds)", "#8f8e88", ":", 1.2, None),
    "oracle": ("Oracle (wrong voxels first)", INK, "-", 0.9, None),
}


def plot_disagreement(summary: pd.DataFrame, mc: dict, label_names: dict, path: Path, title: str) -> None:
    """Two panels per split and scan group, pooled over the group's disagreeing voxels.

    Left: what the disagreement is. Gray = within grid_band_mm of the 3 mm edge (grid effect); colored = farther
    away, split by ground-truth label (real differences). Right: share of disagreeing voxels where the 3 mm model is
    right, near the edge (open circle) and far from it (filled), with 50 % marked.
    Input: model_disagreement_summary.csv as a DataFrame, the config section, the label names, output path, title.
    """
    use_style()
    pooled = summary[summary.how == "pooled"]
    rows = [(s, g) for s in sorted(pooled.split.unique()) for g in ["bad", "false_alarm", "good", "all"]
            if ((pooled.split == s) & (pooled.group == g)).any()]
    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(DOUBLE, 0.3 * len(rows) + 1.3), sharey=True,
                                   gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.08})
    y, pos = [], 0.0
    for i, (split, _) in enumerate(rows):   # top to bottom, with a small gap where the split changes
        if i > 0 and split != rows[i - 1][0]:
            pos -= 0.6
        y.append(pos)
        pos -= 1.0
    band = mc["grid_band_mm"]
    for yi, (split, group) in zip(y, rows):
        r = pooled[(pooled.split == split) & (pooled.group == group)].iloc[0]
        left = 0.0
        parts = [("near", r.near_share, NEAR_COLOR)]
        parts += [(name, (1 - r.near_share) * np.nan_to_num(r[f"far_label_{name}_share"]), LABEL_COLORS[name])
                  for name in label_names.values()]
        for name, width, color in parts:
            if width <= 0:
                continue
            ax.barh(yi, width, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.0)
            if width >= 0.07:   # direct label only where it fits
                ax.text(left + width / 2, yi, f"{width:.0%}", ha="center", va="center", fontsize=6,
                        color=INK if name in ("near", "cyst", "tumor") else "white")
            left += width
        ax_r.plot([r.near_right_3mm_share], [yi], "o", mfc=SURFACE, mec=INK, mew=1.0, ms=5, zorder=3)
        ax_r.plot([r.far_right_3mm_share], [yi], "o", color=INK, ms=5, zorder=3)
    ax.set_yticks(y, [f"{s} · {GROUP_TITLES[g]} (n = {int(pooled[(pooled.split == s) & (pooled.group == g)].n_scans.iloc[0])})"
                      for s, g in rows])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of the voxels where the two masks disagree")
    ax.set_title(f"Grid effect (≤ {band:g} mm from the 3 mm edge) vs. real differences (by true label)", loc="left")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=NEAR_COLOR, label=f"≤ {band:g} mm (grid effect)")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[n], label=f"> {band:g} mm, true {n}")
                for n in label_names.values()]
    fig.legend(handles=handles, ncol=5, loc="upper left", bbox_to_anchor=(0.0, 0.0), handlelength=1.0,
               columnspacing=1.0)

    ax_r.axvline(0.5, color=GRID, linewidth=1.0, zorder=1)
    ax_r.set_xlim(0, 1)
    ax_r.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_r.set_xlabel(f"3 mm model is right (○ ≤ {band:g} mm, ● > {band:g} mm)")
    ax_r.set_title("Who is right?", loc="left")
    ax_r.grid(axis="y", visible=False)
    fig.suptitle(title + " — voxels pooled per group", x=0.0, ha="left", fontsize=8)
    fig.subplots_adjust(bottom=0.16)
    save(fig, path.parent, path.stem)
    plt.close(fig)


def plot_correction(summary: pd.DataFrame, cases: pd.DataFrame, path: Path, title: str) -> None:
    """Dice gain against the share k of the region corrected, one line per method; median (top) and mean (bottom),
    dev (left) and test (right). The x-axis is logarithmic, since the small k (what a heatmap shows) matter most.

    Input: voxel_correction_summary.csv and voxel_correction_cases.csv as DataFrames, output path, title.
    """
    use_style()
    splits = sorted(summary.split.unique())
    fig, axes = plt.subplots(2, len(splits), figsize=(DOUBLE, 4.2), sharex=True, squeeze=False)
    for col, split in enumerate(splits):
        n = int((cases.split == split).sum())
        for row, stat in enumerate(["median", "mean"]):
            ax = axes[row, col]
            for method, (label, color, ls, lw, marker) in METHOD_STYLE.items():
                d = summary[(summary.split == split) & (summary.method == method)].sort_values("k_percent")
                if d.empty:
                    continue
                ax.plot(d.k_percent, d[f"gain_{stat}"], color=color, linestyle=ls, linewidth=lw, marker=marker,
                        ms=3, label=label)
            ax.set_xscale("log")
            ks = sorted(summary.k_percent.unique())
            ax.set_xticks(ks, [f"{k:g} %" for k in ks])
            ax.minorticks_off()
            ax.set_title(f"{split} (n = {n}) — {stat} Dice gain per scan", loc="left")
            if col == 0:
                ax.set_ylabel(f"{stat} Dice gain")
            if row == 1:
                ax.set_xlabel("share k of the region (organ + border) corrected, most uncertain first")
    fig.suptitle(title, x=0.0, ha="left", fontsize=8)
    fig.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 0.0))
    save(fig, path.parent, path.stem)
    plt.close(fig)
