"""Figures and tables of the G1 results for the report (ACM two-column paper). No new analysis happens here.

Everything is built from the files the G1 evaluation already wrote (results/<config>/g1_scores_<set>.csv,
g1_metrics_<set>.csv, heatmap_eval_<set>_summary.csv) and from the saved model output, so the numbers in the
report are exactly the evaluated ones. The curves are recomputed from the saved scores with the same functions
as the evaluation, and their areas are checked against the saved AUCs.

Style: one look for all figures. Sizes follow the ACM template (single column 3.33 in, double column 7 in),
serif text at 7-8 pt, fonts embedded in the PDF. Color follows the method: the three highlighted measures
have fixed colors (a validated color-blind-safe set), the other GT-free measures are thin gray lines, and the
baselines and the oracle are black/gray with different line styles, so they can be told apart without color.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segreview.evaluation import area, bad_mask, curves_for, random_curves

SINGLE, DOUBLE = 3.33, 7.0          # ACM column widths in inches

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HIGHLIGHT = {                        # method -> color (checked with the palette validator: CVD-safe as a set)
    "model_diff_mean_region": "#2a78d6",
    "combo_mean_missing": "#eb6834",
    "entropy_mean_region": "#1baf7a",
}
REFERENCE = {                        # method -> (color, line style, width)
    "oracle": (INK, "-", 0.9),
    "neg_volume_ml": (INK, "--", 1.0),
    "random": ("#8f8e88", ":", 1.2),
}
OTHER = ("#b9b8b1", "-", 0.6)        # the remaining GT-free measures
LABELS = {
    "model_diff_mean_region": "Model disagreement, mean |Δp| (primary)",
    "combo_mean_missing": "Combination with plausibility (primary)",
    "entropy_mean_region": "Entropy, mean in region",
    "entropy_sum_ml": "Entropy, total",
    "entropy_per_volume": "Entropy per volume",
    "soft_dice_gap": "Soft-Dice gap",
    "model_disagreement": "Model disagreement, 1 − Dice",
    "plaus_asymmetry": "Plausibility: asymmetry",
    "plaus_missing_side": "Plausibility: missing side",
    "neg_volume_ml": "Predicted volume (baseline)",
    "random": "Random order (baseline)",
    "oracle": "Oracle (true Dice)",
}
RULE_TITLES = {"worst10": "worst 10 %", "dice_below_083": "Dice < 0.83"}
# Table order: baselines, GT-free measures, combination, oracle.
TABLE_ORDER = ["random", "neg_volume_ml", "entropy_sum_ml", "entropy_mean_region", "entropy_per_volume",
               "soft_dice_gap", "model_disagreement", "model_diff_mean_region", "plaus_asymmetry",
               "plaus_missing_side", "combo_mean_missing", "oracle"]
PRIMARY = {"model_diff_mean_region", "combo_mean_missing"}


def use_style() -> None:
    """Matplotlib settings shared by all report figures."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Linux Libertine O", "Libertinus Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False,
    })


def save(fig, out: Path, name: str) -> None:
    """Save a figure as PDF (for LaTeX; vector lines and text) and a smaller PNG (for a quick look)."""
    out.mkdir(parents=True, exist_ok=True)
    # No creation date in the PDF, so running the script again gives byte-identical files (clean git history).
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=150)   # preview only
    plt.close(fig)


def method_style(method: str) -> dict:
    if method in HIGHLIGHT:
        return {"color": HIGHLIGHT[method], "ls": "-", "lw": 1.4, "zorder": 4}
    if method in REFERENCE:
        c, ls, lw = REFERENCE[method]
        return {"color": c, "ls": ls, "lw": lw, "zorder": 3}
    return {"color": OTHER[0], "ls": OTHER[1], "lw": OTHER[2], "zorder": 2}


def curves_from_scores(scores: pd.DataFrame, methods: list[str], rules: list[dict], cfg: dict,
                       metrics: pd.DataFrame) -> dict:
    """Review curves per rule and the quality curve per method, recomputed exactly as in the evaluation.

    The random curve uses a fresh generator with the config seed, like evaluate_methods does, so it is the
    same curve. Every recomputed area is checked against the saved AUC (rounded to 4 decimals there).
    Output: {"review": {rule: {method: y}}, "quality": {method: y}}.
    """
    dice = scores["dice"].to_numpy()
    out = {"review": {}, "quality": {}}
    for rule in rules:
        bad = bad_mask(dice, rule["rule"], rule["value"])
        r_rand, q_rand, _ = random_curves(dice, bad, cfg["evaluation"]["n_random_orders"],
                                          np.random.default_rng(cfg["seed"]))
        per = {"random": r_rand}
        out["quality"]["random"] = q_rand
        for m in methods:
            s = -dice if m == "oracle" else scores[m].to_numpy(dtype=float)
            per[m], out["quality"][m] = curves_for(s, dice, bad)
        out["review"][rule["name"]] = per
        saved = metrics[metrics.bad_rule == rule["name"]].set_index("method")
        for m, y in per.items():
            if abs(area(y) - saved.loc[m, "review_auc"]) > 6e-5:
                raise ValueError(f"review AUC of {m} ({rule['name']}) differs from the evaluated value")
            if abs(area(out["quality"][m]) - saved.loc[m, "quality_auc"]) > 6e-5:
                raise ValueError(f"quality AUC of {m} differs from the evaluated value")
    return out


def figure_curves(curves: dict, rules: list[dict], n_bad: dict, n: int, out: Path) -> None:
    """Review curves (one panel per 'bad' rule) and the quality curve, double column, one shared legend."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.35))
    x = np.linspace(0, 100, n + 1)
    draw_order = [m for m in curves["quality"] if m not in HIGHLIGHT and m not in REFERENCE] + \
                 list(REFERENCE) + list(HIGHLIGHT)
    for ax, rule in zip(axes[:2], rules):
        for m in draw_order:
            ax.plot(x, 100 * curves["review"][rule["name"]][m], **method_style(m))
        ax.set_title(f"({'ab'[rules.index(rule)]}) Review curve, bad = {RULE_TITLES[rule['name']]} "
                     f"({n_bad[rule['name']]} of {n})")
        ax.set_xlabel("Scans reviewed, most uncertain first (%)")
        ax.set_ylabel("Bad segmentations found (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
    ax = axes[2]
    for m in draw_order:
        ax.plot(x, curves["quality"][m], **method_style(m))
    ax.set_title("(c) Quality curve")
    ax.set_xlabel("Scans reviewed and corrected (%)")
    ax.set_ylabel("Mean Dice of all scans")
    ax.set_xlim(0, 100)
    ax.set_ylim(min(curves["quality"]["random"][0], 0.9) - 0.002, 1.001)
    # One legend for all panels: highlighted measures, baselines, then one entry for the other measures.
    handles = [plt.Line2D([], [], **{k: v for k, v in method_style(m).items() if k != "zorder"}, label=LABELS[m])
               for m in list(HIGHLIGHT) + ["neg_volume_ml", "random", "oracle"]]
    handles.append(plt.Line2D([], [], color=OTHER[0], lw=OTHER[2], label="Other GT-free measures"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2,
               handlelength=2.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.2)
    save(fig, out, "fig_g1_curves_test")


def figure_auc_vs_volume(metrics: pd.DataFrame, out: Path) -> None:
    """AUC of each GT-free measure minus the volume baseline, with the paired 95 % bootstrap CI.

    A measure beats the baseline when its whole interval lies right of 0 (the locked criterion).
    """
    methods = [m for m in TABLE_ORDER if m not in {"random", "oracle", "neg_volume_ml"}]
    panels = [("worst10", "review", "(a) Review AUC, worst 10 %"), ("dice_below_083", "review", "(b) Review AUC, Dice < 0.83"),
              ("worst10", "quality", "(c) Quality AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (rule, curve, title) in zip(axes, panels):
        m = metrics[metrics.bad_rule == rule].set_index("method")
        for yi, meth in zip(y, methods):
            lo, hi = m.loc[meth, f"{curve}_auc_minus_volume_ci_low"], m.loc[meth, f"{curve}_auc_minus_volume_ci_high"]
            est = m.loc[meth, f"{curve}_auc_minus_volume"]
            color = HIGHLIGHT.get(meth, MUTED)
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.4 if meth in HIGHLIGHT else 0.9, solid_capstyle="butt")
            ax.plot(est, yi, "o", ms=3.8, color=color, mec="white", mew=0.6, zorder=3)
        ax.axvline(0, color=INK, lw=0.7)
        ax.set_title(title)
        ax.set_xlabel("AUC minus volume baseline")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [LABELS[m] for m in methods])
    for tick, meth in zip(axes[0].get_yticklabels(), methods):
        if meth in PRIMARY:
            tick.set_fontweight("bold")
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig_g1_auc_vs_volume_test")


def pick_examples(scores: pd.DataFrame, primary: str, bad_rule: str) -> list[tuple[str, str]]:
    """Three test scans for the heatmap figure, chosen by fixed rules (not by eye):
    - certain and correct: the scan with the lowest primary score (the model's most certain one);
    - uncertain and wrong: the 'bad' scan with the highest primary score;
    - certain but wrong: the 'bad' scan with the lowest primary score (a failure uncertainty cannot see).
    """
    bad = scores[scores[f"bad_{bad_rule}"]]
    return [(scores.loc[scores[primary].idxmin(), "case_id"], "Certain and correct"),
            (bad.loc[bad[primary].idxmax(), "case_id"], "Uncertain and wrong"),
            (bad.loc[bad[primary].idxmin(), "case_id"], "Certain but wrong")]


def figure_heatmap_examples(cfg: dict, examples: list[tuple[str, str]], scores: pd.DataFrame, primary: str,
                            heatmap: str, out: Path) -> pd.DataFrame:
    """For each example scan: the slice with the most wrong voxels, (top) model mask vs ground truth,
    (bottom) the voxel uncertainty heatmap that the review interface shows. Radiological view (patient's
    right on the left, front at the top), cut to the kidneys plus a margin.
    Output: a small table with the chosen scans and slices (written next to the figure).
    """
    from matplotlib.colors import LinearSegmentedColormap

    from segreview.data import list_cases, load_canonical, load_organ_mask
    from segreview.guide import _to_screen
    from segreview.io import load_map

    cases = {c.case_id: c for c in list_cases(cfg)}
    heat_cmap = LinearSegmentedColormap.from_list("blue2cyan", ["#0000ff", "#00ffff"])   # as in the interface
    queue_pos = scores[primary].rank(ascending=False, method="first").astype(int)       # 1 = reviewed first
    aspect, margin_mm = 1.5, 25.0                  # every panel shows the same width:height window
    fig = plt.figure(figsize=(DOUBLE, 3.95))
    left, width, gap = 0.004, (1 - 0.008 - 2 * 0.012) / 3, 0.012
    panel_h = width * DOUBLE / aspect / 3.95
    axes = [[fig.add_axes([left + c * (width + gap), 0.135 + (1 - r) * (panel_h + 0.012), width, panel_h])
             for c in range(3)] for r in range(2)]
    rows = []
    for col, (case_id, what) in enumerate(examples):
        case = cases[case_id]
        img = load_canonical(case.image_path)
        mask = load_map(cfg, "clean", case_id, "mask") > 0
        heat = load_map(cfg, "clean", case_id, heatmap)
        truth, _ = load_organ_mask(cfg, case)
        z = int(np.argmax((mask != truth).sum(axis=(0, 1))))       # the slice with the most wrong voxels
        sx, sy = img.header.get_zooms()[:2]
        ct = _to_screen(np.asanyarray(img.dataobj)[:, :, z].astype(np.float32))
        m, g, h = (_to_screen(a[:, :, z]) for a in (mask, truth, heat))
        ext = (0, ct.shape[1] * sx, ct.shape[0] * sy, 0)
        xs, ys = (np.arange(ct.shape[1]) + 0.5) * sx, (np.arange(ct.shape[0]) + 0.5) * sy
        # Window (in mm) around the kidneys: the box around mask and ground truth plus a margin, widened to the
        # common aspect ratio. Outside the scan the window is black.
        rr, cc = np.nonzero(m | g)
        cx, cy = (cc.min() + cc.max() + 1) / 2 * sx, (rr.min() + rr.max() + 1) / 2 * sy
        half_w = max((cc.max() - cc.min() + 1) * sx / 2 + margin_mm, ((rr.max() - rr.min() + 1) * sy / 2 + margin_mm) * aspect)
        half_h = half_w / aspect
        row = scores[scores.case_id == case_id].iloc[0]
        for r_idx in range(2):
            ax = axes[r_idx][col]
            ax.set_facecolor("black")
            ax.imshow(ct, cmap="gray", vmin=-160, vmax=240, extent=ext, interpolation="bilinear")
            if r_idx == 1:
                shown = np.ma.masked_less(h, 0.1)          # as in the interface: entropy below 0.1 bit not drawn
                im = ax.imshow(shown, cmap=heat_cmap, vmin=0.1, vmax=1, alpha=0.75, extent=ext, interpolation="nearest")
            if g.any() and r_idx == 0:
                ax.contour(xs, ys, g.astype(float), levels=[0.5], colors=["white"], linewidths=0.9, linestyles="--")
            if m.any():
                ax.contour(xs, ys, m.astype(float), levels=[0.5], colors=["#e34948"], linewidths=0.9)
            ax.set_xlim(cx - half_w, cx + half_w)
            ax.set_ylim(cy + half_h, cy - half_h)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0.03, 0.95, "R", transform=ax.transAxes, color="white", fontsize=7, va="top", fontweight="bold")
        axes[0][col].set_title(f"({'abc'[col]}) {what}\n{case_id}: Dice {row.dice:.2f}, queue position "
                               f"{queue_pos[row.name]} of {len(scores)}", fontsize=7.5, pad=3)
        rows.append({"panel": "abc"[col], "category": what, "case_id": case_id, "slice": z, "dice": row.dice,
                     primary: row[primary], "queue_position": int(queue_pos[row.name])})
    handles = [plt.Line2D([], [], color="#e34948", lw=1.0, label="Model mask"),
               plt.Line2D([], [], color=MUTED, lw=1.0, ls="--", label="Ground truth (white, top row)")]
    fig.legend(handles=handles, loc="lower left", ncol=2, bbox_to_anchor=(0.0, 0.0))
    cax = fig.add_axes([0.63, 0.075, 0.3, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Voxel entropy (bits), bottom row", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.5)
    save(fig, out, "fig_heatmap_examples_test")
    return pd.DataFrame(rows)


def _fmt_ci(v, lo, hi, digits=3) -> str:
    if pd.isna(v):
        return "–"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]" if not pd.isna(lo) else f"{v:.{digits}f}"


def g1_table(metrics: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """All methods with Spearman rho, partial rho, review AUC per rule and quality AUC (95 % CI).

    'beats_volume_*' says whether the lower CI bound of the AUC difference to the volume baseline is > 0.
    """
    first = metrics[metrics.bad_rule == rules[0]["name"]].set_index("method")
    rows = []
    for m in TABLE_ORDER:
        f = first.loc[m]
        row = {"method": m, "label": LABELS[m], "primary": m in PRIMARY,
               "spearman_rho": _fmt_ci(f.spearman_rho, f.rho_ci_low, f.rho_ci_high, 2),
               "partial_rho_given_volume": _fmt_ci(f.partial_rho_given_volume, f.partial_rho_ci_low, f.partial_rho_ci_high, 2)}
        for rule in rules:
            r = metrics[metrics.bad_rule == rule["name"]].set_index("method").loc[m]
            row[f"review_auc_{rule['name']}"] = _fmt_ci(r.review_auc, r.review_auc_ci_low, r.review_auc_ci_high)
            row[f"beats_volume_review_{rule['name']}"] = bool(r.review_auc_minus_volume_ci_low > 0)
        row["quality_auc"] = _fmt_ci(f.quality_auc, f.quality_auc_ci_low, f.quality_auc_ci_high)
        row["beats_volume_quality"] = bool(f.quality_auc_minus_volume_ci_low > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _tex_num(text: str) -> str:
    """Minus signs as math minus (a hyphen looks wrong in a table), dashes for missing values."""
    import re
    return re.sub(r"-(?=\d)", "$-$", text.replace("–", "--"))


def _tex_ci(v, lo, hi, digits: int) -> str:
    """Value and CI with the given number of decimals (2 for rho and review AUC, 3 for the quality AUC, whose
    differences are in the third decimal); fewer digits keep the table within the page width."""
    return "--" if pd.isna(v) else _tex_num(f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]")


def g1_latex(metrics: pd.DataFrame, rules: list[dict], n: int, n_bad: dict) -> str:
    """The G1 table as LaTeX (booktabs, table* over both columns). Partial rho is given without its CI here
    (it is in the CSV table) so the table fits the ACM page width."""
    groups = {"random": "Baselines", "entropy_sum_ml": "GT-free uncertainty measures",
              "plaus_asymmetry": "Plausibility", "combo_mean_missing": "Combination", "oracle": "Upper bound"}
    by_rule = {r["name"]: metrics[metrics.bad_rule == r["name"]].set_index("method") for r in rules}
    first = by_rule[rules[0]["name"]]
    head = [RULE_TITLES[r["name"]].replace("%", r"\%").replace("<", "$<$") + f" ({n_bad[r['name']]} bad)" for r in rules]
    lines = [r"\begin{table*}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{G1 on the KiTS test set ($n = %d$ scans). Spearman $\rho$ between score and true Dice (negative ="
             r" good), partial $\rho$ given predicted volume, and areas under the review and quality curves (AUC), with"
             r" 95\%% bootstrap CIs. $^\dagger$: the lower CI bound of the AUC difference to the volume baseline is"
             r" above 0. Bold: primary measures (locked on the development set).}" % n,
             r"\label{tab:g1}", r"\begin{tabular}{lccccc}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Review AUC, bad =} & \\", r"\cmidrule(lr){4-5}",
             r"Method & Spearman $\rho$ & Partial $\rho$ & %s & %s & Quality AUC \\" % tuple(head), r"\midrule"]
    for m in TABLE_ORDER:
        if m in groups:
            if m != "random":
                lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{6}{l}{\emph{%s}} \\" % groups[m])
        f = first.loc[m]
        name = (LABELS[m].replace("|Δp|", r"$|\Delta p|$").replace("−", "$-$").replace(" (primary)", "")
                .replace(" (baseline)", "").replace("Plausibility: ", "").capitalize() if m.startswith("plaus")
                else LABELS[m].replace("|Δp|", r"$|\Delta p|$").replace("−", "$-$").replace(" (primary)", "")
                .replace(" (baseline)", ""))
        name = rf"\textbf{{{name}}}" if m in PRIMARY else name
        cells = [_tex_ci(f.spearman_rho, f.rho_ci_low, f.rho_ci_high, 2),
                 "--" if pd.isna(f.partial_rho_given_volume) else _tex_num(f"{f.partial_rho_given_volume:.2f}")]
        for r in rules:
            x = by_rule[r["name"]].loc[m]
            cells.append(_tex_ci(x.review_auc, x.review_auc_ci_low, x.review_auc_ci_high, 2)
                         + (r"$^\dagger$" if x.review_auc_minus_volume_ci_low > 0 else ""))
        cells.append(_tex_ci(f.quality_auc, f.quality_auc_ci_low, f.quality_auc_ci_high, 3)
                     + (r"$^\dagger$" if f.quality_auc_minus_volume_ci_low > 0 else ""))
        lines.append(f"\\quad {name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def heatmap_table(summary: pd.DataFrame, n: int) -> tuple[pd.DataFrame, str]:
    """Error localization of the candidate heatmaps on the test set (median and range), as CSV table and LaTeX."""
    s = summary.set_index("measure")
    names = {"entropy": "Voxel entropy (chosen)", "m2_diff": r"3 mm vs 6 mm model, $|p_3 - p_6|$",
             "boundary": "Distance to predicted edge (reference)"}
    rows = [{"heatmap": k, "label": v,
             "voxel_auc": f"{s.loc[f'auc_{k}', 'median']:.3f} ({s.loc[f'auc_{k}', 'min']:.2f}--{s.loc[f'auc_{k}', 'max']:.2f})",
             "top10_share": f"{s.loc[f'top_{k}', 'median']:.3f} ({s.loc[f'top_{k}', 'min']:.2f}--{s.loc[f'top_{k}', 'max']:.2f})"}
            for k, v in names.items()]
    table = pd.DataFrame(rows)
    tex = [r"\begin{table}", r"\centering\small",
           r"\caption{Where are the wrong voxels? Heatmaps on the KiTS test set ($n = %d$), inside the organ plus a 5\,mm"
           r" border: median (range) of the voxel-level AUC for wrong voxels, and the share of wrong voxels in the"
           r" heatmap's top 10\%%. The heatmap was chosen on the development set before this evaluation.}" % n,
           r"\label{tab:heatmaps}", r"\begin{tabular}{lcc}", r"\toprule", r"Heatmap & Voxel AUC & Top-10\% share \\",
           r"\midrule"]
    tex += [f"{r['label']} & {r['voxel_auc']} & {r['top10_share']} \\\\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return table, "\n".join(tex) + "\n"
