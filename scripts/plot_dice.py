"""Summarise and plot how Dice is spread over the scans (evaluation only).

G1 ranks scans by uncertainty and checks whether that ranking matches the ranking by true Dice.
That only makes sense if Dice actually differs between scans, so this is the first thing to check.

Reads the newest results/dice_<config name>_<date>.csv (made by scripts/evaluate.py) and writes:
- results/dice_summary_<config name>_<date>.csv : n, min, max, median, mean, std, quartiles
- results/figures/dice_hist_<config name>_<date>.png : histogram with every scan shown as a dot below it

Usage: uv run python scripts/plot_dice.py --config configs/local_all.yaml
"""

from datetime import date

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from segreview.config import config_arg_parser, load_config

# Colours: one series, so one colour. Text and axes stay neutral grey/black so the data stands out.
BAR = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    vis = cfg["visualisation"]
    results = cfg["paths"]["results_dir"]

    files = sorted(results.glob(f"dice_{cfg['name']}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No dice_{cfg['name']}_*.csv in {results}. Run scripts/evaluate.py first.")
    df = pd.read_csv(files[-1])
    d = df["dice"]

    summary = {"n": len(d), "min": d.min(), "q25": d.quantile(0.25), "median": d.median(),
               "q75": d.quantile(0.75), "max": d.max(), "mean": d.mean(), "std": d.std()}
    today = date.today().isoformat()
    pd.DataFrame([summary]).round(4).to_csv(results / f"dice_summary_{cfg['name']}_{today}.csv", index=False)
    print(f"Source: {files[-1].name}")
    for key, value in summary.items():
        print(f"  {key:>6}: {value:.4f}" if key != "n" else f"  {key:>6}: {value}")

    # Bins of fixed width, from just below the worst scan up to a perfect Dice of 1.
    width = vis["dice_bin_width"]
    edges = np.arange(np.floor(d.min() / width) * width, 1.0 + width / 2, width)

    fig, (ax, strip) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, height_ratios=[4, 1],
                                    facecolor=SURFACE)
    ax.hist(d, bins=edges, color=BAR, edgecolor=SURFACE, linewidth=2)
    ax.axvline(summary["median"], color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)  # headroom so the median label does not sit on a bar
    ax.annotate(f"median {summary['median']:.3f}", (summary["median"], ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", ha="left", color=TEXT_MUTED, fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts are whole numbers
    ax.set_ylabel("number of scans", color=TEXT_MUTED)
    ax.set_title(f"Dice per scan, {cfg['model']['organ']}, config '{cfg['name']}' "
                 f"({cfg['model']['resolution']} resolution, n = {len(d)})", color=TEXT, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    # Every scan as a dot, so single outliers are visible even when a bar is only 1 high.
    jitter = np.random.default_rng(cfg["seed"]).uniform(-0.3, 0.3, len(d))
    strip.scatter(d, jitter, s=36, color=BAR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    # Name the worst scans. Labels alternate above/below the dots so neighbours do not overlap.
    worst = df.nsmallest(vis["label_worst_n"], "dice").sort_values("dice")
    for i, (idx, row) in enumerate(worst.iterrows()):
        strip.annotate(row["case_id"], (row["dice"], jitter[df.index.get_loc(idx)]),
                       xytext=(0, 8 if i % 2 == 0 else -14), textcoords="offset points",
                       ha="center", fontsize=8, color=TEXT_MUTED)
    strip.set_ylim(-1.0, 1.0)
    strip.set_yticks([])
    strip.set_xlabel("Dice (model vs. ground truth)", color=TEXT_MUTED)

    for a in (ax, strip):
        a.set_facecolor(SURFACE)
        a.set_axisbelow(True)
        a.tick_params(colors=TEXT_MUTED)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
    strip.spines["left"].set_visible(False)
    fig.tight_layout()

    out = results / "figures" / f"dice_hist_{cfg['name']}_{today}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=vis["dpi"], facecolor=SURFACE)
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
