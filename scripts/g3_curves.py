"""G3 step 4: the learning curves and their summaries (docs/g3_plan.md, "What is reported"), from the Dice per scan
and model in results/g3_kits/dice/. It can run at any time: points that are not evaluated yet are listed and left
out; the primary summary needs every point of the summary's k values on the test set.

Output (results/g3_kits/):
- curves.csv: mean Dice per order and k, for each split (dev, test), organ definition and scan subset ("all", and on
  the test set "bad" = Dice below evaluation.bad_dice with the original model on this machine);
- summary.csv: the primary summary (test, all scans, kidney_tumor_cyst: mean over k = 10..90 of queue - mean of the
  random orders) and the same on the secondary sets, each with a 95 % paired bootstrap interval over the scans, and
  the difference at each k;
- seed_spread.csv: the spread of the random orders at each k (training and order noise);
- models.csv: per model its mean test Dice and its forgetting (share of other structures' voxels relabelled).

Usage: .venv\\Scripts\\python.exe scripts\\g3_curves.py --config configs\\g3_kits.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Colors: the first two slots of the validated categorical palette (queue = blue, random order = orange); text in
# neutral ink, the grid recessive. Each random seed is a thin line of the random color, their mean a dashed one.
QUEUE_COLOR, RANDOM_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_curves(curve, random_orders: list[str], path: Path, subtitle: str = "") -> None:
    """The learning curves on the test set: all test scans and the "bad" ones, side by side (two panels, each with
    its own Dice axis). Saved as PNG and PDF (ACM two-column width, 7 in)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(s, curve[curve.subset == s]) for s in ("all", "bad") if (curve.subset == s).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(7, 2.7), squeeze=False)
    for ax, (subset, c) in zip(axes[0], panels):
        for order in random_orders:
            r = c[c.order == order].sort_values("k")
            ax.plot(r.k, r.mean_dice, color=RANDOM_COLOR, lw=0.8, alpha=0.45)
        mean = c[c.order.isin(random_orders)].groupby("k").mean_dice.mean()
        ax.plot(mean.index, mean.values, color=RANDOM_COLOR, lw=1.8, ls="--",
                label=f"Random order (mean of {len(random_orders)} seeds; thin: each seed)")
        q = c[c.order == "queue"].sort_values("k")
        ax.plot(q.k, q.mean_dice, color=QUEUE_COLOR, lw=1.8, marker="o", ms=3.5,
                label="Queue order (most uncertain first)")
        # Direct labels at the right end; pushed apart when the two curves end close together.
        y_queue, y_random = q.mean_dice.iloc[-1], mean.values[-1]
        low, high = ax.get_ylim()
        nudge = 4 if abs(y_queue - y_random) < 0.05 * (high - low) else 0
        up = 1 if y_queue >= y_random else -1
        for y, text, shift in ((y_queue, "queue", up * nudge), (y_random, "random", -up * nudge)):
            ax.annotate(text, (q.k.iloc[-1], y), xytext=(4, shift), textcoords="offset points", va="center",
                        fontsize=7, color=INK)
        n = int(c.n_scans.iloc[0])
        ax.set_title(("All test scans" if subset == "all" else "Bad test scans (Dice < 0.83 originally)")
                     + f", n = {n}", fontsize=8, color=INK)
        ax.set_xlabel("Corrected pool scans used for fine-tuning (k)", fontsize=7.5, color=MUTED)
        ax.set_ylabel("Mean Dice (kidney + tumor + cyst)", fontsize=7.5, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(-2, 112)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=2, fontsize=7, frameon=False)
    if subtitle:
        fig.suptitle(subtitle, fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95 if subtitle else 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="configs/g3_kits.yaml")
    args = parser.parse_args()
    g3 = yaml.safe_load(args.config.read_text())

    import pandas as pd

    from segreview.finetune import QUEUE, experiment_runs, read_orders
    from segreview.g3_analysis import (ORIGINAL, curve_points, curves, dice_matrix, seed_spread, summary_weights,
                                       summary_with_ci)

    results = REPO_ROOT / "results" / g3["name"]
    choice_path = results / "settings_choice.json"
    if not choice_path.exists():
        raise SystemExit("No settings_choice.json yet: the curves come after the grid stage.")
    orders = read_orders(results / "orders.csv")
    _, points = experiment_runs(g3, orders, json.loads(choice_path.read_text()))
    random_orders = [o for o in orders if o != QUEUE]
    pts = curve_points(points[points.order != "nocorr"], list(orders))
    files = {p.stem: p for p in (results / "dice").glob("*.csv")}
    dice = pd.concat([pd.read_csv(files[m]) for m in set(points.run_id) | {ORIGINAL} if m in files])
    ev = g3["evaluation"]
    n_test = int(((dice.model == ORIGINAL) & (dice.split == "test")).sum())
    complete = {m for m, g in dice[dice.split == "test"].groupby("model") if len(g) == n_test}
    missing = sorted(set(pts.run_id) - complete)
    if missing:
        print(f"[curves] {len(missing)} models without all {n_test} test scans yet: {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))
    available = pts[pts.run_id.isin(complete | {ORIGINAL})]

    curve_rows, summary_rows, spread = [], [], None
    for split in ("test", "dev"):
        for definition in ev["organ_definitions"]:
            column = f"dice_{definition}"
            matrix = dice_matrix(dice, list(available.run_id.unique()), split, column)
            subsets = {"all": matrix}
            if split == "test":
                original = dice_matrix(dice, [ORIGINAL], split, f"dice_{ev['primary_definition']}")[ORIGINAL]
                subsets["bad"] = matrix.loc[original[original < ev["bad_dice"]].index.intersection(matrix.index)]
            for subset, m in subsets.items():
                curve = curves(m, available).assign(split=split, organ_definition=definition, subset=subset)
                curve_rows.append(curve)
                if split == "test" and definition == ev["primary_definition"] and subset == "all":
                    spread = seed_spread(curve, random_orders)
                needed = pts[pts.k.isin(ev["summary_k"])]
                if not set(needed.run_id) <= set(available.run_id) or len(m) < 2:
                    continue
                primary = split == "test" and definition == ev["primary_definition"] and subset == "all"
                for ks, label in [(ev["summary_k"], "mean over k")] + [([k], f"k = {k}") for k in ev["summary_k"]]:
                    weights = summary_weights(pts, QUEUE, random_orders, ks)
                    summary_rows.append({"summary": label, "primary": primary and label == "mean over k",
                                         "split": split, "organ_definition": definition, "subset": subset,
                                         **summary_with_ci(m, weights, ev["n_bootstrap"], g3["seed"], ev["ci_level"])})

    all_curves = pd.concat(curve_rows)
    all_curves.round(5).to_csv(results / "curves.csv", index=False)
    if spread is not None:
        spread.round(5).to_csv(results / "seed_spread.csv", index=False)
    subtitle = ""
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.round(5).to_csv(results / "summary.csv", index=False)
        p = summary[summary.primary].iloc[0]
        subtitle = (f"Queue minus random, mean over k = 10-90: {p.estimate:+.4f} Dice "
                    f"(95 % CI {p.ci_low:+.4f} to {p.ci_high:+.4f}, {p.n_scans} test scans)")
    primary = all_curves[(all_curves.split == "test") & (all_curves.organ_definition == ev["primary_definition"])]
    plot_curves(primary, random_orders, results / "figures" / "learning_curve_test", subtitle)
    test = dice[(dice.split == "test") & dice.model.isin(complete | {ORIGINAL})]
    models = test.groupby("model").agg(mean_dice=(f"dice_{ev['primary_definition']}", "mean"),
                                       forgetting_share=("forgetting_share", "mean"), n_scans=("case_id", "size"))
    models.round(5).to_csv(results / "models.csv")
    print(f"[curves] {len(available)} of {len(pts)} points; summary {'written' if summary_rows else 'not yet'} "
          f"-> {results}")


if __name__ == "__main__":
    main()
