"""G3 phase 3 (docs/g3_plan.md, "Phase 3"; fixed on 2026-11-16 before any phase 3 run): the first ten scans, with
three training seeds for the queue. A pre-specified follow-up, not a new locked test. From the Dice per scan and model
in results/g3_kits/dice/; it can run at any time: points that are not evaluated yet are listed and left out, and a
summary is only written when all its points are evaluated on the test set.

The arms: the queue order with its training seeds (queue, queue_s43, queue_s44) and the random orders (random42, 43,
44). Output (results/g3_kits/):
- phase3_curves.csv: mean Dice per order and k (k = 0 is the original model), per split, organ definition and scan
  subset ("all", and on the test set "bad" = Dice below evaluation.bad_dice with the original model);
- phase3_summary.csv: the primary number (test, all scans, kidney_tumor_cyst, the mean over phase3.summary_k of queue
  arm - random arm) and the same on the other definitions, subsets and the development set, and the difference at
  each k; each with the two-stage interval (scans and the arms' runs: ci_low, ci_high) and the interval over the scans
  only (ci_low_scans, ci_high_scans);
- phase3_spread.csv: at each k, the spread (max - min) of the queue's seeds and of the random orders (test, all scans);
- phase3_small_k.csv: for each run at k <= 5, its mean test Dice, the original model's on the same scans, and the share
  of test scans that get worse than with the original model (the overfitting check);
- figures/phase3_first_scans.png and .pdf.

Usage: .venv\\Scripts\\python.exe scripts\\g3_phase3.py --config configs\\g3_kits.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# As in scripts/g3_curves.py: queue = blue, random order = orange; text in neutral ink, the grid recessive. Each run
# is a thin line of its arm's color, the arm's mean a thick one.
QUEUE_COLOR, RANDOM_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANEL_TITLES = {"kidney_tumor_cyst": "Kidney + tumor + cyst", "kidney_cyst_ignore_tumor": "Tumor left out"}


def plot_first_scans(curve, queue_orders: list[str], random_orders: list[str], ks: list[int], path: Path,
                     subtitle: str = "") -> None:
    """Test Dice against k (0-30, the axis linear up to 1 and logarithmic above, so k = 1, 2, 3, 5 are apart), one
    panel per organ definition, each with its own Dice axis. Saved as PNG and PDF (ACM two-column width, 7 in)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    definitions = list(dict.fromkeys(curve.organ_definition))
    fig, axes = plt.subplots(1, len(definitions), figsize=(7, 2.7), squeeze=False)
    for ax, definition in zip(axes[0], definitions):
        c = curve[curve.organ_definition == definition]
        means = {}
        for arm, orders, color, style in (("random", random_orders, RANDOM_COLOR, "--"),
                                          ("queue", queue_orders, QUEUE_COLOR, "-")):
            for order in orders:
                r = c[c.order == order].sort_values("k")
                ax.plot(r.k, r.mean_dice, color=color, lw=0.8, alpha=0.45)
            mean = c[c.order.isin(orders)].groupby("k").mean_dice.mean()
            label = (f"Queue order (mean of {len(orders)} training seeds; thin: each)" if arm == "queue"
                     else f"Random order (mean of {len(orders)} orders; thin: each)")
            ax.plot(mean.index, mean.values, color=color, lw=1.8, ls=style, marker="o" if arm == "queue" else None,
                    ms=3.5, label=label)
            means[arm] = mean
        # Direct labels at the right end; pushed apart when the two curves end close together.
        y_queue, y_random = means["queue"].values[-1], means["random"].values[-1]
        low, high = ax.get_ylim()
        nudge = 4 if abs(y_queue - y_random) < 0.05 * (high - low) else 0
        up = 1 if y_queue >= y_random else -1
        for y, text, shift in ((y_queue, "queue", up * nudge), (y_random, "random", -up * nudge)):
            ax.annotate(text, (means["queue"].index[-1], y), xytext=(4, shift), textcoords="offset points",
                        va="center", fontsize=7, color=INK)
        ax.set_xscale("symlog", linthresh=1, linscale=0.4)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlim(-0.1, max(ks) * 1.6)
        ax.set_title(f"{PANEL_TITLES.get(definition, definition)}, test scans, n = {int(c.n_scans.iloc[0])}",
                     fontsize=8, color=INK)
        ax.set_xlabel("Corrected pool scans used for fine-tuning (k)", fontsize=7.5, color=MUTED)
        ax.set_ylabel("Mean Dice", fontsize=7.5, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
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

    from segreview.finetune import QUEUE, phase3_runs, read_orders
    from segreview.g3_analysis import ORIGINAL, arm_difference_with_ci, curve_points, curves, dice_matrix

    results = REPO_ROOT / "results" / g3["name"]
    choice_path = results / "settings_choice.json"
    if not choice_path.exists():
        raise SystemExit("No settings_choice.json yet: phase 3 uses the setting chosen in the grid stage.")
    phase3, ev = g3["phase3"], g3["evaluation"]
    _, points = phase3_runs(g3, read_orders(results / "orders.csv"), json.loads(choice_path.read_text()))
    arm_orders = list(dict.fromkeys(points.order))
    queue_orders = [o for o in arm_orders if o == QUEUE or o.startswith(f"{QUEUE}_s")]
    random_orders = [o for o in arm_orders if o not in queue_orders]
    pts = curve_points(points, arm_orders)
    by_point = pts.set_index(["order", "k"]).run_id
    files = {p.stem: p for p in (results / "dice").glob("*.csv")}
    dice = pd.concat([pd.read_csv(files[m]) for m in set(pts.run_id) if m in files])
    n_test = int(((dice.model == ORIGINAL) & (dice.split == "test")).sum())
    complete = {m for m, g in dice[dice.split == "test"].groupby("model") if len(g) == n_test}
    missing = sorted(set(pts.run_id) - complete)
    if missing:
        print(f"[phase3] {len(missing)} models without all {n_test} test scans yet: {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))
    available = pts[pts.run_id.isin(complete)]

    def cells(ks: list[int]) -> list[tuple[list[str], list[str]]]:
        return [([by_point[(o, k)] for o in queue_orders], [by_point[(o, k)] for o in random_orders]) for k in ks]

    curve_rows, summary_rows = [], []
    for split in ("test", "dev"):
        for definition in ev["organ_definitions"]:
            matrix = dice_matrix(dice, list(available.run_id.unique()), split, f"dice_{definition}")
            subsets = {"all": matrix}
            if split == "test":
                original = dice_matrix(dice, [ORIGINAL], split, f"dice_{ev['primary_definition']}")[ORIGINAL]
                subsets["bad"] = matrix.loc[original[original < ev["bad_dice"]].index.intersection(matrix.index)]
            for subset, m in subsets.items():
                curve = curves(m, available).assign(split=split, organ_definition=definition, subset=subset)
                curve_rows.append(curve)
                primary = split == "test" and definition == ev["primary_definition"] and subset == "all"
                for ks, label in [(phase3["summary_k"], "mean over k")] + [([k], f"k = {k}") for k in
                                                                           phase3["k_values"]]:
                    needed = {run for q, r in cells(ks) for run in q + r}
                    if not needed <= set(available.run_id) or len(m) < 2:
                        continue
                    summary_rows.append({"summary": label, "primary": primary and label == "mean over k",
                                         "split": split, "organ_definition": definition, "subset": subset,
                                         "runs_per_arm": f"{len(queue_orders)} / {len(random_orders)}",
                                         **arm_difference_with_ci(m, cells(ks), ev["n_bootstrap"], g3["seed"],
                                                                  ev["ci_level"])})

    all_curves = pd.concat(curve_rows)
    all_curves.round(5).to_csv(results / "phase3_curves.csv", index=False)
    test_all = all_curves[(all_curves.split == "test") & (all_curves.subset == "all")]
    primary_curve = test_all[test_all.organ_definition == ev["primary_definition"]]
    spread_rows = []
    for k, c in primary_curve[primary_curve.k > 0].groupby("k"):
        for arm, orders in (("queue", queue_orders), ("random", random_orders)):
            d = c[c.order.isin(orders)].mean_dice
            spread_rows.append({"k": k, "arm": arm, "n_runs": len(d), "min": d.min(), "max": d.max(),
                                "mean": d.mean(), "spread": d.max() - d.min()})
    pd.DataFrame(spread_rows).round(5).to_csv(results / "phase3_spread.csv", index=False)

    test = dice_matrix(dice, list(available.run_id.unique()), "test", f"dice_{ev['primary_definition']}")
    small = [{"order": o, "k": k, "run_id": by_point[(o, k)], "mean_dice": test[by_point[(o, k)]].mean(),
              "original_mean_dice": test[ORIGINAL].mean(),
              "share_of_scans_worse": (test[by_point[(o, k)]] < test[ORIGINAL]).mean()}
             for k in phase3["k_values"] if k <= 5 for o in arm_orders if by_point[(o, k)] in test.columns]
    pd.DataFrame(small).round(5).to_csv(results / "phase3_small_k.csv", index=False)

    subtitle = ""
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.round(5).to_csv(results / "phase3_summary.csv", index=False)
        p = summary[summary.primary]
        if len(p):
            p = p.iloc[0]
            subtitle = (f"Queue minus random, mean over k = {', '.join(map(str, phase3['summary_k']))}: "
                        f"{p.estimate:+.4f} Dice (95 % CI {p.ci_low:+.4f} to {p.ci_high:+.4f}, scans and "
                        f"training seeds; {p.n_scans} test scans)")
    plot_first_scans(test_all, queue_orders, random_orders, [0] + phase3["k_values"],
                     results / "figures" / "phase3_first_scans", subtitle)
    print(f"[phase3] {len(available)} of {len(pts)} points; summary {'written' if summary_rows else 'not yet'} "
          f"-> {results}")


if __name__ == "__main__":
    main()
