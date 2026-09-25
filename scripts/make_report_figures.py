"""Figures and tables of the G1 results for the report (ACM two-column paper). No new analysis.

Reads the files the G1 evaluation wrote (g1_scores_<set>.csv, g1_metrics_<set>.csv, heatmap_eval_<set>_summary.csv)
and the saved model output. Writes results/<config name>/report/:
  fig_g1_curves_test.{pdf,png}           review curves (both "bad" rules) and quality curve
  fig_g1_auc_vs_volume_test.{pdf,png}    AUC minus the volume baseline with paired 95 % bootstrap CIs
  fig_heatmap_examples_test.{pdf,png}    three test scans: model vs ground truth, and the uncertainty heatmap
  table_g1_test.{csv,tex}                all measures, baselines and the oracle
  table_heatmaps_test.{csv,tex}          error localisation of the candidate heatmaps
  heatmap_examples_test.csv              which scans and slices the example figure shows, and why

Usage: uv run python scripts/make_report_figures.py --config configs/kits100.yaml
"""

import pandas as pd

from segreview.config import config_arg_parser, load_config, results_dir
from segreview.report import (TABLE_ORDER, curves_from_scores, figure_auc_vs_volume, figure_curves,
                              figure_heatmap_examples, g1_latex, g1_table, heatmap_table, pick_examples, use_style)


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    rep = cfg["report"]
    res = results_dir(cfg)
    out = res / "report"
    split = rep["set"].removeprefix("clean_")
    scores = pd.read_csv(res / f"g1_scores_{rep['set']}.csv")
    metrics = pd.read_csv(res / f"g1_metrics_{rep['set']}.csv")
    rules = cfg["evaluation"]["bad_rules"]
    n = len(scores)
    n_bad = {r["name"]: int(scores[f"bad_{r['name']}"].sum()) for r in rules}

    use_style()
    curves = curves_from_scores(scores, [m for m in TABLE_ORDER if m != "random"], rules, cfg, metrics)
    figure_curves(curves, rules, n_bad, n, out)
    figure_auc_vs_volume(metrics, out)
    examples = pick_examples(scores, rep["primary"], rep["example_bad_rule"])
    figure_heatmap_examples(cfg, examples, scores, rep["primary"], rep["heatmap"], out).to_csv(
        out / f"heatmap_examples_{split}.csv", index=False)

    table = g1_table(metrics, rules)
    table.to_csv(out / f"table_g1_{split}.csv", index=False)
    (out / f"table_g1_{split}.tex").write_text(g1_latex(metrics, rules, n, n_bad))
    htable, htex = heatmap_table(pd.read_csv(res / f"heatmap_eval_{split}_summary.csv"), n)
    htable.to_csv(out / f"table_heatmaps_{split}.csv", index=False)
    (out / f"table_heatmaps_{split}.tex").write_text(htex)
    print(table[["label", "spearman_rho", "quality_auc", "beats_volume_quality"]].to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
