"""Compare candidate heatmaps by how well they point at the wrong voxels (G2, evaluation only).

For every clean scan in the chosen split: voxel AUC and top-X % capture of each heatmap
(see src/segreview/heatmap_eval.py). Writes results/<config name>/heatmap_eval_<split>.csv (per scan)
and heatmap_eval_<split>_summary.csv (median, mean and pooled over scans).

Usage: uv run python scripts/evaluate_heatmaps.py --config configs/kits100.yaml --split dev
"""

import pandas as pd

from segreview.config import config_arg_parser, load_config, results_dir
from segreview.data import case_split, evaluation_cases, load_organ_mask
from segreview.heatmap_eval import evaluate_scan
from segreview.io import load_map, voxel_spacing


def main():
    parser = config_arg_parser(__doc__)
    parser.add_argument("--split", default="all", help="dev, test, or all (datasets without a split: all)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    candidates = cfg["heatmap_eval"]["candidates"]

    rows = []
    for case in evaluation_cases(cfg):
        split = case_split(cfg, case.case_id)
        if args.split != "all" and split != args.split:
            continue
        truth, ignore = load_organ_mask(cfg, case)
        mask = load_map(cfg, "clean", case.case_id, "mask").astype(bool)
        maps = {name: load_map(cfg, "clean", case.case_id, name) for name in candidates}
        if ignore.any():  # ignored voxels count as neither error nor correct: treat them as correct background
            truth = truth & ~ignore
            mask = mask & ~ignore
        row = evaluate_scan(maps, mask, truth, voxel_spacing(cfg, "clean", case.case_id),
                            cfg["uncertainty"]["border_mm"], cfg["heatmap_eval"]["top_fraction"])
        rows.append({"case_id": case.case_id, "split": split, **row})
        print(f"  {case.case_id}: " + ", ".join(f"{k}={v:.3f}" for k, v in row.items() if k.startswith(("auc_", "top_"))))

    df = pd.DataFrame(rows)
    out = results_dir(cfg)
    df.to_csv(out / f"heatmap_eval_{args.split}.csv", index=False)
    cols = [c for c in df.columns if c.startswith(("auc_", "top_"))] + ["error_in_region_share"]
    summary = df[cols].agg(["count", "median", "mean", "min", "max"]).T.round(4)
    summary.to_csv(out / f"heatmap_eval_{args.split}_summary.csv", index_label="measure")
    print(f"{len(df)} scans ({args.split}); top fraction = {cfg['heatmap_eval']['top_fraction']}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
