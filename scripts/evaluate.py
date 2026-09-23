"""Compare saved predictions with the ground truth: Dice per scan (evaluation only).

Writes results/dice_<config name>_<date>.csv with one row per scan.
Ground truth is only read here, never in the prediction or uncertainty code.

Usage: uv run python scripts/evaluate.py --config configs/local.yaml
"""

from datetime import date

import numpy as np
import pandas as pd

from segreview.config import config_arg_parser, load_config
from segreview.data import list_cases, load_canonical
from segreview.io import load_mask, mask_path
from segreview.metrics import dice


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    gt_label = cfg["dataset"]["gt_label"]

    rows = []
    for case in list_cases(cfg):
        if not mask_path(cfg, case.case_id).exists():
            print(f"  {case.case_id}: no prediction yet, skipped")
            continue
        # The ground truth is reoriented exactly like the image, so it is on the same voxel grid as the prediction.
        truth = np.asanyarray(load_canonical(case.label_path).dataobj) == gt_label
        pred = load_mask(cfg, case.case_id)
        rows.append({"case_id": case.case_id, "dice": round(dice(pred, truth), 4),
                     "pred_voxels": int(pred.sum()), "truth_voxels": int(truth.sum())})
        print(f"  {case.case_id}: Dice = {rows[-1]['dice']:.4f}")

    df = pd.DataFrame(rows)
    out = cfg["paths"]["results_dir"] / f"dice_{cfg['name']}_{date.today().isoformat()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} scans, mean Dice {df['dice'].mean():.4f} (min {df['dice'].min():.4f}, "
          f"max {df['dice'].max():.4f}) -> {out}")


if __name__ == "__main__":
    main()
