"""G3 task B (2026-11-16): what the first pool scans of each order contain. Allowed only after the joint locked test
on the pool (2026-11-04), because it compares the queue with the pool's ground truth.

The question: is the queue's early lead on the "bad" test scans (Dice < 0.83 with the original model) only that its
first scans teach the model our label definition (tumor counts as kidney) faster? So for the first k = 10, 20, 30
scans of the queue and of each random order: how many have a tumor or a cyst, how large the tumors are, and how many
voxels the simulated correction changes (results/g3_kits/targets_fast.csv, on the 3 mm model's grid). And for the
test scans: how large the tumors of the bad ones are, next to the others.

Output (results/g3_kits/):
- pool_scan_content.csv: per pool scan, the ground truth's kidney, tumor and cyst volume (ml) and the voxels changed;
- queue_content.csv: per order and k, what its first k scans contain (src/segreview/g3_analysis.first_k_content);
- test_tumor_sizes.csv: per test scan, the original model's Dice, whether it is bad, and the label volumes.

Usage: .venv\\Scripts\\python.exe scripts\\g3_queue_content.py --config configs\\g3_kits.yaml
"""

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
KS = [10, 20, 30]


def label_volumes_ml(cfg: dict, cases: list) -> "pd.DataFrame":
    """Volume (ml) of each ground-truth label (dataset.label_names, without background) per scan."""
    import numpy as np
    import pandas as pd

    from segreview.data import load_canonical

    names = {int(v): n for v, n in cfg["dataset"]["label_names"].items() if int(v) != 0}
    rows = []
    for case in cases:
        image = load_canonical(case.label_path)
        counts = np.bincount(np.asanyarray(image.dataobj).astype(np.int64).ravel(), minlength=max(names) + 1)
        voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000
        rows.append({"case_id": case.case_id, **{f"{n}_ml": counts[v] * voxel_ml for v, n in names.items()}})
    return pd.DataFrame(rows).set_index("case_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="configs/g3_kits.yaml")
    args = parser.parse_args()
    g3 = yaml.safe_load(args.config.read_text())

    import pandas as pd

    from segreview.config import load_config
    from segreview.data import list_cases
    from segreview.finetune import QUEUE, read_orders
    from segreview.g3_analysis import ORIGINAL, first_k_content

    results = REPO_ROOT / "results" / g3["name"]
    orders = read_orders(results / "orders.csv")
    random_orders = [o for o in orders if o != QUEUE]

    pool = load_config(REPO_ROOT / g3["pool_config"])
    targets = pd.read_csv(results / f"targets_{g3['model_resolution']}.csv").set_index("case_id")
    per_scan = label_volumes_ml(pool, list_cases(pool)).join(
        targets.voxels_changed_by_correction.rename("voxels_changed"), how="inner")
    if len(per_scan) != len(orders[QUEUE]):
        raise SystemExit(f"{len(per_scan)} pool scans with label volumes and corrections, expected {len(orders[QUEUE])}")
    per_scan.round(3).to_csv(results / "pool_scan_content.csv")

    content = first_k_content(orders, per_scan, KS, random_orders)
    content.round(3).to_csv(results / "queue_content.csv", index=False)

    source = load_config(REPO_ROOT / g3["eval_config"])
    definition = g3["evaluation"]["primary_definition"]
    original = pd.read_csv(results / "dice" / f"{ORIGINAL}.csv")
    original = original[original.split == "test"].set_index("case_id")[f"dice_{definition}"].rename("original_dice")
    test = label_volumes_ml(source, list_cases(source, splits=["test"])).join(original, how="inner")
    test["bad"] = test.original_dice < g3["evaluation"]["bad_dice"]
    test.round(4).to_csv(results / "test_tumor_sizes.csv")

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(content.round(1).to_string(index=False))
        print(test.groupby("bad")[["tumor_ml", "cyst_ml", "kidney_ml"]].agg(["count", "median", "mean"]).round(1))
    print(f"[queue content] -> {results}")


if __name__ == "__main__":
    main()
