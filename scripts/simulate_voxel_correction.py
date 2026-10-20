"""EXPLORATORY: simulated correction of the most uncertain voxels inside each scan (src/segreview/voxel_correction.py).

Follow-up on the supervisor meeting of 2026-10-07, run after the G1 choices were locked. For every scan, the k %
most uncertain voxels of the region (predicted organ + uncertainty.border_mm) are set to the ground truth, and the
Dice gain is measured. Reads only saved predictions (mask, entropy, m2_diff) and the ground truth; no inference.
The dev and test sets are reported separately. The scan-level quality curve is G1's and is not redone here.

Writes to results/<config name>/:
  voxel_correction_cases.csv     per scan: Dice before, error voxels, share of them outside the region, best Dice
                                 reachable inside the region, unfixable share of the Dice gap
  voxel_correction_per_case.csv  per scan, method and k: Dice after and gain
  voxel_correction_auc.csv       per scan and method: area under the gain-vs-k curve (as mean gain over 0..k)
  voxel_correction_summary.csv   per split, method and k: median and mean gain, and the mean as a share of the oracle's
  voxel_correction_auc_summary.csv  per split and method: median and mean area, and the mean as a share of the oracle's
  figures/voxel_correction_gain.png
  voxel_correction_machine.json  machine, library versions and git commit of this run

Usage: uv run python scripts/simulate_voxel_correction.py --config configs/kits100.yaml
"""

import time

import numpy as np
import pandas as pd

from segreview.config import config_arg_parser, figures_dir, load_config, results_dir
from segreview.data import case_split, evaluation_cases, load_organ_mask
from segreview.followup_figures import plot_correction
from segreview.io import load_map, voxel_spacing
from segreview.machine import write_machine_info
from segreview.voxel_correction import GT_FREE, correction_curves, gain_auc

STATUS = "exploratory (after the G1 lock)"


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    vc = cfg["exploratory"]["voxel_correction"]
    if cfg["seed"] not in vc["random_seeds"]:
        raise ValueError("voxel_correction.random_seeds should include the global seed")
    out = results_dir(cfg)
    start = time.perf_counter()
    print(f"*** {STATUS.upper()}: dev and test reported separately ***")

    case_rows, curve_rows = [], []
    for case in evaluation_cases(cfg):
        split = case_split(cfg, case.case_id)
        truth, ignore = load_organ_mask(cfg, case)
        mask = load_map(cfg, "clean", case.case_id, "mask").astype(bool)
        maps = {m: load_map(cfg, "clean", case.case_id, m) for m in vc["methods"] if m in GT_FREE}
        if ignore.any():   # as in the heatmap evaluation: ignored voxels count as correct background
            truth, mask = truth & ~ignore, mask & ~ignore
        info, rows = correction_curves(maps, mask, truth, voxel_spacing(cfg, "clean", case.case_id),
                                       cfg["uncertainty"]["border_mm"], vc["k_percent"], vc["methods"],
                                       vc["random_seeds"])
        case_rows.append({"case_id": case.case_id, "split": split, **info})
        curve_rows += [{"case_id": case.case_id, "split": split, **r} for r in rows]
        at10 = {r["method"]: r["gain"] for r in rows if r["k_percent"] == 10}
        print(f"  {case.case_id} [{split}]: Dice {info['dice_before']:.4f}, outside region "
              f"{info['error_outside_region_share']:.2f}; gain at 10 %: "
              + ", ".join(f"{m} {g:+.4f}" for m, g in at10.items()))

    cases = pd.DataFrame(case_rows)
    curves = pd.DataFrame(curve_rows)
    # dice_before must match the locked G1 Dice (dice.csv is rounded to 4 decimals).
    locked = pd.read_csv(out / "dice.csv").query("variant == 'clean'").set_index("case_id")["dice"]
    diff = (cases.set_index("case_id")["dice_before"].round(4) - locked.reindex(cases.case_id)).abs()
    if not (diff < 1e-9).all():
        raise SystemExit(f"STOP: Dice before differs from dice.csv:\n{diff[diff >= 1e-9]}")
    print(f"Check passed: Dice before matches dice.csv for all {len(cases)} scans.")
    cases.round(6).to_csv(out / "voxel_correction_cases.csv", index=False)
    curves.round(6).to_csv(out / "voxel_correction_per_case.csv", index=False)

    # Area under the gain-vs-k curve per scan and method, for each range 0..up_to.
    auc_rows = []
    for (case_id, split, method), d in curves.groupby(["case_id", "split", "method"], sort=False):
        d = d.sort_values("k_percent")
        auc_rows.append({"case_id": case_id, "split": split, "method": method,
                         **{f"auc_0_{u}": gain_auc(list(d.k_percent), list(d.gain), u) for u in vc["auc_up_to_percent"]}})
    aucs = pd.DataFrame(auc_rows)
    aucs.round(6).to_csv(out / "voxel_correction_auc.csv", index=False)

    # Summary per split: median and mean gain per method and k. The oracle's mean gain is the best any order
    # reaches, so each method's share of it is easier to compare across k.
    summary = curves.groupby(["split", "method", "k_percent"])["gain"].agg(
        gain_median="median", gain_mean="mean").reset_index()
    oracle = summary[summary.method == "oracle"].set_index(["split", "k_percent"])["gain_mean"]
    summary["share_of_oracle_mean"] = (summary["gain_mean"]
                                       / oracle.reindex(pd.MultiIndex.from_frame(summary[["split", "k_percent"]])).to_numpy())
    summary.insert(0, "status", STATUS)
    summary.round(6).to_csv(out / "voxel_correction_summary.csv", index=False)

    auc_cols = [c for c in aucs.columns if c.startswith("auc_")]
    auc_summary = aucs.groupby(["split", "method"])[auc_cols].agg(["median", "mean"])
    auc_summary.columns = [f"{c}_{stat}" for c, stat in auc_summary.columns]
    auc_summary = auc_summary.reset_index()
    for c in auc_cols:
        oracle_auc = auc_summary[auc_summary.method == "oracle"].set_index("split")[f"{c}_mean"]
        auc_summary[f"{c}_share_of_oracle_mean"] = auc_summary[f"{c}_mean"] / auc_summary.split.map(oracle_auc)
    auc_summary.insert(0, "status", STATUS)
    auc_summary.round(6).to_csv(out / "voxel_correction_auc_summary.csv", index=False)

    ctx = cases.groupby("split")[["dice_before", "error_outside_region_share", "dice_ceiling",
                                  "unfixable_dice_gap_share"]].agg(["median", "mean"]).round(4)
    print(ctx.T.to_string())
    print(summary.round(4).to_string(index=False))
    print(auc_summary.round(4).to_string(index=False))
    plot_correction(summary, cases, figures_dir(cfg) / "voxel_correction_gain.png",
                    f"Simulated correction inside a scan (k % of the region set to ground truth). Config "
                    f"'{cfg['name']}', {STATUS}")
    write_machine_info(out / "voxel_correction_machine.json", config=str(args.config),
                       script="scripts/simulate_voxel_correction.py", status=STATUS, random_seeds=vc["random_seeds"],
                       seconds=round(time.perf_counter() - start, 1))
    print(f"Done in {time.perf_counter() - start:.0f} s -> {out}")


if __name__ == "__main__":
    main()
