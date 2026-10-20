"""EXPLORATORY: the 6 mm model alone, and where the 3 mm and 6 mm masks disagree (src/segreview/model_comparison.py).

Follow-up on the supervisor meeting of 2026-10-07, run after the G1 choices were locked. Reads only saved
predictions (mask, m2_mask) and the ground truth; no inference, and the locked G1 files are only read.
The dev and test sets are reported separately.

Writes to results/<config name>/:
  model_dice_3mm_vs_6mm.csv            Dice of both models per scan (case_id, split, dice_3mm, dice_6mm)
  model_dice_3mm_vs_6mm_summary.csv    per split: medians, means, scans below the threshold, which model is better,
                                       Spearman between the two, real finds and false alarms in the queue's top
  queue_top_3mm_vs_6mm.csv             top of the locked queue (report.primary) per split, with both Dice values
  model_disagreement_counts.csv        disagreeing voxels per scan by label, distance to the 3 mm edge, right model
  model_disagreement_per_case.csv      the same as shares per scan, with the scan's group (bad / false_alarm / good)
  model_disagreement_summary.csv       per split and group: pooled shares and the median of the per-scan shares
  figures/model_disagreement_where.png the summary as a figure
  model_comparison_machine.json        machine, library versions and git commit of this run

Usage: uv run python scripts/compare_models.py --config configs/kits100.yaml
"""

import time

import numpy as np
import pandas as pd

from segreview.config import config_arg_parser, figures_dir, load_config, results_dir
from segreview.data import case_split, evaluation_cases, load_label_map, organ_definition
from segreview.io import load_map, voxel_spacing
from segreview.machine import write_machine_info
from segreview.metrics import dice
from segreview.model_comparison import (GROUPS, case_group, disagreement_counts, disagreement_shares,
                                        model_dice_summary, queue_table)
from segreview.followup_figures import plot_disagreement

STATUS = "exploratory (after the G1 lock)"


def bad_threshold(cfg: dict, rule_name: str) -> float:
    """The Dice threshold of a fixed-threshold 'bad' rule in evaluation.bad_rules (so the number lives in one place)."""
    for rule in cfg["evaluation"]["bad_rules"]:
        if rule["name"] == rule_name and rule["rule"] == "threshold":
            return float(rule["value"])
    raise ValueError(f"No threshold rule named '{rule_name}' in evaluation.bad_rules")


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    mc = cfg["exploratory"]["model_comparison"]
    threshold = bad_threshold(cfg, mc["bad_rule"])
    definition = organ_definition(cfg)
    label_names = cfg["dataset"]["label_names"]
    out = results_dir(cfg)
    start = time.perf_counter()
    print(f"*** {STATUS.upper()}: dev and test reported separately ***")

    dice_rows, count_frames = [], []
    for case in evaluation_cases(cfg):
        split = case_split(cfg, case.case_id)
        labels = load_label_map(case)
        truth = np.isin(labels, definition["organ_labels"])
        ignore = np.isin(labels, definition["ignore_labels"])
        mask = load_map(cfg, "clean", case.case_id, "mask").astype(bool)
        m2_mask = load_map(cfg, "clean", case.case_id, "m2_mask").astype(bool)
        # Same Dice function, rounding and ignore mask as the pipeline's step_dice (checked against dice.csv below).
        row = {"case_id": case.case_id, "split": split,
               "dice_3mm": round(dice(mask, truth, ignore), 4), "dice_6mm": round(dice(m2_mask, truth, ignore), 4)}
        dice_rows.append(row)
        counts = disagreement_counts(mask, m2_mask, labels, definition["organ_labels"], definition["ignore_labels"],
                                     label_names, voxel_spacing(cfg, "clean", case.case_id), mc["distance_bins_mm"])
        count_frames.append(counts.assign(case_id=case.case_id, split=split))
        print(f"  {case.case_id} [{split}]: Dice 3 mm {row['dice_3mm']:.4f}, 6 mm {row['dice_6mm']:.4f}, "
              f"{int(counts.n_voxels.sum())} disagreeing voxels")
    dice_df = pd.DataFrame(dice_rows).sort_values("case_id").reset_index(drop=True)

    # The 3 mm Dice must be exactly the locked G1 Dice; otherwise something differs from the G1 run, so stop.
    locked = pd.read_csv(out / "dice.csv").query("variant == 'clean'")[["case_id", "dice"]]
    check = dice_df.merge(locked, on="case_id", how="outer", indicator=True)
    if (check["_merge"] != "both").any() or not np.array_equal(check["dice_3mm"], check["dice"]):
        bad = check[(check["_merge"] != "both") | (check["dice_3mm"] != check["dice"])]
        raise SystemExit(f"STOP: dice_3mm differs from dice.csv for {len(bad)} scans:\n{bad.to_string()}")
    print(f"Check passed: dice_3mm is identical to dice.csv for all {len(dice_df)} scans.")
    dice_df.to_csv(out / "model_dice_3mm_vs_6mm.csv", index=False)

    # Part 1: summary per split and the top of the locked queue.
    primary, top_n = cfg["report"]["primary"], mc["queue_top_n"]
    summary_rows, queues = [], []
    for split, df in dice_df.groupby("split"):
        q = queue_table(pd.read_csv(out / f"g1_scores_clean_{split}.csv"), df, primary, top_n, threshold)
        queues.append(q.assign(split=split, queue_score=primary))
        row = {"status": STATUS, "split": split, "threshold": threshold, **model_dice_summary(df, threshold)}
        for n in sorted({10, top_n}):
            row[f"top{n}_real_finds"] = int((q.head(n).group == "bad").sum())
            row[f"top{n}_false_alarms"] = int((q.head(n).group == "false_alarm").sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).round(4)
    summary.to_csv(out / "model_dice_3mm_vs_6mm_summary.csv", index=False)
    queue = pd.concat(queues)[["split", "queue_score", "rank", "case_id", "score", "dice_3mm", "dice_6mm", "group",
                               "finding"]].round(4)
    queue.to_csv(out / "queue_top_3mm_vs_6mm.csv", index=False)
    print(summary.T.to_string(header=False))
    print(queue[queue.split == "test"].to_string(index=False))

    # Part 2: where the masks disagree, per scan and per group.
    groups = {r.case_id: case_group(r.dice_3mm, r.dice_6mm, threshold) for r in dice_df.itertuples()}
    counts = pd.concat(count_frames, ignore_index=True)
    counts["group"] = counts.case_id.map(groups)
    counts[["case_id", "split", "group", "label", "distance_bin", "right_model", "n_voxels"]].to_csv(
        out / "model_disagreement_counts.csv", index=False)
    share_args = (mc["grid_band_mm"], mc["distance_bins_mm"], label_names)
    per_case = pd.DataFrame([{"case_id": r.case_id, "split": r.split, "group": groups[r.case_id],
                              "dice_3mm": r.dice_3mm, "dice_6mm": r.dice_6mm,
                              **disagreement_shares(counts[counts.case_id == r.case_id], *share_args)}
                             for r in dice_df.itertuples()])
    per_case.round(4).to_csv(out / "model_disagreement_per_case.csv", index=False)

    rows = []
    for split in sorted(dice_df.split.unique()):
        for group in GROUPS + ["all"]:
            members = per_case[(per_case.split == split) & ((per_case.group == group) | (group == "all"))]
            if members.empty:
                continue
            c = counts[counts.case_id.isin(members.case_id)]
            share_cols = [k for k in members.columns if k.endswith("_share")]
            rows.append({"status": STATUS, "split": split, "group": group, "n_scans": len(members),
                         "how": "pooled", **disagreement_shares(c, *share_args)})
            rows.append({"status": STATUS, "split": split, "group": group, "n_scans": len(members),
                         "how": "median_per_scan", "n_disagree": int(members.n_disagree.median()),
                         **members[share_cols].median().to_dict()})
    dis_summary = pd.DataFrame(rows).round(4)
    dis_summary.to_csv(out / "model_disagreement_summary.csv", index=False)
    print(dis_summary[["split", "group", "n_scans", "how", "n_disagree", "near_share", "far_label_tumor_share",
                       "far_label_background_share", "near_right_3mm_share", "far_right_3mm_share"]].to_string(index=False))

    plot_disagreement(dis_summary, mc, label_names, figures_dir(cfg) / "model_disagreement_where.png",
                      f"Where do the 3 mm and 6 mm masks disagree? Config '{cfg['name']}', {STATUS}")
    write_machine_info(out / "model_comparison_machine.json", config=str(args.config), script="scripts/compare_models.py",
                       status=STATUS, seconds=round(time.perf_counter() - start, 1))
    print(f"Done in {time.perf_counter() - start:.0f} s -> {out}")


if __name__ == "__main__":
    main()
