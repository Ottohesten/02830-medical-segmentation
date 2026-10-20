"""EXPLORATORY: scan scores from only the real 3 mm vs. 6 mm differences (src/segreview/disagreement_scores.py).

Follow-up on the disagreement breakdown of 2026-10-19, asked after the G1 lock. The candidates and the selection rule
are fixed in the config (exploratory.disagreement_scores) before this script was first run. kits100 is used as
development data for this question: dev and test are reported separately, and a chosen score must later be tested
once on new scans (the G3 pool). Reads only saved predictions (mask, m2_mask) and the locked G1 result files; no
inference, and no locked file is written.

Checks (the script stops if one fails):
  - the 3 mm mask volume equals pred_volume_ml of the locked scores;
  - the far / directional voxel counts equal those rebuilt from model_disagreement_counts.csv (2026-10-19);
  - the 3 mm Dice equals the locked Dice, and the metric rows of the locked measures, oracle and random are the
    same as in the locked g1_metrics_clean_<split>.csv (same seed, so the same random orders and resamples).

Writes to results/<config name>/:
  exploratory_disagreement_scores.csv     per scan: voxel counts, the 8 GT-free scores, and (evaluation only)
                                          both models' Dice and the scan group
  exploratory_disagreement_metrics.csv    per split and bad rule: Spearman rho, review / quality AUC with CIs
                                          (the G1 evaluate_methods), for new scores, locked measure, entropy,
                                          volume, random and oracle
  exploratory_disagreement_paired.csv     per split: paired bootstrap difference of each new score to the locked
                                          measure and to entropy (rho, review AUC per rule, quality AUC)
  exploratory_disagreement_choice.csv     the candidates on the selection split, ranked by the selection rule
  exploratory_queue_ranks.csv             the top of the locked queue, with each scan's rank under every score
  exploratory_queue_counts.csv            real finds / false alarms / both fine in the top 10 and 20 of each queue
  figures/exploratory_review_curves_test.png  review curves on the test set
  exploratory_disagreement_machine.json   machine, library versions and git commit of this run

Usage: uv run python scripts/explore_disagreement_scores.py --config configs/kits100.yaml
"""

import time

import numpy as np
import pandas as pd

from segreview.config import config_arg_parser, figures_dir, load_config, results_dir
from segreview.data import case_split, evaluation_cases, organ_definition
from segreview.disagreement_scores import (CANDIDATE_COUNTS, candidate_scores, choose_score, counts_from_breakdown,
                                           queue_rank, score_names, top_counts)
from segreview.evaluation import bad_mask, evaluate_methods, paired_differences
from segreview.followup_figures import plot_review_curves_explore
from segreview.io import load_map, voxel_spacing
from segreview.machine import write_machine_info
from segreview.model_comparison import case_group, directional_disagreement

STATUS = "exploratory (after the G1 lock; kits100 used as development data)"


def bad_threshold(cfg: dict, rule_name: str) -> float:
    """The Dice threshold of a fixed-threshold 'bad' rule in evaluation.bad_rules."""
    for rule in cfg["evaluation"]["bad_rules"]:
        if rule["name"] == rule_name and rule["rule"] == "threshold":
            return float(rule["value"])
    raise ValueError(f"No threshold rule named '{rule_name}' in evaluation.bad_rules")


def check_breakdown(scan_df: pd.DataFrame, cfg: dict, out) -> None:
    """Stop unless the new GT-free counts match the ones rebuilt from the 2026-10-19 breakdown (see module docstring)."""
    mc = cfg["exploratory"]["model_comparison"]
    definition = organ_definition(cfg)
    if definition["ignore_labels"]:   # the breakdown left ignored voxels out, so it cannot be compared directly
        print("Breakdown check skipped: the organ definition ignores labels.")
        return
    names = cfg["dataset"]["label_names"]
    organ_names = {names[v] for v in definition["organ_labels"]}
    breakdown = pd.read_csv(out / "model_disagreement_counts.csv")
    bad = []
    for r in scan_df.itertuples():
        expected = counts_from_breakdown(breakdown[breakdown.case_id == r.case_id], mc["grid_band_mm"],
                                         mc["distance_bins_mm"], organ_names)
        if any(expected[k] != getattr(r, k) for k in expected):
            bad.append((r.case_id, expected))
    if bad:
        raise SystemExit(f"STOP: counts differ from model_disagreement_counts.csv for {len(bad)} scans: {bad[:3]}")
    print(f"Check passed: far / directional counts match model_disagreement_counts.csv for all {len(scan_df)} scans.")


def check_locked_rows(rows: pd.DataFrame, locked: pd.DataFrame, methods: list[str], split: str) -> None:
    """Stop unless the metric rows of the already locked methods are the same as in the locked G1 file."""
    cols = [c for c in locked.columns if c in rows.columns and c not in ("bad_rule", "method")]
    a = rows[rows.method.isin(methods)].set_index(["bad_rule", "method"])[cols].round(4)
    b = locked[locked.method.isin(methods)].set_index(["bad_rule", "method"]).loc[a.index, cols]
    diff = (a - b).abs().max().max()
    if not np.isclose(diff, 0, atol=1e-4):
        raise SystemExit(f"STOP: {split}: metrics of {methods} differ from g1_metrics_clean_{split}.csv (max {diff})")
    print(f"Check passed: {split}: metrics of {', '.join(methods)} equal the locked file (max diff {diff:.1e}).")


def main():
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    ex = cfg["exploratory"]
    ds, mc = ex["disagreement_scores"], ex["model_comparison"]
    ev = cfg["evaluation"]
    primary = cfg["report"]["primary"]
    names = score_names(ds["candidates"], ds["normalizations"])
    unknown = set(ds["candidates"]) - set(CANDIDATE_COUNTS)
    if unknown:
        raise ValueError(f"Unknown candidates in the config: {unknown}")
    out = results_dir(cfg)
    start = time.perf_counter()
    print(f"*** {STATUS.upper()}: dev and test reported separately ***")

    # 1. GT-free scores per scan: only the two saved masks are read here.
    rows = []
    for case in evaluation_cases(cfg):
        spacing = voxel_spacing(cfg, "clean", case.case_id)
        mask = load_map(cfg, "clean", case.case_id, "mask").astype(bool)
        m2_mask = load_map(cfg, "clean", case.case_id, "m2_mask").astype(bool)
        counts = directional_disagreement(mask, m2_mask, spacing, mc["grid_band_mm"])
        voxel_ml = float(np.prod(spacing)) / 1000.0
        rows.append({"case_id": case.case_id, "split": case_split(cfg, case.case_id), "voxel_ml": voxel_ml,
                     **counts, **candidate_scores(counts, voxel_ml, ds["candidates"], ds["normalizations"])})
        r = rows[-1]
        print(f"  {case.case_id} [{r['split']}]: far {counts['far_disagree']}, far 6 mm adds {counts['far_b_adds']}, "
              f"6 mm adds (all) {counts['b_adds_all']} voxels")
    scan_df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    check_breakdown(scan_df, cfg, out)

    # 2. Join the locked G1 scores and (evaluation only) both models' Dice.
    locked_scores = pd.concat([pd.read_csv(out / f"g1_scores_clean_{s}.csv") for s in ev["splits"]])
    model_dice = pd.read_csv(out / "model_dice_3mm_vs_6mm.csv")[["case_id", "dice_3mm", "dice_6mm"]]
    df = scan_df.merge(locked_scores[["case_id", "pred_volume_ml", "neg_volume_ml", primary, "entropy_mean_region",
                                      "dice"]], on="case_id", validate="one_to_one")
    df = df.merge(model_dice, on="case_id", validate="one_to_one")
    if len(df) != len(scan_df):
        raise SystemExit("STOP: some scans are missing in the locked score files or model_dice_3mm_vs_6mm.csv")
    if not np.allclose(df.n_mask_a * df.voxel_ml, df.pred_volume_ml, rtol=1e-6):
        raise SystemExit("STOP: the 3 mm mask volume differs from pred_volume_ml in the locked scores")
    if not np.array_equal(df.dice, df.dice_3mm):
        raise SystemExit("STOP: dice_3mm differs from the locked Dice")
    print("Check passed: 3 mm volume and Dice equal the locked G1 files.")
    threshold = bad_threshold(cfg, mc["bad_rule"])
    df["group"] = [case_group(a, b, threshold) for a, b in zip(df.dice_3mm, df.dice_6mm)]
    df.drop(columns=["dice", "pred_volume_ml", "neg_volume_ml", primary, "entropy_mean_region"]).round(6).to_csv(
        out / "exploratory_disagreement_scores.csv", index=False)

    # 3. Evaluate per split with the G1 functions, plus paired differences to the locked measure and entropy.
    known = [primary, "entropy_mean_region", "neg_volume_ml"]
    metric_frames, paired_frames, curves_by_split = [], [], {}
    for split in ev["splits"]:
        d_split = df[df.split == split].reset_index(drop=True)
        dice = d_split.dice_3mm.to_numpy(dtype=float)
        scores = {m: d_split[m].to_numpy(dtype=float) for m in names + known}
        split_rows, curves_by_split[split] = [], {}
        for rule in ev["bad_rules"]:
            r, curves = evaluate_methods(scores, dice, cfg, rule)
            n_bad = int(bad_mask(dice, rule["rule"], rule["value"]).sum())
            split_rows += [{"split": split, "bad_rule": rule["name"], "n_bad": n_bad, **x} for x in r]
            curves_by_split[split][rule["name"]] = curves
        split_df = pd.DataFrame(split_rows)
        check_locked_rows(split_df, pd.read_csv(out / f"g1_metrics_clean_{split}.csv"), known + ["oracle", "random"],
                          split)
        metric_frames.append(split_df)
        # Only the new scores are compared (the same resamples for all, so the differences are paired).
        pr = paired_differences({m: scores[m] for m in names + ds["compare_to"]}, ds["compare_to"], dice, ev["bad_rules"], ev["n_bootstrap"], ev["ci_level"],
                                np.random.default_rng(cfg["seed"]))
        paired_frames.append(pd.DataFrame(pr).assign(split=split))
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.insert(0, "status", STATUS)
    metrics.round(4).to_csv(out / "exploratory_disagreement_metrics.csv", index=False)
    paired = pd.concat(paired_frames, ignore_index=True)
    paired = paired[["split", "method", "reference", "measure", "bad_rule", "diff", "diff_ci_low", "diff_ci_high",
                     "share_boot_better"]]
    paired.round(4).to_csv(out / "exploratory_disagreement_paired.csv", index=False)

    # 4. The pre-registered choice: on the selection split and rule only.
    sel = ds["selection"]
    sel_rows = metrics[(metrics.split == sel["split"]) & (metrics.bad_rule == sel["bad_rule"])]
    choice = choose_score(sel_rows, names, sel["metric"], "quality_auc")
    ref = sel_rows.set_index("method")
    choice["locked_primary_" + sel["metric"]] = ref.loc[primary, sel["metric"]]
    choice["entropy_" + sel["metric"]] = ref.loc["entropy_mean_region", sel["metric"]]
    choice.insert(0, "split", sel["split"])
    choice.insert(1, "bad_rule", sel["bad_rule"])
    choice.round(4).to_csv(out / "exploratory_disagreement_choice.csv", index=False)
    chosen = choice.loc[choice.chosen, "method"].iloc[0]
    print(choice.round(4).to_string(index=False))
    print(f"Chosen on {sel['split']} ({sel['metric']}, {sel['bad_rule']}): {chosen}")

    # 5. The queue: where the top of the locked queue ends up, and false alarms in each score's top 10 / 20.
    rank_frames, count_rows = [], []
    for split in ev["splits"]:
        d_split = df[df.split == split].reset_index(drop=True)
        groups = d_split.group.to_numpy()
        ranks = pd.DataFrame({"split": split, "case_id": d_split.case_id, "dice_3mm": d_split.dice_3mm,
                              "dice_6mm": d_split.dice_6mm, "group": groups})
        for m in [primary, "entropy_mean_region"] + names:
            ranks[f"rank_{m}"] = queue_rank(d_split[m].to_numpy(dtype=float))
        top = ranks.sort_values(f"rank_{primary}").head(max(ds["queue_top_n"]))
        rank_frames.append(top)
        for m in [primary, "entropy_mean_region", "neg_volume_ml"] + names:
            for n in ds["queue_top_n"]:
                count_rows.append({"split": split, "method": m, "top_n": min(n, len(d_split)),
                                   **top_counts(d_split[m].to_numpy(dtype=float), groups, n)})
    queue_ranks = pd.concat(rank_frames, ignore_index=True)
    queue_ranks["finding"] = queue_ranks.group.map({"bad": "real find", "false_alarm": "false alarm", "good": ""})
    queue_ranks.to_csv(out / "exploratory_queue_ranks.csv", index=False)
    queue_counts = pd.DataFrame(count_rows).round(2)
    queue_counts.to_csv(out / "exploratory_queue_counts.csv", index=False)
    print(queue_counts.to_string(index=False))

    # 6. One figure: review curves on the figure split, the candidates in the chosen normalization.
    fig = ds["figure"]
    norm = next(n for n in ds["normalizations"] if chosen.endswith(f"_{n}"))
    rule_order = [fig["bad_rule"]] + [r["name"] for r in ev["bad_rules"] if r["name"] != fig["bad_rule"]]
    fig_metrics = metrics[metrics.split == fig["split"]]
    plot_review_curves_explore({r: curves_by_split[fig["split"]][r] for r in rule_order}, fig_metrics,
                               ds["candidates"], norm, primary,
                               figures_dir(cfg) / f"exploratory_review_curves_{fig['split']}.png",
                               f"Review curves, {fig['split']} set (n = {int((df.split == fig['split']).sum())}), "
                               f"config '{cfg['name']}' — {STATUS}")

    write_machine_info(out / "exploratory_disagreement_machine.json", config=str(args.config),
                       script="scripts/explore_disagreement_scores.py", status=STATUS, chosen=chosen,
                       seconds=round(time.perf_counter() - start, 1))
    print(f"Done in {time.perf_counter() - start:.0f} s -> {out}")


if __name__ == "__main__":
    main()
