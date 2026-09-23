"""The G1 pipeline as a sequence of steps. Each script in scripts/ runs one step; run_g1.py runs all.

Steps (all read the same config):
1. inference    predict the organ in every scan: clean scans and every configured perturbation.
2. tta          extra test-time augmentation passes on the clean scans (if uncertainty.tta.n_passes > 0).
3. second model predict the clean scans with a second network (uncertainty.second_model_resolution,
                e.g. the 6 mm model), to measure how much two models disagree.
4. dice         true Dice of every prediction against ground truth       (uses ground truth)
5. label check  if the dataset has several organ definitions: Dice under each, and how much of each
                ground-truth label the model calls organ                  (uses ground truth)
6. uncertainty  GT-free scores per scan + the voxel-wise heatmaps         (no ground truth)
7. evaluate     rank the scans by each score and compare with Dice        (uses ground truth)

Every step skips work that is already saved (except evaluate, which is fast), so an interrupted run
can simply be restarted. Clean and perturbed scans are evaluated separately: clean scans show how
the method does on real data, perturbed scans show whether it notices errors we caused on purpose.

Result files (results/<config name>/):
  inference_times.csv, tta_times.csv, second_model_times.csv   runtime per scan
  dice.csv                             Dice per scan and variant (organ definition from the config)
  label_check.csv, label_check_summary.csv, figures/label_check.png   (only with several definitions)
  scores.csv                           GT-free scores per scan and variant
  g1_scores_<set>.csv                  scores + Dice + "bad" flag, for set = clean / perturbed
  g1_metrics_<set>.csv                 Spearman, curve areas and CIs per method
  figures/g1_curves_<set>.png, figures/g1_scatter_<set>.png
"""

import time

import numpy as np
import pandas as pd
import torch

from segreview.augment import perturb, sample_tta_passes
from segreview.config import figures_dir, results_dir
from segreview.data import list_cases, load_canonical, load_label_map, load_organ_mask, organ_definition, organ_from_labels
from segreview.evaluation import bad_mask, evaluate_methods
from segreview.figures import plot_curves, plot_label_check, plot_scatter
from segreview.io import affine_of, has_prediction, load_map, map_path, save_map, voxel_spacing
from segreview.metrics import dice
from segreview.uncertainty import model_difference, scan_scores

CLEAN = "clean"
# Columns in scores.csv that are information, not ranking methods.
INFO_COLUMNS = {"case_id", "variant", "pred_volume_ml"}


def variants(cfg: dict) -> list[str]:
    """'clean' plus the name of every configured perturbation."""
    return [CLEAN] + [p["name"] for p in (cfg.get("perturbations") or [])]


def _update_csv(path, new: pd.DataFrame, keys: list[str]) -> None:
    """Add rows to a CSV file, replacing old rows with the same keys (e.g. case_id + variant)."""
    if path.exists():
        new = pd.concat([pd.read_csv(path), new]).drop_duplicates(keys, keep="last")
    new.sort_values(keys).to_csv(path, index=False)


def step_inference(cfg: dict, device: torch.device, overwrite: bool = False) -> None:
    """Predict mask + organ probability for every scan and variant that has no saved prediction yet."""
    from segreview.inference import OrganSegmenter

    cases = list_cases(cfg)
    specs = {p["name"]: p for p in (cfg.get("perturbations") or [])}
    todo = [(i, c, v) for v in variants(cfg) for i, c in enumerate(cases)
            if overwrite or not has_prediction(cfg, v, c.case_id)]
    print(f"[inference] device={device}, {len(cases)} scans x {len(variants(cfg))} variants, {len(todo)} to predict")
    if not todo:
        return

    segmenter = OrganSegmenter(cfg, device)
    for i, case, variant in todo:
        image = load_canonical(case.image_path)
        if variant != CLEAN:
            image = perturb(image, specs[variant], cfg, i)
        start = time.perf_counter()
        pred = segmenter.predict(image)
        seconds = time.perf_counter() - start
        save_map(cfg, variant, case.case_id, "mask", pred.mask, pred.affine)
        save_map(cfg, variant, case.case_id, "prob", pred.prob, pred.affine)
        print(f"  {case.case_id} [{variant}]: {seconds:.1f} s")
        # Written after every scan, so the log is complete even if the run stops halfway.
        _update_csv(results_dir(cfg) / "inference_times.csv",
                    pd.DataFrame([{"case_id": case.case_id, "variant": variant, "seconds": round(seconds, 2),
                                   "device": str(device), "resolution": cfg["model"]["resolution"],
                                   "shape": "x".join(map(str, image.shape)), "source": "run"}]),
                    ["case_id", "variant"])


def step_tta(cfg: dict, device: torch.device, overwrite: bool = False) -> None:
    """Run the TTA passes on the clean scans and save vote count + standard deviation maps.

    For each voxel we keep: in how many passes it was organ (tta_votes), and the standard deviation
    of the organ probability over the passes (tta_std). Both are summaries, so the individual passes
    never need to be stored.
    """
    n_passes = cfg["uncertainty"]["tta"]["n_passes"]
    if n_passes == 0:
        print("[tta] n_passes = 0, skipped")
        return
    from segreview.inference import OrganSegmenter

    cases = list_cases(cfg)
    todo = [(i, c) for i, c in enumerate(cases)
            if overwrite or not map_path(cfg, CLEAN, c.case_id, "tta_std").exists()]
    print(f"[tta] {n_passes} passes x {len(todo)} scans to do")
    if not todo:
        return

    segmenter = OrganSegmenter(cfg, device)
    for i, case in todo:
        image = load_canonical(case.image_path)
        votes = np.zeros(image.shape, dtype=np.uint8)
        total = np.zeros(image.shape, dtype=np.float32)
        total_sq = np.zeros(image.shape, dtype=np.float32)
        start = time.perf_counter()
        for tta_pass in sample_tta_passes(cfg, i):
            pred = segmenter.predict(image, tta=tta_pass)
            votes += pred.mask
            total += pred.prob
            total_sq += pred.prob ** 2
        seconds = time.perf_counter() - start
        mean = total / n_passes
        std = np.sqrt(np.maximum(total_sq / n_passes - mean ** 2, 0.0))  # var = E[p^2] - E[p]^2
        save_map(cfg, CLEAN, case.case_id, "tta_votes", votes, image.affine)
        save_map(cfg, CLEAN, case.case_id, "tta_std", std, image.affine)
        print(f"  {case.case_id}: {n_passes} passes in {seconds:.1f} s")
        _update_csv(results_dir(cfg) / "tta_times.csv",
                    pd.DataFrame([{"case_id": case.case_id, "n_passes": n_passes, "seconds": round(seconds, 2),
                                   "device": str(device)}]), ["case_id"])


def step_second_model(cfg: dict, device: torch.device, overwrite: bool = False) -> None:
    """Predict the clean scans with the second model and save its mask and probability (m2_mask, m2_prob).

    The public TotalSegmentator weights have only one fold, so fold disagreement is impossible. The 6 mm
    network is a separately trained model of the same structures, so the disagreement between it and the
    main model can play the role of an ensemble. Skipped if uncertainty.second_model_resolution is null.
    """
    resolution = cfg["uncertainty"].get("second_model_resolution")
    if not resolution:
        print("[second model] not configured, skipped")
        return
    from segreview.inference import OrganSegmenter

    cases = list_cases(cfg)
    todo = [c for c in cases if overwrite or not map_path(cfg, CLEAN, c.case_id, "m2_prob").exists()]
    print(f"[second model] resolution={resolution}, {len(todo)} scans to predict")
    if not todo:
        return
    segmenter = OrganSegmenter(cfg, device, resolution=resolution)
    for case in todo:
        image = load_canonical(case.image_path)
        start = time.perf_counter()
        pred = segmenter.predict(image)
        seconds = time.perf_counter() - start
        save_map(cfg, CLEAN, case.case_id, "m2_mask", pred.mask, pred.affine)
        save_map(cfg, CLEAN, case.case_id, "m2_prob", pred.prob, pred.affine)
        print(f"  {case.case_id}: {seconds:.1f} s")
        _update_csv(results_dir(cfg) / "second_model_times.csv",
                    pd.DataFrame([{"case_id": case.case_id, "resolution": resolution, "seconds": round(seconds, 2),
                                   "device": str(device)}]), ["case_id"])


def step_label_check(cfg: dict) -> pd.DataFrame | None:
    """Compare ground-truth definitions of the organ (only if the dataset lists more than one).

    For each clean scan:
    - Dice of the model's mask under every definition in dataset.organ_definitions.
    - coverage_label_<v>: the share of the voxels with ground-truth label v that the model calls organ.
      For KiTS, coverage_label_2 answers "how much of the tumour does TotalSegmentator call kidney or cyst?".
      Empty (NaN) if the scan has no voxels with that label.
    Writes label_check.csv, label_check_summary.csv (min/median/max per definition) and figures/label_check.png.
    """
    definitions = list(cfg["dataset"]["organ_definitions"])
    if len(definitions) < 2:
        return None
    label_values = sorted({v for d in definitions for key in ("organ_labels", "ignore_labels")
                           for v in organ_definition(cfg, d)[key]})
    rows = []
    for case in list_cases(cfg):
        if not has_prediction(cfg, CLEAN, case.case_id):
            continue
        labels = load_label_map(case)
        pred = load_map(cfg, CLEAN, case.case_id, "mask").astype(bool)
        row = {"case_id": case.case_id}
        for d in definitions:
            truth, ignore = organ_from_labels(labels, organ_definition(cfg, d))
            row[f"dice_{d}"] = round(dice(pred, truth, ignore), 4)
        for v in label_values:
            in_label = labels == v
            n = int(in_label.sum())
            row[f"voxels_label_{v}"] = n
            row[f"coverage_label_{v}"] = round(float(pred[in_label].sum()) / n, 4) if n else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    out = results_dir(cfg)
    df.to_csv(out / "label_check.csv", index=False)

    stat_cols = [f"dice_{d}" for d in definitions] + [f"coverage_label_{v}" for v in label_values]
    summary = df[stat_cols].agg(["count", "min", "median", "max", "mean"]).T.round(4)
    summary.to_csv(out / "label_check_summary.csv", index_label="measure")
    plot_label_check(df, definitions, [f"coverage_label_{v}" for v in label_values],
                     cfg["visualisation"]["dice_bin_width"], cfg["visualisation"]["share_bin_width"], cfg["seed"],
                     f"Organ definitions, config '{cfg['name']}' (clean scans, n = {len(df)})",
                     figures_dir(cfg) / "label_check.png", cfg["visualisation"]["dpi"])
    print(f"[label check] {len(df)} scans -> {out / 'label_check.csv'}")
    print(summary.to_string())
    return df


def step_dice(cfg: dict) -> pd.DataFrame:
    """True Dice of every saved prediction (all variants) against ground truth -> dice.csv.

    Perturbed scans are on the same voxel grid as the originals, so the same ground truth is used.
    """
    rows = []
    for case in list_cases(cfg):
        truth, ignore = load_organ_mask(cfg, case)
        for variant in variants(cfg):
            if not has_prediction(cfg, variant, case.case_id):
                continue
            pred = load_map(cfg, variant, case.case_id, "mask")
            rows.append({"case_id": case.case_id, "variant": variant, "dice": round(dice(pred, truth, ignore), 4),
                         "pred_voxels": int(pred.sum()), "truth_voxels": int(truth.sum())})
    df = pd.DataFrame(rows).sort_values(["variant", "case_id"])
    df.to_csv(results_dir(cfg) / "dice.csv", index=False)
    print(f"[dice] {len(df)} predictions -> {results_dir(cfg) / 'dice.csv'}")
    return df


def step_uncertainty(cfg: dict) -> pd.DataFrame:
    """GT-free scores for every saved prediction, and the entropy heatmap per scan -> scores.csv.

    Uses only the saved mask and probability (and TTA / second-model maps, if present). No ground truth.
    """
    n_tta = cfg["uncertainty"]["tta"]["n_passes"]
    rows = []
    for case in list_cases(cfg):
        for variant in variants(cfg):
            if not has_prediction(cfg, variant, case.case_id):
                continue
            mask = load_map(cfg, variant, case.case_id, "mask")
            prob = load_map(cfg, variant, case.case_id, "prob")
            has_tta = n_tta > 0 and map_path(cfg, variant, case.case_id, "tta_std").exists()
            has_m2 = map_path(cfg, variant, case.case_id, "m2_prob").exists()
            m2_prob = load_map(cfg, variant, case.case_id, "m2_prob") if has_m2 else None
            scores, entropy = scan_scores(
                mask, prob, voxel_spacing(cfg, variant, case.case_id), cfg["uncertainty"]["border_mm"],
                tta_votes=load_map(cfg, variant, case.case_id, "tta_votes") if has_tta else None,
                tta_std=load_map(cfg, variant, case.case_id, "tta_std") if has_tta else None,
                n_tta=n_tta if has_tta else 0,
                m2_mask=load_map(cfg, variant, case.case_id, "m2_mask") if has_m2 else None,
                m2_prob=m2_prob)
            affine = affine_of(cfg, variant, case.case_id)
            save_map(cfg, variant, case.case_id, "entropy", entropy, affine)
            if has_m2:  # second heatmap: where the two models disagree
                save_map(cfg, variant, case.case_id, "m2_diff", model_difference(prob, m2_prob), affine)
            rows.append({"case_id": case.case_id, "variant": variant, **scores})
    df = pd.DataFrame(rows).sort_values(["variant", "case_id"])
    df.to_csv(results_dir(cfg) / "scores.csv", index=False)
    print(f"[uncertainty] {len(df)} predictions -> {results_dir(cfg) / 'scores.csv'} (+ entropy heatmaps)")
    return df


def step_evaluate(cfg: dict) -> None:
    """Evaluate every score on the clean scans and, separately, on the perturbed scans.

    A method is only evaluated on a set if it has a value for every scan in that set
    (e.g. TTA scores only exist where TTA was run).
    """
    out = results_dir(cfg)
    merged = pd.read_csv(out / "scores.csv").merge(pd.read_csv(out / "dice.csv"), on=["case_id", "variant"])
    sets = {"clean": merged[merged.variant == CLEAN], "perturbed": merged[merged.variant != CLEAN]}
    ev = cfg["evaluation"]

    for set_name, df in sets.items():
        if df.empty:
            continue
        df = df.reset_index(drop=True)
        d = df["dice"].to_numpy()
        methods = [c for c in df.columns if c not in INFO_COLUMNS | {"dice", "pred_voxels", "truth_voxels"}
                   and df[c].notna().all()]
        scores = {m: df[m].to_numpy(dtype=float) for m in methods}
        rows, curves = evaluate_methods(scores, d, cfg)

        bad = bad_mask(d, ev["bad"]["rule"], ev["bad"]["value"])
        df.assign(bad=bad).to_csv(out / f"g1_scores_{set_name}.csv", index=False)
        pd.DataFrame(rows).round(4).to_csv(out / f"g1_metrics_{set_name}.csv", index=False)

        ids = df["case_id"] if set_name == CLEAN else df["case_id"] + " / " + df["variant"]
        note = "  -- only a pipeline test, too few scans to interpret" if len(df) < 10 else ""
        title = (f"G1 on {set_name} scans, config '{cfg['name']}' (n = {len(df)}, bad = {int(bad.sum())} by "
                 f"{ev['bad']['rule']} {ev['bad']['value']}){note}")
        plot_curves(rows, curves, title, figures_dir(cfg) / f"g1_curves_{set_name}.png", cfg["visualisation"]["dpi"])
        plot_scatter({**scores}, d, bad, list(ids), rows, title,
                     figures_dir(cfg) / f"g1_scatter_{set_name}.png", cfg["visualisation"]["dpi"])
        print(f"[evaluate] {set_name}: n = {len(df)}, bad = {int(bad.sum())} -> {out / f'g1_metrics_{set_name}.csv'}")
        cols = ["method", "spearman_rho", "rho_ci_low", "rho_ci_high", "partial_rho_given_volume",
                "review_auc", "quality_auc",
                "quality_auc_minus_volume", "quality_auc_minus_volume_ci_low", "quality_auc_minus_volume_ci_high"]
        print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))
