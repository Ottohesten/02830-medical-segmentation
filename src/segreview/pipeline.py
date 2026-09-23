"""The G1 pipeline as a sequence of steps. Each script in scripts/ runs one step; run_g1.py runs all.

Steps (all read the same config):
1. inference    predict the organ in every scan: clean scans and every configured perturbation.
2. tta          extra test-time augmentation passes on the clean scans (if uncertainty.tta.n_passes > 0).
3. dice         true Dice of every prediction against ground truth       (uses ground truth)
4. uncertainty  GT-free scores per scan + the voxel-wise entropy heatmap  (no ground truth)
5. evaluate     rank the scans by each score and compare with Dice        (uses ground truth)

Every step skips work that is already saved (except evaluate, which is fast), so an interrupted run
can simply be restarted. Clean and perturbed scans are evaluated separately: clean scans show how
the method does on real data, perturbed scans show whether it notices errors we caused on purpose.

Result files (results/<config name>/):
  inference_times.csv, tta_times.csv   runtime per scan
  dice.csv                             Dice per scan and variant
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
from segreview.data import list_cases, load_canonical, load_organ_mask
from segreview.evaluation import bad_mask, evaluate_methods
from segreview.figures import plot_curves, plot_scatter
from segreview.io import affine_of, has_prediction, load_map, map_path, save_map, voxel_spacing
from segreview.metrics import dice
from segreview.uncertainty import scan_scores

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


def step_dice(cfg: dict) -> pd.DataFrame:
    """True Dice of every saved prediction (all variants) against ground truth -> dice.csv.

    Perturbed scans are on the same voxel grid as the originals, so the same ground truth is used.
    """
    rows = []
    for case in list_cases(cfg):
        truth = load_organ_mask(cfg, case)
        for variant in variants(cfg):
            if not has_prediction(cfg, variant, case.case_id):
                continue
            pred = load_map(cfg, variant, case.case_id, "mask")
            rows.append({"case_id": case.case_id, "variant": variant, "dice": round(dice(pred, truth), 4),
                         "pred_voxels": int(pred.sum()), "truth_voxels": int(truth.sum())})
    df = pd.DataFrame(rows).sort_values(["variant", "case_id"])
    df.to_csv(results_dir(cfg) / "dice.csv", index=False)
    print(f"[dice] {len(df)} predictions -> {results_dir(cfg) / 'dice.csv'}")
    return df


def step_uncertainty(cfg: dict) -> pd.DataFrame:
    """GT-free scores for every saved prediction, and the entropy heatmap per scan -> scores.csv.

    Uses only the saved mask and probability (and TTA maps, if present). No ground truth.
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
            scores, entropy = scan_scores(
                mask, prob, voxel_spacing(cfg, variant, case.case_id), cfg["uncertainty"]["border_mm"],
                tta_votes=load_map(cfg, variant, case.case_id, "tta_votes") if has_tta else None,
                tta_std=load_map(cfg, variant, case.case_id, "tta_std") if has_tta else None,
                n_tta=n_tta if has_tta else 0)
            save_map(cfg, variant, case.case_id, "entropy", entropy, affine_of(cfg, variant, case.case_id))
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
