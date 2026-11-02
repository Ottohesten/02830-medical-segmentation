"""G3 step 3: predict kits100 scans with the original model and with trained G3 models, and measure Dice per scan
and per model (docs/g3_plan.md, "Design: learning curve"). Finished (model, scan) pairs are skipped, so a stopped
evaluation continues with the next scan.

Prediction uses G1's inference code (src/segreview/inference.py) on this machine, for every model, also the original
one (never the Mac's saved predictions: machines differ). A trained model is the pretrained network with the run's
weights (checkpoint_final.pth). Per scan and model:
- Dice for each organ definition in configs/g3_kits.yaml (kidney_tumor_cyst, kidney_cyst_ignore_tumor);
- forgetting: the share of voxels the original model gives another structure (not kidney) whose label the model
  changes, on the network's grid (finetune.other_organs_changed); the original model's labels are kept for this;
- the predicted kidney mask, saved as work_dir/predictions/<model>/<case>.nii.gz (not in git).
Output (git): results/g3_kits/dice/<model>.csv (one row per scan) and <model>.json (machine, commit, checkpoint).

Usage: .venv\\Scripts\\python.exe scripts\\g3_evaluate.py --config configs\\g3_kits.yaml --models original <run_id> ...
                 [--splits dev test] [--max-scans N]
"""

import argparse
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "original"


def checkpoint_of(g3: dict, model: str) -> Path | None:
    """The trained weights of a G3 run (None for the original model)."""
    if model == ORIGINAL:
        return None
    found = sorted((REPO_ROOT / g3["work_dir"] / "runs" / model).rglob(g3["training"]["checkpoint"]))
    if not found:
        raise SystemExit(f"{model}: no {g3['training']['checkpoint']} (not trained yet?)")
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="configs/g3_kits.yaml")
    parser.add_argument("--models", nargs="+", required=True, help="'original' and/or run ids")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], help="kits100 splits to predict")
    parser.add_argument("--resolution", default=None, help="model resolution (default: the config's model_resolution)")
    parser.add_argument("--max-scans", type=int, default=None, help="only the first N scans (pipeline test)")
    args = parser.parse_args()
    g3 = yaml.safe_load(args.config.read_text())

    import nibabel as nib
    import numpy as np
    import pandas as pd
    import torch

    from segreview.config import load_config, setup_compute
    from segreview.data import case_split, list_cases, load_canonical, load_organ_mask
    from segreview.finetune import other_organs_changed, save_atomically
    from segreview.inference import OrganSegmenter
    from segreview.machine import machine_info, write_machine_info
    from segreview.metrics import dice

    machine = machine_info()
    source = load_config(REPO_ROOT / g3["eval_config"])
    device = setup_compute(source)            # as G1's inference: the source config's device and CPU threads
    resolution = args.resolution or g3["model_resolution"]
    definitions = g3["evaluation"]["organ_definitions"]
    cases = list_cases(source, splits=args.splits)[:args.max_scans]
    work = REPO_ROOT / g3["work_dir"]
    out = REPO_ROOT / "results" / g3["name"] / "dice"
    out.mkdir(parents=True, exist_ok=True)
    labels_dir = work / "original_labels" / resolution

    segmenter = OrganSegmenter(source, device, resolution)
    kidney_ids = segmenter.spec.organ_channels
    pretrained = segmenter.predictor.list_of_parameters
    for model in args.models:
        checkpoint = checkpoint_of(g3, model)
        # nnU-Net loads list_of_parameters into the network before every prediction, so this swaps the weights.
        segmenter.predictor.list_of_parameters = pretrained if checkpoint is None else [
            torch.load(checkpoint, map_location="cpu", weights_only=False)["network_weights"]]
        table_path = out / f"{model}.csv"
        rows = pd.read_csv(table_path).to_dict("records") if table_path.exists() else []
        done = {r["case_id"] for r in rows}
        todo = [c for c in cases if c.case_id not in done]
        print(f"[evaluate] {model}: {len(cases)} scans, {len(todo)} to predict ({resolution})")
        if not todo:
            continue
        write_machine_info(out / f"{model}.json", info=machine, config=str(args.config), model=model,
                           model_resolution=resolution, splits=args.splits,
                           checkpoint=None if checkpoint is None else str(checkpoint.relative_to(REPO_ROOT)))
        for case in todo:
            t0 = time.perf_counter()
            pred = segmenter.predict(load_canonical(case.image_path), labels=True)
            seconds = time.perf_counter() - t0
            row = {"model": model, "case_id": case.case_id, "split": case_split(source, case.case_id)}
            for definition in definitions:
                truth, ignore = load_organ_mask(source, case, definition)
                row[f"dice_{definition}"] = dice(pred.mask, truth, ignore)
            original_labels = labels_dir / f"{case.case_id}.npz"
            if model == ORIGINAL:
                labels_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(original_labels, labels=pred.labels)
                row["forgetting_share"] = 0.0
            else:
                row["forgetting_share"] = (other_organs_changed(np.load(original_labels)["labels"], pred.labels,
                                                                kidney_ids)
                                           if original_labels.exists() else float("nan"))
            row["predict_s"] = round(seconds, 2)
            mask_path = work / "predictions" / model / f"{case.case_id}.nii.gz"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(pred.mask, pred.affine), mask_path)
            rows.append(row)
            # After every scan, through a temporary file: the evaluation can stop at any time and continue.
            save_atomically(lambda p: pd.DataFrame(rows).to_csv(p, index=False), table_path)
            print(f"  {case.case_id} [{row['split']}]: {seconds:.1f} s")


if __name__ == "__main__":
    main()
