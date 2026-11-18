"""G3 step 1: prepare the learning curve (docs/g3_plan.md, "Design: learning curve"). Finished parts are skipped.

1. The pool (configs/g3_pool.yaml, positions 101-200 of kits100's seeded draw) is checked to share no scan with
   kits100. G1's ground-truth-free steps run on it, with kits100's settings: the 3 mm model, the 6 mm model and the
   scan scores -> data/predictions/g3_pool/, results/g3_pool/scores.csv. This is also the model output the group's
   locked ranking test uses later. No Dice and no ranking metric on the pool (log 2026-10-21).
2. The orders: the queue (most uncertain first by the locked G1 score) and the random orders ->
   results/g3_kits/orders.csv.
3. The training data in nnU-Net's preprocessed format (work_dir/nnUNet_preprocessed/): for every pool scan, the same
   preprocessing as inference (resampling to the network's grid, nnU-Net's normalization), the model's own prediction
   of all 117 structures, and two targets: corrected (kidney replaced by the ground truth; the reviewer) and, as a copy
   named <case>_nocorr, uncorrected (the control). Only the kidney is "foreground" for nnU-Net's patch sampling.
   For the 6 mm pipeline test only the first k scans of the queue are prepared. The number of voxels each correction
   changes is saved (results/g3_kits/targets_<resolution>.csv) but, until the locked test, not compared with the queue.

Usage: .venv\\Scripts\\python.exe scripts\\g3_prepare.py --config configs\\g3_kits.yaml
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_nnunet_env(work_dir: Path) -> None:
    """nnU-Net reads its folders from environment variables when it is first imported, so they are set first."""
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ[name] = str(work_dir / name)
    os.environ["nnUNet_compile"] = "false"


def prepare_targets(g3: dict, source: dict, device, resolution: str, case_ids: list[str], work_dir: Path,
                    with_control: bool) -> None:
    """Write the training scans of one model (resolution) into nnU-Net's preprocessed folder (see the docstring).

    Input: the G3 config, kits100's config (weights, dataset paths), the device, the resolution ('fast' = 3 mm,
           'fastest' = 6 mm), the pool scans to prepare, the work folder, whether to add the uncorrected copies.
    """
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import torch
    from nnunetv2.training.dataloading.nnunet_dataset import comp_blosc2_params, nnUNetDatasetBlosc2
    from totalsegmentator.resampling import change_spacing

    from segreview.config import results_dir
    from segreview.data import Case, dataset_dir, load_canonical, load_label_map
    from segreview.finetune import NOCORR_SUFFIX, corrected_target, kidney_only, own_target, side_labeled_truth
    from segreview.inference import OrganSegmenter
    from segreview.weights import model_folder

    segmenter = OrganSegmenter(source, device, resolution)
    predictor, spacing = segmenter.predictor, segmenter.spec.spacing_mm
    plans, dataset_json = predictor.plans_manager.plans, predictor.dataset_json
    class_ids = {name: int(v) for name, v in dataset_json["labels"].items()}
    kidney_ids = [class_ids[f"{base}_{side}"] for base in g3["truth_to_classes"] for side in ("left", "right")]
    foreground = predictor.plans_manager.get_label_manager(dataset_json).foreground_labels

    # nnU-Net's preprocessed folder for this model: plans, dataset description and fingerprint from the weights.
    pre_base = work_dir / "nnUNet_preprocessed" / plans["dataset_name"]
    pre_dir = pre_base / predictor.configuration_manager.data_identifier
    pre_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = model_folder(source, segmenter.spec)
    for name in ("dataset.json", "dataset_fingerprint.json"):
        shutil.copy(weights_dir / name, pre_base / name)
    (pre_base / f"{plans['plans_name']}.json").write_text(json.dumps(plans, indent=2))

    root, ds = dataset_dir(source), source["dataset"]
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    patch = tuple(predictor.configuration_manager.patch_size)
    stats_path = results_dir(g3) / f"targets_{resolution}.csv"
    rows = pd.read_csv(stats_path).to_dict("records") if stats_path.exists() else []
    done = {r["case_id"] for r in rows}
    names = lambda c: [c] + ([c + NOCORR_SUFFIX] if with_control else [])
    todo = [c for c in case_ids if c not in done or not all((pre_dir / f"{n}.pkl").exists() for n in names(c))]
    print(f"[targets {resolution}] {len(case_ids)} scans, {len(todo)} to prepare -> {pre_dir}")
    for case_id in todo:
        t0 = time.perf_counter()
        case = Case(case_id, root / ds["image_glob"].replace("*", case_id), root / ds["label_path"].format(stem=case_id))
        image = load_canonical(case.image_path)
        labels = load_label_map(case).astype(np.uint8)
        image_rs = change_spacing(image, spacing, order=3, dtype=np.int32)
        labels_rs = change_spacing(nib.Nifti1Image(labels, image.affine), spacing, target_shape=image_rs.shape,
                                   order=0, dtype=np.uint8)
        ct = np.asanyarray(image_rs.dataobj).astype(np.float32)
        truth = side_labeled_truth(np.asanyarray(labels_rs.dataobj), ct, class_ids, g3["truth_to_classes"],
                                   g3["body_threshold_hu"])
        # nnU-Net's reader gives (z, y, x), as in inference.py. Signed labels: nnU-Net marks voxels outside the scan's
        # nonzero region with -1 (in uint8 that would become 255).
        data, seg = ct.transpose(2, 1, 0)[None], truth.astype(np.int16).transpose(2, 1, 0)[None]
        data_pp, seg_pp, props = preprocessor.run_case_npy(data, seg, {"spacing": [spacing] * 3},
                                                           predictor.plans_manager,
                                                           predictor.configuration_manager, dataset_json)
        t1 = time.perf_counter()
        with torch.no_grad():
            pred = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data_pp)).argmax(0).numpy()
        t2 = time.perf_counter()
        blocks, chunks = comp_blosc2_params(data_pp.shape, patch, 4)
        blocks_seg, chunks_seg = comp_blosc2_params(data_pp.shape, patch, 2)
        targets = {case_id: corrected_target(pred, seg_pp[0], kidney_ids)}
        if with_control:
            targets[case_id + NOCORR_SUFFIX] = own_target(pred, seg_pp[0])
        for name, target in targets.items():
            # G3's foreground: only the target's kidney classes have locations, so nnU-Net's foreground patches are
            # centred on the kidney (the same sampler and seed as nnU-Net's own preprocessing).
            props_t = {**props, "class_locations": preprocessor._sample_foreground_locations(
                kidney_only(target, kidney_ids)[None], foreground, seed=1234)}
            nnUNetDatasetBlosc2.save_case(data_pp.astype(np.float32), target[None], props_t, str(pre_dir / name),
                                          chunks=chunks, blocks=blocks, chunks_seg=chunks_seg, blocks_seg=blocks_seg)
        inside = seg_pp[0] != -1
        rows = [r for r in rows if r["case_id"] != case_id] + [{
            "case_id": case_id, "resolution": resolution, "grid_zyx": "x".join(map(str, data_pp.shape[1:])),
            "voxels_changed_by_correction": int(((targets[case_id] != pred) & inside).sum()),
            "prepare_s": round(t1 - t0, 1), "predict_s": round(t2 - t1, 1)}]
        # Sorted by scan id and written after every scan; not printed (no comparison with the queue before the
        # locked test).
        pd.DataFrame(rows).sort_values("case_id").to_csv(stats_path, index=False)
        print(f"  {case_id}: {time.perf_counter() - t0:.0f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="configs/g3_kits.yaml")
    args = parser.parse_args()
    g3 = yaml.safe_load(args.config.read_text())
    work_dir = REPO_ROOT / g3["work_dir"]
    setup_nnunet_env(work_dir)
    started = time.perf_counter()

    import pandas as pd

    from segreview.config import load_config, results_dir, setup_compute
    from segreview.data import list_cases, selected_case_ids
    from segreview.finetune import orders_table, pool_orders
    from segreview.machine import machine_info, write_machine_info
    from segreview.pipeline import step_inference, step_second_model, step_uncertainty

    machine = machine_info()
    g3["paths"] = {"results_dir": REPO_ROOT / "results"}            # so results_dir(g3) is results/g3_kits/
    source = load_config(REPO_ROOT / g3["eval_config"])
    pool = load_config(REPO_ROOT / g3["pool_config"])
    pool_ids = selected_case_ids(pool)
    overlap = set(pool_ids) & set(selected_case_ids(source))
    if overlap:
        raise SystemExit(f"The pool shares scans with {source['name']}: {sorted(overlap)}")
    if pool["evaluation"]["splits"]:
        raise SystemExit("The pool config must not allow ground truth for G1's evaluation (evaluation.splits: []).")

    # ---- 1. G1's ground-truth-free steps on the pool --------------------------------------------------------------
    device = setup_compute(pool)
    step_inference(pool, device)
    step_second_model(pool, device)
    step_uncertainty(pool, cases=list_cases(pool))

    # ---- 2. orders --------------------------------------------------------------------------------------------------
    scores = pd.read_csv(results_dir(pool) / "scores.csv")
    scores = scores[scores.variant == "clean"]
    if sorted(scores.case_id) != sorted(pool_ids):
        raise SystemExit("scores.csv does not have exactly the pool's scans.")
    orders = pool_orders(scores, g3["orders"]["queue_score"], g3["orders"]["random_seeds"])
    orders_table(orders).to_csv(results_dir(g3) / "orders.csv", index=False)
    print(f"[orders] {', '.join(orders)} -> {results_dir(g3) / 'orders.csv'}")

    # ---- 3. training data ---------------------------------------------------------------------------------------------
    import torch
    device = torch.device(g3["compute"]["device"])
    prepare_targets(g3, source, device, g3["model_resolution"], pool_ids, work_dir, with_control=True)
    test = g3["pipeline_test"]
    prepare_targets(g3, source, device, test["model_resolution"], orders["queue"][:test["k"]], work_dir,
                    with_control=False)
    write_machine_info(results_dir(g3) / "prepare_machine.json", info=machine, config=str(args.config),
                       run_seconds=round(time.perf_counter() - started))
    print(f"-> {results_dir(g3)}")


if __name__ == "__main__":
    main()
