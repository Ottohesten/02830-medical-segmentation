"""G3 feasibility smoke test: can nnU-Net's own trainer fine-tune the pretrained TotalSegmentator model on a few
corrected KiTS scans on this machine, and how long does one training step take? (see docs/g3_plan.md)

This is NOT the G3 pipeline. It prepares 2 scans the way the pipeline would, starts nnU-Net's trainer on them,
runs a few training steps per batch size and measures time and memory. No model is saved.

Steps:
1. Training data, per scan, with the same preprocessing as inference (src/segreview/inference.py): canonical RAS
   orientation, resampling to the network's voxel size, nnU-Net's intensity normalization. The TARGET is the
   corrected segmentation a reviewer would hand back: the model's own prediction of all 117 structures, with the
   kidney replaced by the ground truth (the ground truth plays the reviewer). Kidney voxels the model predicted
   where the ground truth has none become background. The ground-truth kidney is split into the model's left and
   right classes at the body's midline. Keeping the model's own labels for all other structures means the
   fine-tuning is not taught to forget them.
2. The data are saved in nnU-Net's preprocessed format in work_dir, so nnU-Net's trainer (the NoMirroring variant,
   as the model was trained) reads them with its own data loader and data augmentation.
3. Sanity check on one training batch: the loss with random weights, with nnU-Net's standard start of a fine-tuning
   (load_pretrained_weights, which re-initializes the last layer, "seg_layers"), and with ALL pretrained weights.
   With all weights the loss must already be low; if not, the data or labels do not line up with the model.
4. Timing: for each batch size, a fresh trainer with all pretrained weights runs a few training steps. Measured:
   the time to get a batch (loading + augmentation), the time of the step (forward, backward, update), and the peak
   memory of the process (on a GPU: the peak GPU memory).

Output: results/g3_smoke/{cases,sanity,timing}.csv. nnU-Net's results folder is set to work_dir, so the pretrained
weights on disk are never touched.

Usage: uv run python scripts/g3_smoke_test.py --config configs/g3_smoke.yaml
"""

import argparse
import os
import platform
import resource
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_nnunet_env(work_dir: Path, processes: int) -> None:
    """nnU-Net reads its folders from environment variables when it is first imported, so they are set first."""
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ[name] = str(work_dir / name)
    os.environ["nnUNet_n_proc_DA"] = str(processes)
    os.environ["nnUNet_compile"] = "false"       # torch.compile gives nothing on a CPU and slows the start


def peak_memory_gb(device) -> float:
    """Peak memory so far: GPU memory on CUDA, otherwise the process's peak RAM (resident set size)."""
    import torch
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9   # bytes on macOS, KiB on Linux


def side_labeled_truth(labels, ct, class_ids: dict, truth_to_classes: dict, body_threshold_hu: float):
    """The ground-truth kidney in the model's own classes, split into patient left and right.

    The midline is the mean left-right position of the body voxels (CT above body_threshold_hu) in the slices that
    contain kidney, as in plausibility.py. The arrays are in canonical RAS order (x, y, z), where a larger x is
    further towards the patient's right.
    Input: ground-truth label map and CT on the same grid, the model's class ids by name, which ground-truth labels
           become which class (e.g. {"kidney": [1, 2], "kidney_cyst": [3]}), body threshold.
    Output: uint8 map with the model's class ids (0 elsewhere).
    """
    import numpy as np
    organ = np.isin(labels, [v for values in truth_to_classes.values() for v in values])
    out = np.zeros(labels.shape, np.uint8)
    if not organ.any():
        return out
    body = ct[:, :, organ.any(axis=(0, 1))] > body_threshold_hu
    midline = np.nonzero(body)[0].mean()
    right = (np.arange(labels.shape[0]) > midline)[:, None, None]
    for base, truth_labels in truth_to_classes.items():
        part = np.isin(labels, truth_labels)
        out[part & right] = class_ids[f"{base}_right"]
        out[part & ~right] = class_ids[f"{base}_left"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="e.g. configs/g3_smoke.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    work_dir = REPO_ROOT / cfg["work_dir"]
    setup_nnunet_env(work_dir, cfg["compute"]["augmentation_processes"])

    import json
    import shutil

    import nibabel as nib
    import numpy as np
    import pandas as pd
    import torch
    from nnunetv2.training.dataloading.nnunet_dataset import comp_blosc2_params, nnUNetDatasetBlosc2
    from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
        nnUNetTrainerNoMirroring)
    from totalsegmentator.resampling import change_spacing

    from segreview.config import load_config, pick_device
    from segreview.data import Case, dataset_dir, list_cases, load_canonical, load_label_map
    from segreview.inference import OrganSegmenter
    from segreview.metrics import dice
    from segreview.weights import model_folder

    source = load_config(REPO_ROOT / cfg["source_config"])
    device = pick_device(cfg["compute"]["device"])
    if cfg["compute"]["num_threads"] > 0:
        torch.set_num_threads(cfg["compute"]["num_threads"])
    out_dir = REPO_ROOT / "results" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # The G3 scans must never be among the scans G1/G2 evaluate on (kits100).
    used = {c.case_id for c in list_cases(source)}
    clash = set(cfg["cases"]) & used
    if clash:
        raise SystemExit(f"{sorted(clash)} are part of {source['name']}; G3 must use other scans.")
    root = dataset_dir(source)
    ds = source["dataset"]
    cases = [Case(c, root / ds["image_glob"].replace("*", c), root / ds["label_path"].format(stem=c))
             for c in cfg["cases"]]

    # The pretrained network, as in inference (weights, plans, preprocessing).
    segmenter = OrganSegmenter(source, device, cfg["model_resolution"])
    predictor, spacing = segmenter.predictor, segmenter.spec.spacing_mm
    plans, dataset_json = predictor.plans_manager.plans, predictor.dataset_json
    class_ids = {name: int(v) for name, v in dataset_json["labels"].items()}
    kidney_ids = [class_ids[f"{base}_{side}"] for base in cfg["truth_to_classes"] for side in ("left", "right")]

    # nnU-Net's preprocessed folder for this test: <work_dir>/nnUNet_preprocessed/<dataset name>/.
    pre_base = work_dir / "nnUNet_preprocessed" / plans["dataset_name"]
    pre_dir = pre_base / predictor.configuration_manager.data_identifier
    if pre_base.exists():
        shutil.rmtree(pre_base)
    pre_dir.mkdir(parents=True)
    weights_dir = model_folder(source, segmenter.spec)
    for name in ("dataset.json", "dataset_fingerprint.json"):
        shutil.copy(weights_dir / name, pre_base / name)
    (pre_base / f"{plans['plans_name']}.json").write_text(json.dumps(plans, indent=2))
    checkpoint = weights_dir / "fold_0" / "checkpoint_final.pth"

    # ---- 1-2: training data -------------------------------------------------------------------------------
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    case_rows = []
    for case in cases:
        t0 = time.perf_counter()
        image = load_canonical(case.image_path)
        labels = load_label_map(case).astype(np.uint8)
        image_rs = change_spacing(image, spacing, order=3, dtype=np.int32)
        labels_rs = change_spacing(nib.Nifti1Image(labels, image.affine), spacing, target_shape=image_rs.shape,
                                   order=0, dtype=np.uint8)
        ct = np.asanyarray(image_rs.dataobj).astype(np.float32)
        truth = side_labeled_truth(np.asanyarray(labels_rs.dataobj), ct, class_ids, cfg["truth_to_classes"],
                                   cfg["body_threshold_hu"])
        # nnU-Net's reader gives (z, y, x); inference.py does the same transposition. The labels must be a signed
        # type: nnU-Net marks voxels outside the scan's nonzero region with -1 (in uint8 that would become 255).
        data, seg = ct.transpose(2, 1, 0)[None], truth.astype(np.int16).transpose(2, 1, 0)[None]
        data_pp, seg_pp, props = preprocessor.run_case_npy(data, seg, {"spacing": [spacing] * 3},
                                                           predictor.plans_manager, predictor.configuration_manager,
                                                           dataset_json)
        t1 = time.perf_counter()
        with torch.no_grad():
            pred = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data_pp)).argmax(0).numpy()
        t2 = time.perf_counter()
        gt = seg_pp[0] > 0
        target = np.where(gt, seg_pp[0], np.where(np.isin(pred, kidney_ids), 0, pred))
        target = np.where(seg_pp[0] == -1, -1, target).astype(np.int16)      # keep nnU-Net's "outside" marker
        model_kidney = np.isin(pred, kidney_ids)
        blocks, chunks = comp_blosc2_params(data_pp.shape, tuple(predictor.configuration_manager.patch_size), 4)
        blocks_seg, chunks_seg = comp_blosc2_params(data_pp.shape, tuple(predictor.configuration_manager.patch_size), 2)
        nnUNetDatasetBlosc2.save_case(data_pp.astype(np.float32), target[None], props, str(pre_dir / case.case_id),
                                      chunks=chunks, blocks=blocks, chunks_seg=chunks_seg, blocks_seg=blocks_seg)
        assert target.min() >= -1 and target.max() < len(class_ids), "target labels out of range"
        case_rows.append({"case_id": case.case_id, "shape_zyx": "x".join(map(str, data_pp.shape[1:])),
                          "kidney_dice_model_vs_truth": dice(model_kidney, gt),
                          "voxels_changed_by_correction": int((target != pred).sum()),
                          "kidney_voxels_truth": int(gt.sum()), "kidney_voxels_model": int(model_kidney.sum()),
                          "prepare_s": round(t1 - t0, 1), "predict_s": round(t2 - t1, 1)})
        print(f"{case.case_id}: grid {data_pp.shape[1:]}, kidney Dice model vs truth "
              f"{case_rows[-1]['kidney_dice_model_vs_truth']:.3f}, prepare {t1 - t0:.0f} s, predict {t2 - t1:.0f} s")
    prep_peak = peak_memory_gb(device)

    def make_trainer(batch_size: int):
        """A fresh nnU-Net trainer (random weights) that reads the data above; results go to work_dir."""
        # nnU-Net's own start script adds continue_training to the plans, and the trainer removes it again.
        trainer = nnUNetTrainerNoMirroring({**plans, "continue_training": False}, "3d_fullres", "all", dataset_json,
                                           device)
        # The trainer takes its batch size from the plans (again, when it initializes), so it is changed there.
        trainer.configuration_manager.configuration["batch_size"] = batch_size
        trainer.initial_lr = cfg["training"]["initial_lr"]
        trainer.num_epochs = 1
        trainer.on_train_start()             # builds network, loss, optimizer and the data loaders
        assert trainer.batch_size == batch_size, f"batch size {trainer.batch_size}, wanted {batch_size}"
        return trainer

    def batch_loss(trainer, batch) -> tuple[float, float]:
        """Loss of the network on one batch without training (forward pass only), and the seconds it took."""
        t0 = time.perf_counter()
        with torch.no_grad():
            output = trainer.network(batch["data"].to(device))
            loss = float(trainer.loss(output, [t.to(device) for t in batch["target"]]))
        return loss, time.perf_counter() - t0

    full_weights = torch.load(checkpoint, map_location=device, weights_only=False)["network_weights"]

    # ---- 3: sanity check of the starting point -------------------------------------------------------------
    trainer = make_trainer(cfg["training"]["batch_sizes"][0])
    batch = next(trainer.dataloader_train)
    sanity = [{"weights": "random (no pretraining)", **dict(zip(("loss", "forward_s"), batch_loss(trainer, batch)))}]
    # What nnU-Net's load_pretrained_weights does: copy every weight except the last layer (seg_layers). (Its own
    # function cannot be used here: it loads the checkpoint without map_location, which fails without a GPU.)
    state = trainer.network.state_dict()
    state.update({k: v for k, v in full_weights.items() if ".seg_layers." not in k})
    trainer.network.load_state_dict(state)
    sanity.append({"weights": "nnU-Net default fine-tuning start (last layer re-initialized)",
                   **dict(zip(("loss", "forward_s"), batch_loss(trainer, batch)))})
    trainer.network.load_state_dict(full_weights)
    sanity.append({"weights": "all pretrained weights", **dict(zip(("loss", "forward_s"), batch_loss(trainer, batch)))})
    for s in sanity:
        s["batch_size"] = trainer.batch_size
        print(f"loss on one batch, {s['weights']}: {s['loss']:.3f} (forward pass {s['forward_s']:.1f} s, "
              f"batch {trainer.batch_size})")

    # ---- 4: timing ------------------------------------------------------------------------------------------
    rows = []
    for batch_size in cfg["training"]["batch_sizes"]:
        trainer = make_trainer(batch_size)
        trainer.network.load_state_dict(full_weights)
        trainer.on_train_epoch_start()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for i in range(cfg["training"]["iterations"]):
            t0 = time.perf_counter()
            batch = next(trainer.dataloader_train)
            t1 = time.perf_counter()
            loss = float(trainer.train_step(batch)["loss"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            rows.append({"batch_size": int(batch["data"].shape[0]), "iteration": i, "warmup": i == 0, "data_s": t1 - t0,
                         "step_s": t2 - t1, "loss": loss, "peak_memory_gb": peak_memory_gb(device)})
            print(f"batch {rows[-1]['batch_size']}, step {i}: data {t1 - t0:.1f} s, step {t2 - t1:.1f} s, loss {loss:.3f}, "
                  f"peak {rows[-1]['peak_memory_gb']:.2f} GB")
        for loader in (trainer.dataloader_train, trainer.dataloader_val):
            if hasattr(loader, "_finish"):
                loader._finish()

    machine = {"device": str(device), "machine": platform.machine(), "processor": platform.processor(),
               "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
               "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "",
               "model_resolution": cfg["model_resolution"], "patch_size": str(predictor.configuration_manager.patch_size),
               "peak_memory_preparation_gb": round(prep_peak, 2)}
    pd.DataFrame(case_rows).to_csv(out_dir / "cases.csv", index=False)
    pd.DataFrame(sanity).to_csv(out_dir / "sanity.csv", index=False)
    timing = pd.DataFrame(rows).assign(**machine)
    timing.to_csv(out_dir / "timing.csv", index=False)
    summary = timing[~timing.warmup].groupby("batch_size")[["data_s", "step_s", "peak_memory_gb"]].agg(
        {"data_s": "mean", "step_s": "mean", "peak_memory_gb": "max"})
    summary["step_s_per_sample"] = summary.step_s / summary.index
    print(summary.round(2).to_string())
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
