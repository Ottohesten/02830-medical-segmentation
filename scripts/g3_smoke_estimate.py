"""Planning estimates for G3 from the GPU smoke test (phase 1, step 5 of docs/tasks/g3_finetuning.md): the time of one
fine-tuning run and of the whole plan, and the disk space. These are ESTIMATES for planning, computed from measured
numbers; they are not results. The formulas are in src/segreview/g3_estimates.py.

What it reads or measures (settings: section `estimate` of configs/g3_smoke_gpu.yaml):
1. Training: summary.csv of each smoke test (results/g3_smoke_gpu/, results/g3_smoke_gpu_3mm/), i.e. the wall time
   per training step and per validation step with the configured number of augmentation workers.
2. Prediction: the time to predict one scan with each model on this GPU, measured here with G1's inference code
   (src/segreview/inference.py: resampling, network, reduction to the kidney maps, resampling back), on the 2
   smoke-test scans (not in kits100). Nothing is saved. The first prediction of each model is a warm-up (not counted).
3. Disk: the KiTS files of kits100 and of the G3 pool (the next scans of the same seeded draw), asked from the
   dataset's server without downloading them; the preprocessed smoke-test scans; the pretrained checkpoints.

Output: results/<config name>/: estimates.csv (time), disk.csv (disk space), inference_times_gpu.csv (2.) and
estimate_machine.json; also printed.
Usage: .venv\\Scripts\\python.exe scripts\\g3_smoke_estimate.py --config configs\\g3_smoke_gpu.yaml
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from segreview.config import REPO_ROOT, load_config, setup_compute
from segreview.data import Case, all_case_ids, dataset_dir, load_canonical, random_draw, selected_case_ids
from segreview.g3_estimates import evaluation_hours, folder_bytes, remote_sizes, run_hours
from segreview.inference import OrganSegmenter
from segreview.machine import machine_info, write_machine_info
from segreview.weights import model_folder, model_spec


def wall_times(summary: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Per batch size: wall seconds per training step (and its data wait) and per validation step, at `workers`."""
    rows = summary[summary.augmentation_processes == workers]
    wall = rows.pivot(index="batch_size", columns="phase", values="wall_s")
    data = rows[rows.phase == "train"].set_index("batch_size").data_s
    return pd.DataFrame({"train_wall_s": wall["train"], "train_data_s": data,
                         "validation_wall_s": wall["validation"]}).reset_index()


def time_predictions(source: dict, device, resolution: str, cases: list[Case]) -> list[dict]:
    """Seconds to predict each scan with one model (G1's inference code). The first prediction is a warm-up."""
    segmenter = OrganSegmenter(source, device, resolution)
    images = [load_canonical(c.image_path) for c in cases]
    segmenter.predict(images[0])
    rows = []
    for case, image in zip(cases, images):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        segmenter.predict(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rows.append({"model_resolution": resolution, "case_id": case.case_id,
                     "shape": "x".join(map(str, image.shape)), "seconds": round(time.perf_counter() - t0, 1)})
        print(f"predict {case.case_id} ({resolution}): {rows[-1]['seconds']} s")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="configs/g3_smoke_gpu.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    est = cfg["estimate"]
    machine = machine_info()                   # before any output is written, so git's status is that of the code
    started = time.perf_counter()
    source = load_config(REPO_ROOT / cfg["source_config"])
    device = setup_compute(source)            # as G1's inference: the source config's device and threads
    out_dir = REPO_ROOT / "results" / cfg["name"]
    smokes = [yaml.safe_load((REPO_ROOT / p).read_text()) for p in est["smoke_configs"]]
    label = {s["model_resolution"]: f"{model_spec(source, s['model_resolution']).spacing_mm:g} mm" for s in smokes}

    # ---- 2. prediction time per scan on this machine ------------------------------------------------------------
    root, ds = dataset_dir(source), source["dataset"]
    cases = [Case(c, root / ds["image_glob"].replace("*", c), root / ds["label_path"].format(stem=c))
             for c in cfg["cases"]]
    predictions = pd.DataFrame([row for s in smokes for row in time_predictions(source, device,
                                                                                s["model_resolution"], cases)])
    predict_s = predictions.groupby("model_resolution").seconds.mean()

    # ---- 1. one fine-tuning run -----------------------------------------------------------------------------------
    rows = []
    for smoke in smokes:
        res = smoke["model_resolution"]
        walls = wall_times(pd.read_csv(REPO_ROOT / "results" / smoke["name"] / "summary.csv"), est["workers"])
        for w in walls.itertuples():
            for steps in est["steps"]:
                rows.append({"what": "one fine-tuning run", "model": label[res], "batch_size": w.batch_size,
                             "augmentation_workers": est["workers"], "steps": steps,
                             "epochs": steps / est["steps_per_epoch"], "train_wall_s": w.train_wall_s,
                             "train_data_wait_s": w.train_data_s, "validation_wall_s": w.validation_wall_s,
                             "hours": run_hours(steps, w.train_wall_s, w.validation_wall_s, est["steps_per_epoch"],
                                                est["validation_steps_per_epoch"])})
    for res, seconds in predict_s.items():
        rows.append({"what": "predict one scan", "model": label[res], "hours": seconds / 3600,
                     "seconds_per_scan": seconds})

    # ---- the whole plan (3 mm models) ----------------------------------------------------------------------------
    plan = est["plan"]
    one = next(r for r in rows if r["what"] == "one fine-tuning run" and r["model"] == label[plan["model_resolution"]]
               and r["batch_size"] == plan["batch_size"] and r["steps"] == plan["steps"])
    parts = {
        f"plan: train {plan['models']} models x {plan['steps']} steps": plan["models"] * one["hours"],
        f"plan: predict kits100 ({plan['evaluation_scans']} scans) with {plan['models']} models + the original":
            evaluation_hours(plan["models"] + 1, plan["evaluation_scans"], predict_s[plan["model_resolution"]]),
        f"plan: queue = predict the pool ({plan['pool_scans']} scans) with the 3 mm and 6 mm models":
            sum(evaluation_hours(1, plan["pool_scans"], predict_s[s["model_resolution"]]) for s in smokes),
    }
    parts["plan: total"] = sum(parts.values())
    rows += [{"what": what, "model": label[plan["model_resolution"]], "hours": hours} for what, hours in parts.items()]
    estimates = pd.DataFrame(rows)

    # ---- 3. disk space ---------------------------------------------------------------------------------------------
    draw = random_draw(source, all_case_ids(source))
    n = len(selected_case_ids(source))
    if sorted(draw[:n]) != selected_case_ids(source):
        raise SystemExit("The seeded draw does not start with the config's selection.")
    if set(draw[n:n + len(cfg["cases"])]) != set(cfg["cases"]):
        raise SystemExit(f"The smoke-test scans are not positions {n + 1}-{n + len(cfg['cases'])} of the draw.")
    pool = draw[n:n + plan["pool_scans"]]
    dl = ds["download"]

    def kits_gb(ids):
        urls = [f"{dl['base_url']}/{t.format(case_id=c)}" for c in ids for t in (dl["image_file"], dl["label_file"])]
        return sum(remote_sizes(urls)) / 1e9

    disk = [{"what": f"KiTS files, {source['name']} ({n} scans)", "count": n, "gb_total": kits_gb(draw[:n])},
            {"what": f"KiTS files, G3 pool (draw positions {n + 1}-{n + len(pool)})", "count": len(pool),
             "gb_total": kits_gb(pool)}]
    for smoke in smokes:
        res = smoke["model_resolution"]
        per_scan = folder_bytes(REPO_ROOT / smoke["work_dir"] / "nnUNet_preprocessed") / len(smoke["cases"]) / 1e9
        disk.append({"what": f"preprocessed training scans, {label[res]} (measured per scan x pool)",
                     "count": len(pool), "gb_each": per_scan, "gb_total": per_scan * len(pool)})
        # A fine-tuned model has the same size as the pretrained checkpoint (same network, same weights).
        per_model = est["checkpoints_per_model"] * (model_folder(source, model_spec(source, res)) / "fold_0"
                                                     / "checkpoint_final.pth").stat().st_size / 1e9
        models = plan["models"] if res == plan["model_resolution"] else 1
        disk.append({"what": f"checkpoints of fine-tuned {label[res]} models ({est['checkpoints_per_model']} each)"
                             + (" in the plan" if models > 1 else ", per model"),
                     "count": models, "gb_each": per_model, "gb_total": models * per_model})
    disk = pd.DataFrame(disk)

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "inference_times_gpu.csv", index=False)
    estimates.round(4).to_csv(out_dir / "estimates.csv", index=False)
    disk.round(3).to_csv(out_dir / "disk.csv", index=False)
    write_machine_info(out_dir / "estimate_machine.json", info=machine, config=str(args.config),
                       run_seconds=round(time.perf_counter() - started))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(estimates.round(3).to_string(index=False))
        print(disk.round(2).to_string(index=False))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
