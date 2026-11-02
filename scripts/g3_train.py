"""G3 step 2: train ONE run of the learning curve (docs/g3_plan.md, "Design: learning curve"). One run per process,
so every run starts clean (GPU memory, nnU-Net's data workers); scripts/g3_run.py starts the runs one after another.

A run fine-tunes the pretrained model from ALL its weights (also the last layer) with nnU-Net's own trainer on the
first k scans of one order of the pool (corrected, or uncorrected for the control), for the setting's number of
steps and learning rate. Runs are defined by finetune.plan_runs from configs/g3_kits.yaml, results/g3_kits/orders.csv
and the setting (the grid's, or the chosen one in results/g3_kits/settings_choice.json).

A finished run (its checkpoint_final.pth and results/g3_kits/training/<run_id>.json exist) is skipped. A stopped run
continues where it stopped: nnU-Net saves the whole training state (weights, optimizer, epoch, log) after every epoch
as checkpoint_latest.pth, and the run continues from it, so a stop loses at most the epoch in progress.

Output: the trained model in work_dir/runs/<run_id>/ (nnU-Net's folder: checkpoint_final.pth, checkpoint_best.pth,
training log, progress.png; not in git) and, in git, results/g3_kits/training/<run_id>.csv (per epoch: losses,
learning rate, pseudo Dice, seconds) and <run_id>.json (the run's scans, settings, time, peak memory, machine, commit;
resumed_after_epochs: the epochs finished before a stop, or null; after a stop, time and peak memory cover only the
part after it).

Usage: .venv\\Scripts\\python.exe scripts\\g3_train.py --config configs\\g3_kits.yaml --run <run_id>
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_folder(g3: dict, run_id: str) -> Path:
    """nnU-Net's results folder of one run (work_dir/runs/<run_id>)."""
    return REPO_ROOT / g3["work_dir"] / "runs" / run_id


def setup_nnunet_env(g3: dict, run_id: str) -> None:
    """nnU-Net reads its folders from environment variables when it is first imported, so they are set first:
    the shared preprocessed data, and this run's own results folder."""
    work = REPO_ROOT / g3["work_dir"]
    os.environ["nnUNet_raw"] = str(work / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(work / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(run_folder(g3, run_id))
    os.environ["nnUNet_n_proc_DA"] = str(g3["compute"]["augmentation_processes"])
    # torch.compile: not available with torch 2.2 on Python 3.12 (and nnU-Net turns it off on Windows anyway).
    os.environ["nnUNet_compile"] = "false"


def load_choice(g3: dict) -> dict | None:
    """The chosen setting (results/g3_kits/settings_choice.json, written by scripts/g3_run.py), or None before."""
    path = REPO_ROOT / "results" / g3["name"] / "settings_choice.json"
    return json.loads(path.read_text()) if path.exists() else None


def find_run(g3: dict, run_id: str) -> dict:
    """The definition of a run, from the config, the saved orders and the chosen setting (finetune.experiment_runs)."""
    from segreview.finetune import experiment_runs, read_orders
    orders = read_orders(REPO_ROOT / "results" / g3["name"] / "orders.csv")
    for run in experiment_runs(g3, orders, load_choice(g3))[0]:
        if run["run_id"] == run_id:
            return run
    raise SystemExit(f"Unknown run '{run_id}'. Runs come from configs/g3_kits.yaml and results/g3_kits/orders.csv.")


def readable(checkpoint: Path) -> bool:
    """Whether a saved training state can be loaded (it can be broken after a power cut); if not, the run starts
    again from the pretrained weights."""
    import torch
    try:
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        return True
    except Exception as error:
        print(f"[train] {checkpoint} cannot be read ({error}); the run starts again")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="configs/g3_kits.yaml")
    parser.add_argument("--run", required=True, help="the run's id, e.g. queue_k050_S2500_L0.001")
    args = parser.parse_args()
    g3 = yaml.safe_load(args.config.read_text())
    run = find_run(g3, args.run)
    out = run_folder(g3, run["run_id"])
    setup_nnunet_env(g3, run["run_id"])

    import numpy as np
    import torch

    from segreview.config import load_config, pick_device
    from segreview.finetune import g3_trainer_class, save_atomically, training_keys, training_log
    from segreview.machine import machine_info, write_machine_info
    from segreview.memory import PeakMemory
    from segreview.weights import model_folder, model_spec

    results = REPO_ROOT / "results" / g3["name"] / "training"
    run_json = results / f"{run['run_id']}.json"
    final = next(out.rglob("checkpoint_final.pth"), None) if out.exists() else None
    if final is not None and run_json.exists():
        print(f"[train] {run['run_id']}: already trained ({final}), skipped")
        return
    if final is not None:
        print(f"[train] {run['run_id']}: stopped after its last epoch, before its log was written: trained again")
    machine = machine_info()                  # before this run writes anything (git status of the code)
    source = load_config(REPO_ROOT / g3["eval_config"])
    resolution = run.get("model_resolution", g3["model_resolution"])
    spec = model_spec(source, resolution)
    weights = model_folder(source, spec)
    plans = json.loads((weights / "plans.json").read_text())
    dataset_json = json.loads((weights / "dataset.json").read_text())

    device = pick_device(g3["compute"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("compute.device is 'cuda', but torch does not see a GPU.")
    torch.set_num_threads(g3["compute"]["num_threads"])
    torch.backends.cudnn.benchmark = True     # as nnU-Net's own training start on a GPU
    random.seed(run["seed"])
    np.random.seed(run["seed"])
    torch.manual_seed(run["seed"])            # nnU-Net's data workers seed themselves; GPU training is not exactly
                                              # repeatable (CONTRIBUTING.md), the seed fixes what can be fixed

    trainer = g3_trainer_class()({**plans, "continue_training": False}, "3d_fullres", "all", dataset_json, device)
    trainer.g3_cases = training_keys(run)
    # The trainer takes its batch size from the plans when it initializes, so it is changed there (smoke test).
    trainer.configuration_manager.configuration["batch_size"] = run.get("batch_size", g3["training"]["batch_size"])
    trainer.initial_lr = run["lr"]
    trainer.num_iterations_per_epoch = g3["training"]["steps_per_epoch"]
    trainer.num_val_iterations_per_epoch = g3["training"]["validation_steps_per_epoch"]
    trainer.num_epochs = run["steps"] // g3["training"]["steps_per_epoch"]
    trainer.initialize()                      # builds the network and the optimizer; on_train_start does not rebuild
    trainer.save_every = 1                    # the training state after every epoch, so a stopped run can continue
    latest = Path(trainer.output_folder) / "checkpoint_latest.pth"
    resumed_after = None
    if latest.exists() and readable(latest):
        trainer.load_checkpoint(str(latest))  # weights, optimizer, epoch and log; nnU-Net only loads a str path
        resumed_after = trainer.current_epoch
    else:
        pretrained = torch.load(weights / "fold_0" / "checkpoint_final.pth", map_location=device, weights_only=False)
        trainer.network.load_state_dict(pretrained["network_weights"])   # ALL weights, also the last layer
        del pretrained
    print(f"[train] {run['run_id']}: {len(trainer.g3_cases)} scans, {run['steps']} steps "
          f"({trainer.num_epochs} epochs), learning rate {run['lr']}, batch {trainer.batch_size}, device {device}; "
          + ("starts from the pretrained weights" if resumed_after is None else
             f"continues after {resumed_after} of {trainer.num_epochs} epochs"))
    started = time.perf_counter()
    with PeakMemory(device, interval=2.0) as memory:
        trainer.run_training()
    seconds = time.perf_counter() - started

    results.mkdir(parents=True, exist_ok=True)
    log = training_log(trainer)
    save_atomically(lambda p: log.round(6).to_csv(p, index=False), results / f"{run['run_id']}.csv")
    # The JSON last: it marks the run as finished.
    save_atomically(lambda p: write_machine_info(
        p, info=machine, config=str(args.config), run={k: v for k, v in run.items() if k != "cases"},
        cases=run["cases"], training_keys=trainer.g3_cases, model_resolution=resolution,
        batch_size=trainer.batch_size, epochs=trainer.num_epochs, resumed_after_epochs=resumed_after,
        seconds=round(seconds), final_train_loss=float(log.train_loss.iloc[-1]),
        final_validation_loss=float(log.validation_loss.iloc[-1]),
        model_folder=str(Path(trainer.output_folder).relative_to(REPO_ROOT)), peak_memory=memory.as_dict(),
        augmentation_processes=int(os.environ["nnUNet_n_proc_DA"])), run_json)
    print(f"[train] {run['run_id']}: done in {seconds / 60:.1f} min -> {trainer.output_folder}")


if __name__ == "__main__":
    main()
