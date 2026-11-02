"""G3: run one stage of the learning curve, step by step (docs/g3_plan.md, "Design: learning curve"; task document
docs/tasks/g3_finetuning.md). Every training run and every evaluation is its own process (scripts/g3_train.py,
scripts/g3_evaluate.py) and finished work is skipped, so a stage can be stopped and started again.

Stages:
  prepare  scripts/g3_prepare.py: the pool's model output and scores, the orders, the training data.
  test     the pipeline test with the 6 mm model (a few minutes): one short run, then 2 development scans. Not a result.
  grid     the settings grid (configs/g3_kits.yaml, settings_grid): its runs at k = 50 in queue and random order; the
           original model and these models on the 20 development scans; then the choice by the rule fixed in the
           config -> results/g3_kits/settings_choice.json and settings_grid.csv. No trained model predicts a test scan.
  curves   needs settings_choice.json: the original model on the development and test scans, then every curve run
           (k ascending, all orders at each k, then the control), each followed by its predictions of the
           development and test scans. Interrupted curves are therefore filled from left to right.
The processes' output goes to work_dir/logs/<stage>.log.

Usage: .venv\\Scripts\\python.exe scripts\\g3_run.py --config configs\\g3_kits.yaml --stage grid
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "test", "grid", "curves")


def run_step(cmd: list[str], log_path: Path) -> None:
    """Run one step as its own process, its output appended to the stage's log file; stop the stage if it fails."""
    started = datetime.now()
    print(f"{started:%Y-%m-%d %H:%M:%S}  {' '.join(cmd[1:])}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {started:%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
        log.flush()
        code = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    minutes = (datetime.now() - started).total_seconds() / 60
    print(f"    -> exit code {code} after {minutes:.1f} min", flush=True)
    if code != 0:
        raise SystemExit(f"Step failed: {' '.join(cmd)} (see {log_path})")


def ram_in_use() -> str:
    """The PC's RAM in use right now (other programs count; the PC's RAM is G3's limit)."""
    try:
        import psutil
    except ImportError:
        return "unknown"
    memory = psutil.virtual_memory()
    return f"{(memory.total - memory.available) / 1e9:.1f} of {memory.total / 1e9:.1f} GB"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="configs/g3_kits.yaml")
    parser.add_argument("--stage", required=True, choices=STAGES)
    args = parser.parse_args()
    g3 = yaml.safe_load(args.config.read_text())
    config = str(args.config)
    py = sys.executable
    results = REPO_ROOT / "results" / g3["name"]
    log_dir = REPO_ROOT / g3["work_dir"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{args.stage}.log"
    print(f"[g3_run] stage {args.stage}; RAM in use {ram_in_use()}; log {log}", flush=True)

    if args.stage == "prepare":
        run_step([py, "scripts/g3_prepare.py", "--config", config], log)
        return

    import pandas as pd

    from segreview.finetune import choose_setting, experiment_runs, read_orders

    orders = read_orders(results / "orders.csv")
    choice_path = results / "settings_choice.json"
    choice = json.loads(choice_path.read_text()) if choice_path.exists() else None
    runs, points = experiment_runs(g3, orders, choice)

    def train(run_id: str) -> None:
        run_step([py, "scripts/g3_train.py", "--config", config, "--run", run_id], log)

    def evaluate(models: list[str], splits: list[str], *extra: str) -> None:
        run_step([py, "scripts/g3_evaluate.py", "--config", config, "--models", *models, "--splits", *splits,
                  *extra], log)

    if args.stage == "test":
        test = g3["pipeline_test"]
        for run in (r for r in runs if r["stage"] == "test"):
            train(run["run_id"])
            evaluate([run["run_id"]], ["dev"], "--resolution", test["model_resolution"],
                     "--max-scans", str(test["dev_scans"]))
        return

    if args.stage == "grid":
        grid_runs = [r for r in runs if r["stage"] == "grid"]
        for run in grid_runs:
            print(f"[g3_run] RAM in use before training: {ram_in_use()}", flush=True)
            train(run["run_id"])
        evaluate(["original"] + [r["run_id"] for r in grid_runs], ["dev"])
        if choice is not None:
            print(f"[g3_run] the setting was already chosen: {choice}")
            return
        definition = g3["evaluation"]["primary_definition"]
        dice = pd.concat([pd.read_csv(results / "dice" / f"{m}.csv") for m in
                          ["original"] + [r["run_id"] for r in grid_runs]])
        dice = dice.rename(columns={f"dice_{definition}": "dice"})
        n_dev = int((dice.model == "original").sum())
        chosen, table = choose_setting(dice, grid_runs, n_dev, g3["settings_grid"]["tie_dice"])
        table.round(5).to_csv(results / "settings_grid.csv", index=False)
        original = float(dice.loc[(dice.model == "original") & (dice.split == "dev"), "dice"].mean())
        choice_path.write_text(json.dumps({
            "date": datetime.now().isoformat(timespec="seconds"), **chosen,
            "rule": "highest mean development Dice over the setting's runs (queue and random order at k = "
                    f"{g3['settings_grid']['k']}); within {g3['settings_grid']['tie_dice']}: fewer steps",
            "organ_definition": definition, "n_development_scans": n_dev,
            "original_model_dev_mean_dice": original}, indent=2) + "\n")
        print(f"[g3_run] chosen setting: {chosen} (original model: {original:.4f}) -> {choice_path}")
        return

    # curves
    if choice is None:
        raise SystemExit("No settings_choice.json: run the grid stage first.")
    evaluate(["original"], ["dev", "test"])
    by_id = {r["run_id"]: r for r in runs}
    rank = {name: i for i, name in enumerate(list(orders) + ["nocorr"])}     # at each k: queue, random..., control
    ordered = points.assign(rank=points.order.map(rank)).sort_values(["k", "rank"])
    curve_ids = list(dict.fromkeys(ordered.run_id))                          # a shared run (k = 100) once
    for run_id in curve_ids:
        print(f"[g3_run] {run_id} ({by_id[run_id]['k']} scans); RAM in use before training: {ram_in_use()}",
              flush=True)
        train(run_id)
        evaluate([run_id], ["dev", "test"])
    print("[g3_run] curves stage finished; next: scripts/g3_curves.py", flush=True)


if __name__ == "__main__":
    main()
