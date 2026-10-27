"""Plan and run development-only robustness strength selection (task step 3).

Runs only the 3 mm model, evaluates development Dice, and suggests strengths by the config's rule.
It never computes uncertainty scores or reads test scans. Ask Otto before starting runs over 30 minutes.

Run: PYTHONPATH=src .venv-gpu/bin/python scripts/select_robustness_strengths.py --config configs/kits_robustness_dev.yaml --plan
Replace --plan with --run only after the long run is approved; --summarize uses saved predictions only.
"""

import hashlib
import json
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from segreview.config import REPO_ROOT, config_arg_parser, load_config, results_dir, setup_compute
from segreview.data import list_cases
from segreview.machine import machine_info
from segreview.pipeline import step_dice, step_inference
from segreview.robustness import summarize_degradation, suggest_strengths, validate_development_run


def write_plan(cfg: dict, cases: list, out: Path) -> dict:
    """Estimate candidate-run time from measured noisy 3 mm times and image sizes, without opening labels."""
    reference = pd.read_csv(REPO_ROOT / cfg['robustness_selection']['timing_csv'])
    reference = reference[(reference.resolution == cfg['model']['resolution'])
                          & (reference.variant == cfg['robustness_selection']['timing_variant'])]
    sizes = [{'case_id': c.case_id, 'voxels': int(np.prod(nib.load(c.image_path).shape))} for c in cases]
    frame = pd.DataFrame(sizes)
    sample = frame[frame.case_id.isin(reference.case_id)]
    if len(sample) != len(reference) or sample.empty:
        raise ValueError('Timing reference cases must be present in the development selection.')
    count = len(cfg['perturbations']) + 1
    plan = {'scope': '3 mm model only; dev Dice only; no uncertainty scores or test inference',
            'n_cases': len(cases), 'n_variants_including_clean': count, 'n_predictions': len(cases) * count,
            'reference_timing_csv': cfg['robustness_selection']['timing_csv'],
            'unscaled_minutes': len(cases) * count * reference.total_seconds.mean() / 60,
            'mean_voxel_ratio_dev_to_reference': frame.voxels.mean() / sample.voxels.mean(),
            'voxel_scaled_minutes': count * frame.voxels.sum() * reference.total_seconds.sum() / sample.voxels.sum() / 60,
            'estimate_limitations': 'Two measured scans, one noise type; raw-voxel scaling is a rough planning proxy; excludes Dice analysis',
            'max_run_minutes': cfg['robustness_selection']['max_run_minutes'],
            'candidate_specs': cfg['perturbations'], 'selection_rule': cfg['robustness_selection']}
    frame.to_csv(out / 'planned_cases.csv', index=False)
    (out / 'run_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def write_dice_summary(cfg: dict, out: Path) -> dict:
    """Evaluate saved dev masks with the existing Dice function, then write paired drops and rule suggestions."""
    rows = step_dice(cfg)
    paired, summary = summarize_degradation(rows, cfg)
    paired.to_csv(out / 'dev_dice_drops.csv', index=False)
    summary.to_csv(out / 'dev_degradation_summary.csv', index=False)
    choice = suggest_strengths(summary, cfg)
    (out / 'strength_suggestion.json').write_text(json.dumps(choice, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(choice, indent=2), flush=True)
    return choice


def run_candidates(cfg: dict, config_path: Path, out: Path) -> None:
    """Save exact provenance, run the development model predictions, and summarize Dice without scoring."""
    device = setup_compute(cfg)
    if device.type == 'cuda':
        (torch.ones(4, device=device) + 1).cpu()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    info = machine_info()
    files = [Path(__file__), REPO_ROOT / 'src/segreview/robustness.py', REPO_ROOT / 'src/segreview/pipeline.py',
             REPO_ROOT / 'src/segreview/augment.py', REPO_ROOT / 'src/segreview/inference.py']
    info.update({'status': 'running', 'config': str(config_path), 'device': str(device),
                 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 'source_sha256': {str(p.resolve().relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                 'torch_threads': torch.get_num_threads(), 'nnUNet_compile': 'false',
                 'scope': '3 mm dev inference and Dice only; no score computation or test inference'})
    cpu_path = Path('/proc/cpuinfo')
    if cpu_path.exists():
        info['cpu_model'] = next((line.split(':', 1)[1].strip() for line in cpu_path.read_text().splitlines()
                                  if line.startswith('model name')), info['cpu'])
    if device.type == 'cuda':
        info['nvidia_driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    manifest = out / 'machine.json'
    history_path = out / 'run_attempts.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    try:
        step_inference(cfg, device)
        write_dice_summary(cfg, out)
        info['status'] = 'complete'
    except BaseException as exc:
        info.update({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        info.update({'elapsed_seconds': time.perf_counter() - start,
                     'process_peak_rss_mib': peak / 2**20 if sys.platform == 'darwin' else peak / 1024,
                     'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None})
        manifest.write_text(json.dumps(info, indent=2) + '\n')
        history.append(info)
        history_path.write_text(json.dumps(history, indent=2) + '\n')


def stop_at_limit(signum, frame):
    """Stop at the approved config limit, keeping completed predictions and per-scan timings."""
    raise TimeoutError('Candidate run reached robustness_selection.max_run_minutes; ask Otto before extending it.')


def main():
    """Validate dev-only scope and config identity before planning, inferring, or summarizing."""
    import os

    parser = config_arg_parser(__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan', action='store_true', help='save run size, estimate, and rule without inference')
    action.add_argument('--run', action='store_true', help='run dev inference and Dice; requires approval for long runs')
    action.add_argument('--summarize', action='store_true', help='evaluate saved development masks; no inference')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cases = list_cases(cfg)
    validate_development_run(cfg, cases)
    out = results_dir(cfg)
    snapshot = out / 'run_config.yaml'
    if snapshot.exists() and snapshot.read_bytes() != args.config.read_bytes():
        raise ValueError('Config differs from the saved candidate plan; use a separate config/output name for changes.')
    snapshot.write_bytes(args.config.read_bytes())
    if args.plan:
        print(json.dumps(write_plan(cfg, cases, out), indent=2))
    elif args.summarize:
        write_dice_summary(cfg, out)
    else:
        os.environ['nnUNet_compile'] = 'false'
        signal.signal(signal.SIGALRM, stop_at_limit)
        signal.alarm(int(cfg['robustness_selection']['max_run_minutes'] * 60))
        try:
            run_candidates(cfg, args.config, out)
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    main()
