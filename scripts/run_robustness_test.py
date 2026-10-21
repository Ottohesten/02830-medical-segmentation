"""Plan, run, and verify held-out inference for robustness step 4; never read labels or compute scores.

Run with --plan first and ask Otto before --run if the estimate exceeds 30 minutes.
Both models receive clean and identically perturbed scans. Completed predictions can be resumed
under an identical config and source snapshot. --verify checks saved outputs without inference.
"""

import hashlib
import json
import os
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
from segreview.data import CASE_LIST_FILE, dataset_dir, list_cases
from segreview.io import map_path
from segreview.machine import machine_info
from segreview.pipeline import step_inference, step_second_model, variants
from segreview.robustness import validate_test_run


def sha256(path: Path) -> str:
    """Return a file's SHA-256, reading in chunks so large prediction files do not fill memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    """Write a readable JSON artifact with a final newline."""
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_test_scope(cfg: dict) -> tuple[list, dict]:
    """Verify archived strength/evidence hashes and scientific settings before returning the 80 test cases."""
    settings = cfg['robustness_test']
    lock = json.loads((REPO_ROOT / settings['strength_lock_file']).read_text())
    locked_path = REPO_ROOT / settings['locked_config']
    if sha256(locked_path) != lock['final_config_sha256']:
        raise ValueError('The archived development choice config has changed.')
    for file_key, hash_key in [('source_config', 'source_config_sha256'), ('summary_file', 'summary_sha256')]:
        if sha256(REPO_ROOT / lock[file_key]) != lock[hash_key]:
            raise ValueError('The development selection evidence has changed.')
    locked = load_config(locked_path)
    reference = load_config(REPO_ROOT / settings['reference_config'])
    cases = validate_test_run(cfg, list_cases(cfg), reference, list_cases(reference), locked, lock)
    return cases, locked


def source_hashes(cfg: dict) -> dict:
    """Identify every project module used in inference, the dataset description, and selection lock."""
    paths = [Path(__file__), *(REPO_ROOT / 'src/segreview' / f'{name}.py' for name in
             ['pipeline', 'robustness', 'augment', 'inference', 'weights', 'data', 'io', 'config', 'machine']),
             REPO_ROOT / cfg['dataset']['file'],
             REPO_ROOT / cfg['robustness_test']['reference_config'],
             REPO_ROOT / cfg['robustness_test']['strength_lock_file'],
             REPO_ROOT / cfg['robustness_test']['locked_config'],
             *(REPO_ROOT / cfg['robustness_test'][key] for key in
               ['development_timing_csv', 'benchmark_timing_csv', 'benchmark_model_load_csv'])]
    listing = dataset_dir(cfg) / CASE_LIST_FILE
    if listing.exists():
        paths.append(listing)
    return {str(path.resolve().relative_to(REPO_ROOT)): sha256(path) for path in paths}


def estimate_runtime(cfg: dict, sizes: pd.DataFrame, development: pd.DataFrame,
                     benchmark: pd.DataFrame, model_load_seconds: float) -> dict:
    """Project main-model times from 20 dev cases, adding size-scaled I/O and two-scan second-model times."""
    settings = cfg['robustness_test']
    dev_sizes = sizes[sizes['split'] == 'dev'].set_index('case_id').voxels
    test_voxels = float(sizes.loc[sizes['split'] == 'test', 'voxels'].sum())
    components = []
    for variant in variants(cfg):
        measured = development[(development.variant == variant)
                               & (development.resolution == cfg['model']['resolution'])]
        if len(measured) != len(dev_sizes) or set(measured.case_id) != set(dev_sizes.index):
            raise ValueError('Planning requires one measured main-model time per dev case and variant.')
        main_prediction = float(measured.seconds.sum() / dev_sizes.sum() * test_voxels)
        proxy = 'clean' if variant == 'clean' else settings['benchmark_perturbation']
        for resolution in [cfg['model']['resolution'], cfg['uncertainty']['second_model_resolution']]:
            rows = benchmark[(benchmark.resolution == resolution) & (benchmark.variant == proxy)]
            if rows.empty or rows.case_id.duplicated().any() or not set(rows.case_id).issubset(dev_sizes.index):
                raise ValueError('Benchmark reference must contain distinct development cases for both models.')
            reference_voxels = float(dev_sizes.loc[rows.case_id].sum())
            if resolution == cfg['model']['resolution']:
                seconds = main_prediction + float((rows.total_seconds - rows.prediction_seconds).sum()
                                                   / reference_voxels * test_voxels)
                basis = 'All 20 dev predictions for this variant plus two-scan I/O proxy'
            else:
                seconds = float(rows.total_seconds.sum() / reference_voxels * test_voxels)
                basis = 'Two-scan clean/noise benchmark proxy; no measured second-model lowres times'
            components.append({'variant': variant, 'resolution': resolution,
                               'estimated_seconds': seconds, 'basis': basis})
    return {'estimated_minutes': (sum(c['estimated_seconds'] for c in components) + model_load_seconds) / 60,
            'model_load_seconds': model_load_seconds, 'components': components,
            'limitations': 'Raw-voxel scaling is approximate. Main-model data cover 20 dev scans; second-model '
                           'and I/O proxies cover two dev scans. Noise is a proxy for lowres overhead. '
                           'Excludes final header/hash verification; this is a plan, not a measured result.'}


def make_plan(cfg: dict, cases: list, locked: dict, config_path: Path, out: Path) -> dict:
    """Write test IDs, image-header sizes, runtime projection, exact config, and source hashes without inference."""
    sizes = pd.DataFrame([{'case_id': c.case_id, 'split': split, 'voxels': int(np.prod(nib.load(c.image_path).shape))}
                          for split, group in [('dev', list_cases(locked)), ('test', cases)] for c in group])
    settings = cfg['robustness_test']
    estimate = estimate_runtime(cfg, sizes, pd.read_csv(REPO_ROOT / settings['development_timing_csv']),
                                pd.read_csv(REPO_ROOT / settings['benchmark_timing_csv']),
                                float(pd.read_csv(REPO_ROOT / settings['benchmark_model_load_csv']).seconds.sum()))
    sizes.to_csv(out / 'planning_image_sizes.csv', index=False)
    sizes[sizes['split'] == 'test'].to_csv(out / 'planned_cases.csv', index=False)
    plan = {'scope': 'Test inference only; no ground-truth reads, Dice, uncertainty scores, or evaluation',
            'n_cases': len(cases), 'case_ids': [c.case_id for c in cases], 'variants': variants(cfg),
            'resolutions': [cfg['model']['resolution'], cfg['uncertainty']['second_model_resolution']],
            'n_predictions': len(cases) * len(variants(cfg)) * 2,
            'n_prediction_files': len(cases) * len(variants(cfg)) * 4,
            'max_run_minutes': settings['max_run_minutes'], 'runtime_projection': estimate,
            'config_sha256': sha256(config_path), 'source_sha256': source_hashes(cfg)}
    write_json(out / 'run_plan.json', plan)
    return plan


def verify_predictions(cfg: dict, cases: list, out: Path) -> dict:
    """Require complete per-model timing rows and saved map grids; save file sizes and hashes without reading labels."""
    expected = {(c.case_id, v) for c in cases for v in variants(cfg)}
    for filename, resolution in [('inference_times.csv', cfg['model']['resolution']),
                                 ('second_model_times.csv', cfg['uncertainty']['second_model_resolution'])]:
        rows = pd.read_csv(out / filename)
        keys = list(zip(rows.case_id, rows.variant))
        if (len(keys) != len(expected) or set(keys) != expected or not rows.resolution.eq(resolution).all()
                or not np.isfinite(rows.seconds).all() or (rows.seconds < 0).any()):
            raise ValueError(f'Incomplete or invalid per-scan timings: {filename}')
    manifest = []
    for case in cases:
        original = nib.as_closest_canonical(nib.load(case.image_path))
        for variant in variants(cfg):
            for kind in ['mask', 'prob', 'm2_mask', 'm2_prob']:
                path = map_path(cfg, variant, case.case_id, kind)
                image = nib.load(path)
                if image.shape != original.shape or not np.allclose(image.affine, original.affine):
                    raise ValueError(f'Prediction grid differs from the input: {path}')
                manifest.append({'case_id': case.case_id, 'variant': variant, 'kind': kind,
                                 'path': str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
                                 'bytes': path.stat().st_size,
                                 'sha256': sha256(path)})
    pd.DataFrame(manifest).to_csv(out / 'prediction_manifest.csv', index=False)
    report = {'status': 'passed', 'n_cases': len(cases), 'n_predictions': len(expected) * 2,
              'n_files': len(manifest), 'prediction_bytes': sum(row['bytes'] for row in manifest),
              'checks': 'Exact case/variant/model timing coverage; matching input/output shapes and affines; file SHA-256',
              'ground_truth_read': False, 'dice_or_uncertainty_computed': False}
    write_json(out / 'verification.json', report)
    return report


def run_test(cfg: dict, config_path: Path, cases: list, out: Path) -> None:
    """Save machine provenance, run both models on test only, then verify maps and record time/memory per attempt."""
    device = setup_compute(cfg)
    if device.type == 'cuda':
        (torch.ones(4, device=device) + 1).cpu()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    info = machine_info()
    cpu_path = Path('/proc/cpuinfo')
    if cpu_path.exists():
        info['cpu_model'] = next((line.split(':', 1)[1].strip() for line in cpu_path.read_text().splitlines()
                                  if line.startswith('model name')), info['cpu'])
    if device.type == 'cuda':
        info['nvidia_driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    info.update({'status': 'running', 'config': str(config_path), 'config_sha256': sha256(config_path),
                 'source_sha256': source_hashes(cfg), 'torch_threads': torch.get_num_threads(),
                 'nnUNet_compile': os.environ['nnUNet_compile'], 'case_ids': [c.case_id for c in cases],
                 'scope': 'Both models, test inference only; no ground truth, Dice, or uncertainty scores'})
    history_path = out / 'run_attempts.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    write_json(out / 'machine.json', info)
    start = time.perf_counter()
    try:
        for name, step in [('main_model', step_inference), ('second_model', step_second_model)]:
            stage_start = time.perf_counter()
            step(cfg, device, splits=['test'])
            info[f'{name}_wall_seconds'] = time.perf_counter() - stage_start
        info['verification'] = verify_predictions(cfg, cases, out)
        info['status'] = 'complete'
    except BaseException as exc:
        info.update({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        info.update({'elapsed_seconds': time.perf_counter() - start,
                     'process_peak_rss_mib': peak / 2**20 if sys.platform == 'darwin' else peak / 1024,
                     'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None})
        write_json(out / 'machine.json', info)
        history.append(info)
        write_json(history_path, history)


def stop_at_limit(signum, frame):
    """Stop at the configured reserve limit, retaining completed predictions for a later approved resume."""
    raise TimeoutError('Test inference reached robustness_test.max_run_minutes; ask Otto before extending it.')


def main():
    """Validate the archived development choice, then plan, run, or verify with immutable config/source snapshots."""
    parser = config_arg_parser(__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan', action='store_true', help='plan using image headers and saved dev timings only')
    action.add_argument('--run', action='store_true', help='run inference after approval of the long run')
    action.add_argument('--verify', action='store_true', help='verify saved test outputs without inference')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cases, locked = load_test_scope(cfg)
    out = results_dir(cfg)
    snapshot = out / 'run_config.yaml'
    if snapshot.exists() and snapshot.read_bytes() != args.config.read_bytes():
        raise ValueError('Config differs from the saved test plan; preserve provenance with a separate output name.')
    if args.plan:
        if (out / 'machine.json').exists():
            raise ValueError('A run has started; do not replace its original plan.')
        snapshot.write_bytes(args.config.read_bytes())
        print(json.dumps(make_plan(cfg, cases, locked, args.config, out), indent=2))
    else:
        plan = json.loads((out / 'run_plan.json').read_text())
        if (plan['config_sha256'] != sha256(args.config) or plan['source_sha256'] != source_hashes(cfg)
                or plan['case_ids'] != [c.case_id for c in cases]):
            raise ValueError('Config or source changed since planning; review and regenerate the plan before running.')
        if args.verify:
            print(json.dumps(verify_predictions(cfg, cases, out), indent=2))
            return
        os.environ['nnUNet_compile'] = 'false'
        signal.signal(signal.SIGALRM, stop_at_limit)
        signal.alarm(int(cfg['robustness_test']['max_run_minutes'] * 60))
        try:
            run_test(cfg, args.config, cases, out)
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    main()
