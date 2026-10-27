"""Time both G1 models on two development scans, clean and with one perturbation (robustness step 2).

No ground truth or uncertainty scores are read. Predictions go to the config's own folders.
Timings include a separate prediction interval and an interval with loading, perturbation, and saving.
The first prediction per model includes CUDA warm-up; no warm-up runs are discarded.

Run: PYTHONPATH=src .venv-gpu/bin/python scripts/benchmark_robustness.py --config configs/kits_robustness_timing.yaml
"""

import gc
import hashlib
import json
import resource
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from segreview.augment import perturb
from segreview.config import REPO_ROOT, config_arg_parser, load_config, results_dir, setup_compute
from segreview.data import case_split, list_cases, load_canonical
from segreview.io import save_map
from segreview.machine import machine_info


def select_timing_cases(cfg: dict, cases: list) -> list:
    """Return (original index, case) pairs; reject test IDs, duplicates, or unsafe output paths."""
    ids = cfg['timing']['case_ids']
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError('Timing requires exactly two distinct development cases.')
    if len(cfg.get('perturbations') or []) != 1:
        raise ValueError('Timing requires exactly one perturbation.')
    if cfg['name'] == 'kits100':
        raise ValueError('Timing must use its own config name, not kits100.')
    for path_key, locked in [('results_dir', 'results/kits100'),
                             ('predictions_dir', 'data/predictions/kits100')]:
        target = (cfg['paths'][path_key] / cfg['name']).resolve()
        if target.is_relative_to((REPO_ROOT / locked).resolve()):
            raise ValueError('Timing must not write to locked kits100 folders.')
    indexed = {case.case_id: (i, case) for i, case in enumerate(cases)}
    if any(case_id not in indexed or case_split(cfg, case_id) != 'dev' for case_id in ids):
        raise ValueError('Every timing case must be an available development case.')
    return [indexed[case_id] for case_id in ids]


def runtime_projections(rows: pd.DataFrame, cfg: dict, model_load_seconds: float) -> pd.DataFrame:
    """Extrapolate measured per-scan times to clean plus P perturbations; estimates are not accuracy results."""
    expected = {(c, v, r) for c in cfg['timing']['case_ids']
                for v in ['clean', cfg['perturbations'][0]['name']]
                for r in [cfg['model']['resolution'], cfg['uncertainty']['second_model_resolution']]}
    observed = list(zip(rows.case_id, rows.variant, rows.resolution))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Projections require all eight distinct case/variant/model timings.')
    paired = rows.groupby(['case_id', 'variant'])[['prediction_seconds', 'total_seconds']].sum()
    means = paired.groupby('variant').mean()
    degraded = cfg['perturbations'][0]['name']
    projections = []
    for count in cfg['timing']['projection_case_counts']:
        for perturbations in cfg['timing']['projection_perturbation_counts']:
            projections.append({
                'n_cases': count, 'n_perturbations': perturbations, 'includes_clean_reference': True,
                'prediction_seconds': count * (means.loc['clean', 'prediction_seconds']
                                              + perturbations * means.loc[degraded, 'prediction_seconds']),
                'end_to_end_seconds': model_load_seconds + count * (means.loc['clean', 'total_seconds']
                                                       + perturbations * means.loc[degraded, 'total_seconds']),
                'basis': 'Two dev scans; clean plus P variants; both models; load each model once',
            })
    return pd.DataFrame(projections)


def run_benchmark(cfg: dict, config_path: Path, overwrite: bool = False) -> None:
    """Run eight predictions, save each timing immediately, and preserve config and machine provenance."""
    from segreview.inference import OrganSegmenter

    selected = select_timing_cases(cfg, list_cases(cfg))
    out = results_dir(cfg)
    timings_path = out / 'timings.csv'
    if timings_path.exists() and not overwrite:
        raise ValueError('Timings already exist; use --overwrite only for a deliberate complete rerun.')
    device = setup_compute(cfg)
    if device.type == 'cuda':
        # Availability alone is insufficient: old torch builds can detect an unsupported GPU.
        (torch.ones(4, device=device) + 1).cpu()
        torch.cuda.synchronize()
    settings = cfg['timing']
    start = time.perf_counter()
    info = machine_info()
    info.update({'config': str(config_path), 'status': 'running', 'device': str(device),
                 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                    if line.startswith('model name')), info['cpu']),
                 'torch_threads': torch.get_num_threads(), 'case_ids': settings['case_ids'],
                 'warmup': 'First prediction per model is included, not discarded',
                 'timing_scope': 'prediction includes resampling, preprocessing, network, reduction, back-resampling'})
    (out / 'run_config.yaml').write_bytes(config_path.read_bytes())
    manifest = out / 'machine.json'
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    rows, loads = [], []
    resolutions = [cfg['model']['resolution'], cfg['uncertainty']['second_model_resolution']]
    specs = cfg['perturbations']
    try:
        for model_index, resolution in enumerate(resolutions):
            t0 = time.perf_counter()
            segmenter = OrganSegmenter(cfg, device, resolution=resolution)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            loads.append({'resolution': resolution, 'seconds': time.perf_counter() - t0})
            pd.DataFrame(loads).to_csv(out / 'model_load_times.csv', index=False)
            for variant, spec in [('clean', None), (specs[0]['name'], specs[0])]:
                for case_index, case in selected:
                    t0 = time.perf_counter()
                    image = load_canonical(case.image_path)
                    t1 = time.perf_counter()
                    if spec is not None:
                        image = perturb(image, spec, cfg, case_index)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    t2 = time.perf_counter()
                    pred = segmenter.predict(image)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    prefix = '' if model_index == 0 else 'm2_'
                    save_map(cfg, variant, case.case_id, prefix + 'mask', pred.mask, pred.affine)
                    save_map(cfg, variant, case.case_id, prefix + 'prob', pred.prob, pred.affine)
                    t4 = time.perf_counter()
                    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    rows.append({'case_id': case.case_id, 'case_index': case_index, 'variant': variant,
                                 'resolution': resolution, 'spacing_mm': segmenter.spec.spacing_mm,
                                 'device': str(device), 'shape': 'x'.join(map(str, image.shape)),
                                 'load_seconds': t1 - t0, 'perturb_seconds': t2 - t1,
                                 'prediction_seconds': t3 - t2, 'save_seconds': t4 - t3,
                                 'total_seconds': t4 - t0,
                                 'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20
                                 if device.type == 'cuda' else None,
                                 'process_peak_rss_mib': peak_rss / 2**20 if sys.platform == 'darwin' else peak_rss / 1024})
                    pd.DataFrame(rows).to_csv(timings_path, index=False)
                    print(f'{case.case_id} {variant} {resolution}: {t3-t2:.2f} s prediction, {t4-t0:.2f} s total', flush=True)
                    del image, pred
                    gc.collect()
            del segmenter
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        frame = pd.DataFrame(rows)
        runtime_projections(frame, cfg, sum(row['seconds'] for row in loads)).to_csv(out / 'runtime_projections.csv', index=False)
        frame.groupby(['resolution', 'variant'])[['prediction_seconds', 'total_seconds', 'gpu_peak_allocated_mib']].agg(
            ['mean', 'min', 'max']).to_csv(out / 'timing_summary.csv')
        info['status'] = 'complete'
    except BaseException as exc:
        info.update({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        info.update({'elapsed_seconds': time.perf_counter() - start, 'completed_predictions': len(rows)})
        manifest.write_text(json.dumps(info, indent=2) + '\n')


def stop_at_limit(signum, frame):
    """Stop the bounded timing run rather than continue beyond the configured session limit."""
    raise TimeoutError('Timing run reached timing.max_run_minutes; ask Otto before a longer run.')


def main():
    """Read the timing config and run the benchmark with its configured time limit."""
    parser = config_arg_parser(__doc__)
    parser.add_argument('--overwrite', action='store_true', help='deliberately rerun all eight predictions')
    args = parser.parse_args()
    cfg = load_config(args.config)
    signal.signal(signal.SIGALRM, stop_at_limit)
    signal.alarm(int(cfg['timing']['max_run_minutes'] * 60))
    try:
        run_benchmark(cfg, args.config, args.overwrite)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
