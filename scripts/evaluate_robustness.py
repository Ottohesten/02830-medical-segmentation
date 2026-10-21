"""Step 5: prepare selected maps, time CPU scoring, then evaluate fixed robustness questions.

Run --plan, --prepare (only missing dev second-model inference), --benchmark (two dev scans and
synthetic G1 timings), then --evaluate. Ask Otto before any stage expected to exceed 30 minutes.
Inference configs/results remain intact; this experiment copies maps into its own prediction folder.
"""

import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from segreview.config import REPO_ROOT, config_arg_parser, load_config, results_dir, setup_compute
from segreview.data import case_split, list_cases, load_organ_mask
from segreview.evaluation import evaluate_methods
from segreview.io import load_map, map_path, voxel_spacing
from segreview.machine import machine_info
from segreview.metrics import dice
from segreview.pipeline import step_second_model, variants
from segreview.robustness_evaluation import change_summaries, evaluate_queues, paired_changes
from segreview.uncertainty import scan_scores


def digest(path: Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path: Path, value) -> None:
    """Write a readable JSON artifact with a final newline."""
    path.write_text(json.dumps(value, indent=2) + '\n')


def validate_scope(cfg: dict) -> tuple[list, dict]:
    """Require the archived strength/scientific settings, original 100 IDs, separate folders, and fixed methods."""
    settings = cfg['robustness_evaluation']
    locked_path = REPO_ROOT / settings['locked_config']
    lock_path = REPO_ROOT / cfg['strength_lock']['summary_file']
    lock = json.loads((lock_path.parent / 'strength_lock.json').read_text())
    if digest(locked_path) != lock['final_config_sha256']:
        raise ValueError('Archived locked config changed.')
    for key in ['source_config', 'summary_file']:
        if digest(REPO_ROOT / lock[key]) != lock[f'{key}_sha256' if key == 'source_config' else 'summary_sha256']:
            raise ValueError('Development selection evidence changed.')
    locked = load_config(locked_path)
    comparison = deepcopy(cfg)
    comparison.pop('robustness_evaluation')
    comparison['name'] = locked['name']
    comparison['dataset']['n_cases'] = locked['dataset']['n_cases']
    comparison['evaluation']['splits'] = locked['evaluation']['splits']
    comparison['report']['set'] = locked['report']['set']
    if comparison != locked or settings['methods'] != [lock['primary_score'], lock['secondary_score'], 'neg_volume_ml']:
        raise ValueError('Use only the locked strengths, scientific settings, primary/secondary scores, and volume baseline.')
    if (cfg['evaluation']['splits'] != ['dev', 'test'] or settings['bootstrap_unit'] != 'case_id'
            or settings['inference_only_splits'] != ['dev']):
        raise ValueError('Keep dev/test reporting separate, bootstrap patients, and infer only missing dev maps.')
    sources = {split: load_config(REPO_ROOT / settings[key]) for split, key in
               [('dev', 'development_config'), ('test', 'test_config')]}
    reference = load_config(REPO_ROOT / settings['reference_config'])
    if Path(cfg['name']).name != cfg['name'] or cfg['name'] in {'.', '..'}:
        raise ValueError('Use a separate simple experiment name.')
    for prior in [locked, reference, *sources.values()]:
        for key in ['predictions_dir', 'results_dir']:
            target = (cfg['paths'][key] / cfg['name']).resolve()
            protected = (prior['paths'][key] / prior['name']).resolve()
            if target.is_relative_to(protected) or protected.is_relative_to(target):
                raise ValueError('Evaluation outputs must not overlap archived or locked outputs.')
    cases = list_cases(cfg)
    if ([c.case_id for c in cases] != [c.case_id for c in list_cases(reference)]
            or cfg['dataset']['split'] != reference['dataset']['split'] or cfg['seed'] != reference['seed']):
        raise ValueError('Case selection and split must match the original study.')
    for split in ['dev', 'test']:
        if sum(case_split(cfg, c.case_id) == split for c in cases) != settings[f'expected_{split}_cases']:
            raise ValueError('Unexpected split size.')
    return cases, sources


def source_hashes() -> dict:
    """Identify the script and every project module used for inference, scoring, and evaluation."""
    paths = [Path(__file__), *(REPO_ROOT / 'src/segreview' / f'{name}.py' for name in
             ['robustness_evaluation', 'evaluation', 'pipeline', 'uncertainty', 'metrics', 'data', 'io',
              'config', 'machine', 'inference', 'weights', 'augment'])]
    return {str(p.resolve().relative_to(REPO_ROOT)): digest(p) for p in paths}


@contextmanager
def record_stage(cfg: dict, config_path: Path, out: Path, stage: str):
    """Record stage provenance, elapsed time, status, and peak process/GPU memory, including interrupted stages."""
    device = setup_compute(cfg)
    gpu = stage == 'prepare' and device.type == 'cuda'
    if gpu:
        (torch.ones(4, device=device) + 1).cpu()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    info = machine_info()
    cpu = Path('/proc/cpuinfo')
    if cpu.exists():
        info['cpu_model'] = next((line.split(':', 1)[1].strip() for line in cpu.read_text().splitlines()
                                  if line.startswith('model name')), info['cpu'])
    if gpu:
        info['nvidia_driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    info.update({'stage': stage, 'device_used': str(device) if gpu else 'cpu', 'status': 'running',
                 'config_sha256': digest(config_path), 'source_sha256': source_hashes(),
                 'torch_threads': torch.get_num_threads(), 'seed': cfg['seed']})
    path = out / f'{stage}_machine.json'
    if path.exists():
        raise ValueError(f'{stage} already started; inspect its saved status before deliberately resuming.')
    write_json(path, info)
    start = time.perf_counter()
    try:
        yield info
        info['status'] = 'complete'
    except BaseException as exc:
        info.update({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        info.update({'elapsed_seconds': time.perf_counter() - start,
                     'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                     'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if gpu else None})
        write_json(path, info)


def prepare_maps(cfg: dict, cases: list, sources: dict, out: Path) -> None:
    """Hash-check and copy the 720 selected input maps, then infer only the 40 missing dev second-model pairs."""
    imported = []
    for split, source in sources.items():
        manifest = pd.read_csv(results_dir(source) / 'prediction_manifest.csv')
        ids = {c.case_id for c in cases if case_split(cfg, c.case_id) == split}
        kinds = ['mask', 'prob'] if split == 'dev' else ['mask', 'prob', 'm2_mask', 'm2_prob']
        selected = manifest[manifest.case_id.isin(ids) & manifest.variant.isin(variants(cfg)) & manifest.kind.isin(kinds)]
        expected = {(c, v, k) for c in ids for v in variants(cfg) for k in kinds}
        if len(selected) != len(expected) or set(zip(selected.case_id, selected.variant, selected.kind)) != expected:
            raise ValueError('Incomplete source prediction manifest.')
        for row in selected.itertuples():
            source_path = REPO_ROOT / row.path
            target = map_path(cfg, row.variant, row.case_id, row.kind)
            if digest(source_path) != row.sha256:
                raise ValueError(f'Prediction differs from the archived manifest: {source_path}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f'Target unexpectedly exists: {target}')
            shutil.copy2(source_path, target)
            if digest(target) != row.sha256:
                raise ValueError('Prediction copy checksum mismatch.')
            imported.append({'case_id': row.case_id, 'variant': row.variant, 'kind': row.kind,
                             'source_path': row.path, 'sha256': row.sha256})
    pd.DataFrame(imported).to_csv(out / 'imported_predictions.csv', index=False)
    print(f'[prepare] {len(imported)} archived maps copied and hash-checked', flush=True)
    step_second_model(cfg, setup_compute(cfg), splits=['dev'])
    manifest = []
    for case in cases:
        for variant in variants(cfg):
            for kind in ['mask', 'prob', 'm2_mask', 'm2_prob']:
                path = map_path(cfg, variant, case.case_id, kind)
                manifest.append({'case_id': case.case_id, 'variant': variant, 'kind': kind,
                                 'path': str(path.relative_to(REPO_ROOT)), 'bytes': path.stat().st_size,
                                 'sha256': digest(path)})
    pd.DataFrame(manifest).to_csv(out / 'prediction_manifest.csv', index=False)
    print(f'[prepare] {len(manifest)} maps ready; no labels or scores read', flush=True)


def score_prediction(cfg: dict, case_id: str, variant: str) -> dict:
    """Compute the locked GT-free scores with the existing G1 function, requiring four finite aligned maps."""
    images = {kind: nib.load(map_path(cfg, variant, case_id, kind))
              for kind in ['mask', 'prob', 'm2_mask', 'm2_prob']}
    ref = images['mask']
    if any(i.shape != ref.shape or not np.allclose(i.affine, ref.affine) for i in images.values()):
        raise ValueError('Model maps must share their grid.')
    arrays = {k: i.get_fdata(dtype=np.float32) if k in ['prob', 'm2_prob'] else np.asanyarray(i.dataobj)
              for k, i in images.items()}
    if any(not np.isfinite(a).all() or a.min() < 0 or a.max() > 1 for a in arrays.values()):
        raise ValueError('Model maps must be finite within [0, 1].')
    scores, _ = scan_scores(arrays['mask'], arrays['prob'], tuple(ref.header.get_zooms()[:3]),
                            cfg['uncertainty']['border_mm'], m2_mask=arrays['m2_mask'], m2_prob=arrays['m2_prob'])
    return {k: scores[k] for k in ['pred_volume_ml', *cfg['robustness_evaluation']['methods']]}


def benchmark(cfg: dict, cases: list, out: Path) -> dict:
    """Time four dev score computations and one synthetic mixed-queue evaluation; project the full CPU stage."""
    settings = cfg['robustness_evaluation']
    selected = [c for c in cases if c.case_id in settings['benchmark_dev_cases']]
    if len(selected) != len(settings['benchmark_dev_cases']) or any(case_split(cfg, c.case_id) != 'dev' for c in selected):
        raise ValueError('Benchmark only the configured development cases.')
    times = []
    for case in selected:
        for variant in variants(cfg):
            start = time.perf_counter()
            score_prediction(cfg, case.case_id, variant)
            times.append({'case_id': case.case_id, 'variant': variant,
                          'voxels': int(np.prod(nib.load(case.image_path).shape)),
                          'seconds': time.perf_counter() - start})
    frame = pd.DataFrame(times)
    frame.to_csv(out / 'benchmark_times.csv', index=False)
    rng = np.random.default_rng(cfg['seed'])
    n_cases = settings['expected_test_cases']
    n = n_cases * len(variants(cfg))
    scores = {m: rng.uniform(1, 2, size=n) for m in settings['methods']}
    scores['neg_volume_ml'] *= -1
    start = time.perf_counter()
    evaluate_methods(scores, rng.uniform(.6, 1, size=n), cfg, cfg['evaluation']['bad_rules'][0],
                     groups=np.repeat(np.arange(n_cases), len(variants(cfg))))
    eval_seconds = time.perf_counter() - start
    total_voxels = sum(int(np.prod(nib.load(c.image_path).shape)) for c in cases) * len(variants(cfg))
    scoring_seconds = frame.seconds.sum() / frame.voxels.sum() * total_voxels
    n_evaluations = len(cfg['evaluation']['splits']) * (len(variants(cfg)) + 1) * len(cfg['evaluation']['bad_rules'])
    estimate = settings['timing_safety_factor'] * (scoring_seconds + n_evaluations * eval_seconds
                                                + len(cases) * settings['dice_allowance_seconds_per_case']) / 60
    result = {'estimated_evaluation_minutes': estimate, 'score_proxy_seconds': scoring_seconds,
              'synthetic_queue_evaluation_seconds': eval_seconds, 'n_queue_evaluations': n_evaluations,
              'safety_factor': settings['timing_safety_factor'], 'test_labels_or_scores_read': False,
              'limitations': 'Two dev scans, voxel-scaled CPU scoring; synthetic largest queue; fixed Dice allowance.'}
    write_json(out / 'runtime_projection.json', result)
    return result


def evaluate(cfg: dict, cases: list, out: Path) -> None:
    """Score every saved view without labels first, then evaluate Dice and fixed paired/queue questions once."""
    manifest = pd.read_csv(out / 'prediction_manifest.csv')
    for row in manifest.itertuples():
        if digest(REPO_ROOT / row.path) != row.sha256:
            raise ValueError('Prepared prediction changed before evaluation.')
    score_rows = []
    for case in cases:
        for variant in variants(cfg):
            score_rows.append({'case_id': case.case_id, 'variant': variant, 'split': case_split(cfg, case.case_id),
                               **score_prediction(cfg, case.case_id, variant)})
            pd.DataFrame(score_rows).to_csv(out / 'scores.csv', index=False)
            print(f'[scores] {len(score_rows)} / {len(cases) * len(variants(cfg))}: {case.case_id} {variant}', flush=True)
    dice_rows, per_model = [], []
    for case in cases:
        truth, ignore = load_organ_mask(cfg, case)
        for variant in variants(cfg):
            row = {'case_id': case.case_id, 'variant': variant, 'split': case_split(cfg, case.case_id)}
            for kind, key, resolution in [('mask', 'dice', cfg['model']['resolution']),
                                          ('m2_mask', 'dice_second_model', cfg['uncertainty']['second_model_resolution'])]:
                value = round(dice(load_map(cfg, variant, case.case_id, kind), truth, ignore), 4)
                row[key] = value
                per_model.append({**{k: row[k] for k in ['case_id', 'variant', 'split']},
                                  'resolution': resolution, 'dice': value})
            dice_rows.append(row)
        pd.DataFrame(dice_rows).to_csv(out / 'dice.csv', index=False)
        pd.DataFrame(per_model).to_csv(out / 'dice_per_model.csv', index=False)
        print(f'[dice] {len(dice_rows)} / {len(cases) * len(variants(cfg))}', flush=True)
    scores, dice_frame = pd.DataFrame(score_rows), pd.DataFrame(dice_rows)
    paired = paired_changes(scores, dice_frame, cfg)
    paired.to_csv(out / 'paired_changes.csv', index=False)
    degradation, correlations = change_summaries(paired, cfg)
    degradation.to_csv(out / 'dice_degradation.csv', index=False)
    correlations.to_csv(out / 'change_correlations.csv', index=False)
    metrics, curves, shifts, tops = evaluate_queues(scores, dice_frame, cfg, progress=lambda text: print(text, flush=True))
    for name, frame in [('g1_metrics', metrics), ('g1_curves', curves), ('queue_rank_changes', shifts), ('queue_top_summary', tops)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    write_json(out / 'verification.json', {'status': 'passed', 'n_patients': len(cases), 'n_score_rows': len(scores),
                                         'n_dice_rows': len(dice_frame), 'n_model_dice_rows': len(per_model),
                                         'n_paired_rows': len(paired), 'queue_sets': sorted(metrics['set'].unique()),
                                         'mixed_queue_bootstrap_unit': 'case_id', 'no_additional_strength_or_score_selection': True})


def stop_at_limit(signum, frame):
    """Stop at the stage cap and require Otto's approval before extending a long run."""
    raise TimeoutError('Stage reached robustness_evaluation.max_stage_minutes; ask Otto before extending it.')


def main():
    """Require an unchanged plan and run only the requested, separately timed step 5 stage."""
    parser = config_arg_parser(__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ['plan', 'prepare', 'benchmark', 'evaluate']:
        actions.add_argument(f'--{action}', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cases, sources = validate_scope(cfg)
    out = results_dir(cfg)
    plan_path = out / 'analysis_plan.json'
    if args.plan:
        if plan_path.exists():
            raise ValueError('Analysis plan already exists; retain the first pre-evaluation plan.')
        plan = {'config_sha256': digest(args.config), 'source_sha256': source_hashes(),
                'case_ids': [c.case_id for c in cases], 'settings': cfg['robustness_evaluation'],
                'perturbations': cfg['perturbations'], 'evaluation': cfg['evaluation'],
                'sign_convention': 'Positive rho(Dice clean-minus-perturbed, score perturbed-minus-clean) supports detection.',
                'mixed_queue': 'Two views per patient; bootstrap case IDs, retain both; no independent-row rho p-value.',
                'bad_flags': 'Follow existing G1: define flags in the full queue; retain flags in bootstrap.',
                'notes': 'Fixed primary/secondary only plus volume, random, oracle; no combinations or candidate search.'}
        write_json(plan_path, plan)
        (out / 'run_config.yaml').write_bytes(args.config.read_bytes())
        print(json.dumps(plan, indent=2))
        return
    plan = json.loads(plan_path.read_text())
    if (plan['config_sha256'] != digest(args.config) or plan['source_sha256'] != source_hashes()
            or plan['case_ids'] != [c.case_id for c in cases]):
        raise ValueError('Config/source/case selection changed since the pre-evaluation plan.')
    stage = next(s for s in ['prepare', 'benchmark', 'evaluate'] if getattr(args, s))
    if stage != 'prepare' and json.loads((out / 'prepare_machine.json').read_text())['status'] != 'complete':
        raise ValueError('Complete and verify preparation first.')
    if stage == 'evaluate':
        estimate = json.loads((out / 'runtime_projection.json').read_text())['estimated_evaluation_minutes']
        if estimate > cfg['robustness_evaluation']['max_stage_minutes']:
            raise ValueError('Projected evaluation exceeds the stage cap; ask Otto before starting a longer run.')
    os.environ['nnUNet_compile'] = 'false'
    signal.signal(signal.SIGALRM, stop_at_limit)
    signal.alarm(int(cfg['robustness_evaluation']['max_stage_minutes'] * 60))
    try:
        with record_stage(cfg, args.config, out, stage):
            if stage == 'prepare':
                prepare_maps(cfg, cases, sources, out)
            elif stage == 'benchmark':
                print(json.dumps(benchmark(cfg, cases, out), indent=2), flush=True)
            else:
                evaluate(cfg, cases, out)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
