"""Screen GT-free size/shape checks for missed-errors step 3, using kits100 as development data.

Benchmark and feature stages never load Dice or labels. Evaluation attaches saved Dice only after
features are saved, reuses the G1 engine, and reports tie-fair budgets and the four earlier misses.
Only aggregate tables/metadata are written; no masks, probability maps, CT images or combinations.
"""

import hashlib
import itertools
import json
import math
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from scipy.spatial import ConvexHull
from scipy.stats import rankdata

from segreview.config import REPO_ROOT, config_arg_parser, load_config, setup_compute
from segreview.data import list_cases
from segreview.evaluation import evaluate_methods, paired_differences
from segreview.io import map_path
from segreview.machine import machine_info

STATUS = 'exploratory screen; all kits100 cases are development data; no fresh-pool result'


def digest(path: Path) -> str:
    """Return a file's SHA-256 by streaming its bytes."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def settings_digest(cfg: dict) -> str:
    """Fingerprint all resolved settings so a changed setup requires a new runtime benchmark."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def component_geometry(mask: np.ndarray, spacing: tuple) -> dict:
    """Return voxel volume, voxel-face area and exact convex hull of the occupied voxel boxes.

    Hull vertices of boundary voxel centers suffice: adding their eight corners gives the same
    hull as adding all occupied voxel corners. Keeping only extreme centers makes Qhull cheaper.
    Physical spacing is used for area/volume; normal kidney concavity is not assumed to be an error.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any() or len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError('Need a nonempty 3D component and positive 3D spacing')
    volume = int(mask.sum()) * float(np.prod(spacing))
    boundary = mask & ~binary_erosion(mask, border_value=0)
    centers = np.argwhere(boundary).astype(float)
    if len(centers) >= 4 and np.linalg.matrix_rank(centers - centers[0]) == 3:
        centers = centers[ConvexHull(centers).vertices]
    offsets = np.array(list(itertools.product((-.5, .5), repeat=3)))
    corners = ((centers[:, None, :] + offsets) * np.asarray(spacing)).reshape(-1, 3)
    hull_volume = float(ConvexHull(corners).volume)
    if hull_volume < volume * (1 - 1e-8):
        raise ValueError('Voxel hull cannot be smaller than the occupied voxel volume')
    padded = np.pad(mask, 1)
    area = 0.0
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        faces = np.count_nonzero(padded[tuple(a)] != padded[tuple(b)])
        area += faces * np.prod([spacing[i] for i in range(3) if i != axis])
    return {'volume_ml': volume / 1000, 'hull_volume_ml': hull_volume / 1000,
            'surface_mm2': float(area), 'hull_deficit': float(np.clip(1 - volume / hull_volume, 0, 1)),
            'compactness_deficit': float(np.clip(1 - 36 * np.pi * volume ** 2 / area ** 3, 0, 1))}


def mask_features(mask: np.ndarray, spacing: tuple, settings: dict) -> tuple[dict, list[dict]]:
    """Compute shape scores from at most two large predicted components, without CT or GT inputs.

    A whole empty/tiny prediction gets maximal deficits. A single well-shaped component does not:
    the separate size/missing-side check is responsible for noticing a missing other kidney.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError('Need a 3D mask')
    voxel_ml = float(np.prod(spacing)) / 1000
    total = int(mask.sum()) * voxel_ml
    if mask.any():
        box = []
        for axis in range(3):
            occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
            box.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
        cropped = mask[tuple(box)]
        pieces, _ = label(cropped, structure=generate_binary_structure(3, settings['connectivity']))
        sizes = np.bincount(pieces.ravel())
        eligible = [i for i in range(1, len(sizes)) if sizes[i] * voxel_ml >= settings['min_component_ml']]
        eligible.sort(key=lambda i: (-sizes[i], i))
        selected = eligible[:settings['max_components']]
    else:
        eligible, selected = [], []
    rows = []
    for index in selected:
        part = pieces == index
        # Tight component boxes keep surface computations independent of the inter-kidney gap.
        positions = np.nonzero(part)
        box = tuple(slice(int(x.min()), int(x.max()) + 1) for x in positions)
        rows.append({'component_id': index, **component_geometry(part[box], spacing)})
    result = {'mask_volume_ml': total, 'n_large_components': len(eligible), 'n_shape_components': len(rows),
              'shape_hull_deficit_max': max((r['hull_deficit'] for r in rows), default=1.0),
              'shape_compactness_deficit_max': max((r['compactness_deficit'] for r in rows), default=1.0)}
    return result, rows


def inclusion_probabilities(score: np.ndarray, budget: int) -> np.ndarray:
    """Give each case its probability of being in the first k under uniform score-tie breaking."""
    score = np.asarray(score)
    if not np.isfinite(score).all() or not 0 <= budget <= len(score):
        raise ValueError('Need finite scores and a valid budget')
    return np.array([np.clip((budget - np.count_nonzero(score > value)) / np.count_nonzero(score == value), 0, 1)
                     for value in score], dtype=float)


def exact_table(frame: pd.DataFrame, expected: set[str], name: str) -> pd.DataFrame:
    """Return a case-sorted table, rejecting missing, duplicate or extra case IDs."""
    if frame.case_id.duplicated().any() or set(frame.case_id) != expected:
        raise ValueError(f'{name}: need the exact selected cases, without duplicates')
    return frame.sort_values('case_id').reset_index(drop=True)


def load_scope(cfg: dict) -> tuple[dict, Path, list, pd.DataFrame]:
    """Validate independent output and exact cohort; read only GT-free source table columns."""
    settings = cfg['checks']
    source = load_config(REPO_ROOT / settings['source_config'])
    output = (cfg['paths']['results_dir'] / cfg['name']).resolve()
    protected = [(source['paths'][key] / source['name']).resolve() for key in ('results_dir', 'predictions_dir')]
    for path in protected:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output overlaps a read-only source folder')
    if output == (REPO_ROOT / settings['inspection_queue']).resolve().parent:
        raise ValueError('Keep completed inspection outputs separate')
    cases = list_cases(source)
    expected = {c.case_id for c in cases}
    if expected != {c.case_id for c in list_cases(cfg)} or cfg['seed'] != source['seed']:
        raise ValueError('Source and check cohort/seed must match')
    folder = source['paths']['results_dir'] / source['name']
    cols = ['case_id', 'variant', 'split', 'pred_volume_ml', 'neg_volume_ml', 'left_ml', 'right_ml',
            'plaus_missing_side', 'plaus_asymmetry', settings['primary'], 'entropy_mean_region']
    saved = pd.read_csv(folder / 'scores.csv', usecols=cols)
    saved = exact_table(saved.loc[saved.variant.eq(settings['variant'])], expected, 'scores')
    far_cols = ['case_id', 'far_disagree_per_volume', 'far_6mm_adds_per_volume']
    far = exact_table(pd.read_csv(folder / 'exploratory_disagreement_scores.csv', usecols=far_cols), expected, 'far scores')
    saved = saved.merge(far, on='case_id', validate='one_to_one')
    saved['size_min_side_neg_ml'] = -saved[['left_ml', 'right_ml']].min(axis=1)
    return source, output, cases, saved


def extract_case(case, cfg: dict, source: dict) -> tuple[dict, list[dict], dict]:
    """Read one saved mask and its spacing; return GT-free aggregate features and input/timing evidence."""
    start = time.monotonic()
    path = map_path(source, cfg['checks']['variant'], case.case_id, 'mask')
    image = nib.load(path)
    mask = np.asanyarray(image.dataobj)
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected a binary saved mask')
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
    values, parts = mask_features(mask, spacing, cfg['checks'])
    proof = {'case_id': case.case_id, 'mask_path': str(path.relative_to(REPO_ROOT)), 'mask_sha256': digest(path),
             'grid_voxels': int(np.prod(mask.shape)), 'seconds': time.monotonic() - start}
    return {'case_id': case.case_id, **values}, [{'case_id': case.case_id, **r} for r in parts], proof


def evaluate_scores(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate saved GT-free columns with attached Dice using unchanged G1 metrics/paired bootstrap."""
    methods = cfg['checks']['methods']
    scores = {name: frame[name].to_numpy(dtype=float) for name in methods}
    if not all(np.isfinite(s).all() for s in scores.values()):
        raise ValueError('All scores must be finite')
    outcome = frame['dice'].to_numpy()
    if not np.isfinite(outcome).all() or not np.logical_and(outcome >= 0, outcome <= 1).all():
        raise ValueError('Need finite Dice in [0, 1]')
    metrics, curves = [], []
    for rule in cfg['evaluation']['bad_rules']:
        rows, outputs = evaluate_methods(scores, outcome, cfg, rule)
        metrics.extend({'bad_rule': rule['name'], 'status': STATUS, **row} for row in rows)
        for method, (review, quality) in outputs.items():
            curves.extend({'bad_rule': rule['name'], 'method': method, 'fraction_reviewed': k / len(frame),
                           'review': r, 'quality': q} for k, (r, q) in enumerate(zip(review, quality)))
    ev = cfg['evaluation']
    paired = paired_differences(scores, cfg['checks']['references'], outcome, ev['bad_rules'],
                                ev['n_bootstrap'], ev['ci_level'], np.random.default_rng(cfg['seed']))
    return pd.DataFrame(metrics), pd.DataFrame(paired), pd.DataFrame(curves)


def budget_tables(frame: pd.DataFrame, inspection: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fair top-budget finds/false alarms and per-miss ranks/probabilities for each standalone score."""
    settings = cfg['checks']
    n = len(frame)
    budget = math.ceil(settings['review_fraction'] * n)
    bad = frame.dice.lt(settings['bad_dice_threshold']).to_numpy()
    missed = set(inspection.loc[inspection.missed_bad, 'case_id'])
    previous_rank = inspection.set_index('case_id')['rank']
    target = frame.case_id.isin(missed).to_numpy()
    summaries, targets = [], []
    for method in settings['methods']:
        score = frame[method].to_numpy(dtype=float)
        weights = inclusion_probabilities(score, budget)
        ranks = rankdata(-score, method='average')
        summaries.append({'method': method, 'review_budget': budget, 'expected_bad_found': float(weights[bad].sum()),
                          'expected_false_alarms': float(weights[~bad].sum()),
                          'expected_original_misses_found': float(weights[target].sum()),
                          'distinct_score_values': int(len(np.unique(score))), 'status': STATUS})
        for i in np.flatnonzero(target):
            case_id = frame.iloc[i].case_id
            targets.append({'method': method, 'case_id': case_id, 'dice': frame.iloc[i].dice,
                            'original_rank': int(previous_rank.loc[case_id]), 'fair_rank': float(ranks[i]),
                            'rank_gain': float(previous_rank.loc[case_id] - ranks[i]),
                            'top_budget_probability': weights[i], 'status': STATUS})
    return pd.DataFrame(summaries), pd.DataFrame(targets)


def save_machine(output: Path, stage: str, cfg: dict, config_path: Path, start: float, **extra) -> None:
    """Save CPU/code/config versions and hashes for the stage, with Copenhagen time and explicit GT usage."""
    info = machine_info()
    paths = [config_path.resolve(), REPO_ROOT / cfg['checks']['source_config'], Path(__file__).resolve(),
             REPO_ROOT / 'uv.lock', REPO_ROOT / 'src/segreview/evaluation.py']
    folder = load_config(paths[1])['paths']['results_dir'] / load_config(paths[1])['name']
    paths += [folder / 'scores.csv', folder / 'exploratory_disagreement_scores.csv']
    if stage == 'evaluate':
        paths += [folder / 'dice.csv', output / 'exploratory_features.csv', REPO_ROOT / cfg['checks']['inspection_queue']]
    info.update(date=datetime.now(ZoneInfo('Europe/Copenhagen')).isoformat(timespec='seconds'),
                scipy=scipy.__version__, execution_device='cpu', seed=cfg['seed'], threads=cfg['compute']['num_threads'],
                status=STATUS, stage=stage, seconds=time.monotonic()-start,
                source_hashes={str(p.relative_to(REPO_ROOT)): digest(p) for p in paths}, **extra)
    if Path('/proc/cpuinfo').exists():
        info['cpu_model'] = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                                  if s.startswith('model name')), info['cpu'])
    (output / f'exploratory_{stage}_machine.json').write_text(json.dumps(info, indent=2, default=str) + '\n')


def benchmark(cfg: dict, source: dict, output: Path, cases: list) -> None:
    """Time small/median/large grids plus a synthetic full-width evaluation, then save a stage projection."""
    settings = cfg['checks']['benchmark']
    grids = [(int(np.prod(nib.load(map_path(source, cfg['checks']['variant'], c.case_id, 'mask')).shape)), c)
             for c in cases]
    grids.sort(key=lambda row: (row[0], row[1].case_id))
    positions = np.linspace(0, len(grids)-1, settings['n_cases']).round().astype(int)
    proofs = [extract_case(grids[i][1], cfg, source)[2] for i in positions]
    pd.DataFrame(proofs).to_csv(output / 'exploratory_benchmark_cases.csv', index=False)
    small = deepcopy(cfg)
    small['evaluation']['n_bootstrap'] = settings['n_bootstrap']
    rng = np.random.default_rng(cfg['seed'])
    synthetic = pd.DataFrame({name: rng.random(len(cases)) for name in cfg['checks']['methods']})
    synthetic['dice'] = np.linspace(.5, .99, len(cases))
    begin = time.monotonic()
    evaluate_scores(synthetic, small)
    evaluation_seconds = time.monotonic() - begin
    feature_seconds = sum(p['seconds'] for p in proofs) * sum(n for n, _ in grids) / sum(p['grid_voxels'] for p in proofs)
    eval_projection = evaluation_seconds * cfg['evaluation']['n_bootstrap'] / settings['n_bootstrap']
    projection = {'status': 'runtime planning only; no real Dice/GT loaded', 'n_cases': len(cases),
                  'settings_sha256': settings_digest(cfg), 'script_sha256': digest(Path(__file__)),
                  'case_seconds': sum(p['seconds'] for p in proofs), 'synthetic_evaluation_seconds': evaluation_seconds,
                  'feature_stage_projected_seconds': feature_seconds * settings['safety_factor'],
                  'evaluation_stage_projected_seconds': eval_projection * settings['safety_factor'],
                  'safety_factor': settings['safety_factor'], 'stage_cap_seconds': settings['max_stage_seconds']}
    projection['within_stage_cap'] = max(projection['feature_stage_projected_seconds'],
                                         projection['evaluation_stage_projected_seconds']) < settings['max_stage_seconds']
    (output / 'exploratory_runtime_projection.json').write_text(json.dumps(projection, indent=2) + '\n')
    print(json.dumps(projection, indent=2), flush=True)


def main() -> None:
    """Run a bounded benchmark, GT-free feature extraction, or evaluation from saved aggregate features."""
    parser = config_arg_parser(__doc__)
    parser.add_argument('--stage', choices=['benchmark', 'features', 'evaluate'], required=True)
    args = parser.parse_args()
    start = time.monotonic()
    cfg = load_config(args.config)
    setup_compute(cfg)
    source, output, cases, saved = load_scope(cfg)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'benchmark':
        benchmark(cfg, source, output, cases)
    else:
        projection = json.loads((output / 'exploratory_runtime_projection.json').read_text())
        if projection['settings_sha256'] != settings_digest(cfg) or projection['script_sha256'] != digest(Path(__file__)):
            raise ValueError('Settings/code changed since benchmark; benchmark again')
        if not projection['within_stage_cap']:
            raise ValueError('Projected stage exceeds 30-minute cap; ask before a longer run')
        if args.stage == 'features':
            rows, parts, proofs = [], [], []
            for case in cases:
                values, components, proof = extract_case(case, cfg, source)
                rows.append(values)
                parts.extend(components)
                proofs.append(proof)
                print(f"Features {case.case_id}: {proof['seconds']:.2f} s", flush=True)
            frame = saved.merge(pd.DataFrame(rows), on='case_id', validate='one_to_one')
            if not np.allclose(frame.mask_volume_ml, frame.pred_volume_ml,
                               rtol=cfg['checks']['pred_volume_rtol'], atol=cfg['checks']['pred_volume_atol_ml']):
                raise ValueError('Loaded mask volumes differ from saved source scores')
            frame['status'] = STATUS
            frame.to_csv(output / 'exploratory_features.csv', index=False)
            pd.DataFrame(parts).to_csv(output / 'exploratory_components.csv', index=False)
            pd.DataFrame(proofs).to_csv(output / 'exploratory_feature_inputs.csv', index=False)
        else:
            feature_path = output / 'exploratory_features.csv'
            feature_info = json.loads((output / 'exploratory_features_machine.json').read_text())
            for name, expected_hash in feature_info['source_hashes'].items():
                if digest(REPO_ROOT / name) != expected_hash:
                    raise ValueError('Feature-stage code/config/source changed; regenerate features')
            frame = exact_table(pd.read_csv(feature_path), {c.case_id for c in cases}, 'features')
            folder = source['paths']['results_dir'] / source['name']
            outcomes = pd.read_csv(folder / 'dice.csv', usecols=['case_id', 'variant', 'dice'])
            outcomes = exact_table(outcomes.loc[outcomes.variant.eq(cfg['checks']['variant'])], set(frame.case_id), 'Dice')
            frame = frame.merge(outcomes[['case_id', 'dice']], on='case_id', validate='one_to_one')
            inspection = exact_table(pd.read_csv(REPO_ROOT / cfg['checks']['inspection_queue']), set(frame.case_id), 'inspection')
            if not np.allclose(frame.dice, inspection.dice, rtol=0, atol=1e-12):
                raise ValueError('Outcome Dice differs from inspection')
            metrics, paired, curves = evaluate_scores(frame, cfg)
            metrics.to_csv(output / 'exploratory_metrics.csv', index=False)
            paired.to_csv(output / 'exploratory_paired_vs_locked.csv', index=False)
            curves.to_csv(output / 'exploratory_curves.csv', index=False)
            budgets, targets = budget_tables(frame, inspection, cfg)
            budgets.to_csv(output / 'exploratory_budget.csv', index=False)
            targets.to_csv(output / 'exploratory_miss_ranks.csv', index=False)
            frame.to_csv(output / 'exploratory_evaluation_cases.csv', index=False)
            flags = frame.plaus_missing_side.gt(0)
            bad = frame.dice.lt(cfg['checks']['bad_dice_threshold'])
            summary = {'status': STATUS, 'n_cases': len(frame), 'n_bad': int(bad.sum()),
                       'n_original_misses': int(inspection.missed_bad.sum()),
                       'missing_side_flagged_bad': int((flags & bad).sum()),
                       'missing_side_flagged_nonbad': int((flags & ~bad).sum()),
                       'new_combinations_evaluated': False, 'fresh_pool_used': False}
            (output / 'exploratory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(budgets.round(4).to_string(index=False), flush=True)
    save_machine(output, args.stage, cfg, args.config, start,
                 actual_outcomes_used=args.stage == 'evaluate', ground_truth_label_maps_used=False)


if __name__ == '__main__':
    main()
