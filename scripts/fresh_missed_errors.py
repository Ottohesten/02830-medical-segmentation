"""Freeze and execute the single joint fresh-pool ranking check (missed-errors step 5).

Freeze/old-data benchmarks need no fresh inputs. Preflight requires real group coordination;
fresh scores are saved without labels, and an exclusive marker guards the one outcome stage.
Only aggregate tables and provenance are written. Inference and downloads are never performed.
"""

import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from segreview.config import REPO_ROOT, config_arg_parser, load_config, setup_compute
from segreview.data import CASE_LIST_FILE, dataset_dir, list_cases, load_canonical
from segreview.disagreement_scores import candidate_scores
from segreview.evaluation import bootstrap_indices
from segreview.machine import machine_info
from segreview.metrics import dice
from segreview.model_comparison import directional_disagreement
from segreview.plausibility import paired_organ_check
from segreview.uncertainty import scan_scores

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.check_missed_errors import digest, evaluate_scores, exact_table, inclusion_probabilities, mask_features
from scripts.combine_missed_errors import build_combinations, queue_changes

STATUS = 'locked joint fresh-pool ranking test; no score selection on this pool'
INPUT_COLUMNS = ['ct', 'label', 'mask', 'prob', 'm2_mask', 'm2_prob']


def now() -> str:
    """Return an explicit Copenhagen timestamp for the scientific audit trail."""
    return datetime.now(ZoneInfo('Europe/Copenhagen')).isoformat(timespec='seconds')


def write_json(path: Path, value: dict) -> None:
    """Save small JSON metadata with real newlines and readable indentation."""
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def pool_allocation(all_ids: list[str], old_ids: set[str], seed: int, skip: int, count: int) -> pd.DataFrame:
    """Reproduce the seeded kits100 draw and return its next cases without looking at any outcomes."""
    ids = sorted(all_ids)
    if len(set(ids)) != len(ids) or skip != len(old_ids) or len(ids) < skip + count:
        raise ValueError('Invalid universe/offset/cohort size')
    order = np.random.default_rng(seed).permutation(len(ids))
    if {ids[i] for i in order[:skip]} != old_ids:
        raise ValueError('Original cohort is not reproduced by this source list/seed')
    rows = [{'case_id': ids[index], 'draw_position': position}
            for position, index in enumerate(order[skip:skip+count], start=skip+1)]
    return pd.DataFrame(rows).sort_values('case_id').reset_index(drop=True)


def experiment(cfg: dict) -> tuple[Path, dict]:
    """Return independent output/source config, rejecting overlap with locked or earlier result folders."""
    source = load_config(REPO_ROOT / cfg['fresh']['source_config'])
    output = (cfg['paths']['results_dir'] / cfg['name']).resolve()
    protected = [source['paths']['results_dir'] / source['name'], source['paths']['predictions_dir'] / source['name']]
    protected += [cfg['paths']['results_dir'] / name for name in ['kits_missed_errors', 'kits_missed_checks',
                                                                'kits_missed_combinations']]
    for folder in protected:
        folder = folder.resolve()
        if output == folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output overlaps a protected source folder')
    return output, source


def scientific_hashes(cfg: dict, config_path: Path, source: dict) -> dict:
    """Fingerprint the scientific recipe, implementation, source list and original model checkpoints."""
    paths = [config_path.resolve(), REPO_ROOT / cfg['fresh']['protocol'],
             REPO_ROOT / cfg['fresh']['source_config'], dataset_dir(source) / CASE_LIST_FILE]
    paths += [REPO_ROOT / name for name in cfg['fresh']['implementation_files']]
    paths += [REPO_ROOT / name for name in cfg['fresh']['reference_checkpoints'].values()]
    return {str(p.relative_to(REPO_ROOT)): digest(p) for p in paths}


def require_lock(cfg: dict, config_path: Path, output: Path, source: dict) -> dict:
    """Require unchanged scientific files and the frozen cohort before any fresh input operation."""
    lock = json.loads((output / 'lock.json').read_text())
    if lock['scientific_hashes'] != scientific_hashes(cfg, config_path, source):
        raise ValueError('Scientific lock changed; log any amendment before fresh outcomes, never silently replace it')
    if digest(output / 'cohort.csv') != lock['cohort_sha256']:
        raise ValueError('Frozen cohort changed')
    return lock


def freeze(cfg: dict, config_path: Path, output: Path, source: dict) -> dict:
    """Write an immutable scientific lock/cohort; repeat calls verify rather than replace the lock."""
    if (output / 'lock.json').exists():
        return require_lock(cfg, config_path, output, source)
    old = pd.read_csv(source['paths']['results_dir'] / source['name'] / 'scores.csv', usecols=['case_id', 'variant'])
    old_ids = set(old.loc[old.variant.eq('clean'), 'case_id'])
    all_ids = (dataset_dir(source) / CASE_LIST_FILE).read_text().split()
    settings = cfg['fresh']
    if cfg['seed'] != source['seed'] or settings['skip_cases'] != source['dataset']['n_cases']:
        raise ValueError('Seed/offset must match the original draw')
    cohort = pool_allocation(all_ids, old_ids, cfg['seed'], settings['skip_cases'], settings['n_cases'])
    cohort['prior_g3_smoke_use'] = cohort.case_id.isin(settings['known_g3_smoke_cases'])
    if set(settings['known_g3_smoke_cases']) - set(cohort.case_id):
        raise ValueError('Expected smoke-test cases are not in the agreed pool')
    cohort.to_csv(output / 'cohort.csv', index=False)
    (output / 'run_config.yaml').write_bytes(config_path.read_bytes())
    lock = {'created_at': now(), 'state_at_lock': 'scientific recipe frozen; coordination/data may still be pending',
            'config': str(config_path), 'seed': cfg['seed'], 'n_cases': len(cohort), 'kits100_overlap': 0,
            'cohort_sha256': digest(output / 'cohort.csv'), 'scientific_hashes': scientific_hashes(cfg, config_path, source),
            'checkpoint_sha256': {name: digest(REPO_ROOT / path) for name, path in settings['reference_checkpoints'].items()},
            'prior_g3_smoke_cases': settings['known_g3_smoke_cases'], 'fresh_outcomes_accessed_before_lock': False}
    write_json(output / 'lock.json', lock)
    template = {'bjorn_notified_at': None, 'frederik_confirmed_at': None, 'evidence': None,
                'ranking_metrics_not_previously_evaluated': None, 'checkpoint_sha256': lock['checkpoint_sha256'],
                'inference_metadata': None, 'inputs_manifest': None}
    write_json(output / 'handoff_template.json', template)
    return lock


def handoff_status(handoff: dict, lock: dict) -> list[str]:
    """List missing real coordination/provenance facts; never infer contact or model provenance."""
    issues = [f'Missing {key}' for key in ['bjorn_notified_at', 'frederik_confirmed_at', 'evidence',
                                         'inference_metadata', 'inputs_manifest'] if not handoff.get(key)]
    if handoff.get('ranking_metrics_not_previously_evaluated') is not True:
        issues.append('Fresh ranking-metric status not confirmed')
    if handoff.get('checkpoint_sha256') != lock['checkpoint_sha256']:
        issues.append('Original baseline checkpoint hashes not confirmed')
    return issues


def relative_file(value: str, parent: Path) -> Path:
    """Resolve an input manifest/metadata path relative to the file that declares it."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def canonical_geometry(path: Path) -> tuple:
    """Return canonical shape/affine from NIfTI headers only, without reading label/image voxels."""
    image = nib.load(path)
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                nib.orientations.axcodes2ornt(('R', 'A', 'S')))
    shape = tuple(np.asarray(image.shape)[np.argsort(transform[:, 0])])
    return shape, image.affine @ nib.orientations.inv_ornt_aff(transform, image.shape)


def preflight(cfg: dict, output: Path, lock: dict, handoff_path: Path | None) -> dict:
    """Verify actual coordination before fresh headers/files; record pending facts without opening the pool."""
    if handoff_path is None:
        result = {'ready': False, 'issues': ['Need actual Bjørn/Frederik coordination and an input handoff'],
                  'fresh_headers_or_voxels_read': False, 'checked_at': now()}
    else:
        handoff_path = handoff_path.resolve()
        handoff = json.loads(handoff_path.read_text())
        issues = handoff_status(handoff, lock)
        result = {'ready': False, 'issues': issues, 'fresh_headers_or_voxels_read': False, 'checked_at': now()}
        if not issues:
            manifest_path = relative_file(handoff['inputs_manifest'], handoff_path.parent)
            metadata_path = relative_file(handoff['inference_metadata'], handoff_path.parent)
            expected = set(pd.read_csv(output / 'cohort.csv').case_id)
            rows = exact_table(pd.read_csv(manifest_path), expected, 'fresh handoff')
            inventory = []
            for row in rows.to_dict('records'):
                paths = {key: relative_file(row[key], manifest_path.parent) for key in INPUT_COLUMNS}
                reference = canonical_geometry(paths['ct'])
                entry = {'case_id': row['case_id'], 'grid_voxels': int(np.prod(reference[0]))}
                for key, path in paths.items():
                    if key in ['ct', 'label']:
                        shape, affine = canonical_geometry(path)
                    else:
                        image = nib.load(path)
                        shape, affine = image.shape, image.affine
                        if key in ['prob', 'm2_prob'] and (image.get_data_dtype() != np.dtype('uint8') or
                                not np.isclose(image.dataobj.slope, 1/255, rtol=1e-6) or image.dataobj.inter != 0):
                            raise ValueError('Need the standard 1/255 probability representation')
                    if shape != reference[0] or not np.allclose(affine, reference[1], rtol=0, atol=1e-4):
                        raise ValueError(f'Geometry mismatch: {row["case_id"]} {key}')
                    entry[key] = str(path)
                    entry[key + '_sha256'] = digest(path)
                inventory.append(entry)
            pd.DataFrame(inventory).to_csv(output / 'input_inventory.csv', index=False)
            result.update(ready=True, input_inventory_sha256=digest(output / 'input_inventory.csv'),
                          handoff=str(handoff_path), handoff_sha256=digest(handoff_path),
                          inference_metadata=str(metadata_path), inference_metadata_sha256=digest(metadata_path),
                          fresh_headers_or_voxels_read='headers/bytes only; no voxel arrays', n_cases=len(rows))
    write_json(output / 'preflight.json', result)
    return result


def require_ready(output: Path) -> pd.DataFrame:
    """Load only a verified source inventory, requiring unchanged handoff and inference metadata."""
    info = json.loads((output / 'preflight.json').read_text())
    if not info['ready']:
        raise ValueError('Coordination/data pending; do not open fresh scores or outcomes')
    for path, key in [(Path(info['handoff']), 'handoff_sha256'),
                      (Path(info['inference_metadata']), 'inference_metadata_sha256'),
                      (output / 'input_inventory.csv', 'input_inventory_sha256')]:
        if digest(path) != info[key]:
            raise ValueError('Preflight inputs changed')
    return pd.read_csv(output / 'input_inventory.csv')


def case_scores(paths: dict, cfg: dict) -> dict:
    """Compute all frozen GT-free scores from CT and both original model maps, never labels."""
    f = cfg['fresh']
    ct = load_canonical(Path(paths['ct'])).get_fdata(dtype=np.float32)
    mask_image = nib.load(paths['mask'])
    raw_mask = np.asanyarray(mask_image.dataobj)
    raw_second = np.asanyarray(nib.load(paths['m2_mask']).dataobj)
    if not np.isin(raw_mask, [0, 1]).all() or not np.isin(raw_second, [0, 1]).all():
        raise ValueError('Need binary predicted masks')
    mask, second = raw_mask.astype(bool), raw_second.astype(bool)
    del raw_mask, raw_second
    spacing = tuple(float(x) for x in mask_image.header.get_zooms()[:3])
    plausibility = paired_organ_check(mask, ct, spacing, f)
    del ct
    prob = nib.load(paths['prob']).get_fdata(dtype=np.float32)
    prob2 = nib.load(paths['m2_prob']).get_fdata(dtype=np.float32)
    if not all(np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 for p in [prob, prob2]):
        raise ValueError('Need finite organ probabilities in [0, 1]')
    scores, entropy_map = scan_scores(mask, prob, spacing, f['border_mm'], m2_mask=second, m2_prob=prob2)
    del entropy_map, prob, prob2
    shape, _ = mask_features(mask, spacing, f)
    counts = directional_disagreement(mask, second, spacing, f['grid_band_mm'])
    far = candidate_scores(counts, float(np.prod(spacing))/1000, ['far_disagree'], ['per_volume'])
    return {**{name: scores[name] for name in ['model_diff_mean_region', 'entropy_mean_region', 'neg_volume_ml']},
            'plaus_missing_side': plausibility['plaus_missing_side'],
            'shape_hull_deficit_max': shape['shape_hull_deficit_max'], **far}


def old_benchmark(cfg: dict, output: Path, source: dict) -> dict:
    """Time/reproduce scores on old development cases and synthetic outcomes; no fresh arrays are read."""
    cases = list_cases(source, splits=['dev'])
    cases.sort(key=lambda c: int(np.prod(nib.load(c.image_path).shape)))
    settings = cfg['fresh']['benchmark']
    positions = np.linspace(0, len(cases)-1, settings['n_cases']).round().astype(int)
    times, actual, dice_checks = [], [], []
    source_folder = source['paths']['predictions_dir'] / source['name'] / 'clean'
    for i in positions:
        case = cases[i]
        paths = {'ct': str(case.image_path)}
        paths.update({kind: str(source_folder / f'{case.case_id}_{kind}.nii.gz') for kind in cfg['fresh']['required_maps']})
        begin = time.monotonic()
        actual.append({'case_id': case.case_id, **case_scores(paths, cfg)})
        score_seconds = time.monotonic()-begin
        begin = time.monotonic()
        labels = np.asanyarray(load_canonical(case.label_path).dataobj)
        truth = np.isin(labels, source['dataset']['organ_definitions'][source['dataset']['organ_definition']]['organ_labels'])
        saved_mask = np.asanyarray(nib.load(paths['mask']).dataobj).astype(bool)
        dice_checks.append({'case_id': case.case_id, 'dice_3mm': dice(saved_mask, truth),
                            'dice_6mm': dice(np.asanyarray(nib.load(paths['m2_mask']).dataobj).astype(bool), truth)})
        times.append({'case_id': case.case_id, 'grid_voxels': int(np.prod(nib.load(case.image_path).shape)),
                      'seconds': score_seconds, 'dice_seconds': time.monotonic()-begin})
    # Reproduction against saved development scores catches pipeline drift before using fresh data.
    saved = pd.read_csv(REPO_ROOT / 'results/kits_missed_checks/exploratory_features.csv')
    comparison = pd.DataFrame(actual).merge(saved, on='case_id', suffixes=('_actual', '_saved'), validate='one_to_one')
    for name in actual[0]:
        atol = cfg['fresh']['old_far_rounding_atol'] if name == 'far_disagree_per_volume' else 1e-9
        if name != 'case_id' and not np.allclose(comparison[name+'_actual'], comparison[name+'_saved'], rtol=1e-6, atol=atol):
            raise ValueError(f'Old-data score reproduction failed: {name}')
    reference_dice = pd.read_csv(source['paths']['results_dir'] / source['name'] / 'model_dice_3mm_vs_6mm.csv')
    checked = pd.DataFrame(dice_checks).merge(reference_dice, on='case_id', suffixes=('_actual', '_saved'), validate='one_to_one')
    for column in ['dice_3mm', 'dice_6mm']:
        if not np.allclose(checked[column+'_actual'], checked[column+'_saved'], rtol=0, atol=cfg['fresh']['old_dice_rounding_atol']):
            raise ValueError('Old-data Dice reproduction failed')
    small = deepcopy(cfg)
    small['evaluation']['n_bootstrap'] = settings['n_bootstrap']
    rng = np.random.default_rng(cfg['seed'])
    synthetic = pd.DataFrame({name: rng.random(cfg['fresh']['n_cases']) for name in cfg['checks']['methods']})
    synthetic['dice'] = np.linspace(.5, .99, len(synthetic))
    synthetic['case_id'] = [f'synthetic_{i}' for i in range(len(synthetic))]
    begin = time.monotonic()
    evaluate_scores(synthetic, small)
    budget_statistics(synthetic, small)
    queue_changes(synthetic, small)
    elapsed = time.monotonic()-begin
    pd.DataFrame(times).to_csv(output / 'old_benchmark_cases.csv', index=False)
    result = {'old_scores_reproduced': True, 'fresh_arrays_read': False,
              'seconds_per_grid_voxel': sum(x['seconds'] for x in times)/sum(x['grid_voxels'] for x in times),
              'dice_seconds_per_grid_voxel': sum(x['dice_seconds'] for x in times)/sum(x['grid_voxels'] for x in times),
              'evaluation_projected_seconds': elapsed * cfg['evaluation']['n_bootstrap']/settings['n_bootstrap'] * settings['safety_factor'],
              'safety_factor': settings['safety_factor']}
    write_json(output / 'runtime_benchmark.json', result)
    return result


def check_runtime(cfg: dict, output: Path, inventory: pd.DataFrame, stage: str) -> None:
    """Require a measured stage projection below the configured 30-minute cap."""
    timing = json.loads((output / 'runtime_benchmark.json').read_text())
    if stage == 'evaluate':
        seconds = (timing['evaluation_projected_seconds'] + timing['dice_seconds_per_grid_voxel'] *
                   inventory.grid_voxels.sum() * timing['safety_factor'])
    else:
        seconds = timing['seconds_per_grid_voxel'] * inventory.grid_voxels.sum() * timing['safety_factor']
    if seconds >= cfg['fresh']['benchmark']['max_stage_seconds']:
        raise ValueError(f'Projected {stage} takes {seconds:.1f} s; ask before a longer run')


def verify_inventory(inventory: pd.DataFrame, columns: list[str]) -> None:
    """Check the exact input bytes before using arrays in a scientific stage."""
    for row in inventory.to_dict('records'):
        for key in columns:
            if digest(Path(row[key])) != row[key + '_sha256']:
                raise ValueError(f'Changed input: {row["case_id"]} {key}')


def score_fresh(cfg: dict, output: Path, inventory: pd.DataFrame) -> dict:
    """Save all locked fresh scores once, using no GT voxels, before the outcome-stage marker exists."""
    path = output / 'scores.csv'
    if path.exists() or (output / 'evaluation_started.json').exists():
        raise ValueError('Scores/outcomes already exist; do not silently replace fresh results')
    verify_inventory(inventory, ['ct', *cfg['fresh']['required_maps']])
    rows = []
    for row in inventory.to_dict('records'):
        rows.append({'case_id': row['case_id'], **case_scores(row, cfg)})
        print(f'Scored {row["case_id"]}', flush=True)
    scores = build_combinations(pd.DataFrame(rows), cfg['combinations'], cfg['fresh']['rank_lattice_atol'])
    scores.to_csv(path, index=False)
    info = {'scores_sha256': digest(path), 'lock_sha256': digest(output / 'lock.json'),
            'input_inventory_sha256': digest(output / 'input_inventory.csv'), 'label_voxels_read': False,
            'n_cases': len(scores), 'saved_at': now()}
    write_json(output / 'scores_provenance.json', info)
    return info


def consume_outcome_stage(path: Path, payload: dict) -> None:
    """Create an exclusive one-time outcome marker; an existing marker requires logged technical recovery."""
    with path.open('x') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


def budget_statistics(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Report fixed-budget capture/quality and paired bootstrap differences for the prespecified controls."""
    n = len(frame)
    k = math.ceil(cfg['checks']['review_fraction'] * n)
    outcome = frame.dice.to_numpy()
    bad = outcome < cfg['checks']['bad_dice_threshold']
    indices = bootstrap_indices(n, cfg['evaluation']['n_bootstrap'], np.random.default_rng(cfg['seed']))
    stats = {}
    for method in cfg['checks']['methods']:
        score = frame[method].to_numpy()
        def measure(index):
            """Return bad capture and simulated post-correction quality for one resampled case set."""
            weights = inclusion_probabilities(score[index], k)
            capture = float(weights @ bad[index])
            return np.array([capture, k-capture, float((outcome[index] + weights*(1-outcome[index])).mean())])
        stats[method] = (measure(np.arange(n)), np.array([measure(i) for i in indices]))
    rows = []
    base = cfg['checks']['primary']
    tail = (1-cfg['evaluation']['ci_level'])/2*100
    for method, (full, boot) in stats.items():
        for column, name in enumerate(['bad_capture', 'false_alarms', 'quality_after_correction']):
            diff = boot[:, column]-stats[base][1][:, column]
            low, high = np.percentile(diff, [tail, 100-tail])
            rows.append({'method': method, 'measure': name, 'review_budget': k, 'value': full[column],
                         'difference_to_g1': full[column]-stats[base][0][column],
                         'difference_ci_low': low, 'difference_ci_high': high})
    return pd.DataFrame(rows)


def primary_decisions(paired: pd.DataFrame, bad_count: int, n_cases: int, cfg: dict) -> list[dict]:
    """Apply only the two frozen per-comparison criteria; absent discrimination is not an improvement."""
    decisions = []
    for method in [cfg['combinations'][0]['name'], 'far_disagree_per_volume']:
        row = paired.loc[paired.method.eq(method)&paired.measure.eq('review_auc')&paired.bad_rule.eq('dice_below_083')].iloc[0]
        informative = 0 < bad_count < n_cases
        decisions.append({'method': method, 'difference': None if pd.isna(row['diff']) else float(row['diff']),
                          'ci_low': None if pd.isna(row.diff_ci_low) else float(row.diff_ci_low),
                          'ci_high': None if pd.isna(row.diff_ci_high) else float(row.diff_ci_high),
                          'primary_informative': informative,
                          'meets_per_comparison_criterion': bool(informative and
                                                               row.diff_ci_low > cfg['fresh']['comparison_atol'])})
    return decisions


def evaluate_fresh(cfg: dict, output: Path, inventory: pd.DataFrame) -> dict:
    """Open all fresh outcomes in one joint, fixed analysis after verifying the frozen score table."""
    info = json.loads((output / 'scores_provenance.json').read_text())
    if info['scores_sha256'] != digest(output / 'scores.csv') or info['lock_sha256'] != digest(output / 'lock.json'):
        raise ValueError('Frozen score table/lock changed')
    if info['input_inventory_sha256'] != digest(output / 'input_inventory.csv'):
        raise ValueError('Input inventory changed after scoring')
    verify_inventory(inventory, ['mask', 'm2_mask', 'label'])
    consume_outcome_stage(output / 'evaluation_started.json', {'started_at': now(), 'lock_sha256': info['lock_sha256'],
                                                             'scores_sha256': info['scores_sha256']})
    outcomes = []
    for row in inventory.to_dict('records'):
        truth = np.isin(np.asanyarray(load_canonical(Path(row['label'])).dataobj), cfg['dataset']['organ_definitions'][cfg['dataset']['organ_definition']]['organ_labels'])
        mask = np.asanyarray(nib.load(row['mask']).dataobj).astype(bool)
        second = np.asanyarray(nib.load(row['m2_mask']).dataobj).astype(bool)
        outcomes.append({'case_id': row['case_id'], 'dice': dice(mask, truth), 'dice_6mm': dice(second, truth)})
        pd.DataFrame(outcomes).to_csv(output / 'dice.csv', index=False)
    scores = pd.read_csv(output / 'scores.csv')
    frame = scores.merge(pd.DataFrame(outcomes), on='case_id', validate='one_to_one')
    metrics, paired, curves = evaluate_scores(frame, cfg)
    metrics['status'] = STATUS
    metrics.to_csv(output / 'metrics.csv', index=False)
    paired.to_csv(output / 'paired.csv', index=False)
    curves.to_csv(output / 'curves.csv', index=False)
    budget_statistics(frame, cfg).to_csv(output / 'budget_statistics.csv', index=False)
    changes, tradeoffs = queue_changes(frame, cfg)
    changes.to_csv(output / 'queue_changes.csv', index=False)
    tradeoffs.to_csv(output / 'tradeoffs.csv', index=False)
    bad_count = int(frame.dice.lt(cfg['checks']['bad_dice_threshold']).sum())
    decisions = primary_decisions(paired, bad_count, len(frame), cfg)
    # Count finite primary resamples explicitly, including those without a bad case.
    ids = bootstrap_indices(len(frame), cfg['evaluation']['n_bootstrap'], np.random.default_rng(cfg['seed']))
    bad = frame.dice.lt(cfg['checks']['bad_dice_threshold']).to_numpy()
    summary = {'status': 'joint outcome stage completed; per-comparison CIs, no familywise claim',
               'n_cases': len(frame), 'n_bad': bad_count, 'decisions': decisions,
               'finite_primary_bootstrap_samples': sum(bool(bad[i].any()) for i in ids), 'completed_at': now()}
    write_json(output / 'evaluation_completed.json', summary)
    return summary


def main() -> None:
    """Run only the requested frozen stage; missing coordination never opens fresh headers or outcomes."""
    parser = config_arg_parser(__doc__)
    parser.add_argument('--stage', required=True, choices=['freeze', 'benchmark', 'preflight', 'score', 'evaluate'])
    parser.add_argument('--handoff', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    cfg = load_config(args.config)
    setup_compute(cfg)
    output, source = experiment(cfg)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'freeze':
        result = freeze(cfg, args.config, output, source)
    else:
        lock = require_lock(cfg, args.config, output, source)
        if args.stage == 'benchmark':
            result = old_benchmark(cfg, output, source)
        elif args.stage == 'preflight':
            result = preflight(cfg, output, lock, args.handoff)
        else:
            inventory = require_ready(output)
            check_runtime(cfg, output, inventory, args.stage)
            result = score_fresh(cfg, output, inventory) if args.stage == 'score' else evaluate_fresh(cfg, output, inventory)
    info = machine_info()
    info.update(date=now(), execution_device='cpu', seed=cfg['seed'], threads=cfg['compute']['num_threads'],
                stage=args.stage, seconds=time.monotonic()-start, scientific_lock_sha256=digest(output/'lock.json'),
                fresh_label_voxels_read=args.stage=='evaluate')
    if Path('/proc/cpuinfo').exists():
        info['cpu_model'] = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                                  if s.startswith('model name')), info['cpu'])
    write_json(output / f'{args.stage}_machine.json', info)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
