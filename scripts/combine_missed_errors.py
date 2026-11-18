"""Compare six predeclared G1 + anatomical-check combinations (missed-errors step 4).

Build rank scores from the committed GT-free step 3 table before attaching saved outcomes.
Reuse the unchanged G1 engine and isolated step 3 helpers; write aggregate tables only.
This is development selection, with no fresh-pool evaluation or final prospective lock.
"""

import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

from segreview.config import REPO_ROOT, config_arg_parser, load_config, setup_compute
from segreview.evaluation import combine_scores
from segreview.machine import machine_info

# Task helpers live outside the installed shared package; keep both direct CLI and test imports usable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.check_missed_errors import (budget_tables, digest, evaluate_scores, exact_table,
                                         inclusion_probabilities, settings_digest)

STATUS = 'exploratory combinations on kits100 development data; no fresh-pool test or lock'


def build_combinations(features: pd.DataFrame, specs: list[dict], tolerance: float) -> pd.DataFrame:
    """Add GT-free G1 rank combinations and preserve mathematically exact ties on their rank lattice."""
    if {'dice', 'bad', 'missed_bad', 'truth_voxels'}.intersection(features.columns):
        raise ValueError('Do not attach ground-truth outcomes before building scores')
    frame = features.copy()
    original = {name: frame[name].to_numpy(dtype=float) for spec in specs for name in spec['scores']}
    if not len(frame) or not all(np.isfinite(x).all() for x in original.values()):
        raise ValueError('Need nonempty finite score columns')
    for spec in specs:
        if spec['name'] in frame.columns:
            raise ValueError('Combination names must be unique and distinct from base columns')
        values = combine_scores(original, spec)
        # Average tied ranks are half-integers. This removes floating-point noise, not real differences.
        scale = 2 * len(frame) * (len(spec['scores']) if spec['rule'] == 'mean_rank' else 1)
        ticks = np.rint(values * scale)
        if not np.allclose(values * scale, ticks, rtol=0, atol=tolerance):
            raise ValueError('Unexpected values outside the exact rank lattice')
        frame[spec['name']] = ticks / scale
    return frame


def choose_recommendation(metrics: pd.DataFrame, budgets: pd.DataFrame, cfg: dict) -> dict:
    """Choose at most one development candidate by the predeclared full-cohort AUC/budget rule."""
    selection = cfg['selection']
    if selection['eligibility'] != 'higher_review_auc_and_no_fewer_bad_in_budget_than_locked':
        raise ValueError('Unsupported eligibility rule')
    if selection['order'] != ['review_auc_descending', 'quality_auc_descending', 'config_order']:
        raise ValueError('Unsupported selection order')
    if selection['max_recommendations'] != 1:
        raise ValueError('This screen recommends at most one candidate')
    base = cfg['checks']['primary']
    table = metrics.loc[metrics.bad_rule.eq(selection['bad_rule'])].set_index('method')
    budget = budgets.set_index('method')
    if table.index.duplicated().any() or budget.index.duplicated().any():
        raise ValueError('Selection inputs must have unique method rows')
    eps = selection['comparison_atol']
    eligible = []
    considered = []
    for order, spec in enumerate(cfg['combinations']):
        name = spec['name']
        row = table.loc[name]
        gain = float(row.review_auc - table.loc[base, 'review_auc'])
        count = float(budget.loc[name, 'expected_bad_found'])
        ok = gain > eps and count >= budget.loc[base, 'expected_bad_found'] - eps
        considered.append({'method': name, 'eligible': bool(ok), 'review_auc_gain': gain,
                           'expected_bad_found': count})
        if ok:
            eligible.append((float(row.review_auc), float(row.quality_auc), order, name))
    chosen = None
    if eligible:
        best_auc = max(x[0] for x in eligible)
        tied = [x for x in eligible if abs(x[0]-best_auc) <= eps]
        best_quality = max(x[1] for x in tied)
        chosen = min((x for x in tied if abs(x[1]-best_quality) <= eps), key=lambda x: x[2])[3]
    return {'status': 'provisional development recommendation; not a fresh-test lock',
            'selected': chosen, 'considered': considered,
            'reason': 'highest eligible review-AUC, then quality-AUC, then config order' if chosen else
                      'no combination improved review-AUC without losing top-budget bad-case capture'}


def queue_changes(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Account for every new/lost bad or nonbad inclusion relative to locked G1 at the fixed budget."""
    settings = cfg['checks']
    budget = math.ceil(len(frame) * settings['review_fraction'])
    base = inclusion_probabilities(frame[settings['primary']].to_numpy(), budget)
    bad = frame.dice.lt(settings['bad_dice_threshold']).to_numpy()
    rows, summaries = [], []
    for spec in cfg['combinations']:
        name = spec['name']
        prob = inclusion_probabilities(frame[name].to_numpy(), budget)
        delta = prob-base
        ranks = rankdata(-frame[name].to_numpy(), method='average')
        for i, case in enumerate(frame.itertuples()):
            rows.append({'method': name, 'case_id': case.case_id, 'dice': case.dice, 'bad': bool(bad[i]),
                         'fair_rank': ranks[i], 'locked_top_probability': base[i], 'top_probability': prob[i],
                         'inclusion_change': delta[i]})
        summaries.append({'method': name, 'new_bad': float(np.maximum(delta[bad], 0).sum()),
                          'lost_bad': float(np.maximum(-delta[bad], 0).sum()),
                          'new_false_alarms': float(np.maximum(delta[~bad], 0).sum()),
                          'removed_false_alarms': float(np.maximum(-delta[~bad], 0).sum())})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, Path, dict]:
    """Read verified GT-free aggregate columns and protect all source folders; no scan data is needed."""
    settings = cfg['checks']
    source = load_config(REPO_ROOT / settings['source_config'])
    output = (cfg['paths']['results_dir'] / cfg['name']).resolve()
    protected = [(source['paths'][key] / source['name']).resolve() for key in ['results_dir', 'predictions_dir']]
    protected += [(REPO_ROOT / settings[key]).resolve().parent for key in ['source_features', 'inspection_queue']]
    for path in protected:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output must not overlap a read-only source folder')
    feature_path = REPO_ROOT / settings['source_features']
    proof = json.loads((REPO_ROOT / settings['source_verification']).read_text())
    if not proof['passed'] or digest(feature_path) != proof['output_hashes'][feature_path.name]:
        raise ValueError('Source feature table differs from the verified step 3 result')
    source_scores = source['paths']['results_dir'] / source['name'] / 'scores.csv'
    ids = pd.read_csv(source_scores, usecols=['case_id', 'variant', 'split'])
    ids = ids.loc[ids.variant.eq(settings['variant'])]
    expected = set(ids.case_id)
    if len(ids) != len(expected) or len(ids) != cfg['dataset']['n_cases'] or cfg['seed'] != source['seed']:
        raise ValueError('Source cohort count/seed differs')
    cols = ['case_id', 'split', *settings['base_methods']]
    frame = exact_table(pd.read_csv(feature_path, usecols=cols), expected, 'features')
    ids = exact_table(ids, expected, 'source')
    if not np.array_equal(frame['split'], ids['split']):
        raise ValueError('Original split labels differ')
    return frame, output, source


def source_hashes(cfg: dict, config_path: Path, include_outcomes: bool) -> dict:
    """Fingerprint the config, scripts, shared engine and aggregate inputs used by this stage."""
    settings = cfg['checks']
    source = load_config(REPO_ROOT / settings['source_config'])
    paths = [config_path.resolve(), Path(__file__).resolve(), REPO_ROOT / 'scripts/check_missed_errors.py',
             REPO_ROOT / 'src/segreview/evaluation.py', REPO_ROOT / 'uv.lock']
    paths += [REPO_ROOT / settings[k] for k in ['source_config', 'source_features', 'source_verification']]
    paths += [source['paths']['results_dir'] / source['name'] / 'scores.csv']
    if include_outcomes:
        paths += [source['paths']['results_dir'] / source['name'] / 'dice.csv', REPO_ROOT / settings['inspection_queue']]
    return {str(p.relative_to(REPO_ROOT)): digest(p) for p in paths}


def main() -> None:
    """Benchmark synthetic evaluation or compare the fixed family, saving scores before real outcomes."""
    parser = config_arg_parser(__doc__)
    parser.add_argument('--stage', required=True, choices=['benchmark', 'evaluate'])
    args = parser.parse_args()
    start = time.monotonic()
    cfg = load_config(args.config)
    settings = cfg['checks']
    settings['methods'] = settings['base_methods'] + [s['name'] for s in cfg['combinations']]
    if len(set(settings['methods'])) != len(settings['methods']):
        raise ValueError('All base/combination names must be unique')
    setup_compute(cfg)
    features, output, source = load_inputs(cfg)
    output.mkdir(parents=True, exist_ok=True)
    settings_sha = settings_digest(cfg)
    if args.stage == 'benchmark':
        small = deepcopy(cfg)
        small['evaluation']['n_bootstrap'] = settings['benchmark']['n_bootstrap']
        rng = np.random.default_rng(cfg['seed'])
        synthetic = pd.DataFrame({name: rng.random(len(features)) for name in settings['base_methods']})
        synthetic['plaus_missing_side'] = rng.integers(0, 3, len(features))
        synthetic = build_combinations(synthetic, cfg['combinations'], settings['rank_lattice_atol'])
        synthetic['dice'] = np.linspace(.5, .99, len(features))
        begin = time.monotonic()
        evaluate_scores(synthetic, small)
        elapsed = time.monotonic()-begin
        estimate = elapsed * cfg['evaluation']['n_bootstrap'] / small['evaluation']['n_bootstrap'] * settings['benchmark']['safety_factor']
        projection = {'synthetic_evaluation_seconds': elapsed, 'projected_seconds': estimate,
                      'stage_cap_seconds': settings['benchmark']['max_stage_seconds'],
                      'within_cap': estimate < settings['benchmark']['max_stage_seconds'],
                      'settings_sha256': settings_sha, 'source_hashes': source_hashes(cfg, args.config, False)}
        (output / 'exploratory_runtime_projection.json').write_text(json.dumps(projection, indent=2)+'\n')
        (output / 'run_config.yaml').write_bytes(args.config.read_bytes())
        print(json.dumps(projection, indent=2), flush=True)
    else:
        projection = json.loads((output / 'exploratory_runtime_projection.json').read_text())
        if not projection['within_cap'] or projection['settings_sha256'] != settings_sha:
            raise ValueError('Need a matching benchmark below the stage cap; ask before longer runs')
        if projection['source_hashes'] != source_hashes(cfg, args.config, False):
            raise ValueError('Code/config/inputs changed since benchmark')
        scores = build_combinations(features, cfg['combinations'], settings['rank_lattice_atol'])
        scores['status'] = STATUS
        score_path = output / 'exploratory_combined_scores.csv'
        scores.to_csv(score_path, index=False)
        (output / 'exploratory_scores_provenance.json').write_text(json.dumps({
            'outcomes_attached': False, 'input_columns': features.columns.tolist(),
            'score_table_sha256': digest(score_path), 'source_hashes': source_hashes(cfg, args.config, False),
            'rank_ties': 'exact rational lattice; shared combine_scores used before canonicalization'}, indent=2)+'\n')
        folder = source['paths']['results_dir'] / source['name']
        outcomes = pd.read_csv(folder / 'dice.csv', usecols=['case_id', 'variant', 'dice'])
        outcomes = exact_table(outcomes.loc[outcomes.variant.eq(settings['variant'])], set(scores.case_id), 'Dice')
        frame = scores.merge(outcomes[['case_id', 'dice']], on='case_id', validate='one_to_one')
        inspection = exact_table(pd.read_csv(REPO_ROOT / settings['inspection_queue']), set(frame.case_id), 'inspection')
        if not np.allclose(frame.dice, inspection.dice, atol=1e-12, rtol=0):
            raise ValueError('Saved Dice differs from inspection')
        metrics, paired, curves = evaluate_scores(frame, cfg)
        metrics['status'] = STATUS
        metrics.to_csv(output / 'exploratory_metrics.csv', index=False)
        paired.to_csv(output / 'exploratory_paired.csv', index=False)
        curves.to_csv(output / 'exploratory_curves.csv', index=False)
        budgets, targets = budget_tables(frame, inspection, cfg)
        budgets['status'], targets['status'] = STATUS, STATUS
        budgets.to_csv(output / 'exploratory_budget.csv', index=False)
        targets.to_csv(output / 'exploratory_miss_ranks.csv', index=False)
        changes, tradeoffs = queue_changes(frame, cfg)
        changes.to_csv(output / 'exploratory_queue_changes.csv', index=False)
        tradeoffs.to_csv(output / 'exploratory_tradeoffs.csv', index=False)
        recommendation = choose_recommendation(metrics, budgets, cfg)
        (output / 'exploratory_recommendation.json').write_text(json.dumps(recommendation, indent=2)+'\n')
        frame.to_csv(output / 'exploratory_evaluation_cases.csv', index=False)
        print(json.dumps(recommendation, indent=2), flush=True)
        print(budgets.drop(columns='status').round(4).to_string(index=False), flush=True)
    info = machine_info()
    info.update(date=datetime.now(ZoneInfo('Europe/Copenhagen')).isoformat(timespec='seconds'),
                execution_device='cpu', threads=cfg['compute']['num_threads'], seed=cfg['seed'], scipy=scipy.__version__,
                status=STATUS, stage=args.stage, seconds=time.monotonic()-start,
                actual_outcomes_used=args.stage=='evaluate', image_or_label_maps_read=False,
                source_hashes=source_hashes(cfg, args.config, args.stage=='evaluate'))
    if Path('/proc/cpuinfo').exists():
        info['cpu_model'] = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                                  if s.startswith('model name')), info['cpu'])
    (output / f'exploratory_{args.stage}_machine.json').write_text(json.dumps(info, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
