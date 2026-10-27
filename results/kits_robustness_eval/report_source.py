"""Verify completed step 5 artifacts and extract key results without rerunning inference, scoring, or bootstraps.

From the repository root: PYTHONPATH=src:. .venv-gpu/bin/python
results/kits_robustness_eval/report_source.py --config configs/kits_robustness_eval.yaml
"""

import json

import numpy as np
import pandas as pd

from scripts.evaluate_robustness import digest, source_hashes, validate_scope, write_json
from segreview.config import REPO_ROOT, config_arg_parser, load_config, results_dir
from segreview.data import case_split
from segreview.io import prediction_dir
from segreview.pipeline import variants


def main():
    """Require complete maps/tables and unchanged provenance, then save an audit and a small test-results extract."""
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    cases, _ = validate_scope(cfg)
    out = results_dir(cfg)
    plan = json.loads((out / 'analysis_plan.json').read_text())
    assert plan['config_sha256'] == digest(args.config)
    assert plan['source_sha256'] == source_hashes()
    for stage in ['prepare', 'benchmark', 'evaluate']:
        info = json.loads((out / f'{stage}_machine.json').read_text())
        assert info['status'] == 'complete'
        assert info['config_sha256'] == plan['config_sha256']
        assert info['source_sha256'] == plan['source_sha256']
    ids = [c.case_id for c in cases]
    assert ids == plan['case_ids']
    expected = {(c.case_id, v, case_split(cfg, c.case_id)) for c in cases for v in variants(cfg)}
    scores = pd.read_csv(out / 'scores.csv')
    dice = pd.read_csv(out / 'dice.csv')
    methods = cfg['robustness_evaluation']['methods']
    for frame in [scores, dice]:
        assert len(frame) == len(expected)
        assert set(zip(frame.case_id, frame.variant, frame['split'])) == expected
    assert np.isfinite(scores[methods].to_numpy()).all()
    assert dice[['dice', 'dice_second_model']].apply(lambda c: c.between(0, 1).all()).all()
    model_dice = pd.read_csv(out / 'dice_per_model.csv')
    resolutions = [cfg['model']['resolution'], cfg['uncertainty']['second_model_resolution']]
    expected_models = {(c, v, s, r) for c, v, s in expected for r in resolutions}
    assert len(model_dice) == len(expected_models)
    assert set(zip(model_dice.case_id, model_dice.variant, model_dice['split'], model_dice.resolution)) == expected_models
    paired = pd.read_csv(out / 'paired_changes.csv')
    expected_pairs = {(c, v, s) for c, v, s in expected if v != 'clean'}
    assert len(paired) == len(expected_pairs)
    assert set(zip(paired.case_id, paired.variant, paired['split'])) == expected_pairs
    metrics = pd.read_csv(out / 'g1_metrics.csv')
    expected_metrics = {(f'{v}_{s}', r['name'], m) for s in cfg['evaluation']['splits']
                        for v in [*variants(cfg), 'mixed'] for r in cfg['evaluation']['bad_rules']
                        for m in [*methods, 'oracle', 'random']}
    assert len(metrics) == len(expected_metrics)
    assert set(zip(metrics['set'], metrics.bad_rule, metrics.method)) == expected_metrics
    assert metrics.review_auc.between(0, 1).all() and metrics.quality_auc.between(0, 1).all()
    mixed = metrics[metrics.variant == 'mixed']
    assert mixed.rho_p_value.isna().all() and mixed.bootstrap_unit.eq('case_id').all()
    for split in cfg['evaluation']['splits']:
        n = sum(case_split(cfg, c.case_id) == split for c in cases)
        frame = metrics[metrics['split'] == split]
        assert frame.n_patients.eq(n).all()
        assert frame[frame.variant == 'mixed'].n_rows.eq(n * len(variants(cfg))).all()
    manifest = pd.read_csv(out / 'prediction_manifest.csv')
    expected_maps = {(c, v, k) for c, v, _ in expected for k in ['mask', 'prob', 'm2_mask', 'm2_prob']}
    assert len(manifest) == len(expected_maps)
    assert set(zip(manifest.case_id, manifest.variant, manifest.kind)) == expected_maps
    assert len(list(prediction_dir(cfg).parent.rglob('*.nii.gz'))) == len(manifest)
    for row in manifest.itertuples():
        assert digest(REPO_ROOT / row.path) == row.sha256
    imported = pd.read_csv(out / 'imported_predictions.csv')
    for row in imported.itertuples():
        assert digest(REPO_ROOT / row.source_path) == row.sha256
    write_json(out / 'post_run_checks.json', {
        'status': 'passed', 'config_source_and_stage_hashes_match_pre_outcome_plan': True,
        'n_cases': len(cases), 'n_score_rows': len(scores), 'n_model_dice_rows': len(model_dice),
        'n_paired_rows': len(paired), 'n_metric_rows': len(metrics), 'n_prediction_files': len(manifest),
        'prediction_bytes': int(manifest.bytes.sum()), 'n_archived_inputs_unchanged': len(imported),
        'mixed_bootstrap_unit': 'case_id', 'mixed_independent_row_rho_p_values_omitted': True})
    degradation = pd.read_csv(out / 'dice_degradation.csv')
    correlations = pd.read_csv(out / 'change_correlations.csv')
    tops = pd.read_csv(out / 'queue_top_summary.csv')
    threshold = cfg['robustness_evaluation']['bad_rule']
    curves = pd.read_csv(out / 'g1_curves.csv')
    top_review = curves[(curves['set'] == 'mixed_test') & (curves.bad_rule == threshold)
                        & np.isclose(curves.reviewed_fraction, cfg['robustness_evaluation']['queue_top_fraction'])].copy()
    bad_counts = metrics[(metrics['set'] == 'mixed_test') & (metrics.bad_rule == threshold)].set_index('method').n_bad
    top_review['expected_bad_found'] = top_review.review_recall * top_review.method.map(bad_counts)
    extract = {'scope': 'Locked lowres_8mm; 80 test cases; task threshold; secondary entropy remains secondary',
               'test_dice': degradation[degradation['split'] == 'test'].to_dict('records'),
               'test_change_correlations': correlations[correlations['split'] == 'test'].to_dict('records'),
               'test_top_queue': tops[tops['split'] == 'test'].to_dict('records'),
               'test_mixed_top_review': top_review.to_dict('records'),
               'test_queue_metrics': metrics[(metrics['split'] == 'test') & (metrics.bad_rule == threshold)]
                   [['set', 'method', 'n_bad', 'review_auc', 'review_auc_ci_low', 'review_auc_ci_high',
                     'review_auc_minus_volume', 'review_auc_minus_volume_ci_low', 'review_auc_minus_volume_ci_high']]
                   .to_dict('records')}
    # CSV NaNs denote undefined fields (e.g. a random-vs-volume bootstrap CI); JSON uses null.
    clean_json = json.loads(json.dumps(extract), parse_constant=lambda _: None)
    write_json(out / 'key_results.json', clean_json)
    print((out / 'post_run_checks.json').read_text())


if __name__ == '__main__':
    main()
