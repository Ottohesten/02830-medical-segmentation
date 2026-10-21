"""Check paired robustness signs, split isolation, patient bootstrap, and GT-free scoring with synthetic arrays."""

import numpy as np
import pandas as pd
import pytest
import nibabel as nib

from scripts.evaluate_robustness import score_prediction, validate_scope
from segreview.config import REPO_ROOT, load_config
from segreview.evaluation import bootstrap_indices, evaluate_methods, expected_top_k_sums
from segreview.io import map_path, save_map
from segreview.robustness_evaluation import change_summaries, evaluate_queues, paired_changes
from segreview.uncertainty import scan_scores


@pytest.fixture
def example():
    """Return four patients with known deterioration and opposite primary/secondary score responses."""
    cfg = {'seed': 42, 'model': {'resolution': 'fast'}, 'uncertainty': {'second_model_resolution': 'fastest'},
           'perturbations': [{'name': 'lowres', 'type': 'lowres', 'strength': 8}],
           'evaluation': {'n_bootstrap': 20, 'n_random_orders': 20, 'ci_level': .95,
                          'bad_rules': [{'name': 'bad', 'rule': 'threshold', 'value': .83}]},
           'robustness_evaluation': {'methods': ['primary', 'secondary', 'neg_volume_ml'],
                                     'bad_rule': 'bad', 'queue_top_fraction': .5}}
    scores, dice_rows = [], []
    for i, (clean, drop) in enumerate(zip([.9, .85, .8, .9], [.02, .06, .05, .1])):
        for variant in ['clean', 'lowres']:
            changed = variant != 'clean'
            keys = {'case_id': f'case{i}', 'variant': variant, 'split': 'dev' if i < 2 else 'test'}
            scores.append({**keys, 'primary': .2 + i*.05 + drop * changed,
                           'secondary': .4 - drop * changed, 'neg_volume_ml': -100 + 10 * changed})
            dice_rows.append({**keys, 'dice': round(clean - drop * changed, 4),
                              'dice_second_model': round(clean - .01 - drop * changed, 4)})
    return cfg, pd.DataFrame(scores), pd.DataFrame(dice_rows)


def test_patient_bootstrap_keeps_both_views_and_legacy_draws_unchanged():
    """A sampled patient contributes both rows every time; ungrouped draws retain the old seeded sequence."""
    groups = np.array(['a', 'b', 'c', 'a', 'b', 'c'])
    draws = bootstrap_indices(6, 40, np.random.default_rng(42), groups)
    for draw in draws:
        assert len(draw) == 6
        for first, second in [(0, 3), (1, 4), (2, 5)]:
            assert (draw == first).sum() == (draw == second).sum()
    rng = np.random.default_rng(42)
    expected = [rng.integers(0, 6, size=6) for _ in range(4)]
    actual = bootstrap_indices(6, 4, np.random.default_rng(42))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='group ID'):
        bootstrap_indices(6, 4, np.random.default_rng(42), groups[:3])


def test_paired_signs_bad_counts_and_split_separation(example):
    """Positive Dice loss and positive primary change support detection; test patients never enter dev summaries."""
    cfg, scores, dice_rows = example
    paired = paired_changes(scores, dice_rows, cfg)
    np.testing.assert_allclose(paired.dice_drop, [.02, .06, .05, .1])
    np.testing.assert_allclose(paired.primary_change, paired.dice_drop)
    degradation, correlations = change_summaries(paired, cfg)
    assert len(degradation) == 4
    dev = degradation[(degradation['split'] == 'dev') & (degradation.resolution == 'fast')].iloc[0]
    assert dev.n_cases == 2 and dev.n_clean_bad == 0 and dev.n_perturbed_bad == dev.n_new_bad == 1
    assert dev.median_paired_drop == pytest.approx(.04)
    np.testing.assert_allclose(correlations[correlations.method == 'primary'].rho_dice_drop_vs_score_rise, 1)
    np.testing.assert_allclose(correlations[correlations.method == 'secondary'].rho_dice_drop_vs_score_rise, -1)
    assert correlations[correlations.method == 'neg_volume_ml'].rho_dice_drop_vs_score_rise.isna().all()


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'nan', 'dice_range', 'cross_split'])
def test_rejects_incomplete_invalid_or_cross_split_pairs(example, problem):
    """Never produce a valid-looking paired analysis from incomplete, invalid, or split-mismatched rows."""
    cfg, scores, dice_rows = example
    if problem == 'missing':
        scores = scores.iloc[:-1]
    elif problem == 'duplicate':
        scores = pd.concat([scores, scores.iloc[:1]], ignore_index=True)
    elif problem == 'nan':
        scores.loc[0, 'primary'] = np.nan
    elif problem == 'dice_range':
        dice_rows.loc[0, 'dice'] = 1.1
    else:
        scores.loc[0, 'split'] = 'test'
        dice_rows.loc[0, 'split'] = 'test'
    with pytest.raises(ValueError):
        paired_changes(scores, dice_rows, cfg)


def test_mixed_queue_uses_existing_engine_and_tie_fair_top_k(example):
    """All six queues remain split-specific; mixed full-sample AUC matches G1 and dependent-row p-values are absent."""
    cfg, scores, dice_rows = example
    metrics, curves, shifts, tops = evaluate_queues(scores, dice_rows, cfg)
    assert set(metrics['set']) == {'clean_dev', 'lowres_dev', 'mixed_dev', 'clean_test', 'lowres_test', 'mixed_test'}
    mixed = metrics[metrics['set'] == 'mixed_dev'].set_index('method')
    assert mixed.n_rows.eq(4).all() and mixed.n_patients.eq(2).all()
    assert mixed.rho_p_value.isna().all()
    group = scores[scores['split'] == 'dev'].sort_values(['case_id', 'variant'])
    truths = dice_rows[dice_rows['split'] == 'dev'].sort_values(['case_id', 'variant']).dice.to_numpy()
    direct, _ = evaluate_methods({m: group[m].to_numpy() for m in cfg['robustness_evaluation']['methods']},
                                 truths, cfg, cfg['evaluation']['bad_rules'][0])
    assert mixed.loc['primary', 'review_auc'] == pytest.approx(next(r['review_auc'] for r in direct if r['method'] == 'primary'))
    assert shifts.groupby(['split', 'method']).size().eq(2).all()
    assert not curves.empty and len(tops) == 6
    # If all scores tie, each of two perturbed views has 1/2 probability of being in the top half.
    assert expected_top_k_sums(np.ones(4), np.array([0, 1, 0, 1]))[2] == 1


def test_locked_scores_match_g1_without_ground_truth(tmp_path):
    """Score only saved synthetic model outputs and compare the selected values with the existing G1 function."""
    cfg = {'name': 'synthetic', 'paths': {'predictions_dir': tmp_path},
           'uncertainty': {'border_mm': 1},
           'robustness_evaluation': {'methods': ['model_diff_mean_region', 'entropy_mean_region', 'neg_volume_ml']}}
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    for kind, array in [('mask', mask), ('m2_mask', mask), ('prob', mask*.8), ('m2_prob', mask*.6)]:
        save_map(cfg, 'clean', 'a', kind, array, np.eye(4))
    actual = score_prediction(cfg, 'a', 'clean')
    # Quantization is part of saved G1 inputs; compare using the same decoded arrays.
    from segreview.io import load_map
    expected, _ = scan_scores(mask, load_map(cfg, 'clean', 'a', 'prob'), (1, 1, 1), 1,
                              m2_mask=mask, m2_prob=load_map(cfg, 'clean', 'a', 'm2_prob'))
    for method in cfg['robustness_evaluation']['methods']:
        assert actual[method] == pytest.approx(expected[method])
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), map_path(cfg, 'clean', 'a', 'm2_prob'))
    with pytest.raises(ValueError, match='share their grid'):
        score_prediction(cfg, 'a', 'clean')


def test_evaluation_scope_rejects_changed_primary_before_any_data_access():
    """The archived locked score list cannot silently become a new test-set score selection."""
    cfg = load_config(REPO_ROOT / 'configs/kits_robustness_eval.yaml')
    cfg['robustness_evaluation']['methods'][0] = 'soft_dice_gap'
    with pytest.raises(ValueError, match='locked strengths'):
        validate_scope(cfg)
