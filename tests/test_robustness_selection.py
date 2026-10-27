"""Check development isolation, paired Dice degradation, and strength selection using four synthetic scans."""

from types import SimpleNamespace

import pandas as pd
import pytest

from segreview.config import REPO_ROOT
from segreview.robustness import summarize_degradation, suggest_strengths, validate_development_run


@pytest.fixture
def cfg(tmp_path):
    """Return a small selection config with thresholds whose outcomes can be checked by hand."""
    return {'name': 'dev_candidates', 'paths': {'results_dir': tmp_path / 'results',
                                              'predictions_dir': tmp_path / 'predictions'},
            'dataset': {'split': {'dev': ['a', 'b', 'c', 'd']}},
            'evaluation': {'splits': ['dev'], 'bad_rules': [{'name': 'bad', 'rule': 'threshold', 'value': 0.83}]},
            'perturbations': [{'name': 'noise_30', 'type': 'noise', 'strength': 30},
                              {'name': 'noise_60', 'type': 'noise', 'strength': 60},
                              {'name': 'lowres_6', 'type': 'lowres', 'strength': 6},
                              {'name': 'contrast_80', 'type': 'contrast', 'strength': 0.8}],
            'robustness_selection': {'expected_dev_cases': 4, 'min_dice_drop': 0.02, 'min_affected_cases': 2,
                                     'min_remaining_good_cases': 1, 'catastrophic_dice': 0.5,
                                     'max_new_catastrophic_cases': 1, 'bad_rule': 'bad',
                                     'rule': 'mildest_eligible_per_type'}}


@pytest.fixture
def rows():
    """Return paired examples with mild/effective noise, ineffective lowres, and destructive contrast."""
    values = {'clean': [0.9, 0.85, 0.7, 0.4], 'noise_30': [0.88, 0.80, 0.69, 0.39],
              'noise_60': [0.85, 0.78, 0.65, 0.30], 'lowres_6': [0.899, 0.849, 0.699, 0.399],
              'contrast_80': [0.1, 0.1, 0.1, 0.1]}
    return pd.DataFrame([{'case_id': case, 'variant': variant, 'split': 'dev', 'dice': value}
                         for variant, dice in values.items() for case, value in zip(['a', 'b', 'c', 'd'], dice)])


def test_requires_exact_development_set(cfg):
    """A test scan cannot replace a dev scan, even when the total count is correct."""
    cases = [SimpleNamespace(case_id=x) for x in ['a', 'b', 'c', 'd']]
    validate_development_run(cfg, cases)
    cases[-1] = SimpleNamespace(case_id='held_out')
    with pytest.raises(ValueError, match='exactly the development'):
        validate_development_run(cfg, cases)


def test_rejects_test_evaluation_before_inference(cfg):
    """The development runner must reject configurations whose evaluation includes test."""
    cfg['evaluation']['splits'] = ['dev', 'test']
    with pytest.raises(ValueError, match='Only development'):
        validate_development_run(cfg, [SimpleNamespace(case_id=x) for x in ['a', 'b', 'c', 'd']])


@pytest.mark.parametrize('key,locked', [('results_dir', 'results/kits100'),
                                      ('predictions_dir', 'data/predictions/kits100')])
def test_rejects_locked_subfolders(cfg, key, locked):
    """An alternative name must not make a path beneath a locked folder writable."""
    cfg['paths'][key] = REPO_ROOT / locked
    with pytest.raises(ValueError, match='locked'):
        validate_development_run(cfg, [SimpleNamespace(case_id=x) for x in ['a', 'b', 'c', 'd']])


def test_paired_drops_and_new_failures_are_counted_correctly(cfg, rows):
    """Use median paired drops, not a difference of medians; do not count existing failures as new."""
    paired, summary = summarize_degradation(rows, cfg)
    noise = summary.set_index('variant').loc['noise_30']
    assert noise.median_paired_drop == pytest.approx(0.015)
    assert noise.median_clean_dice - noise.median_dice == pytest.approx(0.03)
    assert noise.n_clear_drops == 2
    assert noise.n_new_bad == 1
    assert noise.n_bad == 3
    assert noise.n_new_catastrophic == 0
    assert paired[(paired.variant == 'noise_30') & (paired.case_id == 'a')].dice_drop.iloc[0] == 0.02


def test_mildest_eligible_type_is_chosen_and_other_types_are_explicitly_excluded(cfg, rows):
    """Do not pick stronger noise merely because its degradation is larger, or force ineffective/destructive types."""
    _, summary = summarize_degradation(rows, cfg)
    choice = suggest_strengths(summary, cfg)
    assert choice['selected_variants'] == ['noise_30']
    assert choice['types_without_eligible_candidate'] == ['lowres', 'contrast']
    contrast = summary.set_index('variant').loc['contrast_80']
    assert contrast.n_new_catastrophic == 3
    assert not contrast.eligible
    assert not choice['uncertainty_scores_computed'] and not choice['test_set_used']


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'test', 'nan', 'out_of_range'])
def test_summary_rejects_incomplete_invalid_or_test_rows(cfg, rows, problem):
    """Incomplete or invalid evaluation must not yield a seemingly valid strength choice."""
    if problem == 'missing':
        rows = rows.iloc[:-1]
    elif problem == 'duplicate':
        rows = pd.concat([rows.iloc[:-1], rows.iloc[:1]], ignore_index=True)
    elif problem == 'test':
        rows.loc[0, 'split'] = 'test'
    elif problem == 'nan':
        rows.loc[0, 'dice'] = float('nan')
    else:
        rows.loc[0, 'dice'] = 1.1
    with pytest.raises(ValueError, match='exactly one finite dev Dice'):
        summarize_degradation(rows, cfg)
