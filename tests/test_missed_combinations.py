"""Check GT-free rank combinations, exact ties, fixed development selection and lost-case accounting."""

import numpy as np
import pandas as pd
import pytest

import scripts.combine_missed_errors as script
from scripts.check_missed_errors import inclusion_probabilities


def specs():
    """Return two simple alternatives for a four-case example with a separate missing-side alert."""
    return [{'name': 'mean', 'rule': 'mean_rank', 'scores': ['g1', 'flag']},
            {'name': 'max', 'rule': 'max_rank', 'scores': ['g1', 'flag']}]


def test_mean_and_max_have_different_alert_behavior():
    """A high flag alone makes the last G1 case joint first under maximum rank, but not under mean rank."""
    frame = pd.DataFrame({'g1': [4, 3, 2, 1], 'flag': [0, 0, 0, 1]})
    out = script.build_combinations(frame, specs(), 1e-9)
    assert np.allclose(out['mean'], [.75, .625, .5, .625])
    assert np.allclose(out['max'], [1, .75, .5, 1])
    assert np.allclose(inclusion_probabilities(out['max'], 2), [1, 0, 0, 1])
    assert list(frame.columns) == ['g1', 'flag']


def test_mathematically_equal_rank_sums_stay_exactly_tied():
    """Opposite rankings average to one constant; floating roundoff must not favor particular case IDs."""
    frame = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100)[::-1]})
    out = script.build_combinations(frame, [{'name': 'mean', 'rule': 'mean_rank', 'scores': ['a', 'b']}], 1e-9)
    assert out['mean'].nunique() == 1
    assert np.allclose(inclusion_probabilities(out['mean'], 20), .2)


@pytest.mark.parametrize('column', ['dice', 'bad', 'missed_bad', 'truth_voxels'])
def test_outcomes_cannot_enter_combination_stage(column):
    """Outcome columns are rejected before constructing or saving any rank combination."""
    frame = pd.DataFrame({'g1': [1, 2], 'flag': [0, 1], column: [0, 1]})
    with pytest.raises(ValueError, match='ground-truth'):
        script.build_combinations(frame, specs(), 1e-9)


def test_combination_rejects_nonfinite_inputs_and_colliding_names():
    """Invalid raw scores and name collisions must stop rather than silently change a method."""
    with pytest.raises(ValueError, match='finite'):
        script.build_combinations(pd.DataFrame({'g1': [np.nan], 'flag': [0]}), specs(), 1e-9)
    with pytest.raises(ValueError, match='unique'):
        script.build_combinations(pd.DataFrame({'g1': [1], 'flag': [0], 'mean': [2]}), specs(), 1e-9)


@pytest.fixture
def selection_inputs():
    """Provide an AUC winner, a budget failure and two potentially tied eligible methods."""
    cfg = {'checks': {'primary': 'locked'},
           'combinations': [{'name': name} for name in ['mean', 'max', 'both']],
           'selection': {'bad_rule': 'threshold', 'comparison_atol': 1e-12,
                         'eligibility': 'higher_review_auc_and_no_fewer_bad_in_budget_than_locked',
                         'order': ['review_auc_descending', 'quality_auc_descending', 'config_order'],
                         'max_recommendations': 1}}
    metrics = pd.DataFrame({'method': ['locked', 'mean', 'max', 'both'], 'bad_rule': ['threshold'] * 4,
                            'review_auc': [.80, .82, .83, .84], 'quality_auc': [.95, .94, .945, .955]})
    budget = pd.DataFrame({'method': ['locked', 'mean', 'max', 'both'], 'expected_bad_found': [11, 11, 10, 11]})
    return cfg, metrics, budget


def test_selection_uses_full_queue_auc_and_preserves_budget_capture(selection_inputs):
    """A larger AUC cannot win if it sacrifices top-budget bad-case capture."""
    cfg, metrics, budget = selection_inputs
    metrics.loc[metrics.method.eq('max'), 'review_auc'] = .99
    out = script.choose_recommendation(metrics, budget, cfg)
    assert out['selected'] == 'both'
    assert not next(x for x in out['considered'] if x['method'] == 'max')['eligible']
    assert 'not a fresh-test lock' in out['status']


def test_selection_ties_use_quality_then_config_order(selection_inputs):
    """Tied AUCs use quality, then the predeclared order, without looking at individual missed cases."""
    cfg, metrics, budget = selection_inputs
    metrics.loc[metrics.method.eq('mean'), 'review_auc'] = .84
    assert script.choose_recommendation(metrics, budget, cfg)['selected'] == 'both'
    metrics.loc[metrics.method.eq('mean'), 'quality_auc'] = .955
    assert script.choose_recommendation(metrics, budget, cfg)['selected'] == 'mean'


def test_no_eligible_candidate_is_a_valid_result(selection_inputs):
    """The script must return no recommendation rather than manufacture an improvement."""
    cfg, metrics, budget = selection_inputs
    metrics.loc[metrics.method.ne('locked'), 'review_auc'] = .79
    assert script.choose_recommendation(metrics, budget, cfg)['selected'] is None


def test_tradeoffs_count_lost_bad_cases_as_well_as_rescues():
    """Finding a new bad scan is no net improvement when another bad scan is displaced."""
    frame = pd.DataFrame({'case_id': list('abcde'), 'dice': [.9, .7, .6, .95, .9],
                           'locked': [5, 4, 3, 2, 1], 'combo': [3, 2, 4, 5, 1]})
    cfg = {'checks': {'primary': 'locked', 'review_fraction': .4, 'bad_dice_threshold': .83},
           'combinations': [{'name': 'combo'}]}
    detail, summary = script.queue_changes(frame, cfg)
    assert len(detail) == 5
    assert summary.new_bad.item() == 1 and summary.lost_bad.item() == 1
    assert summary.new_false_alarms.item() == 1 and summary.removed_false_alarms.item() == 1


def test_tradeoffs_keep_fractional_tie_changes():
    """Half-inclusion from a boundary tie is carried into both gained and lost bad-case counts."""
    frame = pd.DataFrame({'case_id': list('abcd'), 'dice': [.9, .7, .6, .95],
                           'locked': [4, 3, 2, 1], 'combo': [4, 3, 3, 1]})
    cfg = {'checks': {'primary': 'locked', 'review_fraction': .5, 'bad_dice_threshold': .83},
           'combinations': [{'name': 'combo'}]}
    _, summary = script.queue_changes(frame, cfg)
    assert summary.new_bad.item() == .5 and summary.lost_bad.item() == .5


def test_source_folders_are_protected_before_input_reads(tmp_path, monkeypatch):
    """An alternate experiment name cannot route outputs into completed step 3 results."""
    source = {'name': 'locked', 'paths': {'results_dir': tmp_path/'results', 'predictions_dir': tmp_path/'predictions'}}
    monkeypatch.setattr(script, 'load_config', lambda path: source)
    cfg = {'name': 'child', 'paths': {'results_dir': tmp_path/'checks'},
           'checks': {'source_config': 'source.yaml', 'source_features': str(tmp_path/'checks/features.csv'),
                      'inspection_queue': str(tmp_path/'inspection/queue.csv')}}
    with pytest.raises(ValueError, match='read-only'):
        script.load_inputs(cfg)
