"""Synthetic checks of missed-case selection, GT-independent ties and error-location diagnostics."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.inspect_missed_errors import build_queue, error_diagnostics, inspection_slices, validate_output


@pytest.fixture
def tables():
    """Return five cases with a score tie, an exact threshold case, and two missed bad cases."""
    scores = pd.DataFrame({'case_id': list('abcde'), 'split': ['dev'] * 5, 'variant': ['clean'] * 5,
                           'score': [.9, .7, .7, .2, .1]})
    outcomes = scores[['case_id', 'split', 'variant']].copy()
    outcomes['dice'] = [.9, .7, .5, .83, .8]
    settings = {'score': 'score', 'variant': 'clean', 'tie_break': 'case_id_ascending',
                'review_fraction': .4, 'bad_dice_threshold': .83}
    return scores, outcomes, settings


def test_queue_uses_scores_and_ids_not_dice_for_ties(tables):
    """The worse tied case cannot move ahead merely because its ground-truth Dice is lower."""
    scores, outcomes, settings = tables
    q = build_queue(scores.sample(frac=1, random_state=42), outcomes, set('abcde'), settings)
    assert q.case_id.tolist() == list('abcde')
    assert q['rank'].tolist() == [1, 2, 3, 4, 5]
    assert q.loc[q.missed_bad, 'case_id'].tolist() == ['c', 'e']
    assert not q.loc[q.case_id.eq('d'), 'bad'].item()
    outcomes['dice'] = outcomes['dice'].iloc[::-1].to_numpy()
    changed = build_queue(scores, outcomes, set('abcde'), settings)
    assert changed.case_id.tolist() == q.case_id.tolist()


def test_budget_rounds_up_and_keeps_only_requested_variant(tables):
    """A fractional budget rounds up; extra variants cannot duplicate the clean cohort."""
    scores, outcomes, settings = tables
    settings['review_fraction'] = .21
    scores = pd.concat([scores, scores.assign(variant='other')])
    q = build_queue(scores, outcomes, set('abcde'), settings)
    assert len(q) == 5 and q.in_top_budget.sum() == 2


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'extra', 'split', 'nan_score', 'nan_dice', 'range'])
def test_queue_rejects_incomplete_or_invalid_inputs(tables, problem):
    """A silent join loss or invalid outcome must stop rather than change which cases count as missed."""
    scores, outcomes, settings = deepcopy(tables)
    if problem == 'missing':
        outcomes = outcomes.iloc[:-1]
    elif problem == 'duplicate':
        scores = pd.concat([scores, scores.iloc[:1]])
    elif problem == 'extra':
        scores.loc[0, 'case_id'] = 'unexpected'
    elif problem == 'split':
        outcomes.loc[0, 'split'] = 'test'
    elif problem == 'nan_score':
        scores.loc[0, 'score'] = np.nan
    elif problem == 'nan_dice':
        outcomes.loc[0, 'dice'] = np.nan
    else:
        outcomes.loc[0, 'dice'] = 1.1
    with pytest.raises(ValueError):
        build_queue(scores, outcomes, set('abcde'), settings)


def test_diagnostics_distinguish_label_errors_confidence_and_second_model():
    """Three missed voxels include two low-probability errors, one exact zero, and two recovered by 6 mm."""
    labels = np.array([0, 1, 1, 2, 2, 3, 0, 0], dtype=np.uint8).reshape(1, 1, -1)
    pred = np.isin(np.arange(8), [2, 3, 6]).reshape(labels.shape)
    second = np.isin(np.arange(8), [4, 5]).reshape(labels.shape)
    prob = np.array([0, 0, 1, 1, .05, .4, 1, 0]).reshape(labels.shape)
    result = error_diagnostics(labels, pred, second, prob, (1, 1, 2),
                               {'organ_labels': [1, 2, 3], 'ignore_labels': []}, 0, .1,
                               {1: 'kidney', 2: 'tumor', 3: 'cyst'})
    assert result['dice_recomputed'] == .5
    assert result['false_negative_voxels'] == 3 and result['false_positive_voxels'] == 1
    assert result['false_negative_ml'] == pytest.approx(.006)
    assert result['fn_low_probability_share'] == pytest.approx(2/3)
    assert result['fn_zero_probability_share'] == pytest.approx(1/3)
    assert result['fn_outside_region_share'] == 1
    assert result['fn_recovered_by_6mm_share'] == pytest.approx(2/3)
    assert result['kidney_missed_share'] == .5 and result['tumor_missed_share'] == .5
    assert result['cyst_missed_share'] == 1


def test_no_false_negatives_has_undefined_fn_shares():
    """Undefined FN fractions must not masquerade as evidence of confident errors."""
    labels = np.ones((2, 2, 2), dtype=np.uint8)
    pred = np.ones_like(labels, dtype=bool)
    result = error_diagnostics(labels, pred, pred, labels, (1, 1, 1),
                               {'organ_labels': [1], 'ignore_labels': []}, 0, .1, {1: 'kidney'})
    assert result['dice_recomputed'] == 1 and result['false_negative_voxels'] == 0
    assert np.isnan(result['fn_low_probability_share'])


def test_error_slices_include_separate_missed_structures():
    """A missed tumor or distant kidney is shown even if it is absent from the maximal-organ slice."""
    labels = np.zeros((3, 3, 8), dtype=np.uint8)
    labels[:, :, 0] = 1
    labels[0, 0, 2] = 2
    labels[0, 0, 6] = 3
    pred = np.zeros_like(labels, dtype=bool)
    pred[0, 0, 7] = True
    slices = inspection_slices(labels, pred, [1, 2, 3], {1: 'kidney', 2: 'tumor', 3: 'cyst'},
                               ['kidney', 'tumor', 'cyst', 'false_positive', 'organ'])
    assert [index for axis, index, _ in slices if axis == 2] == [0, 2, 6, 7]
    assert sum(axis == 1 for axis, _, _ in slices) == 1


@pytest.mark.parametrize('target', ['results/locked', 'results/locked/child', 'predictions/locked', 'results'])
def test_output_cannot_overwrite_or_contain_sources(tmp_path, target):
    """A new experiment name cannot bypass source protection through alternate output paths."""
    source = {'name': 'locked', 'paths': {'results_dir': tmp_path / 'results',
                                        'predictions_dir': tmp_path / 'predictions'}}
    path = tmp_path / target
    cfg = {'name': path.name, 'paths': {'results_dir': path.parent}}
    with pytest.raises(ValueError, match='read-only'):
        validate_output(cfg, source)
