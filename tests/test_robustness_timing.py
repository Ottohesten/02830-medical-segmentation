"""Check dev/test isolation and measured-time extrapolation for robustness step 2 using synthetic cases."""

from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.benchmark_robustness import runtime_projections, select_timing_cases
from segreview.config import REPO_ROOT


@pytest.fixture
def cfg(tmp_path):
    """Return an isolated timing config with two development IDs and one held-out ID."""
    return {'name': 'timing_test', 'paths': {'results_dir': tmp_path / 'results',
                                          'predictions_dir': tmp_path / 'predictions'},
            'dataset': {'split': {'dev': ['a', 'b']}}, 'perturbations': [{'name': 'noise'}],
            'model': {'resolution': 'fast'}, 'uncertainty': {'second_model_resolution': 'fastest'},
            'timing': {'case_ids': ['a', 'b'], 'projection_case_counts': [20, 80],
                       'projection_perturbation_counts': [3, 4]}}


def test_preserves_original_case_indices_for_noise(cfg):
    """Subsetting must preserve the random seed's original scan index for both models."""
    cases = [SimpleNamespace(case_id=x) for x in ['held_out', 'b', 'a']]
    selected = select_timing_cases(cfg, cases)
    assert [(i, case.case_id) for i, case in selected] == [(2, 'a'), (1, 'b')]


@pytest.mark.parametrize('ids', [['a', 'held_out'], ['a', 'missing'], ['a', 'a']])
def test_rejects_held_out_missing_and_duplicate_cases(cfg, ids):
    """Timing must fail before running any model when IDs are invalid or belong to test."""
    cfg['timing']['case_ids'] = ids
    with pytest.raises(ValueError):
        select_timing_cases(cfg, [SimpleNamespace(case_id=x) for x in ['a', 'b', 'held_out']])


@pytest.mark.parametrize('path_key,locked', [('results_dir', 'results/kits100'),
                                           ('predictions_dir', 'data/predictions/kits100')])
def test_rejects_locked_output_subfolders(cfg, path_key, locked):
    """A different config name must not permit writing beneath either locked output folder."""
    cfg['paths'][path_key] = REPO_ROOT / locked
    with pytest.raises(ValueError, match='locked'):
        select_timing_cases(cfg, [SimpleNamespace(case_id=x) for x in ['a', 'b']])


def synthetic_timings():
    """Return eight known times: paired clean predictions 3/6 s, paired noisy predictions 6/12 s."""
    return pd.DataFrame([{'case_id': case, 'variant': variant, 'resolution': model,
                          'prediction_seconds': duration * multiplier * severity,
                          'total_seconds': duration * multiplier * severity + 1}
                         for case, multiplier in [('a', 1), ('b', 2)]
                         for variant, severity in [('clean', 1), ('noise', 2)]
                         for model, duration in [('fast', 2), ('fastest', 1)]])


def test_projects_both_models_clean_reference_and_io(cfg):
    """Include both models, one clean reference, P noisy variants, I/O, and one model-load cost."""
    result = runtime_projections(synthetic_timings(), cfg, model_load_seconds=10)
    row = result[(result.n_cases == 20) & (result.n_perturbations == 3)].iloc[0]
    assert row.prediction_seconds == pytest.approx(630)
    assert row.end_to_end_seconds == pytest.approx(800)
    assert len(result) == 4
    test_row = result[(result.n_cases == 80) & (result.n_perturbations == 3)].iloc[0]
    assert test_row.end_to_end_seconds == pytest.approx(3170)


@pytest.mark.parametrize('duplicate', [False, True])
def test_incomplete_or_duplicated_measurements_cannot_produce_estimates(cfg, duplicate):
    """An interrupted run or duplicated row must not be mistaken for all eight predictions."""
    rows = synthetic_timings().iloc[:-1]
    if duplicate:
        rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match='eight distinct'):
        runtime_projections(rows, cfg, model_load_seconds=10)
