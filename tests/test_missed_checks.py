"""Hand-check geometry, separate-kidney handling and fair budget accounting for exploratory checks."""

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.check_missed_errors import (budget_tables, component_geometry, exact_table,
                                         inclusion_probabilities, load_scope, mask_features)


def settings():
    """Use a tiny component threshold for synthetic voxel examples, not the actual experiment."""
    return {'min_component_ml': .02, 'max_components': 2, 'connectivity': 1}


def test_solid_cube_and_internal_hole_have_known_hull_volumes():
    """A filled cube has no hull deficit; removing its central voxel gives deficit 1/27."""
    cube = np.ones((3, 3, 3), dtype=bool)
    full = component_geometry(cube, (1, 1, 1))
    assert full['volume_ml'] == pytest.approx(.027)
    assert full['hull_volume_ml'] == pytest.approx(.027)
    assert full['hull_deficit'] == pytest.approx(0)
    assert full['surface_mm2'] == 54
    assert full['compactness_deficit'] == pytest.approx(1 - np.pi/6)
    cube[1, 1, 1] = False
    hole = component_geometry(cube, (1, 1, 1))
    assert hole['hull_volume_ml'] == pytest.approx(.027)
    assert hole['hull_deficit'] == pytest.approx(1/27)
    assert hole['surface_mm2'] == 60


def test_corner_indentation_and_scale_invariance():
    """The L-shaped footprint has area 8 and hull area 8.5; scaling leaves both shape scores unchanged."""
    part = np.ones((3, 3, 3), dtype=bool)
    part[2, 2, :] = False
    a = component_geometry(part, (1, 1, 1))
    b = component_geometry(part, (2, 2, 2))
    assert a['hull_deficit'] == pytest.approx(1/17)
    assert b['volume_ml'] == pytest.approx(a['volume_ml'] * 8)
    assert b['surface_mm2'] == pytest.approx(a['surface_mm2'] * 4)
    for key in ('hull_deficit', 'compactness_deficit'):
        assert a[key] == pytest.approx(b[key])


@pytest.mark.parametrize('shape', [(1, 1, 1), (1, 2, 2), (2, 3, 4)])
def test_voxel_corners_handle_thin_and_anisotropic_components(shape):
    """Using voxel centers alone would underestimate these hull volumes or fail for planar inputs."""
    spacing = (2, 3, 4)
    result = component_geometry(np.ones(shape, bool), spacing)
    sides = np.asarray(shape) * spacing
    assert result['hull_deficit'] == pytest.approx(0)
    assert result['volume_ml'] == pytest.approx(float(np.prod(sides)) / 1000)
    assert result['surface_mm2'] == 2 * (sides[0]*sides[1] + sides[0]*sides[2] + sides[1]*sides[2])


def test_two_kidneys_are_not_wrapped_in_one_large_hull():
    """The gap between two convex kidney proxies must not count as a shape defect."""
    mask = np.zeros((15, 5, 5), dtype=bool)
    mask[:3, :3, :3] = True
    mask[10:13, :3, :3] = True
    mask[7, 4, 4] = True  # a speck below the configured component volume
    features, parts = mask_features(mask, (1, 1, 1), settings())
    assert features['n_large_components'] == 2 and len(parts) == 2
    assert features['shape_hull_deficit_max'] == pytest.approx(0)
    assert features['mask_volume_ml'] == pytest.approx(.055)


def test_single_convex_component_is_left_to_missing_side_check():
    """A missing kidney cannot be inferred from the shape of the other convex component alone."""
    features, parts = mask_features(np.ones((3, 3, 3), bool), (1, 1, 1), settings())
    assert len(parts) == 1 and features['shape_hull_deficit_max'] == pytest.approx(0)


@pytest.mark.parametrize('empty', [True, False])
def test_empty_or_only_tiny_predictions_are_suspicious(empty):
    """A small/empty outline cannot appear perfect just because no component passed the threshold."""
    mask = np.zeros((3, 3, 3), bool)
    if not empty:
        mask[0, 0, 0] = True
    features, parts = mask_features(mask, (1, 1, 1), settings())
    assert not parts and features['shape_hull_deficit_max'] == 1
    assert features['shape_compactness_deficit_max'] == 1


def test_tied_inclusion_weights_and_false_alarms_are_fractional():
    """At budget 2, three tied flags each have 2/3 inclusion probability, not case-ID preference."""
    score = np.array([1, 1, 1, 0, 0])
    weights = inclusion_probabilities(score, 2)
    assert np.allclose(weights, [2/3, 2/3, 2/3, 0, 0])
    assert weights.sum() == 2
    frame = pd.DataFrame({'case_id': list('abcde'), 'dice': [.9, .9, .7, .5, .8], 'flag': score})
    inspection = pd.DataFrame({'case_id': list('abcde'), 'rank': [1, 2, 3, 4, 5],
                               'missed_bad': [False, False, True, True, True]})
    cfg = {'checks': {'review_fraction': .4, 'bad_dice_threshold': .83, 'methods': ['flag']}}
    budget, misses = budget_tables(frame, inspection, cfg)
    assert budget.expected_bad_found.item() == pytest.approx(2/3)
    assert budget.expected_false_alarms.item() == pytest.approx(4/3)
    assert budget.expected_original_misses_found.item() == pytest.approx(2/3)
    assert misses.loc[misses.case_id.eq('c'), 'fair_rank'].item() == 2
    assert misses.loc[misses.case_id.eq('d'), 'fair_rank'].item() == 4.5


def test_all_tied_scores_have_random_budget_inclusion():
    """A constant flag carries no information and must not look better through file ordering."""
    assert np.allclose(inclusion_probabilities(np.zeros(5), 2), .4)
    assert np.allclose(inclusion_probabilities(np.zeros(5), 0), 0)
    assert np.allclose(inclusion_probabilities(np.zeros(5), 5), 1)


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'extra'])
def test_exact_cohort_joins_cannot_silently_drop_cases(problem):
    """A feature/source mismatch must stop instead of changing the sample used for evaluation."""
    ids = {'missing': ['a'], 'duplicate': ['a', 'b', 'b'], 'extra': ['a', 'b', 'c']}[problem]
    with pytest.raises(ValueError, match='exact'):
        exact_table(pd.DataFrame({'case_id': ids}), {'a', 'b'}, 'test')


def test_read_only_source_output_is_rejected_before_data_access(tmp_path):
    """A changed config name/path cannot write beneath the source results."""
    source = {'name': 'locked', 'paths': {'raw_dir': str(tmp_path / 'raw'), 'models_dir': str(tmp_path / 'models'),
                                         'predictions_dir': str(tmp_path / 'predictions'),
                                         'results_dir': str(tmp_path / 'results')},
              'dataset': {'file': 'configs/datasets/kits23.yaml'}}
    path = tmp_path / 'source.yaml'
    path.write_text(yaml.safe_dump(source))
    cfg = {'name': 'child', 'paths': {'results_dir': tmp_path / 'results/locked'},
           'checks': {'source_config': str(path)}}
    with pytest.raises(ValueError, match='read-only'):
        load_scope(cfg)
