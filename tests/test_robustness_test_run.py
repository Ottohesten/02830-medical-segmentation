"""Test held-out isolation, identical model inputs, resumable variant maps, and timing plans on tiny synthetic scans."""

from copy import deepcopy
from types import SimpleNamespace
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_robustness_test import estimate_runtime, load_test_scope, verify_predictions
from segreview import pipeline
from segreview.config import REPO_ROOT, load_config
from segreview.data import Case
from segreview.io import map_path
from segreview.robustness import validate_test_run


@pytest.fixture
def scope(tmp_path):
    """Build a one-dev/two-test study with a locked single perturbation and independent output names."""
    locked = {'name': 'chosen_dev', 'seed': 42,
              'paths': {'results_dir': tmp_path / 'results', 'predictions_dir': tmp_path / 'predictions'},
              'dataset': {'n_cases': 1, 'split': {'dev': ['a']}},
              'model': {'resolution': 'fast'}, 'uncertainty': {'second_model_resolution': 'fastest'},
              'report': {'set': 'clean_dev', 'primary': 'model_diff_mean_region'},
              'evaluation': {'splits': ['dev']},
              'perturbations': [{'name': 'noise', 'type': 'noise', 'strength': 10}]}
    cfg = deepcopy(locked)
    cfg.update({'name': 'held_out', 'robustness_test': {'splits': ['test'], 'expected_test_cases': 2,
                                                      'benchmark_perturbation': 'noise'}})
    cfg['dataset']['n_cases'] = 3
    cfg['evaluation']['splits'] = ['test']
    cfg['report']['set'] = 'clean_test'
    reference = deepcopy(cfg)
    reference['name'] = 'kits100'
    cases = [SimpleNamespace(case_id=case) for case in ['a', 'b', 'c']]
    return cfg, locked, reference, cases


def test_scope_returns_test_cases_only(scope):
    """Development IDs remain available for stable indices but are excluded from the returned run scope."""
    cfg, locked, reference, cases = scope
    selected = validate_test_run(cfg, cases, reference, cases, locked, {'selected_variants': ['noise']})
    assert [c.case_id for c in selected] == ['b', 'c']


@pytest.mark.parametrize('change', ['strength', 'seed', 'split', 'score', 'model', 'case_order', 'case_duplicate',
                                   'case_count', 'scope', 'output', 'nested_output'])
def test_scope_rejects_scientific_and_selection_changes(scope, change):
    """Fail before inference when the locked science, full ordered selection, or output isolation changes."""
    cfg, locked, reference, cases = scope
    modified = list(cases)
    if change == 'strength':
        cfg['perturbations'][0]['strength'] = 20
    elif change == 'seed':
        cfg['seed'] = 43
    elif change == 'split':
        cfg['dataset']['split']['dev'] = ['b']
    elif change == 'score':
        cfg['report']['primary'] = 'entropy_mean_region'
    elif change == 'model':
        cfg['uncertainty']['second_model_resolution'] = None
    elif change == 'case_order':
        modified.reverse()
    elif change == 'case_duplicate':
        modified[-1] = modified[0]
    elif change == 'case_count':
        cfg['robustness_test']['expected_test_cases'] = 3
    elif change == 'scope':
        cfg['robustness_test']['splits'] = ['dev', 'test']
    elif change == 'output':
        cfg['name'] = reference['name']
    else:
        cfg['paths']['results_dir'] /= reference['name']
    with pytest.raises(ValueError):
        validate_test_run(cfg, modified, reference, cases, locked, {'selected_variants': ['noise']})


def test_archived_lock_tampering_is_rejected(tmp_path):
    """Changing the selected config bytes must fail even before images or models are loaded."""
    cfg = load_config(REPO_ROOT / 'configs/kits_robustness_test.yaml')
    changed = tmp_path / 'changed.yaml'
    changed.write_text('name: changed\n')
    cfg['robustness_test']['locked_config'] = str(changed)
    with pytest.raises(ValueError, match='archived development choice'):
        load_test_scope(cfg)


@pytest.fixture
def tiny_pipeline(scope, monkeypatch, tmp_path):
    """Replace the network with a recorder while retaining real tiny NIfTI inputs, map saves, and timing CSVs."""
    cfg, _, _, _ = scope
    cases = []
    for case_id in ['a', 'b', 'c']:
        path = tmp_path / f'{case_id}.nii.gz'
        nib.save(nib.Nifti1Image(np.arange(210, dtype=np.float32).reshape(5, 6, 7), np.eye(4)), path)
        cases.append(Case(case_id, path, tmp_path / 'labels_must_not_exist.nii.gz'))
    seen = []
    indices = []

    class Recorder:
        """Record network inputs and return deterministic masks/probabilities on the input grid."""

        def __init__(self, config, device, resolution=None):
            """Keep the model identity so the test can compare inputs across the two models."""
            self.resolution = resolution or config['model']['resolution']

        def predict(self, image):
            """Record the float input and return maps without reading any ground truth."""
            array = image.get_fdata(dtype=np.float32)
            seen.append((self.resolution, array.copy()))
            return SimpleNamespace(mask=(array > 100), prob=np.full(array.shape, 0.6), affine=image.affine)

    def seeded_perturb(image, spec, config, index):
        """Use the project's seed/index convention to expose accidental reindexing in either model."""
        indices.append(index)
        array = image.get_fdata(dtype=np.float32)
        noise = np.random.default_rng([config['seed'], index]).normal(0, spec['strength'], array.shape)
        return nib.Nifti1Image(array + noise, image.affine)

    monkeypatch.setitem(sys.modules, 'segreview.inference', SimpleNamespace(OrganSegmenter=Recorder))
    monkeypatch.setattr(pipeline, 'list_cases', lambda config: cases)
    monkeypatch.setattr(pipeline, 'perturb', seeded_perturb)
    return cfg, cases, seen, indices


def test_both_models_receive_identical_test_variants_and_preserve_indices(tiny_pipeline):
    """Both models must see the same seeded arrays, skip dev, and retain separate clean/noise timing rows."""
    cfg, cases, seen, indices = tiny_pipeline
    pipeline.step_inference(cfg, torch.device('cpu'), splits=['test'])
    pipeline.step_second_model(cfg, torch.device('cpu'), splits=['test'])
    assert [r for r, _ in seen] == ['fast'] * 4 + ['fastest'] * 4
    for (_, main), (_, second) in zip(seen[:4], seen[4:]):
        np.testing.assert_array_equal(main, second)
    assert indices == [1, 2, 1, 2]
    assert not map_path(cfg, 'clean', 'a', 'mask').exists()
    assert not map_path(cfg, 'noise', 'a', 'm2_prob').exists()
    out = cfg['paths']['results_dir'] / cfg['name']
    # Missing label paths prove that neither inference nor verification reads labels.
    report = verify_predictions(cfg, cases[1:], out)
    assert report['n_predictions'] == 8 and report['n_files'] == 16
    assert not report['ground_truth_read']


def test_resume_repairs_missing_second_mask_without_repeating_finished_pairs(tiny_pipeline):
    """A saved second-model probability alone is incomplete; regenerate only its pair with the same seed."""
    cfg, _, seen, indices = tiny_pipeline
    pipeline.step_second_model(cfg, torch.device('cpu'), splits=['test'])
    seen.clear()
    indices.clear()
    pipeline.step_second_model(cfg, torch.device('cpu'), splits=['test'])
    assert not seen
    map_path(cfg, 'noise', 'b', 'm2_mask').unlink()
    pipeline.step_second_model(cfg, torch.device('cpu'), splits=['test'])
    assert len(seen) == 1 and indices == [1]
    rows = pd.read_csv(cfg['paths']['results_dir'] / cfg['name'] / 'second_model_times.csv')
    assert len(rows) == 4 and not rows.duplicated(['case_id', 'variant']).any()


def test_historical_clean_timings_are_preserved_when_resuming(tiny_pipeline):
    """Migrate old clean-only CSVs without losing a completed scan or duplicating its timing."""
    cfg, _, seen, _ = tiny_pipeline
    cfg['perturbations'] = []
    pipeline.step_second_model(cfg, torch.device('cpu'), splits=['test'])
    out = cfg['paths']['results_dir'] / cfg['name']
    path = out / 'second_model_times.csv'
    old = pd.read_csv(path).drop(columns='variant')
    old.to_csv(path, index=False)
    map_path(cfg, 'clean', 'c', 'm2_mask').unlink()
    seen.clear()
    pipeline.step_second_model(cfg, torch.device('cpu'), splits=['test'])
    rows = pd.read_csv(path)
    assert len(seen) == 1
    assert len(rows) == 2 and rows.variant.eq('clean').all()
    assert rows.set_index('case_id').loc['b', 'seconds'] == old.set_index('case_id').loc['b', 'seconds']


def test_projection_combines_twenty_case_model_basis_io_and_second_model_proxy(scope):
    """Hand-check different main/second-model bases, summed test volume, and one model-load cost."""
    cfg, _, _, _ = scope
    sizes = pd.DataFrame([{'case_id': 'a', 'split': 'dev', 'voxels': 100},
                          {'case_id': 'b', 'split': 'test', 'voxels': 200},
                          {'case_id': 'c', 'split': 'test', 'voxels': 100}])
    dev = pd.DataFrame([{'case_id': 'a', 'variant': v, 'resolution': 'fast', 'seconds': s}
                        for v, s in [('clean', 10), ('noise', 20)]])
    bench = pd.DataFrame([{'case_id': 'a', 'variant': v, 'resolution': r,
                          'prediction_seconds': prediction, 'total_seconds': total}
                         for v in ['clean', 'noise']
                         for r, prediction, total in [('fast', 2, 3), ('fastest', 1, 2)]])
    plan = estimate_runtime(cfg, sizes, dev, bench, model_load_seconds=5)
    # Main = (10+1)*3 + (20+1)*3 = 96; second = 2*3 + 2*3 = 12; loads = 5.
    assert plan['estimated_minutes'] == pytest.approx(113 / 60)
    assert len(plan['components']) == 4
    with pytest.raises(ValueError, match='per dev case'):
        estimate_runtime(cfg, sizes, dev.iloc[:1], bench, 5)


@pytest.mark.parametrize('problem', ['timing_missing', 'timing_nonfinite', 'grid', 'missing_map'])
def test_verification_rejects_incomplete_or_misaligned_outputs(tiny_pipeline, problem):
    """An interrupted run, invalid runtime, or wrong map grid must never be marked complete."""
    cfg, cases, _, _ = tiny_pipeline
    pipeline.step_inference(cfg, torch.device('cpu'), splits=['test'])
    pipeline.step_second_model(cfg, torch.device('cpu'), splits=['test'])
    out = cfg['paths']['results_dir'] / cfg['name']
    path = out / 'second_model_times.csv'
    if problem.startswith('timing'):
        rows = pd.read_csv(path)
        if problem == 'timing_missing':
            rows = rows.iloc[:-1]
        else:
            rows.loc[0, 'seconds'] = float('inf')
        rows.to_csv(path, index=False)
    elif problem == 'grid':
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), map_path(cfg, 'noise', 'c', 'm2_prob'))
    else:
        map_path(cfg, 'noise', 'c', 'm2_prob').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_predictions(cfg, cases[1:], out)
