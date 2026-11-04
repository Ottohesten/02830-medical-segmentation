"""Test deterministic pool allocation, genuine coordination gates, GT-free scoring and one outcome opening."""

import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ConstantInputWarning

import scripts.fresh_missed_errors as script
from scripts.check_missed_errors import digest


def test_pool_is_next_seeded_cases_with_no_development_overlap():
    """The pool is a continuation of the original permutation, not a fresh independent draw."""
    ids = [f'case_{i:05}' for i in range(20)]
    order = np.random.default_rng(42).permutation(len(ids))
    old = {ids[i] for i in order[:5]}
    result = script.pool_allocation(ids, old, 42, 5, 6)
    assert set(result.case_id) == {ids[i] for i in order[5:11]}
    assert not set(result.case_id)&old
    assert set(result.draw_position) == set(range(6, 12))


@pytest.mark.parametrize('problem', ['wrong_old', 'duplicate', 'too_short'])
def test_bad_pool_definition_stops_before_data_access(problem):
    """Seed/source mismatches cannot silently substitute another cohort."""
    ids = [str(i) for i in range(10)]
    old = set(np.random.default_rng(42).permutation(10)[:3].astype(str))
    if problem == 'wrong_old':
        old = {'unexpected', 'a', 'b'}
    elif problem == 'duplicate':
        ids[-1] = ids[0]
    else:
        ids = ids[:4]
    with pytest.raises(ValueError):
        script.pool_allocation(ids, old, 42, 3, 4)


def test_missing_coordination_never_opens_fresh_headers(tmp_path, monkeypatch):
    """No handoff remains pending without even asking nibabel to inspect the pool."""
    monkeypatch.setattr(script.nib, 'load', lambda path: pytest.fail('Fresh header opened before coordination'))
    result = script.preflight({}, tmp_path, {}, None)
    assert not result['ready'] and not result['fresh_headers_or_voxels_read']


def test_unconfirmed_handoff_does_not_count_as_notification(tmp_path, monkeypatch):
    """Template hashes are not evidence that Bjørn/Frederik were actually contacted."""
    path = tmp_path/'handoff.json'
    path.write_text(json.dumps({'checkpoint_sha256': {'a': 'hash'}}))
    monkeypatch.setattr(script.nib, 'load', lambda path: pytest.fail('Fresh header opened before confirmation'))
    result = script.preflight({}, tmp_path, {'checkpoint_sha256': {'a': 'hash'}}, path)
    assert not result['ready'] and len(result['issues']) > 0


def test_different_checkpoint_provenance_is_not_accepted():
    """A handoff must identify the original baseline weights, not just a model with the same name."""
    handoff = {key: 'documented' for key in ['bjorn_notified_at','frederik_confirmed_at','evidence','inference_metadata','inputs_manifest']}
    handoff.update(ranking_metrics_not_previously_evaluated=True, checkpoint_sha256={'model': 'fine_tuned'})
    assert 'Original baseline checkpoint hashes not confirmed' in script.handoff_status(handoff, {'checkpoint_sha256': {'model': 'original'}})


def test_one_time_marker_cannot_be_replaced(tmp_path):
    """A second outcome-stage entry is refused and preserves the first marker."""
    path = tmp_path/'evaluation_started.json'
    script.consume_outcome_stage(path, {'run': 1})
    with pytest.raises(FileExistsError):
        script.consume_outcome_stage(path, {'run': 2})
    assert json.loads(path.read_text()) == {'run': 1}


@pytest.mark.parametrize('lower,nbad,expected', [(0,2,False), (1e-16,2,False), (.01,2,True), (.01,0,False), (.01,4,False)])
def test_primary_criterion_is_strict_and_needs_discrimination(lower, nbad, expected):
    """Numerical zero and all/no-bad cohorts cannot produce a positive primary claim."""
    paired = pd.DataFrame([{'method': method, 'measure': 'review_auc', 'bad_rule': 'dice_below_083',
                            'diff': .03, 'diff_ci_low': lower, 'diff_ci_high': .1}
                           for method in ['combo', 'far_disagree_per_volume']])
    cfg = {'combinations': [{'name': 'combo'}], 'fresh': {'comparison_atol': 1e-12}}
    result = script.primary_decisions(paired, nbad, 4, cfg)
    assert all(row['meets_per_comparison_criterion'] == expected for row in result)


def test_budget_quality_counts_displaced_severe_cases():
    """Correcting two bad cases gives quality .94; correcting two already-good cases gives only .80."""
    frame = pd.DataFrame({'dice': [.5,.6,.9,.9,.9], 'g1': [5,4,3,2,1], 'combo': [1,2,3,4,5]})
    cfg = {'checks': {'methods': ['g1','combo'], 'primary': 'g1', 'review_fraction': .4, 'bad_dice_threshold': .83},
           'evaluation': {'n_bootstrap': 20, 'ci_level': .95}, 'seed': 42}
    result = script.budget_statistics(frame, cfg).set_index(['method','measure'])
    assert result.loc[('g1','quality_after_correction'),'value'] == pytest.approx(.94)
    assert result.loc[('combo','quality_after_correction'),'value'] == pytest.approx(.80)
    assert result.loc[('combo','bad_capture'),'difference_to_g1'] == -2
    assert result.loc[('combo','false_alarms'),'difference_to_g1'] == 2


def save_image(path: Path, array: np.ndarray, probability: bool = False, affine=None) -> str:
    """Write only a synthetic test NIfTI, using the project's standard probability encoding when requested."""
    image = nib.Nifti1Image(array.astype(np.uint8), np.eye(4) if affine is None else affine)
    if probability:
        image.header.set_slope_inter(1/255, 0)
    nib.save(image, path)
    return str(path)


def test_case_scores_does_not_read_label_path(tmp_path):
    """The entire GT-free pipeline works even when the supplied label path does not exist."""
    mask = np.zeros((12,6,6), np.uint8)
    mask[1:4,1:4,1:4] = 1
    mask[8:11,1:4,1:4] = 1
    paths = {'ct': save_image(tmp_path/'ct.nii.gz', np.full(mask.shape,40)),
             'mask': save_image(tmp_path/'mask.nii.gz',mask), 'm2_mask': save_image(tmp_path/'m2_mask.nii.gz',mask),
             'prob': save_image(tmp_path/'prob.nii.gz',mask*255,True),
             'm2_prob': save_image(tmp_path/'m2_prob.nii.gz',mask*255,True),
             'label': str(tmp_path/'do_not_read_this_label.nii.gz')}
    cfg = {'fresh': {'body_threshold_hu': -500, 'min_component_ml': .02, 'max_components': 2,
                     'connectivity': 1, 'border_mm': 5, 'grid_band_mm': 6}}
    result = script.case_scores(paths, cfg)
    assert result['model_diff_mean_region'] == 0 and result['far_disagree_per_volume'] == 0
    assert result['plaus_missing_side'] == 0 and result['shape_hull_deficit_max'] == pytest.approx(0)
    assert not Path(paths['label']).exists()


@pytest.fixture
def synthetic_handoff(tmp_path):
    """Provide one synthetic case with test-only coordination evidence for header validation."""
    data = np.zeros((3,3,3),np.uint8)
    row = {'case_id':'synthetic'}
    for key in script.INPUT_COLUMNS:
        row[key] = save_image(tmp_path/f'{key}.nii.gz',data,key in ['prob','m2_prob'])
    pd.DataFrame([row]).to_csv(tmp_path/'manifest.csv',index=False)
    pd.DataFrame({'case_id':['synthetic']}).to_csv(tmp_path/'cohort.csv',index=False)
    (tmp_path/'inference.json').write_text('{"synthetic_test":true}')
    lock = {'checkpoint_sha256':{'model_3mm':'test3','model_6mm':'test6'}}
    handoff = {'bjorn_notified_at':'test only','frederik_confirmed_at':'test only','evidence':'synthetic fixture only',
               'ranking_metrics_not_previously_evaluated':True,'checkpoint_sha256':lock['checkpoint_sha256'],
               'inference_metadata':'inference.json','inputs_manifest':'manifest.csv'}
    script.write_json(tmp_path/'handoff.json',handoff)
    return tmp_path,lock,tmp_path/'handoff.json'


def test_ready_preflight_reads_headers_without_loading_voxels(synthetic_handoff, monkeypatch):
    """Matching grids and standard probability headers pass without requesting any image array."""
    folder,lock,handoff = synthetic_handoff
    monkeypatch.setattr(nib.Nifti1Image,'get_fdata',lambda *args,**kwargs: pytest.fail('Voxel data loaded'))
    result = script.preflight({},folder,lock,handoff)
    assert result['ready'] and result['n_cases']==1
    assert len(script.require_ready(folder))==1


def test_native_flipped_prediction_is_rejected(synthetic_handoff):
    """A map that would only align after reorientation cannot be silently loaded as a canonical array."""
    folder,lock,handoff = synthetic_handoff
    affine = np.diag([-1,1,1,1])
    affine[0,3] = 2
    save_image(folder/'mask.nii.gz',np.zeros((3,3,3)),affine=affine)
    with pytest.raises(ValueError,match='Geometry mismatch'):
        script.preflight({},folder,lock,handoff)


def test_nonstandard_probability_encoding_is_rejected(synthetic_handoff):
    """Unscaled bytes are not interpreted as organ probabilities merely because the grid matches."""
    folder,lock,handoff = synthetic_handoff
    save_image(folder/'prob.nii.gz',np.zeros((3,3,3)))
    with pytest.raises(ValueError,match='probability representation'):
        script.preflight({},folder,lock,handoff)


def test_fresh_outcome_stage_runs_both_questions_then_refuses_reopening(tmp_path, monkeypatch):
    """A tiny synthetic integration covers Dice, metrics, decisions and the single-opening guard."""
    mask = np.zeros((6,6,6),np.uint8)
    mask[1:3,1:3,1:3] = 1
    inventory = []
    for i in range(4):
        truth = mask.copy()
        if i == 1:
            truth[3,1,1:3] = 1
        elif i == 2:
            truth[1,:,:] = 0
        elif i == 3:
            truth = np.zeros_like(mask)
            truth[3:5,3:5,3:5] = 1
        row = {'case_id': f'case_{i}'}
        for key, data in [('mask',mask),('m2_mask',mask),('label',truth)]:
            row[key] = save_image(tmp_path/f'{i}_{key}.nii.gz',data)
            row[key+'_sha256'] = digest(Path(row[key]))
        inventory.append(row)
    inventory = pd.DataFrame(inventory)
    inventory.to_csv(tmp_path/'input_inventory.csv',index=False)
    scores = pd.DataFrame({'case_id': inventory.case_id, 'model_diff_mean_region': [.1,.2,.3,.4],
                            'g1_missing_hull_mean': [.1,.4,.3,.2], 'far_disagree_per_volume': [0,.01,.02,.04],
                            'entropy_mean_region': [.12,.23,.32,.4], 'neg_volume_ml': [-8,-9,-10,-11]})
    scores.to_csv(tmp_path/'scores.csv',index=False)
    (tmp_path/'lock.json').write_text('{}')
    script.write_json(tmp_path/'scores_provenance.json', {'scores_sha256': digest(tmp_path/'scores.csv'),
                      'lock_sha256': digest(tmp_path/'lock.json'), 'input_inventory_sha256': digest(tmp_path/'input_inventory.csv')})
    cfg = {'seed':42, 'fresh':{'comparison_atol':1e-12},
           'dataset': {'organ_definition':'all', 'organ_definitions': {'all':{'organ_labels':[1]}}},
           'checks': {'methods':list(scores.columns[1:]),'primary':'model_diff_mean_region',
                      'references':['model_diff_mean_region'],'review_fraction':.2,'bad_dice_threshold':.83},
           'combinations':[{'name':'g1_missing_hull_mean'}],
           'evaluation': {'bad_rules':[{'name':'dice_below_083','rule':'threshold','value':.83},
                                      {'name':'worst10','rule':'worst_fraction','value':.1}],
                          'n_bootstrap':20,'ci_level':.95,'n_random_orders':4}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConstantInputWarning)
        result = script.evaluate_fresh(cfg,tmp_path,inventory)
    assert result['n_cases']==4 and len(result['decisions'])==2
    assert len(pd.read_csv(tmp_path/'dice.csv'))==4
    assert len(pd.read_csv(tmp_path/'budget_statistics.csv'))==15
    monkeypatch.setattr(script,'load_canonical',lambda path: pytest.fail('Fresh GT reopened'))
    with pytest.raises(FileExistsError):
        script.evaluate_fresh(cfg,tmp_path,inventory)
