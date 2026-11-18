"""Check fresh download logistics: exact frozen IDs, offline planning and preservation of existing data."""

from copy import deepcopy

import pytest

from scripts import prepare_fresh_download as helper
from segreview.data import select_case_ids


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Create a tiny old cohort and disjoint next-draw pool without any image data or network."""
    monkeypatch.setattr(helper, 'REPO_ROOT', tmp_path)
    folder = tmp_path / 'data/raw/kits23'
    folder.mkdir(parents=True)
    all_ids = [f'case_{i:05d}' for i in range(10)]
    (folder / 'case_list.txt').write_text('\n'.join(all_ids) + '\n')
    source = {'seed': 42, 'paths': {'raw_dir': tmp_path / 'data/raw'},
              'dataset': {'root': 'kits23', 'selection': 'random', 'n_cases': 3,
                          'download': {'base_url': 'https://example.invalid/data',
                                       'image_file': 'images/{case_id}.nii.gz',
                                       'label_file': 'segmentations/{case_id}.nii.gz'}}}
    pool = deepcopy(source)
    pool['dataset'].update(offset=3, n_cases=2)
    cfg = {'seed': 42, 'fresh': {'skip_cases': 3, 'n_cases': 2}}
    cohort = helper.pool_allocation(all_ids, set(select_case_ids(source, all_ids)), 42, 3, 2)
    return cfg, pool, source, cohort, folder


def test_plan_selects_exact_next_cases_without_network_or_list_changes(setup, monkeypatch):
    """Planning needs no network and does not silently use the old selector's first cohort."""
    cfg, pool, source, cohort, folder = setup
    before = (folder / 'case_list.txt').read_bytes()
    monkeypatch.setattr(helper, '_download_file', lambda *a: pytest.fail('Planning must be offline'))
    plan = helper.download_plan(cfg, pool, source, cohort)
    assert len(plan) == 4 and set(plan.case_id) == set(cohort.case_id)
    assert not set(plan.case_id) & set(select_case_ids(source, (folder / 'case_list.txt').read_text().split()))
    assert set(plan.kind) == {'ct', 'label'} and not plan.already_present.any()
    assert (folder / 'case_list.txt').read_bytes() == before


@pytest.mark.parametrize('field,value', [('offset', 0), ('n_cases', 3), ('selection', 'first')])
def test_wrong_pool_selection_is_rejected(setup, field, value):
    """A config selecting the wrong draw cannot become an approved download plan."""
    cfg, pool, source, cohort, _ = setup
    pool['dataset'][field] = value
    with pytest.raises(ValueError, match='seed/offset/count'):
        helper.download_plan(cfg, pool, source, cohort)


def test_changed_frozen_cohort_is_rejected(setup):
    """Correct pool settings are insufficient if the cohort IDs have been replaced."""
    cfg, pool, source, cohort, _ = setup
    cohort.loc[0, 'case_id'] = 'case_99999'
    with pytest.raises(ValueError, match='published cohort'):
        helper.download_plan(cfg, pool, source, cohort)


def test_execution_preserves_existing_files_and_only_downloads_missing(setup, monkeypatch):
    """Execution reuses existing bytes and never fetches a case listing or unplanned scan."""
    cfg, pool, source, cohort, folder = setup
    plan = helper.download_plan(cfg, pool, source, cohort)
    existing = helper.REPO_ROOT / plan.iloc[0].path
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'existing input')
    calls = []

    def fake_download(url, path):
        """Record synthetic downloads while creating small stand-ins for the requested files."""
        calls.append((url, path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'new input')

    monkeypatch.setattr(helper, '_download_file', fake_download)
    before = (folder / 'case_list.txt').read_bytes()
    result = helper.execute_plan(plan, 30)
    assert result['downloaded_files'] == len(calls) == 3
    assert result['existing_files_preserved'] == 1 and existing.read_bytes() == b'existing input'
    assert (folder / 'case_list.txt').read_bytes() == before


def test_time_cap_stops_before_another_file(setup, monkeypatch):
    """An elapsed budget stops execution before starting another network transfer."""
    cfg, pool, source, cohort, _ = setup
    plan = helper.download_plan(cfg, pool, source, cohort)
    monkeypatch.setattr(helper, '_download_file', lambda *a: pytest.fail('Expired run must not download'))
    with pytest.raises(RuntimeError, match='time cap'):
        helper.execute_plan(plan, 0)
