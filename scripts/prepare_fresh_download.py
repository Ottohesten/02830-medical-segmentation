"""Plan or download exactly the frozen fresh-pool CTs/labels, without changing shared selection code.

The older data selector on this branch ignores dataset.offset. This logistics helper checks Frederik's
pool config against the published cohort and never rewrites the locked source case list or existing data.
Planning is offline; use --stage download only after Otto approves the download.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

from segreview.config import REPO_ROOT, config_arg_parser, load_config
from segreview.data import CASE_LIST_FILE, _download_file, dataset_dir, select_case_ids
from segreview.machine import machine_info

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.check_missed_errors import digest
from scripts.fresh_missed_errors import experiment, now, pool_allocation, require_lock, write_json


def download_plan(cfg: dict, pool: dict, source: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    """Build an offline URL/path plan, requiring the pool config to reproduce the exact frozen cohort."""
    settings, dataset = cfg['fresh'], pool['dataset']
    if (pool['seed'] != cfg['seed'] or dataset['selection'] != 'random' or
            dataset.get('offset', 0) != settings['skip_cases'] or dataset['n_cases'] != settings['n_cases']):
        raise ValueError('Pool seed/offset/count must match the frozen selection')
    if dataset_dir(pool) != dataset_dir(source):
        raise ValueError('Pool downloads must use the agreed source dataset folder')
    all_ids = (dataset_dir(source) / CASE_LIST_FILE).read_text().split()
    expected = pool_allocation(all_ids, set(select_case_ids(source, all_ids)), pool['seed'],
                               dataset['offset'], dataset['n_cases'])
    if set(expected.case_id) != set(cohort.case_id) or cohort.case_id.duplicated().any():
        raise ValueError('Pool config does not reproduce the published cohort')
    rows = []
    dl = dataset['download']
    for case_id in sorted(cohort.case_id):
        for kind, key in [('ct', 'image_file'), ('label', 'label_file')]:
            relative = dl[key].format(case_id=case_id)
            path = dataset_dir(pool) / relative
            rows.append({'case_id': case_id, 'kind': kind, 'url': f"{dl['base_url']}/{relative}",
                         'path': str(path.relative_to(REPO_ROOT)), 'already_present': path.is_file()})
    return pd.DataFrame(rows)


def execute_plan(plan: pd.DataFrame, max_seconds: float) -> dict:
    """Download only missing planned files; preserve existing inputs and stop between files at the time cap."""
    started = time.monotonic()
    downloaded, skipped = 0, 0
    for row in plan.to_dict('records'):
        path = REPO_ROOT / row['path']
        if path.is_file():
            skipped += 1
            continue
        if time.monotonic() - started >= max_seconds:
            raise RuntimeError('Download time cap reached; ask Otto before continuing a longer run')
        _download_file(row['url'], path)
        downloaded += 1
        print(f"Downloaded {row['case_id']} {row['kind']}", flush=True)
    return {'downloaded_files': downloaded, 'existing_files_preserved': skipped,
            'seconds': time.monotonic() - started, 'fresh_voxel_arrays_read': False}


def main() -> None:
    """Save a reviewable offline plan, or execute that unchanged plan after explicit download approval."""
    parser = config_arg_parser(__doc__)
    parser.add_argument('--pool-config', type=Path, required=True)
    parser.add_argument('--stage', choices=['plan', 'download'], default='plan')
    args = parser.parse_args()
    cfg, pool = load_config(args.config), load_config(args.pool_config)
    output, source = experiment(cfg)
    require_lock(cfg, args.config, output, source)
    plan = download_plan(cfg, pool, source, pd.read_csv(output / 'cohort.csv'))
    path = output / 'download_plan.csv'
    if args.stage == 'plan':
        plan.to_csv(path, index=False)
        result = {'planned_files': len(plan), 'missing_files': int((~plan.already_present).sum()),
                  'download_started': False, 'fresh_voxel_arrays_read': False}
    else:
        reviewed = pd.read_csv(path)
        pd.testing.assert_frame_equal(plan.drop(columns='already_present'), reviewed.drop(columns='already_present'))
        result = execute_plan(plan, cfg['fresh']['benchmark']['max_stage_seconds'])
        write_json(output / 'download_result.json', result)
    info = machine_info()
    info.update(date=now(), stage=args.stage, seed=cfg['seed'], execution_device='cpu',
                scientific_lock_sha256=digest(output / 'lock.json'), pool_config_sha256=digest(args.pool_config),
                download_plan_sha256=digest(path), **result)
    write_json(output / f'download_{args.stage}_machine.json', info)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
