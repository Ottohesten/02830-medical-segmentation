"""Reproduce the step 4 timing summary and provenance checks from saved artifacts; no inference or evaluation.

From the repository root: PYTHONPATH=src:. .venv-gpu/bin/python results/kits_robustness_test/report_source.py
--config configs/kits_robustness_test.yaml
"""

import json

import pandas as pd

from scripts.run_robustness_test import sha256, source_hashes
from segreview.config import config_arg_parser, load_config, results_dir
from segreview.io import prediction_dir


def main():
    """Read completed run artifacts, require matching provenance/coverage, and write timing and audit summaries."""
    args = config_arg_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    out = results_dir(cfg)
    machine = json.loads((out / 'machine.json').read_text())
    plan = json.loads((out / 'run_plan.json').read_text())
    verification = json.loads((out / 'verification.json').read_text())
    assert machine['status'] == 'complete' and verification['status'] == 'passed'
    assert machine['config_sha256'] == plan['config_sha256'] == sha256(args.config)
    assert machine['source_sha256'] == plan['source_sha256'] == source_hashes(cfg)
    rows = pd.concat([pd.read_csv(out / file) for file in ['inference_times.csv', 'second_model_times.csv']],
                     ignore_index=True)
    expected = {(case, variant, resolution) for case in plan['case_ids'] for variant in plan['variants']
                for resolution in plan['resolutions']}
    assert len(rows) == len(expected) and set(zip(rows.case_id, rows.variant, rows.resolution)) == expected
    summary = rows.groupby(['resolution', 'variant']).seconds.agg(['count', 'sum', 'median', 'min', 'max'])
    summary.to_csv(out / 'timing_summary.csv')
    manifest = pd.read_csv(out / 'prediction_manifest.csv')
    prediction_root = prediction_dir(cfg).parent
    assert len(list(prediction_root.rglob('*.nii.gz'))) == plan['n_prediction_files'] == len(manifest)
    assert int(manifest.bytes.sum()) == verification['prediction_bytes']
    assert not any(out.glob('*dice*.csv')) and not any(out.glob('*scores*.csv'))
    audit = {'status': 'passed', 'config_and_source_hashes_match_plan': True,
             'exact_case_variant_model_timing_coverage': True, 'n_cases': plan['n_cases'],
             'n_predictions': len(rows), 'n_prediction_files': len(manifest),
             'prediction_bytes': int(manifest.bytes.sum()), 'dice_or_scores_present': False,
             'elapsed_minutes': machine['elapsed_seconds'] / 60,
             'verification_wall_seconds': machine['elapsed_seconds'] - machine['main_model_wall_seconds']
                                          - machine['second_model_wall_seconds'],
             'timing_scope': 'Prediction intervals exclude loading, perturbing, saving, and model setup; '
                             'model wall intervals include them. Total wall time also includes verification.'}
    (out / 'post_run_checks.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))
    print(summary.to_string())


if __name__ == '__main__':
    main()
