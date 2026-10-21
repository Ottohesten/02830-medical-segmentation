"""Verify and visualize the completed development-only candidate run without computing scores."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import yaml

root = Path.cwd()
out = root / 'results/kits_robustness_dev'
cfg_path = root / 'configs/kits_robustness_dev.yaml'
cfg = yaml.safe_load(cfg_path.read_text())
machine = json.loads((out / 'machine.json').read_text())
assert machine['status'] == 'complete'
assert machine['config_sha256'] == hashlib.sha256(cfg_path.read_bytes()).hexdigest()
for path, digest in machine['source_sha256'].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
timings = pd.read_csv(out / 'inference_times.csv')
dice = pd.read_csv(out / 'dice.csv')
variants = ['clean'] + [s['name'] for s in cfg['perturbations']]
expected = {(case, variant) for case in cfg['dataset']['split']['dev'] for variant in variants}
for frame in [timings, dice]:
    assert len(frame) == len(expected)
    assert set(zip(frame.case_id, frame.variant)) == expected
assert dice['split'].eq('dev').all()
assert dice.dice.between(0, 1).all()
manifest = []
for row in timings.itertuples(index=False):
    maps = []
    for kind in ['mask', 'prob']:
        path = root / cfg['paths']['predictions_dir'] / cfg['name'] / row.variant / f'{row.case_id}_{kind}.nii.gz'
        image = nib.load(path)
        assert image.shape == tuple(map(int, row.shape.split('x')))
        maps.append(image)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest.append({'case_id': row.case_id, 'variant': row.variant, 'kind': kind,
                         'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': digest})
    assert np.allclose(maps[0].affine, maps[1].affine)
pd.DataFrame(manifest).to_csv(out / 'prediction_manifest.csv', index=False)
check = {'status': 'passed', 'dev_cases': len(cfg['dataset']['split']['dev']),
         'variants_including_clean': len(variants), 'timing_rows': len(timings), 'dice_rows': len(dice),
         'prediction_maps': len(manifest), 'paired_prediction_grids_match': True,
         'config_and_code_hashes_match': True, 'test_dice_rows': int((dice['split'] == 'test').sum()),
         'score_files_present': [p.name for p in out.glob('*score*')],
         'prediction_bytes': sum(row['bytes'] for row in manifest)}
assert check['test_dice_rows'] == 0 and not check['score_files_present']
(out / 'verification.json').write_text(json.dumps(check, indent=2) + '\n')

drops = pd.read_csv(out / 'dev_dice_drops.csv')
summary = pd.read_csv(out / 'dev_degradation_summary.csv')
choice = json.loads((out / 'strength_suggestion.json').read_text())
names = [s['name'] for s in cfg['perturbations']]
values = [drops[drops.variant == name].dice_drop.to_numpy() for name in names]
fig, ax = plt.subplots(figsize=(11, 5))
boxes = ax.boxplot(values, patch_artist=True, showfliers=False)
for name, box in zip(names, boxes['boxes']):
    box.set_facecolor('#a7d7bd' if name in choice['selected_variants'] else '#dce2e8')
rng = np.random.default_rng(cfg['seed'])
for index, points in enumerate(values, 1):
    ax.scatter(index + rng.uniform(-0.12, 0.12, len(points)), points, s=16, color='#283747', alpha=0.65)
ax.axhline(0, color='#555555', linewidth=0.7)
ax.axhline(cfg['robustness_selection']['min_dice_drop'], color='#b04232', linestyle='--',
           label='Clear-drop threshold (0.02)')
ax.set_xticks(range(1, len(names) + 1), labels=names, rotation=30, ha='right')
ax.set_ylabel('Dice(clean) - Dice(perturbed)')
ax.set_title('Development only: paired Dice degradation (20 scans per candidate)')
ax.text(0.01, 0.98, 'Green: selected by the predeclared Dice-only rule', transform=ax.transAxes, va='top', fontsize=9)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.2)
fig.tight_layout()
(out / 'figures').mkdir(exist_ok=True)
fig.savefig(out / 'figures/dev_dice_degradation.png', dpi=cfg['visualization']['dpi'])
plt.close(fig)
print(json.dumps(check, indent=2))
print(summary.to_string(index=False))
print(json.dumps(choice, indent=2))
