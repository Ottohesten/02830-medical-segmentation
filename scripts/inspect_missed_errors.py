"""Inspect exploratory kits100 errors missed by the locked G1 queue (task step 2).

Read saved scores, masks and CTs without inference or writes to the source experiment.
Write a pooled queue, error diagnostics and multiple-slice figures to this config's results.
Ground truth selects/illustrates errors only; no new GT-free score is defined here.
"""

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from segreview.config import REPO_ROOT, config_arg_parser, load_config, setup_compute
from segreview.data import case_split, list_cases, load_canonical, organ_definition
from segreview.io import load_map, map_path
from segreview.machine import machine_info
from segreview.metrics import dice
from segreview.uncertainty import uncertainty_region

STATUS = 'exploratory; all kits100 cases are development data for this follow-up'


def to_display(plane: np.ndarray) -> np.ndarray:
    """Orient a canonical axial/coronal plane with patient right on the left, as in show_case.py."""
    return plane.T[::-1, ::-1]


def digest(path: Path) -> str:
    """Return the SHA-256 of an input/output file without loading all its bytes at once."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_output(cfg: dict, source: dict) -> Path:
    """Return an independent output folder, rejecting overlap with either locked source folder."""
    output = (cfg['paths']['results_dir'] / cfg['name']).resolve()
    protected = [(source['paths']['results_dir'] / source['name']).resolve(),
                 (source['paths']['predictions_dir'] / source['name']).resolve()]
    for path in protected:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Output must not overlap a read-only source folder')
    return output


def build_queue(scores: pd.DataFrame, outcomes: pd.DataFrame, expected_ids: set[str],
                settings: dict) -> pd.DataFrame:
    """Rank the exact cohort by saved score, then attach Dice and mark misses after a fixed budget.

    Inputs: saved score/Dice tables, expected case IDs, variant/score/budget settings.
    Output: one row per case; score ties use ascending case ID and never use ground truth.
    """
    score = settings['score']
    a = scores.loc[scores['variant'].eq(settings['variant']), ['case_id', 'split', score]].copy()
    b = outcomes.loc[outcomes['variant'].eq(settings['variant']), ['case_id', 'split', 'dice']].copy()
    for frame, value in ((a, score), (b, 'dice')):
        if set(frame['case_id']) != expected_ids or frame['case_id'].duplicated().any():
            raise ValueError('Tables must cover exactly the expected cases without duplicates')
        if not np.isfinite(frame[value]).all():
            raise ValueError('Nonfinite score or Dice')
    if not b['dice'].between(0, 1).all():
        raise ValueError('Dice must lie in [0, 1]')
    if settings['tie_break'] != 'case_id_ascending':
        raise ValueError('Unsupported tie rule')
    fraction = settings['review_fraction']
    if not 0 < fraction <= 1:
        raise ValueError('Review fraction must lie in (0, 1]')
    # Sort before joining outcomes so the queue cannot depend on Dice, even in a tie.
    a = a.sort_values([score, 'case_id'], ascending=[False, True]).reset_index(drop=True)
    a['rank'] = np.arange(1, len(a) + 1)
    q = a.merge(b, on=['case_id', 'split'], validate='one_to_one')
    if len(q) != len(a):
        raise ValueError('Score and Dice split labels differ')
    q = q.rename(columns={'split': 'original_g1_split'})
    q['in_top_budget'] = q['rank'].le(math.ceil(len(q) * fraction))
    q['bad'] = q['dice'].lt(settings['bad_dice_threshold'])
    q['missed_bad'] = q['bad'] & ~q['in_top_budget']
    return q


def error_diagnostics(labels: np.ndarray, pred: np.ndarray, second: np.ndarray,
                      prob: np.ndarray, spacing: tuple, definition: dict,
                      border_mm: float, low_threshold: float, label_names: dict) -> dict:
    """Measure error types and confidence using GT for evaluation only; return descriptive counts/shares."""
    if definition['ignore_labels']:
        raise ValueError('This inspection currently requires a union with no ignored labels')
    truth = np.isin(labels, definition['organ_labels'])
    fn = truth & ~pred
    fp = pred & ~truth
    region = uncertainty_region(pred, spacing, border_mm)
    n_fn = int(fn.sum())
    voxel_ml = float(np.prod(spacing)) / 1000
    values = {'dice_recomputed': dice(pred, truth), 'dice_6mm': dice(second, truth),
              'false_negative_voxels': n_fn, 'false_positive_voxels': int(fp.sum()),
              'false_negative_ml': n_fn * voxel_ml, 'false_positive_ml': int(fp.sum()) * voxel_ml}
    for name, mask in [('fn_low_probability', prob <= low_threshold), ('fn_zero_probability', prob == 0),
                       ('fn_outside_region', ~region), ('fn_recovered_by_6mm', second)]:
        count = int((fn & mask).sum())
        values[name + '_voxels'] = count
        values[name + '_share'] = count / n_fn if n_fn else np.nan
    for label in definition['organ_labels']:
        name = label_names[label]
        part = labels == label
        total = int(part.sum())
        missed = int((part & ~pred).sum())
        values.update({name + '_truth_voxels': total, name + '_missed_voxels': missed,
                       name + '_missed_share': missed / total if total else np.nan,
                       name + '_missed_ml': missed * voxel_ml})
    return values


def inspection_slices(labels: np.ndarray, pred: np.ndarray, organ_labels: list[int],
                      label_names: dict, targets: list[str]) -> list[tuple[int, int, str]]:
    """Choose distinct axial error peaks plus one coronal peak, using GT only to illustrate errors."""
    truth = np.isin(labels, organ_labels)
    masks = {label_names[label]: (labels == label) & ~pred for label in organ_labels}
    masks.update(false_positive=pred & ~truth, organ=truth)
    chosen = []
    used = set()
    for target in targets:
        mask = masks[target]
        counts = mask.sum(axis=(0, 1))
        if counts.max() == 0:
            continue
        z = int(counts.argmax())
        if z not in used:
            chosen.append((2, z, f'axial z={z}: peak {target}'))
            used.add(z)
    missed = truth & ~pred
    target = missed if missed.any() else truth
    if target.any():
        y = int(target.sum(axis=(0, 2)).argmax())
        chosen.append((1, y, f'coronal y={y}: peak missed area'))
    return chosen


def plot_inspection(ct: np.ndarray, labels: np.ndarray, pred: np.ndarray, second: np.ndarray,
                    prob: np.ndarray, entropy: np.ndarray, spacing: tuple, row: dict,
                    cfg: dict, source: dict, path: Path) -> list[dict]:
    """Save CT/GT, both masks, probability and entropy at multiple error peaks; return shown slice indices."""
    vis = cfg['visualization']
    definition = organ_definition(source)
    truth = np.isin(labels, definition['organ_labels'])
    slices = inspection_slices(labels, pred, definition['organ_labels'], source['dataset']['label_names'],
                               vis['axial_targets'])
    # A common GT-based crop makes missed tissue visible; this is a diagnostic, not a score input.
    union = truth | pred | second
    box = []
    for axis in range(3):
        occupied = np.flatnonzero(union.any(axis=tuple(i for i in range(3) if i != axis)))
        pad = int(np.ceil(vis['context_margin_mm'] / spacing[axis]))
        box.append(slice(max(0, int(occupied[0]) - pad), min(ct.shape[axis], int(occupied[-1]) + pad + 1)))
    fig, axes = plt.subplots(len(slices), 4, figsize=(16, 3.6 * len(slices)), squeeze=False)
    low = vis['ct_window_level'] - vis['ct_window_width'] / 2
    high = vis['ct_window_level'] + vis['ct_window_width'] / 2
    colors = {1: 'lime', 2: 'orange', 3: 'cyan'}
    for r, (axis, index, description) in enumerate(slices):
        selection = list(box)
        selection[axis] = index
        selection = tuple(selection)
        dims = [i for i in range(3) if i != axis]
        extent = [0, ct[selection].shape[0] * spacing[dims[0]],
                  0, ct[selection].shape[1] * spacing[dims[1]]]
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(to_display(ct[selection]), cmap='gray', vmin=low, vmax=high, extent=extent)
            if c < 2:
                for label, color in colors.items():
                    plane = to_display(labels[selection] == label)
                    if plane.any() and not plane.all():
                        ax.contour(plane, levels=[.5], colors=color, linewidths=1, extent=extent, origin='upper')
                mask = pred if c == 0 else second
                plane = to_display(mask[selection])
                if plane.any() and not plane.all():
                    ax.contour(plane, levels=[.5], colors='red' if c == 0 else 'magenta',
                               linewidths=1, extent=extent, origin='upper')
            else:
                data = prob if c == 2 else entropy
                ax.imshow(to_display(data[selection]), cmap='magma', vmin=0, vmax=1, alpha=.65, extent=extent)
                plane = to_display(truth[selection])
                if plane.any() and not plane.all():
                    ax.contour(plane, levels=[.5], colors='white', linewidths=.8, extent=extent, origin='upper')
            titles = ['3 mm prediction (red)', '6 mm prediction (magenta)', '3 mm probability (0–1)', 'entropy (0–1 bits)']
            ax.set_title(description + '\n' + titles[c], fontsize=9)
            ax.axis('off')
    fig.suptitle(f"EXPLORATORY — {row['case_id']} | queue rank {row['rank']}/{cfg['dataset']['n_cases']} | Dice {row['dice']:.4f}\n"
                 'GT: kidney lime / tumor orange / cyst cyan; white = GT union on heatmaps\n'
                 'GT selects error peaks and crop; these are diagnostic views, not score inputs.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - .8 / (3.6 * len(slices))))
    fig.savefig(path, dpi=vis['dpi'])
    plt.close(fig)
    return [{'axis': axis, 'index': index, 'description': text} for axis, index, text in slices]


def inspect_case(case, row: dict, cfg: dict, source: dict, output: Path) -> tuple[dict, dict]:
    """Load one missed case, verify grids/saved Dice, and return diagnostics plus figure/input provenance."""
    start = time.monotonic()
    variant = cfg['missed_errors']['variant']
    image = load_canonical(case.image_path)
    label_image = load_canonical(case.label_path)
    ct = image.get_fdata(dtype=np.float32)
    labels = np.asanyarray(label_image.dataobj)
    paths = [case.image_path, case.label_path]
    for kind in ('mask', 'm2_mask', 'prob', 'entropy'):
        path = map_path(source, variant, case.case_id, kind)
        saved = nib.load(path)
        if saved.shape != ct.shape or not np.allclose(saved.affine, image.affine, rtol=0, atol=1e-4):
            raise ValueError(f'Prediction geometry differs: {path}')
        paths.append(path)
    if label_image.shape != ct.shape or not np.allclose(label_image.affine, image.affine, rtol=0, atol=1e-4):
        raise ValueError('Label geometry differs')
    pred = load_map(source, variant, case.case_id, 'mask').astype(bool)
    second = load_map(source, variant, case.case_id, 'm2_mask').astype(bool)
    prob = load_map(source, variant, case.case_id, 'prob')
    entropy = load_map(source, variant, case.case_id, 'entropy')
    spacing = image.header.get_zooms()[:3]
    diagnostics = error_diagnostics(labels, pred, second, prob, spacing, organ_definition(source),
                                    source['uncertainty']['border_mm'],
                                    cfg['missed_errors']['low_probability_threshold'], source['dataset']['label_names'])
    if not np.isclose(diagnostics['dice_recomputed'], row['dice'], rtol=0,
                      atol=cfg['missed_errors']['dice_check_atol']):
        raise ValueError('Saved 3 mm Dice does not match the loaded maps')
    path = output / 'figures' / f'exploratory_{case.case_id}_inspection.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = plot_inspection(ct, labels, pred, second, prob, entropy, spacing, row, cfg, source, path)
    evidence = {'case_id': case.case_id, 'slices': shown, 'figure': str(path.relative_to(output)),
                'figure_sha256': digest(path),
                'inputs_sha256': {str(p.relative_to(REPO_ROOT)): digest(p) for p in paths},
                'seconds': time.monotonic() - start}
    return {**row, **diagnostics, 'status': STATUS}, evidence


def main() -> None:
    """Build the saved queue and optionally inspect its misses; write outputs/provenance separately."""
    parser = config_arg_parser(__doc__)
    parser.add_argument('--stage', choices=['queue', 'images', 'all'], default='all')
    parser.add_argument('--case', action='append', help='inspect only these missed cases (repeatable)')
    args = parser.parse_args()
    start = time.monotonic()
    cfg = load_config(args.config)
    setup_compute(cfg)
    source_path = REPO_ROOT / cfg['missed_errors']['source_config']
    source = load_config(source_path)
    output = validate_output(cfg, source)
    output.mkdir(parents=True, exist_ok=True)
    cases = list_cases(source)
    own_cases = list_cases(cfg)
    if {c.case_id for c in cases} != {c.case_id for c in own_cases} or cfg['seed'] != source['seed']:
        raise ValueError('Own/source cohort and seed must agree')
    source_results = source['paths']['results_dir'] / source['name']
    paths = [source_results / name for name in ['scores.csv', 'dice.csv']]
    scores, outcomes = (pd.read_csv(p) for p in paths)
    queue = build_queue(scores, outcomes, {c.case_id for c in cases}, cfg['missed_errors'])
    if not all(row.original_g1_split == case_split(source, row.case_id) for row in queue.itertuples()):
        raise ValueError('Original G1 split differs from source config')
    queue['status'] = STATUS
    queue.to_csv(output / 'exploratory_queue.csv', index=False)
    missed = queue.loc[queue['missed_bad']]
    missed.to_csv(output / 'exploratory_missed_cases.csv', index=False)
    budget = int(queue['in_top_budget'].sum())
    summary = {'status': STATUS, 'n_cases': len(queue), 'review_budget': budget,
               'n_bad': int(queue['bad'].sum()), 'n_bad_in_budget': int((queue['bad'] & queue['in_top_budget']).sum()),
               'n_missed_bad': len(missed), 'score': cfg['missed_errors']['score'],
               'bad_dice_threshold': cfg['missed_errors']['bad_dice_threshold'],
               'tie_rule': cfg['missed_errors']['tie_break'],
               'boundary_tied': bool(0 < budget < len(queue) and
                                     queue.iloc[budget-1][cfg['missed_errors']['score']] ==
                                     queue.iloc[budget][cfg['missed_errors']['score']]),
               'missed_case_ids': missed['case_id'].tolist()}
    (output / 'exploratory_queue_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.case and not set(args.case).issubset(set(missed['case_id'])):
        raise ValueError('--case must select only missed bad cases')
    if args.stage != 'queue':
        selected = missed if args.case is None else missed[missed['case_id'].isin(args.case)]
        by_id = {case.case_id: case for case in cases}
        details_path = output / 'exploratory_error_diagnostics.csv'
        evidence_path = output / 'exploratory_inspection_evidence.json'
        details = pd.read_csv(details_path) if details_path.exists() else pd.DataFrame()
        evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
        for row in selected.to_dict('records'):
            values, proof = inspect_case(by_id[row['case_id']], row, cfg, source, output)
            if not details.empty:
                details = details.loc[~details['case_id'].eq(row['case_id'])]
            details = pd.concat([details, pd.DataFrame([values])], ignore_index=True)
            details.sort_values('rank').to_csv(details_path, index=False)
            evidence[row['case_id']] = proof
            evidence_path.write_text(json.dumps(evidence, indent=2) + '\n')
            print(f"Inspected {row['case_id']}: rank {row['rank']}, {proof['seconds']:.2f} s", flush=True)
    info = machine_info()
    info.update(date=datetime.now(ZoneInfo('Europe/Copenhagen')).isoformat(timespec='seconds'),
                execution_device='cpu', seed=cfg['seed'], status=STATUS, stage=args.stage,
                requested_cases=args.case, seconds=time.monotonic()-start,
                inputs_sha256={str(p.relative_to(REPO_ROOT)): digest(p) for p in
                               [Path(args.config).resolve(), source_path, REPO_ROOT / 'uv.lock', Path(__file__).resolve(), *paths]})
    if Path('/proc/cpuinfo').exists():
        info['cpu_model'] = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                  if line.startswith('model name')), info['cpu'])
    (output / f'exploratory_{args.stage}_machine.json').write_text(json.dumps(info, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
