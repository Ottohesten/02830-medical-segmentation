"""Development-only Dice comparisons and strength suggestions for the G1 robustness task.

The selection rule uses paired Dice degradation only. No uncertainty score is accepted or read.
"""

import numpy as np
import pandas as pd

from segreview.config import REPO_ROOT
from segreview.data import case_split


def validate_development_run(cfg: dict, cases: list) -> None:
    """Reject test cases, incomplete development selections, or writes under locked kits100 paths."""
    if cfg['name'] == 'kits100':
        raise ValueError('Use a separate robustness config name.')
    for key, locked in [('results_dir', 'results/kits100'),
                        ('predictions_dir', 'data/predictions/kits100')]:
        if (cfg['paths'][key] / cfg['name']).resolve().is_relative_to((REPO_ROOT / locked).resolve()):
            raise ValueError('Robustness must not write under locked kits100 paths.')
    dev = cfg['dataset']['split']['dev']
    ids = [case.case_id for case in cases]
    if (len(ids) != cfg['robustness_selection']['expected_dev_cases'] or len(ids) != len(set(ids))
            or set(ids) != set(dev) or any(case_split(cfg, case) != 'dev' for case in ids)):
        raise ValueError('The selected scans must be exactly the development set.')
    if cfg['evaluation'].get('splits') != ['dev']:
        raise ValueError('Only development evaluation is allowed during strength selection.')
    specs = cfg.get('perturbations') or []
    names = [spec['name'] for spec in specs]
    if not names or len(names) != len(set(names)) or 'clean' in names:
        raise ValueError('Candidate variants must have distinct names separate from clean.')
    if cfg['robustness_selection']['rule'] != 'mildest_eligible_per_type':
        raise ValueError('Unknown strength-selection rule.')


def summarize_degradation(dice_rows: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired per-scan Dice drops and one summary per candidate, requiring complete dev rows."""
    ids = cfg['dataset']['split']['dev']
    specs = cfg['perturbations']
    expected = {(case, variant) for case in ids for variant in ['clean'] + [s['name'] for s in specs]}
    observed = list(zip(dice_rows.case_id, dice_rows.variant))
    if (len(observed) != len(expected) or set(observed) != expected or not dice_rows['split'].eq('dev').all()
            or not np.isfinite(dice_rows.dice).all() or not dice_rows.dice.between(0, 1).all()):
        raise ValueError('Need exactly one finite dev Dice in [0, 1] per case and configured variant.')
    settings = cfg['robustness_selection']
    threshold = next(r['value'] for r in cfg['evaluation']['bad_rules'] if r['name'] == settings['bad_rule'])
    clean = dice_rows[dice_rows.variant == 'clean'][['case_id', 'dice']].rename(columns={'dice': 'clean_dice'})
    paired = dice_rows[dice_rows.variant != 'clean'][['case_id', 'variant', 'dice']].merge(clean, on='case_id', validate='many_to_one')
    # Existing Dice rows have four decimals; avoid floating-point noise at the drop threshold.
    paired['dice_drop'] = (paired.clean_dice - paired.dice).round(6)
    paired['new_bad'] = (paired.clean_dice >= threshold) & (paired.dice < threshold)
    paired['new_catastrophic'] = ((paired.clean_dice >= settings['catastrophic_dice'])
                                  & (paired.dice < settings['catastrophic_dice']))
    summaries = []
    for spec in specs:
        group = paired[paired.variant == spec['name']]
        affected = int((group.dice_drop >= settings['min_dice_drop']).sum())
        good = int((group.dice >= threshold).sum())
        catastrophic = int(group.new_catastrophic.sum())
        reasons = []
        if affected < settings['min_affected_cases']:
            reasons.append('too_few_clear_drops')
        if good < settings['min_remaining_good_cases']:
            reasons.append('too_few_good_remaining')
        if catastrophic > settings['max_new_catastrophic_cases']:
            reasons.append('too_many_new_catastrophic')
        summaries.append({'variant': spec['name'], 'type': spec['type'], 'strength': spec['strength'],
                          'n_cases': len(group), 'median_clean_dice': clean.clean_dice.median(),
                          'median_dice': group.dice.median(), 'median_paired_drop': group.dice_drop.median(),
                          'mean_paired_drop': group.dice_drop.mean(), 'n_clear_drops': affected,
                          'n_bad': int((group.dice < threshold).sum()), 'n_new_bad': int(group.new_bad.sum()),
                          'n_remaining_good': good, 'n_new_catastrophic': catastrophic,
                          'eligible': not reasons, 'ineligibility_reason': ';'.join(reasons)})
    return paired, pd.DataFrame(summaries)


def suggest_strengths(summary: pd.DataFrame, cfg: dict) -> dict:
    """Suggest the lowest eligible severity per type; explicitly record types with no eligible candidate."""
    selected, excluded = [], []
    for kind in dict.fromkeys(spec['type'] for spec in cfg['perturbations']):
        eligible = summary[(summary['type'] == kind) & summary.eligible].sort_values(['strength', 'variant'])
        if eligible.empty:
            excluded.append(kind)
        else:
            selected.append(eligible.iloc[0].variant)
    return {'status': 'suggestion_from_dev_dice_only', 'rule': cfg['robustness_selection'],
            'selected_variants': selected, 'types_without_eligible_candidate': excluded,
            'uncertainty_scores_computed': False, 'test_set_used': False}
