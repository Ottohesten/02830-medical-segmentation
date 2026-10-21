"""Step 5 robustness summaries: paired Dice/score changes and G1 queues, separated by development/test.

Higher Dice drop means worse segmentation; higher score change means more uncertainty. Mixed queues
contain two dependent views per patient, so bootstrap samples retain both views together.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from segreview.evaluation import (_ci, bad_mask, bootstrap_indices, evaluate_methods,
                                  expected_top_k_sums)


def paired_changes(scores: pd.DataFrame, dice_rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join complete, finite score/Dice pairs and return clean/perturbed values with signed per-patient changes."""
    methods = cfg['robustness_evaluation']['methods']
    variants = ['clean'] + [s['name'] for s in cfg['perturbations']]
    keys = ['case_id', 'variant', 'split']
    for frame, values in [(scores, methods), (dice_rows, ['dice', 'dice_second_model'])]:
        if frame.empty or frame.duplicated(['case_id', 'variant']).any() or not np.isfinite(frame[values].to_numpy()).all():
            raise ValueError('Need unique finite scores and Dice per case/variant.')
        if not frame.groupby('case_id').variant.apply(lambda v: sorted(v) == sorted(variants)).all():
            raise ValueError('Each patient needs all configured variants.')
    if not dice_rows[['dice', 'dice_second_model']].apply(lambda c: c.between(0, 1).all()).all():
        raise ValueError('Dice must be within [0, 1].')
    merged = scores.merge(dice_rows[keys + ['dice', 'dice_second_model']], on=keys,
                          how='outer', validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('Scores and Dice must have identical case/variant/split coverage.')
    if merged.groupby('case_id')['split'].nunique().ne(1).any():
        raise ValueError('A patient cannot cross development/test splits.')
    merged = merged.drop(columns='_merge')
    clean = merged[merged.variant == 'clean'].drop(columns='variant')
    paired = merged[merged.variant != 'clean'].merge(clean, on=['case_id', 'split'], suffixes=('', '_clean'),
                                                     validate='many_to_one')
    paired['dice_drop'] = (paired.dice_clean - paired.dice).round(6)
    paired['second_model_dice_drop'] = (paired.dice_second_model_clean - paired.dice_second_model).round(6)
    for method in methods:
        paired[f'{method}_change'] = paired[method] - paired[f'{method}_clean']
    return paired.sort_values(['split', 'variant', 'case_id']).reset_index(drop=True)


def change_summaries(paired: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize Dice degradation and bootstrap Spearman(Dice drop, score rise) per split and perturbation."""
    ev = cfg['evaluation']
    settings = cfg['robustness_evaluation']
    threshold = next(r['value'] for r in ev['bad_rules'] if r['name'] == settings['bad_rule'])
    degradation, correlations = [], []
    for (split, variant), group in paired.groupby(['split', 'variant'], sort=True):
        group = group.sort_values('case_id')
        for resolution, clean_col, pert_col, drop_col in [
                (cfg['model']['resolution'], 'dice_clean', 'dice', 'dice_drop'),
                (cfg['uncertainty']['second_model_resolution'], 'dice_second_model_clean',
                 'dice_second_model', 'second_model_dice_drop')]:
            clean, pert = group[clean_col], group[pert_col]
            degradation.append({'split': split, 'variant': variant, 'resolution': resolution,
                                'n_cases': len(group), 'median_clean_dice': clean.median(),
                                'median_perturbed_dice': pert.median(), 'median_paired_drop': group[drop_col].median(),
                                'mean_paired_drop': group[drop_col].mean(), 'n_clean_bad': int((clean < threshold).sum()),
                                'n_perturbed_bad': int((pert < threshold).sum()),
                                'n_new_bad': int(((clean >= threshold) & (pert < threshold)).sum()),
                                'n_recovered': int(((clean < threshold) & (pert >= threshold)).sum())})
        indices = bootstrap_indices(len(group), ev['n_bootstrap'], np.random.default_rng(cfg['seed']))
        drop = group.dice_drop.to_numpy()
        for method in settings['methods']:
            change = group[f'{method}_change'].to_numpy()
            rho = spearmanr(drop, change).statistic if np.ptp(drop) and np.ptp(change) else np.nan
            boots = [spearmanr(drop[idx], change[idx]).statistic if np.ptp(drop[idx]) and np.ptp(change[idx])
                     else np.nan for idx in indices]
            lo, hi = _ci(np.asarray(boots), ev['ci_level'])
            correlations.append({'split': split, 'variant': variant, 'method': method, 'n_cases': len(group),
                                 'rho_dice_drop_vs_score_rise': rho, 'rho_ci_low': lo, 'rho_ci_high': hi,
                                 'n_bootstrap_finite': int(np.isfinite(boots).sum()),
                                 'median_score_change': float(np.median(change)),
                                 'n_score_rises': int((change > 0).sum()),
                                 'n_dice_drops': int((drop > 0).sum()),
                                 'n_drop_and_score_rise': int(((drop > 0) & (change > 0)).sum())})
    return pd.DataFrame(degradation), pd.DataFrame(correlations)


def evaluate_queues(scores: pd.DataFrame, dice_rows: pd.DataFrame, cfg: dict,
                    progress=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the existing G1 engine per variant/split and mixed queue; return metrics, curves, shifts, top-k summaries."""
    merged = scores.merge(dice_rows[['case_id', 'variant', 'split', 'dice']],
                          on=['case_id', 'variant', 'split'], validate='one_to_one')
    methods = cfg['robustness_evaluation']['methods']
    metric_rows, curve_rows, shifts, tops = [], [], [], []
    for split, frame in merged.groupby('split', sort=True):
        frame = frame.sort_values(['case_id', 'variant']).reset_index(drop=True)
        sets = [(v, frame[frame.variant == v].reset_index(drop=True)) for v in sorted(frame.variant.unique())]
        sets.append(('mixed', frame))
        for variant, group in sets:
            name = f'{variant}_{split}'
            d = group.dice.to_numpy()
            score_arrays = {method: group[method].to_numpy() for method in methods}
            groups = group.case_id.to_numpy() if variant == 'mixed' else None
            for rule in cfg['evaluation']['bad_rules']:
                rows, curves = evaluate_methods(score_arrays, d, cfg, rule, groups=groups)
                metric_rows.extend({'set': name, 'split': split, 'variant': variant, 'bad_rule': rule['name'],
                                    'n_rows': len(group), 'n_patients': group.case_id.nunique(),
                                    'n_bad': int(bad_mask(d, rule['rule'], rule['value']).sum()),
                                    'bootstrap_unit': 'case_id', **row} for row in rows)
                curve_rows.extend({'set': name, 'bad_rule': rule['name'], 'method': method,
                                   'reviewed_fraction': i / len(group), 'review_recall': float(r),
                                   'quality_mean_dice': float(q)}
                                  for method, (review, quality) in curves.items()
                                  for i, (r, q) in enumerate(zip(review, quality)))
                if progress:
                    progress(f'[evaluate] {name} / {rule["name"]}: {len(group)} rows, {group.case_id.nunique()} patients')
            if variant == 'mixed':
                degraded = group.variant.ne('clean').to_numpy()
                k = max(1, int(round(len(group) * cfg['robustness_evaluation']['queue_top_fraction'])))
                for method in methods:
                    ranks = group[['case_id', 'variant']].copy()
                    ranks['rank'] = rankdata(-group[method].to_numpy(), method='average')
                    clean = ranks[ranks.variant == 'clean'][['case_id', 'rank']].rename(columns={'rank': 'clean_rank'})
                    changed = ranks[ranks.variant != 'clean'].merge(clean, on='case_id', validate='many_to_one')
                    changed['rank_gain'] = changed.clean_rank - changed['rank']
                    shifts.extend({'split': split, 'method': method, **row} for row in changed.to_dict('records'))
                    count = expected_top_k_sums(group[method].to_numpy(), degraded)[k]
                    tops.append({'split': split, 'method': method, 'n_rows': len(group), 'top_k': k,
                                 'expected_perturbed_in_top_k': float(count), 'share_perturbed_top_k': float(count / k),
                                 'median_rank_gain': float(changed.rank_gain.median()),
                                 'n_perturbed_moved_up': int((changed.rank_gain > 0).sum())})
    return pd.DataFrame(metric_rows), pd.DataFrame(curve_rows), pd.DataFrame(shifts), pd.DataFrame(tops)
