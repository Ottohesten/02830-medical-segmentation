"""Make the step 6 report figure and eight-line summary from saved step 5 tables only.

No scans, labels, predictions, inference, scoring, or bootstrap analysis are opened or rerun.
Run from the repository root with --config configs/kits_robustness_eval.yaml.
"""

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segreview.config import config_arg_parser, load_config, results_dir
from segreview.figures import METHODS
from segreview.machine import machine_info


def digest(path: Path) -> str:
    """Return a file checksum for the figure's input/output provenance."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_tables(cfg: dict) -> dict:
    """Read the fixed test cohort, score intervals, and mixed threshold queue; reject incomplete selections."""
    root = results_dir(cfg)
    names = ['paired_changes', 'dice_degradation', 'change_correlations', 'g1_metrics', 'g1_curves', 'queue_top_summary']
    tables = {name: pd.read_csv(root / f'{name}.csv') for name in names}
    primary, secondary = (cfg['strength_lock'][key] for key in ['primary_score', 'secondary_score'])
    if len(cfg['perturbations']) != 1:
        raise ValueError('This report requires the single locked resolution perturbation.')
    variant = cfg['perturbations'][0]['name']
    strength = cfg['perturbations'][0]['strength']
    if cfg['perturbations'][0]['type'] != 'lowres':
        raise ValueError('This figure summarizes the locked resolution experiment.')
    settings = cfg['robustness_evaluation']
    rule = next(r for r in cfg['evaluation']['bad_rules'] if r['name'] == settings['bad_rule'])
    if rule['rule'] != 'threshold' or cfg['report']['set'] != 'mixed_test':
        raise ValueError('Use the fixed threshold rule and mixed test queue.')
    paired = tables['paired_changes']
    paired = paired[(paired['split'] == 'test') & (paired.variant == variant)].sort_values('case_id')
    if len(paired) != settings['expected_test_cases'] or paired.case_id.duplicated().any():
        raise ValueError('Need exactly the original test-case pairs.')
    degradation = tables['dice_degradation']
    degradation = degradation[(degradation['split'] == 'test') & (degradation.variant == variant)
                              & (degradation.resolution == cfg['model']['resolution'])]
    corr = tables['change_correlations']
    corr = corr[(corr['split'] == 'test') & (corr.variant == variant)].set_index('method')
    metrics = tables['g1_metrics']
    metrics = metrics[(metrics['set'] == cfg['report']['set']) & (metrics.bad_rule == rule['name'])].set_index('method')
    curves = tables['g1_curves']
    curves = curves[(curves['set'] == cfg['report']['set']) & (curves.bad_rule == rule['name'])]
    tops = tables['queue_top_summary']
    tops = tops[tops['split'] == 'test'].set_index('method')
    order = [primary, secondary, 'neg_volume_ml', 'random', 'oracle']
    if (len(degradation) != 1 or not set(order).issubset(metrics.index)
            or not {primary, secondary}.issubset(corr.index) or metrics.index.duplicated().any()):
        raise ValueError('Saved summary rows are incomplete or duplicated.')
    n = len(paired)
    expected_cases = set(json.loads((root / 'analysis_plan.json').read_text())['case_ids'])
    if not set(paired.case_id).issubset(expected_cases) or set(paired.case_id) & set(cfg['dataset']['split']['dev']):
        raise ValueError('The figure must show only held-out cases.')
    for method in order:
        frame = curves[curves.method == method].sort_values('reviewed_fraction')
        if len(frame) != int(metrics.loc[method, 'n_rows']) + 1 or frame.reviewed_fraction.duplicated().any():
            raise ValueError('Need the complete saved review curve for each displayed method.')
    top_fraction = settings['queue_top_fraction']
    top_point = curves[(curves.method == primary) & np.isclose(curves.reviewed_fraction, top_fraction)].iloc[0]
    return {'root': root, 'primary': primary, 'secondary': secondary, 'variant': variant, 'strength': strength,
            'rule': rule, 'paired': paired, 'degradation': degradation.iloc[0], 'corr': corr,
            'metrics': metrics, 'curves': curves, 'tops': tops, 'order': order, 'n': n,
            'n_rows': int(metrics.loc[primary, 'n_rows']), 'n_bad': int(metrics.loc[primary, 'n_bad']),
            'top_k': int(tops.loc[primary, 'top_k']),
            'top_bad_found': top_point.review_recall * metrics.loc[primary, 'n_bad'],
            'input_paths': [root / f'{name}.csv' for name in names] + [root / 'analysis_plan.json']}


def style_axis(ax) -> None:
    """Use simple axes and a light grid so paired observations and method curves remain readable."""
    ax.set_axisbelow(True)
    ax.grid(color='#e6e6e6', linewidth=.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8, color='#666666')


def draw_figure(data: dict, cfg: dict, out: Path, dpi: int) -> list[Path]:
    """Draw one composite figure: paired Dice, score-change rho/CIs, and the saved mixed review curves."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(7.2, 5.4), facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], left=.10, right=.98, bottom=.11,
                            top=.87, hspace=.78, wspace=.9)
    ax_dice = fig.add_subplot(grid[0, 0])
    ax_response = fig.add_subplot(grid[0, 1])
    ax_queue = fig.add_subplot(grid[1, :])
    for ax in [ax_dice, ax_response, ax_queue]:
        style_axis(ax)
    paired, stat = data['paired'], data['degradation']
    jitter = np.random.default_rng(cfg['seed']).uniform(-.055, .055, data['n'])
    for i, row in enumerate(paired.itertuples()):
        ax_dice.plot([jitter[i], 1+jitter[i]], [row.dice_clean, row.dice], color='#777777', alpha=.2, linewidth=.6)
    ax_dice.scatter(jitter, paired.dice_clean, color='#52514e', s=7, alpha=.6, zorder=3)
    ax_dice.scatter(1+jitter, paired.dice, color='#17856b', s=7, alpha=.6, zorder=3)
    ax_dice.scatter([0, 1], [stat.median_clean_dice, stat.median_perturbed_dice],
                    marker='D', color='black', s=25, zorder=5, label='Median')
    ax_dice.axhline(data['rule']['value'], color='#555555', linestyle='--', linewidth=.9,
                    label=f'Bad: Dice < {data["rule"]["value"]:.2f}')
    ax_dice.set(ylim=(0, 1.03), xlim=(-.30, 1.3), ylabel='3 mm model Dice')
    ax_dice.set_xticks([0, 1], [f'Original\n{int(stat.n_clean_bad)}/{data["n"]} bad',
                                f'Lowres {data["strength"]:g} mm\n{int(stat.n_perturbed_bad)}/{data["n"]} bad'])
    ax_dice.set_title('A  Segmentation quality', loc='left', fontsize=10, pad=24)
    ax_dice.text(.5, 1.06, f'Medians: {stat.median_clean_dice:.3f} → {stat.median_perturbed_dice:.3f}',
                  transform=ax_dice.transAxes, ha='center', fontsize=8)
    ax_dice.legend(loc='lower left', frameon=False, fontsize=7, handlelength=1.6)
    for method, y in [(data['primary'], 1), (data['secondary'], 0)]:
        row = data['corr'].loc[method]
        _, color, _ = METHODS[method]
        rho, low, high = row.rho_dice_drop_vs_score_rise, row.rho_ci_low, row.rho_ci_high
        ax_response.errorbar(rho, y, xerr=[[rho-low], [high-rho]], color=color,
                              marker='o' if method == data['primary'] else 's', capsize=3, linewidth=1.7)
        ax_response.text(.18, y+.22, f'{rho:.3f} [{low:.3f}, {high:.3f}]', ha='center', fontsize=7.5)
        ax_response.text(.18, y-.24, f'Score rose on {int(row.n_score_rises)}/{data["n"]}', ha='center', fontsize=7.5)
    ax_response.axvline(0, linestyle='--', color='#666666', linewidth=.9)
    ax_response.set(xlim=(-.20, .60), ylim=(-.55, 1.5), xlabel='ρ(Dice loss, score increase)')
    ax_response.set_yticks([1, 0], ['Model difference\n(primary)', 'Entropy\n(secondary)'])
    ax_response.grid(axis='y', visible=False)
    ax_response.set_title('B  Response to degradation', loc='left', fontsize=10, pad=24)
    ax_response.text(.5, 1.06, 'Positive ρ supports sensitivity; 95% CIs',
                      transform=ax_response.transAxes, ha='center', fontsize=7.5)
    labels = {data['primary']: 'Model difference (primary)', data['secondary']: 'Entropy (secondary)',
              'neg_volume_ml': 'Small-volume baseline', 'random': 'Random order', 'oracle': 'Oracle (true Dice)'}
    for method in data['order']:
        rows = data['curves'][data['curves'].method == method].sort_values('reviewed_fraction')
        _, color, linestyle = METHODS[method]
        if method == data['secondary']:
            linestyle = '--'  # distinguish the close primary/secondary curves in grayscale too.
        auc = data['metrics'].loc[method, 'review_auc']
        ax_queue.plot(100*rows.reviewed_fraction, 100*rows.review_recall, color=color,
                       linestyle=linestyle, linewidth=1.7, label=f'{labels[method]}: AUC {auc:.3f}')
    primary = data['metrics'].loc[data['primary']]
    ax_queue.set(xlim=(0, 100), ylim=(0, 102), xlabel=f'Images reviewed (% of {data["n_rows"]} views)',
                  ylabel=f'Bad segmentations found (% of {data["n_bad"]})')
    ax_queue.set_title('C  Mixed original + lowres test queue', loc='left', fontsize=10, pad=19)
    ax_queue.text(0, 1.035, f'{data["n_rows"]} views from {data["n"]} cases; '
                   f'primary AUC 95% CI [{primary.review_auc_ci_low:.3f}, {primary.review_auc_ci_high:.3f}]',
                   transform=ax_queue.transAxes, fontsize=7.5)
    ax_queue.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none',
                     framealpha=.96, fontsize=7, ncol=1)
    fig.suptitle(f'Resolution stress test on {data["n"]} held-out KiTS23 cases', fontsize=11, y=.98)
    fig.text(.10, .025, f'{data["strength"]:g} mm strength selected on development data. '
             'Bootstrap intervals retain both views of each case.', fontsize=7.5)
    paths = []
    for suffix in ['pdf', 'png']:
        path = out / f'fig_robustness_summary_test.{suffix}'
        fig.savefig(path, dpi=dpi, facecolor='white')
        paths.append(path)
    plt.close(fig)
    return paths


def report_text(data: dict, cfg: dict) -> tuple[list[str], str]:
    """Create eight factual summary lines and an explanatory caption from the displayed saved results."""
    stat = data['degradation']
    p, e = data['primary'], data['secondary']
    response = data['corr'].loc[p]
    metrics = data['metrics']
    dev = cfg['robustness_selection']['expected_dev_cases']
    lines = [
        f'An {data["strength"]:g} mm resolution perturbation was selected on {dev} development scans before test scoring.',
        f'On {data["n"]} test scans, median 3 mm Dice fell from {stat.median_clean_dice:.3f} to '
        f'{stat.median_perturbed_dice:.3f}; median paired loss was {stat.median_paired_drop:.4f}.',
        f'Segmentations below Dice {data["rule"]["value"]:.2f} increased from '
        f'{int(stat.n_clean_bad)}/{data["n"]} to {int(stat.n_perturbed_bad)}/{data["n"]}.',
        f'Primary score rise versus Dice loss was inconclusive: ρ = {response.rho_dice_drop_vs_score_rise:.3f}, '
        f'95% CI [{response.rho_ci_low:.3f}, {response.rho_ci_high:.3f}].',
        f'Entropy rose on {int(data["corr"].loc[e, "n_score_rises"])}/{data["n"]} scans and remains a secondary finding.',
        f'Mixed-queue review-AUC was {metrics.loc[p, "review_auc"]:.3f} for the primary score and '
        f'{metrics.loc[e, "review_auc"]:.3f} for entropy, versus volume {metrics.loc["neg_volume_ml", "review_auc"]:.3f}, '
        f'random {metrics.loc["random", "review_auc"]:.3f}, and oracle {metrics.loc["oracle", "review_auc"]:.3f}.',
        f'The primary found {data["top_bad_found"]:.0f}/{data["n_bad"]} bad segmentations in the first '
        f'{data["top_k"]} views; only {data["tops"].loc[p, "expected_perturbed_in_top_k"]:.0f}/'
        f'{data["top_k"]} of those views were degraded copies.',
        'Failure ranking remains useful in this resolution stress test, with limited evidence for a '
        'response to degradation severity; other domain shifts are outside the tested scope.',
    ]
    caption = (
        f'Resolution robustness on {data["n"]} held-out KiTS23 cases after choosing an {data["strength"]:g} mm perturbation on '
        f'{dev} development cases. (A) Each line joins the original and degraded 3 mm segmentation Dice for '
        'one case; horizontal jitter is cosmetic, diamonds mark medians, and the dashed line is the fixed '
        f'Dice < {data["rule"]["value"]:.2f} bad-segmentation threshold. (B) Spearman correlation between '
        'clean-minus-degraded Dice and degraded-minus-clean score, with saved 95% bootstrap intervals. '
        'Positive correlation supports sensitivity to deterioration; the primary interval includes zero. '
        'The primary score is mean absolute 3 mm–6 mm organ-probability disagreement near the predicted organ; '
        'binary entropy is secondary. (C) Saved G1 review curves for the mixed original/degraded queue, '
        'with volume, random, and oracle references. AUC measures failure discovery, not classification of '
        'degraded images. The oracle uses true Dice only as an evaluation upper bound. '
        f'The {data["n_rows"]} views come from {data["n"]} cases; mixed-queue bootstrap samples retain both '
        f'views together. Intervals come from the original {cfg["evaluation"]["n_bootstrap"]:,} bootstrap resamples; '
        f'random is the mean of {cfg["evaluation"]["n_random_orders"]:,} orders. '
        'No new model selection or statistical analysis was performed for this figure. '
        'Scope is resolution degradation only; lower-resolution, original, and mixed queues have different '
        'bad-segmentation prevalence, so their raw AUCs have different oracle ceilings.'
    )
    return lines, caption


def main():
    """Render report artifacts, save their exact sources/machine metadata, and print the eight-line summary."""
    parser = config_arg_parser(__doc__)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--overwrite', action='store_true', help='replace this report figure/text only')
    args = parser.parse_args()
    start = time.perf_counter()
    cfg = load_config(args.config)
    data = read_tables(cfg)
    out = data['root'] / 'report'
    out.mkdir(exist_ok=True)
    if (out / 'figure_provenance.json').exists() and not args.overwrite:
        raise ValueError('Report already exists; use --overwrite for a deliberate presentation-only edit.')
    paths = draw_figure(data, cfg, out, args.dpi)
    lines, caption = report_text(data, cfg)
    summary_path, caption_path = out / 'summary.md', out / 'caption.md'
    summary_path.write_text('\n'.join(f'{i}. {line}' for i, line in enumerate(lines, 1)) + '\n')
    caption_path.write_text(caption + '\n')
    paths += [summary_path, caption_path]
    info = {**machine_info(), 'device_used': 'cpu (CSV-to-figure only)', 'config_sha256': digest(args.config),
            'script_sha256': digest(Path(__file__)), 'matplotlib': matplotlib.__version__,
            'dpi': args.dpi, 'figure_size_inches': [7.2, 5.4], 'test_cases': data['n'],
            'summary_lines': len(lines), 'new_inference_scoring_or_bootstrap': False,
            'elapsed_seconds': time.perf_counter() - start,
            'presentation_created_after_test_evaluation': True,
            'inputs_sha256': {str(p.relative_to(data['root'])): digest(p) for p in data['input_paths']},
            'outputs_sha256': {p.name: digest(p) for p in paths}}
    cpu = Path('/proc/cpuinfo')
    if cpu.exists():
        info['cpu_model'] = next((line.split(':', 1)[1].strip() for line in cpu.read_text().splitlines()
                                  if line.startswith('model name')), info['cpu'])
    (out / 'figure_provenance.json').write_text(json.dumps(info, indent=2) + '\n')
    print(summary_path.read_text())
    print(f'Figure and caption: {out}')


if __name__ == '__main__':
    main()
