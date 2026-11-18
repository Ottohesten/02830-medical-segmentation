"""Make the missed-errors report figure from verified saved aggregates (step 6); no image/GT access."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from segreview.config import REPO_ROOT, config_arg_parser, load_config
from segreview.machine import machine_info

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.check_missed_errors import digest
from scripts.fresh_missed_errors import experiment, now


def one_row(table: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    """Require exactly one source row so the figure cannot silently mix rules, methods or cohorts."""
    rows = table.loc[mask]
    if len(rows) != 1:
        raise ValueError(f'Expected one {description}; got {len(rows)}')
    return rows.iloc[0]


def load_plot_data(output: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Read only hash-verified outcome tables and the two already locked primary decisions."""
    proof = json.loads((output/'fresh_verification.json').read_text())
    if not proof['passed']:
        raise ValueError('Need a passed outcome verification')
    files = ['metrics.csv','paired.csv','budget_statistics.csv','evaluation_completed.json']
    for name in files:
        if digest(output/name) != proof['output_hashes'][name]:
            raise ValueError(f'Changed verified figure input: {name}')
    metrics, paired, budget = [pd.read_csv(output/name) for name in files[:3]]
    complete = json.loads((output/files[3]).read_text())
    primary = cfg['checks']['primary']
    decisions = complete['decisions']
    methods = [primary, *[row['method'] for row in decisions]]
    if len(methods) != 3 or len(set(methods)) != 3 or methods[1] != cfg['combinations'][0]['name']:
        raise ValueError('Need the baseline and both frozen decisions in their declared order')
    rules = [r for r in cfg['evaluation']['bad_rules'] if r['rule']=='threshold' and
             r['value']==cfg['checks']['bad_dice_threshold']]
    if len(rules) != 1:
        raise ValueError('Need the declared primary threshold rule')
    rule = rules[0]['name']
    labels = ['G1','Combination','Far disagreement']
    rows = []
    for i, method in enumerate(methods):
        metric = one_row(metrics,metrics.method.eq(method)&metrics.bad_rule.eq(rule),'primary metric')
        entry = {'method':method,'label':labels[i],'review_auc':metric.review_auc,
                 'difference':0.0,'ci_low':0.0,'ci_high':0.0,'criterion_met':None}
        if i:
            pair = one_row(paired,paired.method.eq(method)&paired.reference.eq(primary)&
                           paired.measure.eq('review_auc')&paired.bad_rule.eq(rule),'primary paired comparison')
            entry.update(difference=pair['diff'],ci_low=pair.diff_ci_low,ci_high=pair.diff_ci_high,
                         criterion_met=decisions[i-1]['meets_per_comparison_criterion'])
        for measure in ['bad_capture','false_alarms','quality_after_correction']:
            point = one_row(budget,budget.method.eq(method)&budget.measure.eq(measure),'budget point')
            entry[measure] = point.value
            entry['review_budget'] = int(point.review_budget)
        rows.append(entry)
    data = pd.DataFrame(rows)
    if data.review_budget.nunique() != 1:
        raise ValueError('Budget differs across plotted methods')
    meta = {'n_cases':complete['n_cases'],'n_bad':complete['n_bad'],'review_budget':int(data.review_budget.iloc[0]),
            'bad_dice_threshold':cfg['checks']['bad_dice_threshold'],'n_bootstrap':cfg['evaluation']['n_bootstrap'],
            'seed':cfg['seed'],'input_hashes':{name:proof['output_hashes'][name] for name in files}}
    return data,meta


def draw_figure(data: pd.DataFrame, meta: dict, folder: Path, dpi: int) -> list[Path]:
    """Export one two-panel PDF/PNG: locked primary intervals and descriptive fixed-budget capture."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#4b5563','#d55e00','#0072b2']
    with plt.rc_context({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42}):
        fig,(left,right) = plt.subplots(1,2,figsize=(7.4,3.7),gridspec_kw={'width_ratios':[1.2,1]})
        fig.subplots_adjust(left=.15,right=.98,bottom=.23,top=.75,wspace=.65)
        fig.suptitle(f"Fresh-pool ranking test: {meta['n_cases']} scans, {meta['n_bad']} bad",fontsize=11,y=.98)
        fig.text(.5,.88,f"Bad = 3 mm Dice < {meta['bad_dice_threshold']:g}; {meta['n_bootstrap']:,} paired case resamples; seed {meta['seed']}",ha='center',fontsize=8)
        left.axvline(0,color='#777777',linestyle='--',linewidth=1)
        for i in [1,2]:
            row = data.iloc[i]
            y = 2-i
            left.errorbar(row.difference,y,xerr=[[row.difference-row.ci_low],[row.ci_high-row.difference]],
                          fmt='o',color=colors[i],capsize=3,linewidth=1.4,markersize=5)
            left.text(.02,(y+.6)/2.05-.15,f"{row.difference:+.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]",
                      transform=left.transAxes,fontsize=7.5)
        lo,hi = data.ci_low.min(),data.ci_high.max()
        padding = max((hi-lo)*.13,.01)
        left.set(xlim=(lo-padding,hi+padding),ylim=(-.6,1.45),yticks=[1,0],
                 yticklabels=['Combination','Far disagreement'],xlabel='Review-AUC difference vs G1\n95% paired intervals')
        left.set_title(f"A  Primary comparison (G1 = {data.review_auc.iloc[0]:.3f})",loc='left',pad=10)
        left.grid(axis='x',alpha=.18)
        x = np.arange(len(data))
        right.bar(x,data.bad_capture,color=colors,width=.65,zorder=3)
        for i,row in data.iterrows():
            right.text(i,row.bad_capture+.22,f"{row.bad_capture:g}/{meta['n_bad']}",ha='center',fontsize=9)
        right.set(xticks=x,xticklabels=['G1','Combination','Far'],ylim=(0,max(meta['n_bad'],1)+1.8),
                  ylabel='Bad cases found',xlabel='Ranking method')
        right.set_title(f"B  First {meta['review_budget']} reviews",loc='left',pad=10)
        right.grid(axis='y',alpha=.18,zorder=0)
        for axis in [left,right]:
            axis.spines[['top','right']].set_visible(False)
        passed = data.iloc[1:].criterion_met.astype(bool).any()
        footer = 'Primary criterion: lower interval bound above zero.'
        footer += ' Neither candidate meets it.' if not passed else ' See the saved primary decisions.'
        fig.text(.5,.055,footer,ha='center',fontsize=8)
        paths = [folder/'fig_missed_fresh_summary.pdf',folder/'fig_missed_fresh_summary.png']
        for path in paths:
            fig.savefig(path,dpi=dpi)
        plt.close(fig)
    return paths


def main() -> None:
    """Regenerate report assets from verified CSVs, preserving all frozen scientific outputs."""
    start = time.monotonic()
    parser = config_arg_parser(__doc__)
    parser.add_argument('--dpi',type=int,default=300)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output,_ = experiment(cfg)
    # Bind labels/settings to the published recipe without requiring local scans or checkpoint files.
    lock = json.loads((output/'lock.json').read_text())
    proof = json.loads((output/'fresh_verification.json').read_text())
    config_key = str(args.config.resolve().relative_to(REPO_ROOT))
    if digest(args.config) != lock['scientific_hashes'].get(config_key) or digest(output/'lock.json') != proof['scientific_lock_sha256']:
        raise ValueError('Report config/lock differs from the verified scientific recipe')
    data,meta = load_plot_data(output,cfg)
    folder = output/'report'
    folder.mkdir(exist_ok=True)
    data.to_csv(folder/'figure_source_data.csv',index=False)
    paths = draw_figure(data,meta,folder,args.dpi)
    info = machine_info()
    info.update(date=now(),execution_device='cpu',seed=cfg['seed'],seconds=time.monotonic()-start,
                source_commit=info['git_commit'],script_sha256=digest(Path(__file__)),config_sha256=digest(args.config),
                inputs=meta,GT_or_model_arrays_read=False,metrics_recomputed=False,
                outputs={p.name:digest(p) for p in [*paths,folder/'figure_source_data.csv']})
    (folder/'figure_machine.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps({'figure_paths':[str(p) for p in paths],'seconds':info['seconds']},indent=2))


if __name__ == '__main__':
    main()
