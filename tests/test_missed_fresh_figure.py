"""Keep report figures tied to verified inputs and the predeclared primary comparison."""

import json

import pandas as pd
import pytest

from scripts.check_missed_errors import digest
from scripts.plot_missed_fresh import load_plot_data


@pytest.fixture
def report(tmp_path):
    """Create distinct primary/secondary-rule values, two locked methods and an unselected better control."""
    cfg = {'checks':{'primary':'g1','bad_dice_threshold':.83},'combinations':[{'name':'combo'}],
           'evaluation':{'bad_rules':[{'name':'worst','rule':'worst_fraction','value':.1},
                                      {'name':'threshold','rule':'threshold','value':.83}],
                         'n_bootstrap':2000},'seed':42}
    metrics = [{'method':method,'bad_rule':rule,'review_auc':value+(rule=='worst')*.1}
               for method,value in [('g1',.7),('combo',.72),('far',.75),('unused',.99)]
               for rule in ['threshold','worst']]
    pairs = [{'method':method,'reference':'g1','measure':'review_auc','bad_rule':rule,
              'diff':value,'diff_ci_low':-.03,'diff_ci_high':.07}
             for method,value in [('combo',.02),('far',.05)] for rule in ['threshold','worst']]
    budget = [{'method':method,'measure':measure,'value':value,'review_budget':20}
              for method in ['g1','combo','far']
              for measure,value in [('bad_capture',9),('false_alarms',11),('quality_after_correction',.93)]]
    pd.DataFrame(metrics).to_csv(tmp_path/'metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(tmp_path/'paired.csv',index=False)
    pd.DataFrame(budget).to_csv(tmp_path/'budget_statistics.csv',index=False)
    (tmp_path/'evaluation_completed.json').write_text(json.dumps({'n_cases':100,'n_bad':10,
        'decisions':[{'method':method,'meets_per_comparison_criterion':False} for method in ['combo','far']]}))
    files=['metrics.csv','paired.csv','budget_statistics.csv','evaluation_completed.json']
    proof={'passed':True,'output_hashes':{name:digest(tmp_path/name) for name in files}}
    (tmp_path/'fresh_verification.json').write_text(json.dumps(proof))
    return tmp_path,cfg


def test_figure_uses_locked_primary_rule_and_candidates(report):
    """Better-looking secondary/control rows cannot replace the locked comparison in the figure."""
    folder,cfg=report
    data,meta=load_plot_data(folder,cfg)
    assert list(data.method)==['g1','combo','far']
    assert list(data.review_auc)==[.7,.72,.75]
    assert list(data.difference)==[0,.02,.05]
    assert meta['n_bad']==10 and meta['review_budget']==20


def test_changed_verified_source_is_rejected(report):
    """A stale report cannot silently plot a result table changed since outcome verification."""
    folder,cfg=report
    path=folder/'metrics.csv'
    path.write_text(path.read_text().replace('0.72','0.92'))
    with pytest.raises(ValueError,match='Changed verified figure input'):
        load_plot_data(folder,cfg)


def test_missing_primary_pair_is_not_replaced_by_secondary_rule(report):
    """A missing locked interval is a failure, even when another bad-rule row exists."""
    folder,cfg=report
    pairs=pd.read_csv(folder/'paired.csv')
    pairs.loc[~(pairs.method.eq('combo')&pairs.bad_rule.eq('threshold'))].to_csv(folder/'paired.csv',index=False)
    proof=json.loads((folder/'fresh_verification.json').read_text())
    proof['output_hashes']['paired.csv']=digest(folder/'paired.csv')
    (folder/'fresh_verification.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='primary paired comparison'):
        load_plot_data(folder,cfg)
