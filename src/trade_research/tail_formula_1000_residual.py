"""A fixed residual tree separates same-day stock selection from market rallies."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .corporate_cash import save_json, sha
from .tail_formula_1000 import EXPRESSIONS, ROOT as SOURCE
from . import tail_formula_1000_daily as daily

ROOT=Path('data/research/tail_formula_1000_residual')
PROTOCOL=Path('config/tail_formula_1000_residual_protocol.json')


def model():
    if (ROOT/'model_report.json').exists():
        raise ValueError('Do not refit residual tree')
    ROOT.mkdir(parents=True,exist_ok=True)
    _,f,train=daily.inputs()
    baseline=train.groupby('date').opportunity15.transform('mean')
    target=train.opportunity15-baseline
    weights=1/train.groupby('date').code.transform('size')
    tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=2000,criterion='squared_error',random_state=20260927)
    tree.fit(train[list(EXPRESSIONS)],target,sample_weight=weights)
    t=tree.tree_
    report=dict(protocol_sha256=sha(PROTOCOL),source_training_label_report_sha256=sha(SOURCE/'training_label_report.json'),
        source_feature_report_sha256=sha(SOURCE/'feature_report.json'),rows=len(train),days=train.date.nunique(),
        last_observation=train.next_date.max(),parameters=tree.get_params(),
        tree=dict(feature=t.feature.tolist(),threshold=t.threshold.tolist(),children_left=t.children_left.tolist(),
            children_right=t.children_right.tolist(),n_node_samples=t.n_node_samples.tolist(),
            weighted_n_node_samples=t.weighted_n_node_samples.tolist(),value=t.value.reshape(-1).tolist(),impurity=t.impurity.tolist()),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'model_report.json',report)
    return {k:v for k,v in report.items() if k!='tree'}


def verify_model():
    report=json.loads((ROOT/'model_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL)
    assert report['source_training_label_report_sha256']==sha(SOURCE/'training_label_report.json')
    _,f,train=daily.inputs()
    assert train.next_date.lt('2024-07-01').all()
    c=duckdb.connect()
    c.register('training',train)
    expected=c.sql('''SELECT date,code,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target,
        1./count(*) OVER(PARTITION BY date) AS weight FROM training ORDER BY date,code''').df()
    assert expected[['date','code']].equals(train[['date','code']])
    x=train[list(EXPRESSIONS)].to_numpy(dtype='float32')
    nodes={0:np.ones(len(train),dtype=bool)}
    t=report['tree']
    for i in range(len(t['feature'])):
        mask=nodes[i]
        assert int(mask.sum())==t['n_node_samples'][i]
        w=expected.weight[mask]
        y=expected.target[mask]
        mean=float(np.average(y,weights=w))
        variance=float(np.average((y-mean)**2,weights=w))
        np.testing.assert_allclose(w.sum(),t['weighted_n_node_samples'][i],atol=1e-9,rtol=0)
        np.testing.assert_allclose(mean,t['value'][i],atol=2e-11,rtol=0)
        np.testing.assert_allclose(variance,t['impurity'][i],atol=2e-11,rtol=0)
        left=t['children_left'][i]
        if left<0:
            assert mask.sum()>=2000
            continue
        lower=x[:,t['feature'][i]]<=t['threshold'][i]
        nodes[left]=mask&lower
        nodes[t['children_right'][i]]=mask&~lower
    per_day=expected.assign(weighted=lambda x:x.target*x.weight).groupby('date').weighted.sum()
    np.testing.assert_allclose(per_day,0,atol=2e-14,rtol=0)
    result=dict(passed=True,model_report_sha256=sha(ROOT/'model_report.json'),rows=len(train),
        all_node_counts_means_variances_rebuilt=True,every_training_day_has_zero_mean_target=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'model_verification.json',result)
    return result


def freeze():
    proof=json.loads((ROOT/'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256']==sha(ROOT/'model_report.json')
    return daily.freeze(ROOT,PROTOCOL,ROOT/'model_report.json')


def verify():
    return daily.verify(ROOT,PROTOCOL)


def analyze():
    return daily.analyze(ROOT,PROTOCOL)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['model','verify_model','freeze','verify','analyze'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
