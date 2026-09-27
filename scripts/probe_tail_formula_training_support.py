"""Describe trading-day support of trained rule leaves without new outcomes."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha
from trade_research.tail_formula_additive import leaf_indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--features-root',type=Path,required=True)
    a=p.parse_args();r=json.loads((a.root/'model_report.json').read_text())
    proof=json.loads((a.root/'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256']==sha(a.root/'model_report.json')
    fr=json.loads((a.features_root/'feature_report.json').read_text())
    assert r['feature_report_sha256']==sha(a.features_root/'feature_report.json')
    assert fr['features_sha256']==sha(a.features_root/'features.parquet')
    c=duckdb.connect()
    keys=c.execute('''SELECT date,code FROM read_parquet('data/research/tail_formula_1000/full_labels.parquet')
        WHERE known15 AND date>=? AND next_date<?''',[r['training_start'],r['training_end']]).df()
    f=pd.read_parquet(a.features_root/'features.parquet',columns=['date','code','formula_input_valid',*r['feature_names']])
    f=f.loc[f.formula_input_valid].merge(keys,on=['date','code'],validate='one_to_one').sort_values(['date','code'])
    raw=f[r['feature_names']].to_numpy(dtype=float)
    x=np.floor(np.clip(100*raw+10000+.000001,0,999999)).astype('int32')
    codes,dates=pd.factorize(f.date,sort=True);ndays=len(dates)
    w=1/np.bincount(codes)[codes]
    leaves=[];score=np.full(len(f),r['bias'])
    for tree_id,t in enumerate(r['trees']):
        terminal=leaf_indices(x,t)
        counts=np.bincount(terminal*ndays+codes,minlength=len(t['feature'])*ndays).reshape(-1,ndays)
        weights=np.bincount(terminal*ndays+codes,weights=w,minlength=len(t['feature'])*ndays).reshape(-1,ndays)
        for node in np.flatnonzero(np.asarray(t['children_left'])<0):
            leaves.append(dict(tree=tree_id,node=int(node),rows=int(counts[node].sum()),days=int((counts[node]>0).sum()),
                day_weight=float(weights[node].sum()),maximum_date_weight_fraction=float(weights[node].max()/weights[node].sum())))
        score+=r['learning_rate']*np.asarray(t['value'])[terminal]
    selected=f.loc[score>r['thresholds'][3]['threshold']]
    counts=selected.groupby('date').size().sort_values(ascending=False)
    report=dict(model_report_sha256=sha(a.root/'model_report.json'),features_sha256=sha(a.features_root/'features.parquet'),
        training_rows=len(f),training_days=ndays,leaves=leaves,
        leaves_below_10_days=sum(v['days']<10 for v in leaves),leaves_below_20_days=sum(v['days']<20 for v in leaves),
        minimum_leaf_days=min(v['days'] for v in leaves),selected_training_rows=len(selected),selected_training_days=len(counts),
        selected_training_top5_date_share=float(counts.head().sum()/len(selected)) if len(selected) else None,
        posthoc_diagnostic_only=True,evaluation_outcomes_read=False,new_2026_prices_read=False)
    save_json(a.root/'training_support_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='leaves'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
