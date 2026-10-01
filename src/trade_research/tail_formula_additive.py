"""Verified encodings, tree replay and native formula export used by the current study."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from .research_io import save_json, sha
from .tail_formula_baseline import INPUTS as FEATURES, EXPRESSIONS, HEADER

SOURCE = FEATURES
ROOT = Path('data/research/tail_formula_retained_interval')
PROTOCOL = Path('config/tail_formula_retained_interval_model_protocol.json')
QUANTILES = [.97,.98,.99,.995,.998,.999,.9995]

def conn():
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='5GB'")
    return c


def feature_inputs():
    r=json.loads((FEATURES/'feature_report.json').read_text())
    v=json.loads((FEATURES/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(FEATURES/'feature_report.json')
    assert r['features_sha256']==sha(FEATURES/'features.parquet')
    return pd.read_parquet(FEATURES/'features.parquet')


def labels(where):
    r=json.loads((SOURCE/'full_label_report.json').read_text())
    v=json.loads((SOURCE/'full_label_verification.json').read_text())
    assert v['passed'] and v['label_report_sha256']==sha(SOURCE/'full_label_report.json')
    assert r['labels_sha256']==sha(SOURCE/'full_labels.parquet')
    c=conn()
    return c.execute(f'''SELECT date,code,next_date,known15,opportunity15,known_no_trade
        FROM read_parquet(?) WHERE {where}''',[str(SOURCE/'full_labels.parquet')]).df()


def training(f=None,start=None,end='2025-01-01'):
    f=feature_inputs() if f is None else f
    from datetime import date
    assert date.fromisoformat(end).isoformat()==end
    assert start is None or date.fromisoformat(start).isoformat()==start
    where=f"next_date<'{end}'"+(f" AND date>='{start}'" if start else '')
    l=labels(where)
    assert l.next_date.lt(end).all() and (start is None or l.date.ge(start).all())
    t=f.loc[f.formula_input_valid].merge(l.loc[l.known15],on=['date','code'],validate='one_to_one')
    return t.sort_values(['date','code']).reset_index(drop=True)


def encode(f):
    x=f[list(EXPRESSIONS)].to_numpy(dtype=float)
    assert np.isfinite(x).all()
    return np.floor(np.clip(100*x+10000+.000001,0,999999)).astype('int32')


def leaf_indices(x,t):
    node=np.zeros(len(x),dtype='int32')
    left=np.asarray(t['children_left'])
    right=np.asarray(t['children_right'])
    feature=np.asarray(t['feature'])
    threshold=np.asarray(t['threshold'])
    for _ in range(len(left)):
        active=left[node]>=0
        if not active.any():
            break
        n=node[active]
        branch=x[active,feature[n]]<=threshold[n]
        node[active]=np.where(branch,left[n],right[n])
    assert (left[node]<0).all()
    return node


def predict(x,r):
    score=np.full(len(x),r['bias'],dtype=float)
    for t in r['trees']:
        score+=r['learning_rate']*np.asarray(t['value'])[leaf_indices(x,t)]
    return score


def scores():
    if (ROOT/'score_report.json').exists():
        raise ValueError('Do not replace fixed model scores')
    proof=json.loads((ROOT/'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256']==sha(ROOT/'model_report.json')
    r=json.loads((ROOT/'model_report.json').read_text())
    f=feature_inputs()
    out=f[['date','code','half','board','decision_shares','formula_input_valid']].copy()
    out['score']=np.nan
    valid=f.formula_input_valid
    out.loc[valid,'score']=predict(encode(f.loc[valid]),r)
    out.to_parquet(ROOT/'scores.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),model_report_sha256=sha(ROOT/'model_report.json'),
        feature_report_sha256=sha(FEATURES/'feature_report.json'),scores_sha256=sha(ROOT/'scores.parquet'),
        rows=len(out),valid=int(valid.sum()),new_2025_score_groups_read=r.get('new_2025_score_groups_read',False),
        new_2025H2_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'score_report.json',report)
    return report


def native_tree(t,index=0):
    if t['children_left'][index]<0:
        return format(.05*t['value'][index],'.17g')
    threshold=int(np.floor(t['threshold'][index]))
    name=f"X{t['feature'][index]+1:02d}"
    return f"IF({name}<={threshold},{native_tree(t,t['children_left'][index])},{native_tree(t,t['children_right'][index])})"


def native_core(model,threshold,expressions=EXPRESSIONS,header=HEADER):
    core=header+'\n'.join(f'{k}:={v};' for k,v in expressions.items())+'\n'
    core+='\n'.join(f'X{i:02d}:=INTPART(MIN(MAX(100*{name}+10000+0.000001,0),999999));' for i,name in enumerate(expressions,1))+'\n'
    core+='\n'.join(f'T{i:02d}:={native_tree(t)};' for i,t in enumerate(model['trees'],1))+'\n'
    core+='SC:='+format(model['bias'],'.17g')+'+'+'+'.join(f'T{i:02d}' for i in range(1,len(model['trees'])+1))+';\n'
    return core+'CORE:SC>'+format(threshold,'.17g')+';\n'
