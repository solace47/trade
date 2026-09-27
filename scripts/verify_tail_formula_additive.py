"""Rebuild frozen additive scores and the 2025H1 threshold decision in SQL."""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from verify_tick_flow_analysis import interval

ROOT=Path('data/research/tail_formula_additive')
FEATURES=Path('data/research/tail_formula_joint')
SOURCE=Path('data/research/tail_formula_1000')
PROTOCOL=Path('config/tail_formula_additive_protocol.json')


def load(name):
    return json.loads((ROOT/name).read_text())


def tree_sql(tree,node=0):
    left=tree['children_left'][node]
    if left<0:
        # Scientific literals force DOUBLE rather than decimal arithmetic.
        return format(.05*tree['value'][node],'.17e')
    right=tree['children_right'][node]
    feature=tree['feature'][node]+1
    cut=math.floor(tree['threshold'][node])
    return f'(CASE WHEN X{feature:02d}<={cut} THEN {tree_sql(tree,left)} ELSE {tree_sql(tree,right)} END)'


def connection():
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.read_parquet(str(FEATURES/'features.parquet')).create_view('features')
    c.read_parquet(str(ROOT/'scores.parquet')).create_view('scores')
    return c


def scores():
    r=load('score_report.json');m=load('model_report.json');v=load('model_verification.json')
    assert v['passed'] and v['model_report_sha256']==sha(ROOT/'model_report.json')
    for key,path in [('protocol_sha256',PROTOCOL),('model_report_sha256',ROOT/'model_report.json'),
                     ('feature_report_sha256',FEATURES/'feature_report.json'),('scores_sha256',ROOT/'scores.parquet')]:
        assert r[key]==sha(path)
    fr=json.loads((FEATURES/'feature_report.json').read_text())
    assert fr['features_sha256']==sha(FEATURES/'features.parquet')
    assert m['feature_names']==list(fr['expressions'])
    c=connection()
    enc=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}'
                 for i,n in enumerate(m['feature_names'],1))
    c.sql('SELECT date,code,'+enc+' FROM features WHERE formula_input_valid').create_view('encoded')
    score=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
    c.sql('SELECT date,code,'+score+' AS rebuilt_score FROM encoded').create_view('rebuilt')
    expected=c.sql('''SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,
        r.rebuilt_score AS score FROM features f LEFT JOIN rebuilt r USING(date,code) ORDER BY date,code''').df()
    actual=c.sql('SELECT * FROM scores ORDER BY date,code').df()
    pd.testing.assert_frame_equal(actual.drop(columns='score'),expected.drop(columns='score'),check_exact=True,check_dtype=False)
    np.testing.assert_allclose(actual.score,expected.score,rtol=0,atol=2e-11,equal_nan=True)
    assert actual.score.notna().equals(actual.formula_input_valid)
    checks=0
    for t in m['thresholds']:
        np.testing.assert_array_equal(actual.score.gt(t['threshold']),expected.score.gt(t['threshold']))
        checks+=len(actual)
    c.execute(f'''CREATE VIEW training_keys AS SELECT date,code FROM read_parquet('{SOURCE}/full_labels.parquet')
        WHERE next_date<'2025-01-01' AND known15''')
    train=c.sql('SELECT rebuilt_score FROM rebuilt JOIN training_keys USING(date,code) ORDER BY date,code').df()
    for t in m['thresholds']:
        np.testing.assert_allclose(np.quantile(train.rebuilt_score,t['training_quantile']),t['threshold'],rtol=0,atol=2e-11)
    result=dict(passed=True,score_report_sha256=sha(ROOT/'score_report.json'),rows=len(actual),
        valid=int(actual.formula_input_valid.sum()),max_score_difference=float((actual.score-expected.score).abs().max()),
        threshold_flag_checks=checks,all_integer_encodings_tree_scores_and_training_quantiles_rebuilt=True,
        new_2025_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'score_verification.json',result)
    return result


def selection():
    r=load('selection_report.json');m=load('model_report.json');v=load('score_verification.json')
    assert v['passed'] and v['score_report_sha256']==sha(ROOT/'score_report.json')
    for key,path in [('protocol_sha256',PROTOCOL),('model_report_sha256',ROOT/'model_report.json'),
                     ('score_report_sha256',ROOT/'score_report.json'),('calibration_label_report_sha256',SOURCE/'full_label_report.json'),
                     ('calibration_days_sha256',ROOT/'calibration_days.parquet'),('selection_sha256',ROOT/'selection.parquet')]:
        assert r[key]==sha(path)
    lr=json.loads((SOURCE/'full_label_report.json').read_text())
    assert lr['labels_sha256']==sha(SOURCE/'full_labels.parquet')
    c=connection()
    c.execute(f'''CREATE VIEW labels AS SELECT date,code,known15,opportunity15,known_no_trade
        FROM read_parquet('{SOURCE}/full_labels.parquet') WHERE date>='2025-01-01' AND next_date<'2025-07-01' ''')
    c.sql('SELECT date,avg(opportunity15) AS base_rate FROM labels GROUP BY date').create_view('baseline')
    days=[];admitted=[];checks=0
    assert len(r['thresholds'])==len(m['thresholds'])
    for t,s in zip(m['thresholds'],r['thresholds']):
        assert all(t[k]==s[k] for k in t)
        d=c.sql(f'''WITH d AS(SELECT date,count(*) AS rows,count(*) FILTER(WHERE known15) AS known,
            count(*) FILTER(WHERE known15 AND opportunity15=1) AS success,
            count(*) FILTER(WHERE NOT known15 AND NOT known_no_trade) AS unknown,
            count(*) FILTER(WHERE known_no_trade) AS no_trade FROM scores JOIN labels USING(date,code)
            WHERE formula_input_valid AND score>{format(t['threshold'],'.17e')} GROUP BY date)
            SELECT d.*,success/nullif(known,0) AS rate,success/rows AS lower,base_rate,
            success/nullif(known,0)-base_rate AS delta,{t['id']} AS cut_id FROM d JOIN baseline USING(date) ORDER BY date''').df()
        days.append(d)
        ci=interval(d,'delta')
        values=dict(rows=int(d.rows.sum()),known=int(d.known.sum()),unknown=int(d.unknown.sum()),no_trade=int(d.no_trade.sum()),
            days=len(d),rate=d.rate.mean(),lower=d.lower.mean(),delta=d.delta.mean(),mean_selected_per_day=d.rows.mean(),
            p95_selected=np.quantile(d.rows,.95) if len(d) else np.nan,delta_ci=ci)
        for key,value in values.items():
            if s[key] is None:
                assert value is None or np.isnan(value)
            else:
                np.testing.assert_allclose(value,s[key],atol=2e-12,rtol=0)
            checks+=1
        eligible=values['days']>=30 and values['known']>=100 and values['mean_selected_per_day']<=20 and values['p95_selected']<=50
        assert eligible==s['admitted']
        if eligible:
            admitted.append(dict(**t,**values))
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT/'calibration_days.parquet'),pd.concat(days,ignore_index=True),
                                  check_dtype=False,rtol=0,atol=2e-12)
    best=sorted(admitted,key=lambda s:(-s['lower'],-s['delta'],-s['known'],s['threshold']))[0] if admitted else None
    assert len(admitted)==r['admitted_thresholds']
    if best:
        assert best['id']==r['chosen_threshold']['id']
        assert r['core_sha256']==sha(ROOT/'frozen_numeric_core.tdx')
        clause='formula_input_valid AND score>'+format(best['threshold'],'.17e')
    else:
        assert r['chosen_threshold'] is None
        clause='FALSE'
    expected=c.sql('SELECT date,code,half,board,decision_shares,'+clause+' AS selected FROM scores ORDER BY date,code').df()
    actual=pd.read_parquet(ROOT/'selection.parquet').sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True,check_dtype=False)
    assert int(actual.selected.sum())==r['selected']
    result=dict(passed=True,selection_report_sha256=sha(ROOT/'selection_report.json'),rows=len(actual),
        calibration_daily_rows=sum(map(len,days)),scalar_checks=checks,admitted_thresholds=len(admitted),
        all_threshold_statistics_and_selection_flags_rebuilt=True,new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['scores','selection'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--protocol',type=Path,default=PROTOCOL)
    args=p.parse_args();ROOT=args.root;PROTOCOL=args.protocol
    print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
