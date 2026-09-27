"""Independently reconstruct every calibration leaf and the quality decision."""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from verify_tick_flow_analysis import interval

ROOT=Path('data/research/tail_formula_selective')
FEATURE_ROOT=Path('data/research/tail_formula_intraday')
SOURCE=Path('data/research/tail_formula_1000')


def check(ROOT=ROOT,FEATURE_ROOT=FEATURE_ROOT,PROTOCOL=Path('config/tail_formula_selective_protocol.json')):
    report=json.loads((ROOT/'calibration_report.json').read_text())
    assert sha(ROOT/'calibration_report.json')==sha(ROOT/'selection_report.json')
    for key,path in [('protocol_sha256',PROTOCOL),
                     ('model_report_sha256',ROOT/'model_report.json'),
                     ('feature_report_sha256',FEATURE_ROOT/'feature_report.json'),
                     ('full_label_report_sha256',SOURCE/'full_label_report.json'),
                     ('calibration_leaf_days_sha256',ROOT/'calibration_leaf_days.parquet'),
                     ('selection_sha256',ROOT/'selection.parquet')]:
        assert report[key]==sha(path)
    proof=json.loads((ROOT/'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256']==report['model_report_sha256']
    fr=json.loads((FEATURE_ROOT/'feature_report.json').read_text())
    lr=json.loads((SOURCE/'full_label_report.json').read_text())
    assert fr['features_sha256']==sha(FEATURE_ROOT/'features.parquet')
    assert lr['labels_sha256']==sha(SOURCE/'full_labels.parquet')
    model=json.loads((ROOT/'model_report.json').read_text())
    tree=model['tree']
    names=model['feature_names']
    assert names==list(fr['expressions'])
    leaves={}
    for leaf,left in enumerate(tree['children_left']):
        if left>=0:
            continue
        node=leaf
        reverse=[]
        while node:
            if node in tree['children_left']:
                parent=tree['children_left'].index(node)
                op,threshold='<=',math.floor(tree['threshold'][parent]*100)/100
            else:
                parent=tree['children_right'].index(node)
                op,threshold='>',math.ceil(tree['threshold'][parent]*100)/100
            reverse.append(dict(feature=names[tree['feature'][parent]],op=op,threshold=threshold))
            node=parent
        leaves[leaf]=list(reversed(reverse))
    assert len(report['scores'])==len(leaves)
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.read_parquet(str(FEATURE_ROOT/'features.parquet')).create_view('features')
    c.execute(f'''CREATE VIEW labels AS SELECT date,code,next_date,known15,opportunity15,known_no_trade,
        CASE WHEN next_date<'2024-07-01' THEN 'discovery' ELSE 'calibration' END AS phase
        FROM read_parquet('{SOURCE}/full_labels.parquet') WHERE next_date<'2025-01-01'
        AND (next_date<'2024-07-01' OR date>='2024-07-01')''')
    c.execute('''CREATE VIEW baseline AS SELECT date,avg(opportunity15) AS base_rate FROM labels GROUP BY date''')
    daily=[]
    qualified=[]
    scalar_checks=0
    for score in report['scores']:
        node=score['node']
        assert score['conditions']==leaves[node]
        condition=' AND '.join(f"{x['feature']}{x['op']}{x['threshold']}" for x in leaves[node])
        d=c.sql(f'''WITH selected AS(SELECT f.*,l.* EXCLUDE(date,code) FROM features f JOIN labels l USING(date,code)
                WHERE formula_input_valid AND {condition}),
            d AS(SELECT phase,date,count(*) AS rows,count(*) FILTER(WHERE known15) AS known,
                count(*) FILTER(WHERE known15 AND opportunity15=1) AS success,
                count(*) FILTER(WHERE NOT known15 AND NOT known_no_trade) AS unknown,
                count(*) FILTER(WHERE known_no_trade) AS no_trade FROM selected GROUP BY phase,date)
            SELECT d.*,success/nullif(known,0) AS rate,success/rows AS lower,base_rate,
                success/nullif(known,0)-base_rate AS delta,{node} AS node
            FROM d JOIN baseline USING(date) ORDER BY phase,date''').df()
        daily.append(d)
        qualifications={}
        recalibration=None
        for phase in ['discovery','calibration']:
            q=d.loc[d.phase.eq(phase)].set_index('date')
            ci=interval(q.reset_index(),'delta')
            values=dict(rows=int(q.rows.sum()),known=int(q.known.sum()),unknown=int(q.unknown.sum()),no_trade=int(q.no_trade.sum()),
                days=len(q),rate=q.rate.mean(),lower=q.lower.mean(),delta=q.delta.mean(),
                mean_selected_per_day=q.rows.mean(),p95_selected=np.quantile(q.rows,.95) if len(q) else np.nan,delta_ci=ci)
            for key,value in values.items():
                expected=score[phase][key]
                if expected is None:
                    assert value is None or np.isnan(value)
                else:
                    np.testing.assert_allclose(value,expected,atol=2e-12,rtol=0)
                scalar_checks+=1
            gates=dict(enough_days=values['days']>=40,enough_known=values['known']>=300,
                rate=values['rate']>=.6,conservative_rate=values['lower']>=.55,delta=values['delta']>=.08,
                mean_count=values['mean_selected_per_day']<=20,p95_count=values['p95_selected']<=50,
                delta_ci_positive=ci is not None and ci[0]>0)
            assert gates==score[phase]['gates']
            eligible=all(value for key,value in gates.items() if key!='delta_ci_positive')
            assert eligible==score[phase]['eligible']
            qualifications[phase]=eligible
            if phase=='calibration':
                recalibration=values
                qualifications['ci']=gates['delta_ci_positive']
        eligible=all(qualifications.values())
        assert eligible==score['eligible']
        if eligible:
            qualified.append(dict(node=node,**recalibration))
    expected_days=pd.concat(daily,ignore_index=True)
    actual=pd.read_parquet(ROOT/'calibration_leaf_days.parquet')
    pd.testing.assert_frame_equal(actual,expected_days,check_dtype=False,rtol=0,atol=2e-12)
    assert len(qualified)==report['qualified_leaves']
    best=sorted(qualified,key=lambda x:(-x['lower'],-x['delta'],-x['known'],x['node']))[0] if qualified else None
    if best:
        assert best['node']==report['chosen_leaf']['node'] and leaves[best['node']]==report['conditions']
        assert report['core_sha256']==sha(ROOT/'frozen_numeric_core.tdx')
        condition=' AND '.join(f"{x['feature']}{x['op']}{x['threshold']}" for x in report['conditions'])
    else:
        assert report['chosen_leaf'] is None and report['conditions'] is None and report['selected']==0
        condition='FALSE'
    expected=c.sql('SELECT date,code,half,board,decision_shares,formula_input_valid AND '+condition+
        ' AS selected FROM features ORDER BY date,code').df()
    selected=pd.read_parquet(ROOT/'selection.parquet')
    pd.testing.assert_frame_equal(selected,expected,check_exact=True,check_dtype=False)
    result=dict(passed=True,selection_report_sha256=sha(ROOT/'selection_report.json'),leaves=len(leaves),
        calibration_daily_rows=len(actual),scalar_checks=scalar_checks,qualified_leaves=len(qualified),
        all_daily_groups_quality_gates_and_selection_rebuilt=True,new_2025_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--features-root',type=Path,default=FEATURE_ROOT)
    parser.add_argument('--protocol',type=Path,default=Path('config/tail_formula_selective_protocol.json'))
    args=parser.parse_args()
    print(json.dumps(check(args.root,args.features_root,args.protocol),ensure_ascii=False,indent=2))
