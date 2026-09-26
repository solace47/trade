"""Independently rebuild ordered-path labels, fixed pairs and date-weighted summaries."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root=Path('data/research/impulse_retest');labels=Path('data/research/next_day_winner/labels.parquet')
report=json.loads((root/'analysis_report.json').read_text())
inputs=json.loads((root/'input_report.json').read_text())
assert report['input_report_sha256']==sha(root/'input_report.json') and report['labels_sha256']==sha(labels)
assert report['input_verification_sha256']==sha(root/'input_verification.json')
assert json.loads((root/'input_verification.json').read_text())['passed']
for name in ['features','pairs']:assert sha(root/(name+'.parquet'))==inputs[name+'_sha256']
for name,digest in report['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
c=duckdb.connect();c.execute('SET threads=4')
c.read_parquet(str(labels)).create_view('labels')
c.read_parquet(str(root/'features.parquet')).create_view('features')
c.read_parquet(str(root/'pairs.parquet')).create_view('pairs')
c.execute('''CREATE TABLE observations AS SELECT f.*,l.known_label AS known,
    l.known_label AND round(l.next_close*100)::BIGINT*100>=round(l.next_preclose*100)::BIGINT*105 AS up,
    l.known_label AND round(l.next_close*100)::BIGINT*100<=round(l.next_preclose*100)::BIGINT*95 AS down,
    CASE WHEN l.known_label THEN l.next_close/l.next_preclose-1 END AS gain
    FROM features f JOIN labels l USING(date,code)''')
assert c.sql('SELECT count(*) FROM observations').fetchone()[0]==inputs['rows']
metrics=['up_lower','up_upper','down_lower','down_upper','reference_mean']
group_frames=[]
for scope,where in [('all','TRUE'),('necessary','necessary_tradeable')]:
    p=c.sql(f'''SELECT date,half,board,category,'{scope}' AS scope,count(*) AS n,sum(known::INT) AS known,
        sum(up::INT) AS up_count,sum(down::INT) AS down_count,avg(gain) AS reference_mean,
        avg(up::INT) AS up_lower,avg(up::INT+(NOT known)::INT) AS up_upper,
        avg(down::INT) AS down_lower,avg(down::INT+(NOT known)::INT) AS down_upper
        FROM observations WHERE {where} GROUP BY date,half,board,category''').df()
    group_frames.append(p)
daily=pd.concat(group_frames,ignore_index=True)

def compare(p,name,keys):
    stored=pd.read_parquet(root/(name+'.parquet')).set_index(keys).sort_index()
    expected=p.set_index(keys).sort_index()[stored.columns]
    pd.testing.assert_frame_equal(expected,stored,check_dtype=False,atol=2e-12,rtol=0)

compare(daily,'group_daily',['scope','board','half','category','date'])
paired=c.sql('''SELECT p.*,a.known AS candidate_known,b.known AS control_known,
    a.up AS candidate_up,b.up AS control_up,a.down AS candidate_down,b.down AS control_down,
    a.gain AS candidate_gain,b.gain AS control_gain,a.known AND b.known AS both_known,
    CASE WHEN a.known AND b.known THEN a.gain-b.gain END AS reference_difference,
    a.up::INT-b.up::INT-(NOT b.known)::INT AS up_delta_lower,
    a.up::INT-b.up::INT+(NOT a.known)::INT AS up_delta_upper,
    a.down::INT-b.down::INT-(NOT b.known)::INT AS down_delta_lower,
    a.down::INT-b.down::INT+(NOT a.known)::INT AS down_delta_upper
    FROM pairs p JOIN observations a ON p.date=a.date AND p.code=a.code
    JOIN observations b ON p.date=b.date AND p.control_code=b.code''').df()
compare(paired,'paired_outcomes',['date','code'])
c.register('independent_pairs',paired)
pair_metrics=['up_delta_lower','up_delta_upper','down_delta_lower','down_delta_upper','reference_delta',
              'candidate_reference','control_reference']
pdaily=c.sql('''SELECT date,half,board,count(*) AS n,sum(both_known::INT) AS both_known,
    count(DISTINCT control_code) AS distinct_controls,avg(reference_difference) AS reference_delta,
    avg(up_delta_lower) AS up_delta_lower,avg(up_delta_upper) AS up_delta_upper,
    avg(down_delta_lower) AS down_delta_lower,avg(down_delta_upper) AS down_delta_upper,
    avg(candidate_gain) FILTER(WHERE both_known) AS candidate_reference,
    avg(control_gain) FILTER(WHERE both_known) AS control_reference
    FROM independent_pairs GROUP BY date,half,board ORDER BY date''').df()
compare(pdaily,'pair_daily',['board','half','date'])

def equal(actual,expected):
    if expected is None or (np.isscalar(expected) and pd.isna(expected)):assert actual is None
    else:np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0)

def bootstrap(p,column):
    if not len(p) or p[column].isna().any():return None
    week=pd.to_datetime(p.date).dt.to_period('W-SUN').astype(str)
    b=p[column].groupby(week).agg(['sum','size']).sort_index()
    if len(b)<2:return None
    draws=np.random.default_rng(20260926).integers(0,len(b),(10000,len(b)))
    result=b['sum'].to_numpy()[draws].sum(axis=1)/b['size'].to_numpy()[draws].sum(axis=1)
    return np.percentile(result,[2.5,97.5])

for item in report['groups']:
    p=daily.loc[daily.scope.eq(item['scope'])&daily.board.eq(item['board'])&daily.half.eq(item['half'])&daily.category.eq(item['category'])].sort_values('date')
    for name,value in {'stock_days':p.n.sum(),'dates':len(p),'known_labels':p.known.sum(),
        'unknown_labels':(p.n-p.known).sum(),'up_cases':p.up_count.sum(),'down_cases':p.down_count.sum()}.items():assert item[name]==value
    for m in metrics:
        equal(item[m],p[m].mean());equal(item[m+'_week_interval'],bootstrap(p,m))
for item in report['pair_summaries']:
    board,half=item['board'],item['half']
    p=pdaily.loc[pdaily.board.eq(board)&pdaily.half.eq(half)].sort_values('date')
    rows=paired.loc[paired.board.eq(board)&paired.half.eq(half)]
    candidates,days=c.execute("SELECT count(*),count(DISTINCT date) FROM observations WHERE board=? AND half=? AND necessary_tradeable AND category='held_with_volume_confirmation'",[board,half]).fetchone()
    expected={'candidates':candidates,'candidate_dates':days,'pairs':len(rows),'paired_dates':len(p),
        'distinct_control_stock_days':p.distinct_controls.sum(),'unknown_label_pairs':(~rows.both_known).sum(),
        'reference_dates':p.reference_delta.notna().sum()}
    for name,value in expected.items():assert item[name]==value
    for name,number in {'distance_mean':rows.distance.mean(),'distance_median':rows.distance.median(),
        'distance_p90':rows.distance.quantile(.9),'distance_max':rows.distance.max()}.items():equal(item[name],number)
    for name,data in item['input_balance'].items():
        equal(data['mean'],rows[name].mean());equal(data['mean_absolute'],rows[name].abs().mean())
    for m in pair_metrics:
        equal(item[m],p[m].mean());equal(item[m+'_week_interval'],bootstrap(p,m))
for scope,where in [('all','TRUE'),('necessary','necessary_tradeable')]:
    for category in sorted({x['category'] for x in report['portraits']}):
        frame=c.execute(f'''SELECT date,board,half,up,count(*) AS n,sum((category=?)::INT) AS pattern_cases,
            avg((category=?)::INT) AS frequency FROM observations WHERE {where} AND known
            GROUP BY date,board,half,up ORDER BY date''',[category,category]).df()
        for item in [r for r in report['portraits'] if r['scope']==scope and r['category']==category]:
            p=frame.loc[frame.board.eq(item['board'])&frame.half.eq(item['half'])&frame.up.eq(item['winner'])]
            assert item['cases']==p.n.sum() and item['pattern_cases']==p.pattern_cases.sum() and item['dates']==len(p)
            equal(item['pooled_frequency'],p.pattern_cases.sum()/p.n.sum())
            equal(item['daily_frequency'],p.frequency.mean())
            equal(item['daily_frequency_week_interval'],bootstrap(p,'frequency'))
result={'passed':True,'cohort_rows':inputs['rows'],'group_dates':len(daily),'fixed_pairs':len(paired),'pair_dates':len(pdaily),
    'group_values_and_intervals':len(report['groups'])*len(metrics),'pair_values_and_intervals':len(report['pair_summaries'])*len(pair_metrics),
    'inverse_frequencies_and_intervals':len(report['portraits']),
    'analysis_report_sha256':sha(root/'analysis_report.json'),'new_2026_prices_read':False}
save_json(root/'analysis_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
