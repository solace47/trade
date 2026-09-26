"""Complete next-day tails and fixed same-date comparisons of ordered paths."""
import json

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .impulse_retest import ROOT,BASE,CATEGORIES,PROTOCOL
from .reference_gain_accounting import weekly_interval

METRICS=['up_lower','up_upper','down_lower','down_upper','reference_mean']
PAIR_METRICS=['up_delta_lower','up_delta_upper','down_delta_lower','down_delta_upper','reference_delta',
              'candidate_reference','control_reference']
HALVES=['2024H1','2024H2','2025H1','2025H2']
BOARDS=['main','chinext','star']


def value(x):return float(x) if pd.notna(x) and np.isfinite(x) else None


def evaluate():
    if (ROOT/'analysis_report.json').exists():raise ValueError('Do not overwrite an observed path result')
    inputs=json.loads((ROOT/'input_report.json').read_text())
    check=json.loads((ROOT/'input_verification.json').read_text())
    manifest=json.loads((ROOT/'manifest.json').read_text())
    original=json.loads(BASE.with_name('base_report.json').read_text())
    assert check['passed'] and check['input_report_sha256']==sha(ROOT/'input_report.json')
    assert sha(PROTOCOL)==manifest['protocol_sha256']
    for name in ['features','pairs']:assert sha(ROOT/(name+'.parquet'))==inputs[name+'_sha256']
    labels=BASE.with_name('labels.parquet');assert sha(labels)==original['labels_sha256']
    c=duckdb.connect();c.execute('SET threads=4')
    c.read_parquet(str(ROOT/'features.parquet')).create_view('features')
    c.read_parquet(str(ROOT/'pairs.parquet')).create_view('pairs')
    c.read_parquet(str(labels)).create_view('labels')
    c.execute('''CREATE TABLE observations AS SELECT f.*,l.known_label,l.winner IS TRUE AS up,
      l.known_label AND round(l.next_close*100)::BIGINT*100<=round(l.next_preclose*100)::BIGINT*95 AS down,
      CASE WHEN l.known_label THEN l.next_gain END AS reference_gain
      FROM features f JOIN labels l USING(date,code)''')
    daily=[]
    for scope,where in [('all','TRUE'),('necessary','necessary_tradeable')]:
        p=c.sql(f'''SELECT date,half,board,category,'{scope}' AS scope,count(*) AS n,
            sum(known_label::INT) AS known,sum(up::INT) AS up_count,sum(down::INT) AS down_count,
            avg(reference_gain) AS reference_mean FROM observations WHERE {where}
            GROUP BY date,half,board,category''').df()
        p['up_lower']=p.up_count/p.n;p['up_upper']=(p.up_count+p.n-p.known)/p.n
        p['down_lower']=p.down_count/p.n;p['down_upper']=(p.down_count+p.n-p.known)/p.n
        daily.append(p)
    daily=pd.concat(daily,ignore_index=True).sort_values(['scope','board','category','date'])
    daily.to_parquet(ROOT/'group_daily.parquet',index=False,compression='zstd')
    groups=[]
    for scope in ['all','necessary']:
        for board in BOARDS:
            for half in HALVES:
                for category in CATEGORIES:
                    p=daily.loc[daily.scope.eq(scope)&daily.board.eq(board)&daily.half.eq(half)&daily.category.eq(category)]
                    item=dict(scope=scope,board=board,half=half,category=category,stock_days=int(p.n.sum()),dates=len(p),
                        known_labels=int(p.known.sum()),unknown_labels=int((p.n-p.known).sum()),up_cases=int(p.up_count.sum()),down_cases=int(p.down_count.sum()))
                    for m in METRICS:
                        series=p.set_index('date')[m];item[m]=value(series.mean());item[m+'_week_interval']=weekly_interval(series)
                    groups.append(item)
    # The input-only pairs remain present even when either outcome is unknown.
    paired=c.sql('''SELECT p.*,a.known_label AS candidate_known,b.known_label AS control_known,
      a.up AS candidate_up,b.up AS control_up,a.down AS candidate_down,b.down AS control_down,
      a.reference_gain AS candidate_gain,b.reference_gain AS control_gain,
      CASE WHEN a.known_label AND b.known_label THEN a.reference_gain-b.reference_gain END AS reference_difference
      FROM pairs p JOIN observations a ON p.date=a.date AND p.code=a.code
      JOIN observations b ON p.date=b.date AND p.control_code=b.code''').df()
    assert len(paired)==inputs['pairs'] and not paired.duplicated(['date','code']).any()
    paired['both_known']=paired.candidate_known&paired.control_known
    for direction in ['up','down']:
        lower=paired['candidate_'+direction].astype(int)-paired['control_'+direction].astype(int)
        paired[direction+'_delta_lower']=lower-(~paired.control_known).astype(int)
        paired[direction+'_delta_upper']=lower+(~paired.candidate_known).astype(int)
    pairs_daily=paired.groupby(['date','half','board']).agg(n=('code','size'),
        both_known=('both_known','sum'),distinct_controls=('control_code','nunique'),
        reference_delta=('reference_difference','mean'),
        **{name:(name,'mean') for name in PAIR_METRICS if name not in ['reference_delta','candidate_reference','control_reference']}).reset_index()
    joint=paired.loc[paired.both_known].groupby(['date','half','board']).agg(candidate_reference=('candidate_gain','mean'),control_reference=('control_gain','mean')).reset_index()
    pairs_daily=pairs_daily.merge(joint,on=['date','half','board'],how='left',validate='one_to_one').sort_values('date')
    paired.to_parquet(ROOT/'paired_outcomes.parquet',index=False,compression='zstd')
    pairs_daily.to_parquet(ROOT/'pair_daily.parquet',index=False,compression='zstd')
    f=pd.read_parquet(ROOT/'features.parquet');pair_summaries=[]
    for board in BOARDS:
        for half in HALVES:
            p=pairs_daily.loc[pairs_daily.board.eq(board)&pairs_daily.half.eq(half)]
            rows=paired.loc[paired.board.eq(board)&paired.half.eq(half)]
            candidates=f.loc[f.board.eq(board)&f.half.eq(half)&f.necessary_tradeable&f.category.eq(CATEGORIES[-1])]
            item=dict(board=board,half=half,candidates=len(candidates),candidate_dates=candidates.date.nunique(),
                pairs=len(rows),paired_dates=len(p),distinct_control_stock_days=int(p.distinct_controls.sum()),
                unknown_label_pairs=int((~rows.both_known).sum()),reference_dates=int(p.reference_delta.notna().sum()),
                distance_mean=value(rows.distance.mean()),distance_median=value(rows.distance.median()),
                distance_p90=value(rows.distance.quantile(.9)),distance_max=value(rows.distance.max()))
            item['input_balance']={name:{'mean':value(rows[name].mean()),'mean_absolute':value(rows[name].abs().mean())}
                for name in rows.columns if name.endswith('_difference') and name!='reference_difference'}
            for m in PAIR_METRICS:
                series=p.set_index('date')[m];item[m]=value(series.mean());item[m+'_week_interval']=weekly_interval(series)
            pair_summaries.append(item)
    # Complete inverse denominators: how often each path appeared among winners and nonwinners.
    portraits=[]
    for scope,where in [('all','TRUE'),('necessary','necessary_tradeable')]:
        totals=c.sql(f'''SELECT date,half,board,up,count(*) AS n FROM observations
            WHERE {where} AND known_label GROUP BY date,half,board,up''').df()
        categories=c.sql(f'''SELECT date,half,board,up,category,count(*) AS category_n FROM observations
            WHERE {where} AND known_label GROUP BY date,half,board,up,category''').df()
        for category in CATEGORIES:
            p=totals.merge(categories.loc[categories.category.eq(category)],on=['date','half','board','up'],how='left',validate='one_to_one')
            p['category_n']=p.category_n.fillna(0);p['frequency']=p.category_n/p.n
            for (board,half,up),g in p.groupby(['board','half','up']):
                series=g.set_index('date').frequency.sort_index()
                portraits.append(dict(scope=scope,board=board,half=half,winner=bool(up),category=category,
                    cases=int(g.n.sum()),pattern_cases=int(g.category_n.sum()),dates=len(g),
                    pooled_frequency=float(g.category_n.sum()/g.n.sum()),daily_frequency=float(series.mean()),
                    daily_frequency_week_interval=weekly_interval(series)))
    result={'interpretation':'exploratory_ordered_path_association_not_investor_identity_or_executed_profit',
        'input_report_sha256':sha(ROOT/'input_report.json'),'input_verification_sha256':sha(ROOT/'input_verification.json'),
        'labels_sha256':sha(labels),'groups':groups,'pair_summaries':pair_summaries,'portraits':portraits,
        'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in ['group_daily','paired_outcomes','pair_daily']},
        'new_2026_prices_read':False}
    save_json(ROOT/'analysis_report.json',result)
    return {'primary_necessary': [g for g in groups if g['scope']=='necessary' and g['category']==CATEGORIES[-1]],'paired':pair_summaries}


if __name__=='__main__':print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
