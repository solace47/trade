"""Independent source, cash, frozen-tree and daily-statistic checks."""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_1000 import ROOT, OLD, PROTOCOL, EXPRESSIONS
from trade_research.tail_formula_1000_analysis import BUY_COLUMNS, OBS_COLUMNS
from verify_tick_flow_analysis import eq, interval


def frame_equal(a,b):
    pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=2e-10,rtol=0)


def label_check(scope):
    report=json.loads((ROOT/f'{scope}_label_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL)
    assert report['old_labels_sha256']==sha(OLD/'labels.parquet')
    assert report['labels_sha256']==sha(ROOT/f'{scope}_labels.parquet')
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='4GB'")
    got=pd.read_parquet(ROOT/f'{scope}_labels.parquet')
    c.register('keys',got[['date','code']])
    old=c.execute(f"SELECT {','.join('l.'+x for x in BUY_COLUMNS)} FROM read_parquet(?) l JOIN keys USING(date,code) ORDER BY date,code",
        [str(OLD/'labels.parquet')]).df()
    obs=c.execute(f"SELECT {','.join('l.'+x for x in OBS_COLUMNS)} FROM read_parquet(?) l JOIN keys USING(date,code) ORDER BY date,code",
        [str(ROOT/'observations.parquet')]).df()
    frame_equal(got[old.columns],old)
    frame_equal(got[obs.columns],obs)
    source=old.merge(obs,on=['date','code','next_date'],validate='one_to_one')
    c.register('source',source)
    c.execute('''CREATE VIEW facts AS SELECT *,
        CASE WHEN NOT necessary_tradeable THEN 'not_submitted'
             WHEN NOT coalesce(isfinite(entry_vwap) AND entry_vwap>0 AND entry_volume>0,false) THEN 'no_liquidity'
             WHEN NOT coalesce(entry_volume*.1>=decision_shares,false) THEN 'volume_cap'
             WHEN entry_vwap*1.0005>=upper_limit-.005 THEN 'estimated_upper_limit' ELSE 'filled' END AS fill,
        coalesce(isfinite(next_preclose) AND next_preclose>0 AND abs(next_preclose-round(next_preclose,2))<=.0001,false) AS valid_ref,
        coalesce(isfinite(day_close) AND day_close>0 AND abs(day_close-round(day_close,2))<=.0001,false) AS valid_close,
        NOT entry_source_valid OR period_entry_bad_day OR period_bad_symbol AS entry_source_unknown FROM source''')
    c.execute('''CREATE VIEW checked AS SELECT *,
        fill='filled' AS recorded,
        NOT entry_source_unknown AND fill<>'filled' AS known_no_trade,
        NOT catalog_covered OR action_exposure OR NOT valid_ref OR NOT valid_close OR abs(next_preclose-day_close)>.005 AS corporate_unknown,
        coalesce(valid_ref AND next_trade_status=1 AND next_isST IN (0,1) AND next_adjustflag=3,false) AS next_daily_valid,
        fill='filled' AND (NOT entry_bounds_valid OR round(entry_high,2)>=upper_limit) AS queue_unknown FROM facts''')
    c.execute('''CREATE VIEW states AS SELECT *,
        CASE WHEN entry_source_unknown THEN 'entry_source_unknown' WHEN known_no_trade THEN 'no_trade'
            WHEN queue_unknown THEN 'entry_queue_unknown' WHEN corporate_unknown THEN 'corporate_unknown'
            WHEN NOT next_daily_valid THEN 'next_daily_unknown' WHEN NOT source_valid THEN 'morning_source_unknown'
            ELSE 'known' END AS observation_status FROM checked''')
    expected=c.sql('''SELECT date,code,corporate_unknown,next_daily_valid,entry_source_unknown,known_no_trade,observation_status,
        recorded AS entry_recorded,queue_unknown AS entry_queue_unknown,fill AS entry_fill_status FROM states ORDER BY date,code''').df()
    frame_equal(got[expected.columns],expected)
    count=len(expected)*(len(expected.columns)-2)
    for bps in [5,15]:
        c.execute(f'''CREATE OR REPLACE VIEW cash AS WITH b AS (SELECT *,
            entry_vwap+greatest(entry_vwap*{bps}/10000.,.005) AS buy_price FROM states),
            v AS (SELECT *,decision_shares*buy_price AS buy_value FROM b)
            SELECT *,buy_value+CASE WHEN buy_value*.0003>5 THEN buy_value*.0003 ELSE 5 END+buy_value*.00001 AS buy_cash,
                recorded AND buy_price>=upper_limit-.005 AS stress,
                observation_status='known' AND NOT (recorded AND buy_price>=upper_limit-.005) AS known FROM v''')
        ex=c.sql('SELECT date,code,buy_cash,stress,known,known AND NOT period_exit_bad_day AS sensitive_known,NOT known AND NOT known_no_trade AS unknown FROM cash ORDER BY date,code').df()
        ex=ex.rename(columns={k:f'{k}{bps}' for k in ['buy_cash','known','unknown','sensitive_known']}).rename(columns={'stress':f'entry_stress_unknown{bps}'})
        frame_equal(got[ex.columns],ex)
        count+=len(ex)*5
        for name,price in [('sustained','sustained_close'),('any_close','max_close'),('mark_1000','price_1000'),('adverse','min_low')]:
            ex=c.sql(f'''WITH v AS (SELECT *,decision_shares*({price}-greatest({price}*{bps}/10000.,.005)) AS mark_value FROM cash)
                SELECT date,code,CASE WHEN known THEN (mark_value-CASE WHEN mark_value*.0003>5 THEN mark_value*.0003 ELSE 5 END
                    -mark_value*.00051)/buy_cash-1 END AS value FROM v ORDER BY date,code''').df()
            np.testing.assert_allclose(got[f'{name}_return{bps}'],ex.value,atol=2e-10,rtol=0,equal_nan=True)
            count+=len(ex)
            if name in ['sustained','any_close']:
                check=pd.Series(np.where(got[f'known{bps}'],ex.value.gt(0).astype(float),np.nan))
                target=f'opportunity{bps}' if name=='sustained' else f'any_opportunity{bps}'
                np.testing.assert_array_equal(got[target].to_numpy(),check.to_numpy())
                if name=='sustained':
                    expected_one=np.where(got[f'known{bps}'],ex.value.ge(.01).astype(float),np.nan)
                    np.testing.assert_array_equal(got[f'one_percent{bps}'].to_numpy(),expected_one)
                    count+=len(ex)
                count+=len(ex)
    assert len(got)==report['rows']
    assert got.board.eq('main').all() and got.necessary_tradeable.all()
    assert not got.code.str[3:].str.startswith(('92','688','300','301')).any()
    if scope=='training':
        assert got.next_date.lt('2024-07-01').all()
    result=dict(passed=True,label_report_sha256=sha(ROOT/f'{scope}_label_report.json'),rows=len(got),
        numeric_and_status_checks=count,all_sources_rejoined=True,raw_buy_status_rebuilt=True,
        old_tail_outcomes_not_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/f'{scope}_label_verification.json',result)
    return result


def selection_check():
    report=json.loads((ROOT/'selection_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL)
    assert report['training_label_report_sha256']==sha(ROOT/'training_label_report.json')
    assert report['selection_sha256']==sha(ROOT/'selection.parquet')
    assert report['core_sha256']==sha(ROOT/'frozen_numeric_core.tdx')
    assert not (ROOT/'full_labels.parquet').exists()
    f=pd.read_parquet(ROOT/'features.parquet')
    labels=pd.read_parquet(ROOT/'training_labels.parquet')
    train=f.merge(labels[['date','code','known15','opportunity15']],on=['date','code'],validate='one_to_one')
    train=train.loc[train.known15&train.formula_input_valid].sort_values(['date','code']).reset_index(drop=True)
    w=1/train.groupby('date').code.transform('size')
    tree=report['tree']
    names=list(EXPRESSIONS)
    x=train[names].to_numpy(dtype='float32')
    paths={0:[]}
    nodes={0:np.ones(len(train),dtype=bool)}
    leaves=[]
    for i in range(len(tree['feature'])):
        mask=nodes[i]
        assert int(mask.sum())==tree['n_node_samples'][i]
        left=tree['children_left'][i]
        if left<0:
            leaves.append(dict(node=i,rows=int(mask.sum()),weighted_probability=float(np.average(train.opportunity15[mask],weights=w[mask]))))
            continue
        feat=names[tree['feature'][i]]
        t=tree['threshold'][i]
        lower=x[:,tree['feature'][i]]<=t
        nodes[left]=mask&lower
        right=tree['children_right'][i]
        nodes[right]=mask&~lower
        paths[left]=paths[i]+[dict(feature=feat,op='<=',raw_threshold=t,threshold=math.floor(t*100)/100)]
        paths[right]=paths[i]+[dict(feature=feat,op='>',raw_threshold=t,threshold=math.ceil(t*100)/100)]
    chosen=sorted(leaves,key=lambda d:(-d['weighted_probability'],-d['rows'],d['node']))[0]
    assert chosen==report['chosen_leaf'] and leaves==report['leaves']
    conditions=paths[chosen['node']]
    assert conditions==report['conditions'] and len(conditions)<=3
    c=duckdb.connect()
    c.register('f',f)
    expression=' AND '.join(f"{d['feature']}{d['op']}{d['threshold']}" for d in conditions)
    expected=c.sql('SELECT date,code,half,board,decision_shares,formula_input_valid AND '+expression+' AS selected FROM f ORDER BY date,code').df()
    got=pd.read_parquet(ROOT/'selection.parquet')
    frame_equal(got,expected)
    assert int(got.selected.sum())==report['selected']
    result=dict(passed=True,selection_report_sha256=sha(ROOT/'selection_report.json'),rows=len(got),
        selected=int(got.selected.sum()),all_tree_nodes_and_leaf_choice_rebuilt=True,all_rounded_conditions_rebuilt=True,
        evaluation_labels_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_verification.json',result)
    return result


def analysis_check(ROOT=ROOT):
    report=json.loads((ROOT/'analysis_report.json').read_text())
    assert report['daily_summary_sha256']==sha(ROOT/'daily_summary.parquet')
    label_report=json.loads((ROOT/'full_label_report.json').read_text())
    assert label_report['labels_sha256']==sha(ROOT/'full_labels.parquet')
    c=duckdb.connect()
    c.execute("SET memory_limit='4GB'")
    c.read_parquet(str(ROOT/'full_labels.parquet')).create_view('labels')
    c.read_parquet(str(ROOT/'selection.parquet')).create_view('selection')
    c.execute('''CREATE VIEW all_rows AS SELECT l.*,s.selected FROM labels l JOIN selection s USING(date,code)
        WHERE l.date IN (SELECT DISTINCT date FROM selection WHERE selected)''')
    all_daily=[]
    for bps in [5,15]:
        for sensitive in [False,True]:
            known=f'sensitive_known{bps}' if sensitive else f'known{bps}'
            for arm,clause in [('formula','WHERE selected'),('base_same_dates','')]:
                d=c.sql(f'''WITH d AS (SELECT date,half,count(*) AS rows,count(*) FILTER(WHERE {known}) AS known,
                    count(*) FILTER(WHERE {known} AND opportunity{bps}=1) AS success,
                    count(*) FILTER(WHERE NOT {known} AND NOT known_no_trade) AS unknown,
                    count(*) FILTER(WHERE known_no_trade) AS no_trade,
                    count(*) FILTER(WHERE {known} AND one_percent{bps}=1) AS one_percent,
                    count(*) FILTER(WHERE {known} AND any_opportunity{bps}=1) AS any_success,
                    avg(mark_1000_return{bps}) FILTER(WHERE {known}) AS mean1000,
                    avg((mark_1000_return{bps}<0)::INT) FILTER(WHERE {known}) AS negative1000,
                    avg(adverse_return{bps}) FILTER(WHERE {known}) AS adverse_mean,
                    avg((adverse_return{bps}<=-.03)::INT) FILTER(WHERE {known}) AS bad3
                    FROM all_rows {clause} GROUP BY date,half)
                    SELECT *,success/nullif(known,0) AS rate,success/rows AS lower,(success+unknown)/rows AS upper,
                        one_percent/nullif(known,0) AS one_percent_rate,any_success/nullif(known,0) AS any_rate
                    FROM d ORDER BY date''').df()
                d['arm']=arm;d['bps']=bps;d['sensitive']=sensitive
                all_daily.append(d)
    expected=pd.concat(all_daily,ignore_index=True)
    got=pd.read_parquet(ROOT/'daily_summary.parquet')
    frame_equal(got[expected.columns],expected)
    checks=0
    for s in report['summaries']:
        p=s['period']
        f=expected.loc[expected.bps.eq(s['bps'])&expected.sensitive.eq(s['sensitive'])]
        if p=='2024H2_2025':
            f=f.loc[f.date.ge('2024-07-01')]
        elif len(p)==6 and p[4]=='Q' and p[5] in '1234':
            dates=pd.to_datetime(f.date)
            f=f.loc[dates.dt.year.eq(int(p[:4]))&dates.dt.quarter.eq(int(p[5]))]
        elif 'H' in p:
            f=f.loc[f.half.eq(p)]
        else:
            f=f.loc[f.date.str.startswith(p)]
        if s['arm']=='same_day_difference':
            a=f.loc[f.arm.eq('formula')].set_index('date')
            b=f.loc[f.arm.eq('base_same_dates')].set_index('date')
            assert a.index.equals(b.index)
            d=pd.DataFrame({'rate_delta':a.rate-b.rate,'lower_delta':a.lower-b.upper,
                'upper_delta':a.upper-b.lower,'mark1000_delta':a.mean1000-b.mean1000})
            eq(s['days'],len(d),'days')
            for name in d:
                value=d[name].mean()
                eq(s[name],float(value) if pd.notna(value) else None,name)
                checks+=1
                if name!='mark1000_delta':
                    eq(s[name+'_ci'],interval(d.reset_index(),name),name+'_ci')
                    checks+=1
        else:
            d=f.loc[f.arm.eq(s['arm'])].copy()
            for name in ['rows','known','success','unknown','no_trade']:
                eq(s[name],d[name].sum(),name);checks+=1
            eq(s['days'],len(d),'days')
            eq(s['pooled_rate'],d.success.sum()/d.known.sum() if d.known.sum() else None,'pooled_rate')
            eq(s['mean_selected_per_day'],float(d.rows.mean()) if len(d) else None,'mean_selected_per_day')
            for name in ['rate','lower','upper','one_percent_rate','any_rate','mean1000','negative1000','adverse_mean','bad3']:
                value=d[name].mean()
                eq(s[name],float(value) if pd.notna(value) else None,name);checks+=1
                if name in ['rate','lower','upper']:
                    eq(s[name+'_ci'],interval(d,name),name+'_ci');checks+=1
            checks+=3
    result=dict(passed=True,analysis_report_sha256=sha(ROOT/'analysis_report.json'),daily_rows=len(expected),
        all_daily_statistics_rebuilt=True,summary_checks=checks,new_2026_prices_read=bool(expected.date.ge('2026-01-01').any()),no_exit_rules=True)
    save_json(ROOT/'analysis_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['training','full','selection','analysis'])
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    stage=args.stage
    if args.root!=ROOT and stage!='analysis':
        parser.error('--root is only supported for unchanged analysis verification')
    result=label_check(stage) if stage in ['training','full'] else (analysis_check(args.root) if stage=='analysis' else selection_check())
    print(json.dumps(result,ensure_ascii=False,indent=2))
