"""Report a frozen selection's joint morning space/risk event and unknown bounds."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.reference_gain_accounting import weekly_interval
from verify_tick_flow_analysis import eq, interval

SOURCE=Path('data/research/tail_formula_1000')
LABEL_REPORT_SHA='1ae3602185c4162cd7fbfad0cec1061bac246e61c1d4f80c9dd2d88632b3ca3f'
PERIODS={'2025H1':('2025-01-01','2025-07-01'),
         '2025H2':('2025-07-01','2026-01-01'),
         '2025':('2025-01-01','2026-01-01')}


def checked_inputs(root):
    label_report=json.loads((SOURCE/'full_label_report.json').read_text())
    label_proof=json.loads((SOURCE/'full_label_verification.json').read_text())
    assert sha(SOURCE/'full_label_report.json')==LABEL_REPORT_SHA
    assert label_proof['passed'] and label_proof['label_report_sha256']==LABEL_REPORT_SHA
    assert label_report['labels_sha256']==sha(SOURCE/'full_labels.parquet')
    selection=json.loads((root/'selection_report.json').read_text())
    selection_proof=json.loads((root/'selection_verification.json').read_text())
    assert selection_proof['passed'] and selection_proof['selection_report_sha256']==sha(root/'selection_report.json')
    assert selection['selection_sha256']==sha(root/'selection.parquet')
    analysis=json.loads((root/'analysis_report.json').read_text())
    analysis_proof=json.loads((root/'analysis_verification.json').read_text())
    assert analysis_proof['passed'] and analysis_proof['analysis_report_sha256']==sha(root/'analysis_report.json')
    assert analysis['selection_report_sha256']==sha(root/'selection_report.json')
    assert analysis['daily_summary_sha256']==sha(root/'daily_summary.parquet')
    # All models of the new method must already be frozen before either method's
    # additional joint-outcome groups are read.
    combined=Path('data/research/tail_formula_joint_quality_2025')
    common=json.loads((combined/'selection_verification.json').read_text())
    assert common['passed'] and common['selection_report_sha256']==sha(combined/'selection_report.json')
    inputs=dict(selection_report_sha256=sha(root/'selection_report.json'),
        selection_sha256=sha(root/'selection.parquet'),analysis_report_sha256=sha(root/'analysis_report.json'),
        label_report_sha256=LABEL_REPORT_SHA,full_labels_sha256=sha(SOURCE/'full_labels.parquet'),
        protocol_sha256=sha(Path('config/tail_formula_joint_quality_combined_protocol.json')),
        frozen_combined_selection_sha256=sha(combined/'selection_report.json'))
    return inputs


def daily(frame,bps,sensitive):
    known=frame[f'sensitive_known{bps}' if sensitive else f'known{bps}']
    success=known&frame[f'sustained_return{bps}'].ge(.01)&frame[f'adverse_return{bps}'].gt(-.03)
    a=pd.DataFrame(dict(date=frame.date,known=known.astype(int),success=success.astype(int),
        unknown=(~known&~frame.known_no_trade).astype(int),no_trade=frame.known_no_trade.astype(int)))
    d=a.groupby('date').agg(rows=('date','size'),known=('known','sum'),success=('success','sum'),
        unknown=('unknown','sum'),no_trade=('no_trade','sum'))
    assert (d.known+d.unknown+d.no_trade).equals(d.rows)
    d['rate']=d.success/d.known.replace(0,np.nan)
    d['lower']=d.success/d.rows;d['upper']=(d.success+d.unknown)/d.rows
    return d.reset_index()


def connection(root):
    c=duckdb.connect();c.execute('SET threads=4')
    c.read_parquet(str(SOURCE/'full_labels.parquet')).create_view('labels')
    c.read_parquet(str(root/'selection.parquet')).create_view('selection')
    c.execute('''CREATE VIEW joined AS SELECT l.*,s.selected FROM labels l
        JOIN selection s USING(date,code) WHERE l.date IN(SELECT date FROM selection WHERE selected)''')
    return c


def sql_daily(c,arm,bps,sensitive):
    known=f'sensitive_known{bps}' if sensitive else f'known{bps}'
    clause='WHERE selected' if arm=='formula' else ''
    return c.sql(f'''WITH states AS(SELECT date,{known} AS k,known_no_trade AS n,
        {known} AND sustained_return{bps}>=.01 AND adverse_return{bps}>-.03 AS joint
        FROM joined {clause}), counted AS(SELECT date,count(*) AS rows,
        count(*) FILTER(WHERE k) AS known,count(*) FILTER(WHERE joint) AS success,
        count(*) FILTER(WHERE NOT k AND NOT n) AS unknown,count(*) FILTER(WHERE n) AS no_trade
        FROM states GROUP BY date)
        SELECT *,success/nullif(known,0)::DOUBLE AS rate,success/rows::DOUBLE AS lower,
        (success+unknown)/rows::DOUBLE AS upper FROM counted ORDER BY date''').df()


def summarize(group,arm,bps,sensitive,period):
    q=group.set_index('date')
    r=dict(arm=arm,bps=bps,sensitive=sensitive,period=period,days=len(q))
    for n in ['rows','known','success','unknown','no_trade']:
        r[n]=int(q[n].sum())
    r['pooled_rate']=r['success']/r['known'] if r['known'] else None
    for name in ['rate','lower','upper']:
        r[name]=float(q[name].mean()) if q[name].notna().any() else None
        r[name+'_ci']=weekly_interval(q[name])
    return r


def analyze(root):
    if (root/'joint_quality_report.json').exists():
        raise ValueError('Do not replace joint-quality results')
    inputs=checked_inputs(root)
    selection=pd.read_parquet(root/'selection.parquet',columns=['date','code','selected'])
    labels=pd.read_parquet(SOURCE/'full_labels.parquet')
    d=labels.merge(selection,on=['date','code'],validate='one_to_one')
    dates=d.loc[d.selected,'date'].unique()
    assert all('2025-01-01'<=x<'2026-01-01' for x in dates)
    frames=[];summaries=[]
    for bps in [5,15]:
        for sensitive in [False,True]:
            pair={}
            for arm,mask in [('formula',d.selected),('base_same_dates',d.date.isin(dates))]:
                x=daily(d.loc[mask],bps,sensitive)
                pair[arm]=x.set_index('date')
                frames.append(x.assign(arm=arm,bps=bps,sensitive=sensitive))
                for period,(start,end) in PERIODS.items():
                    summaries.append(summarize(x.loc[x.date.ge(start)&x.date.lt(end)],arm,bps,sensitive,period))
            a,b=pair['formula'],pair['base_same_dates'];assert a.index.equals(b.index)
            delta=pd.DataFrame({'date':a.index,'rate_delta':(a.rate-b.rate).values,
                'lower_delta':(a.lower-b.upper).values,'upper_delta':(a.upper-b.lower).values})
            for period,(start,end) in PERIODS.items():
                q=delta.loc[delta.date.ge(start)&delta.date.lt(end)].set_index('date')
                r=dict(arm='same_day_difference',bps=bps,sensitive=sensitive,period=period,days=len(q))
                for n in ['rate_delta','lower_delta','upper_delta']:
                    r[n]=float(q[n].mean()) if q[n].notna().any() else None
                    r[n+'_ci']=weekly_interval(q[n])
                summaries.append(r)
    out=pd.concat(frames,ignore_index=True)
    out.to_parquet(root/'joint_quality_days.parquet',index=False,compression='zstd')
    r=dict(inputs=inputs,summaries=summaries,daily_sha256=sha(root/'joint_quality_days.parquet'),
        space_threshold=.01,adverse_threshold=-.03,year_2025_is_exploratory=True,
        joint_event_is_not_realized_profit=True,no_exit_rules=True,new_2026_prices_read=False)
    save_json(root/'joint_quality_report.json',r)
    return {k:v for k,v in r.items() if k!='summaries'}


def verify(root):
    r=json.loads((root/'joint_quality_report.json').read_text())
    assert r['inputs']==checked_inputs(root) and r['daily_sha256']==sha(root/'joint_quality_days.parquet')
    assert r['space_threshold']==.01 and r['adverse_threshold']==-.03
    c=connection(root);parts=[]
    for bps in [5,15]:
        for sensitive in [False,True]:
            for arm in ['formula','base_same_dates']:
                parts.append(sql_daily(c,arm,bps,sensitive).assign(arm=arm,bps=bps,sensitive=sensitive))
    expected=pd.concat(parts,ignore_index=True)
    actual=pd.read_parquet(root/'joint_quality_days.parquet')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=0,atol=2e-12)
    keys=['bps','sensitive','arm','date'];denoms=['rows','known','unknown','no_trade']
    standard=pd.read_parquet(root/'daily_summary.parquet').sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True)[keys+denoms],
        standard[keys+denoms],check_exact=True,check_dtype=False)
    c.register('expected',expected);checks=0
    for s in r['summaries']:
        start,end=PERIODS[s['period']]
        query=f"bps={s['bps']} AND sensitive={str(s['sensitive']).upper()} AND date>='{start}' AND date<'{end}'"
        if s['arm']=='same_day_difference':
            q=c.sql(f'''WITH a AS(SELECT * FROM expected WHERE {query} AND arm='formula'),
                b AS(SELECT * FROM expected WHERE {query} AND arm='base_same_dates')
                SELECT a.date,a.rate-b.rate AS rate_delta,a.lower-b.upper AS lower_delta,
                a.upper-b.lower AS upper_delta FROM a JOIN b USING(date) ORDER BY date''').df()
            names=['rate_delta','lower_delta','upper_delta']
        else:
            q=c.sql(f"SELECT * FROM expected WHERE {query} AND arm='{s['arm']}' ORDER BY date").df()
            c.register('period_group',q)
            totals=c.sql('SELECT '+','.join(f'coalesce(sum({n}),0)' for n in ['rows','known','success','unknown','no_trade'])+' FROM period_group').fetchone()
            for n,v in zip(['rows','known','success','unknown','no_trade'],totals):
                eq(s[n],int(v),n);checks+=1
            eq(s['pooled_rate'],float(totals[2]/totals[1]) if totals[1] else None,'pooled_rate');checks+=1
            names=['rate','lower','upper']
        eq(s['days'],len(q),'days');checks+=1
        for n in names:
            values=q[n].dropna().to_numpy()
            eq(s[n],float(np.sum(values)/len(values)) if len(values) else None,n)
            eq(s[n+'_ci'],interval(q,n),n+'_ci');checks+=2
    c.close()
    proof=dict(passed=True,joint_quality_report_sha256=sha(root/'joint_quality_report.json'),
        daily_rows=len(actual),summary_checks=checks,all_joint_events_and_unknown_denominators_rebuilt=True,
        all_week_block_intervals_independently_rebuilt=True,standard_analysis_denominators_unchanged=True,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'joint_quality_verification.json',proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['analyze','verify']);p.add_argument('--root',required=True,type=Path)
    a=p.parse_args();print(json.dumps(globals()[a.stage](a.root),ensure_ascii=False,indent=2))
