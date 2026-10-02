"""Describe quote opportunity order without changing stock lists or exits."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.research_io import check_runtime, check_sources, save_json, sha
from trade_research.tail_formula_additive import conn
from trade_research.tail_formula_boundary_evaluation import period, number
from tail_formula_statistics import weekly_interval

ROOT = Path('data/research/tail_formula_opportunity_order')
PROTOCOL = Path('config/tail_formula_opportunity_order.json')
KEYS = ['date', 'code', 'next_date']
NAMES = ['first_positive_end', 'first_space_end', 'first_bad3',
         'positive_before_bad3', 'space_before_bad3', 'positive_then_bad3',
         'bad3_before_positive', 'adverse_until_positive', 'adverse_until_space', 'risk_observed']
PARENT_COLUMNS = ['date','code','next_date','half','board','decision_shares','known5','known15','known_no_trade',
                  'sensitive_known5','sensitive_known15','opportunity5','opportunity15','one_percent5','one_percent15',
                  'adverse_return5','adverse_return15','sustained_return5','sustained_return15','buy_cash5','buy_cash15']


def checked():
    check_runtime()
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    check_sources(p['source_hashes'])
    index = json.loads(Path(p['input_index']).read_text())
    check_sources(index['source_hashes'])
    assert p['window_bars'] == 29 and p['space_threshold'] == .01 and p['adverse_threshold'] == -.03
    return p, index


def mark(price, shares, cash, bps):
    with np.errstate(invalid='ignore', divide='ignore'):
        value = shares * (price - np.maximum(.005, price * bps / 10000))
        return (value - np.maximum(5., value * .0003) - value * .00051) / cash - 1


def first(flags):
    return np.where(flags.any(axis=1), flags.argmax(axis=1), -1)


def produce():
    p, index = checked()
    assert not (ROOT/'label_report.json').exists()
    labels = pd.read_parquet(p['parent_labels'], columns=PARENT_COLUMNS)
    assert len(labels) == 1258085 and labels.date.ge('2024-01-01').all() and labels.date.lt('2026-01-01').all()
    eligible = labels.loc[labels.known5 | labels.known15].copy()
    assert eligible.next_date.lt('2026-01-01').all()
    derived, checks = [], []
    for path in index['observation_parts']:
        obs = pd.read_parquet(path, columns=KEYS+['source_valid_0959','valid_mask','active_mask','close_values','low_values'],
                              filters=[('date','>=','2024-01-01'),('next_date','<','2026-01-01')])
        q = eligible.merge(obs, on=KEYS, how='inner', validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
        if not len(q):
            continue
        assert q.source_valid_0959.all() and np.all((q.valid_mask.to_numpy(dtype='int64') & ((1<<29)-1)) == (1<<29)-1)
        closes = np.stack(q.close_values)[:, :29]; lows = np.stack(q.low_values)[:, :29]
        assert closes.shape == lows.shape == (len(q),29)
        active = (q.active_mask.to_numpy(dtype='int64')[:,None] & (1 << np.arange(29))) != 0
        triple_active = active[:,:-2] & active[:,1:-1] & active[:,2:]
        triple_close = np.minimum(np.minimum(closes[:,:-2],closes[:,1:-1]),closes[:,2:])
        best = np.where(triple_active,triple_close,-np.inf).max(axis=1); best[~np.isfinite(best)] = np.nan
        adverse_price = np.where(active,lows,np.inf).min(axis=1); adverse_price[~np.isfinite(adverse_price)] = np.nan
        out = q[KEYS].copy(); sql = conn(); sql.register('quotes',q)
        for bps in [5,15]:
            known = q[f'known{bps}'].to_numpy(); shares=q.decision_shares.to_numpy()[:,None]; cash=q[f'buy_cash{bps}'].to_numpy()[:,None]
            cm=mark(closes,shares,cash,bps); lm=mark(lows,shares,cash,bps)
            positive=active & (cm>0); space=active & (cm>=.01)
            a=first(positive[:,:-2] & positive[:,1:-1] & positive[:,2:]); a=np.where(a>=0,a+2,-1)
            b=first(space[:,:-2] & space[:,1:-1] & space[:,2:]); b=np.where(b>=0,b+2,-1)
            z=first(active & (lm<=-.03))
            before=lambda x:(x>=0) & ((z<0) | (x<z))
            def until(x):
                value=np.where(active & (np.arange(29)[None,:]<=x[:,None]),lm,np.inf).min(axis=1)
                value[~np.isfinite(value)]=np.nan;return value
            values=[a,b,z,before(a),before(b),(a>=0)&(z>a),(z>=0)&(a>=z),until(a),until(b),active.any(axis=1)]
            for name,value in zip(NAMES,values):out[name+str(bps)]=np.where(known,np.asarray(value,dtype=float),np.nan)
            sustained=mark(best,q.decision_shares.to_numpy(),q[f'buy_cash{bps}'].to_numpy(),bps)
            adverse=mark(adverse_price,q.decision_shares.to_numpy(),q[f'buy_cash{bps}'].to_numpy(),bps)
            np.testing.assert_allclose(sustained[known],q.loc[known,f'sustained_return{bps}'],rtol=0,atol=2e-10,equal_nan=True)
            np.testing.assert_allclose(adverse[known],q.loc[known,f'adverse_return{bps}'],rtol=0,atol=2e-10,equal_nan=True)
            np.testing.assert_array_equal(a[known]>=0,q.loc[known,f'opportunity{bps}'].eq(1))
            np.testing.assert_array_equal(b[known]>=0,q.loc[known,f'one_percent{bps}'].eq(1))
            fields=f'''WITH bars AS(SELECT date,code,pos,known{bps} AS known,
                (active_mask & (1::BIGINT<<pos))<>0 AS active,decision_shares,buy_cash{bps} AS cash,
                list_extract(close_values,pos+1) AS close,list_extract(low_values,pos+1) AS low
                FROM quotes CROSS JOIN range(29) t(pos)),
                cash_marks AS(SELECT *,decision_shares*(close-greatest(.005,close*{bps}/10000.)) AS cv,
                    decision_shares*(low-greatest(.005,low*{bps}/10000.)) AS lv FROM bars),
                marks AS(SELECT *, (cv-greatest(5.,cv*.0003)-cv*.00051)/cash-1 AS cm,
                    (lv-greatest(5.,lv*.0003)-lv*.00051)/cash-1 AS lm FROM cash_marks),
                windows AS(SELECT *,count(*) OVER w AS n,
                    count(*) FILTER(WHERE active AND cm>0) OVER w AS positive,
                    count(*) FILTER(WHERE active AND cm>=.01) OVER w AS space FROM marks
                    WINDOW w AS(PARTITION BY date,code ORDER BY pos ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
                events AS(SELECT date,code,bool_or(known) AS known,
                    coalesce(min(pos) FILTER(WHERE n=3 AND positive=3),-1) AS a,
                    coalesce(min(pos) FILTER(WHERE n=3 AND space=3),-1) AS b,
                    coalesce(min(pos) FILTER(WHERE active AND lm<=-.03),-1) AS z,
                    bool_or(active) AS risk FROM windows GROUP BY date,code),
                prefixes AS(SELECT e.date,e.code,min(m.lm) FILTER(WHERE m.active AND m.pos<=e.a) AS ap,
                    min(m.lm) FILTER(WHERE m.active AND m.pos<=e.b) AS bp
                    FROM events e JOIN marks m USING(date,code) GROUP BY e.date,e.code)
                SELECT e.date,e.code,CASE WHEN known THEN a END AS first_positive_end{bps},
                    CASE WHEN known THEN b END AS first_space_end{bps},CASE WHEN known THEN z END AS first_bad3{bps},
                    CASE WHEN known THEN (a>=0 AND (z<0 OR a<z))::INT END AS positive_before_bad3{bps},
                    CASE WHEN known THEN (b>=0 AND (z<0 OR b<z))::INT END AS space_before_bad3{bps},
                    CASE WHEN known THEN (a>=0 AND z>a)::INT END AS positive_then_bad3{bps},
                    CASE WHEN known THEN (z>=0 AND a>=z)::INT END AS bad3_before_positive{bps},
                    CASE WHEN known THEN ap END AS adverse_until_positive{bps},
                    CASE WHEN known THEN bp END AS adverse_until_space{bps},
                    CASE WHEN known THEN risk::INT END AS risk_observed{bps}
                    FROM events e JOIN prefixes USING(date,code) ORDER BY e.date,e.code'''
            ex=sql.sql(fields).df();pd.testing.assert_frame_equal(out[['date','code']],ex[['date','code']],check_exact=True)
            columns=[name+str(bps) for name in NAMES]
            np.testing.assert_allclose(out[columns].to_numpy(dtype=float),
                ex[columns].to_numpy(dtype=float,na_value=np.nan),rtol=0,atol=2e-12,equal_nan=True)
        sql.close();derived.append(out);checks.append(dict(path=path,eligible_rows=len(q),all_events_and_prefix_marks_SQL_rebuilt=True))
        print(json.dumps(dict(parts_verified=len(checks),eligible_rows_verified=sum(x['eligible_rows'] for x in checks))),flush=True)
    d=pd.concat(derived,ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(d[KEYS],eligible[KEYS].sort_values(KEYS).reset_index(drop=True),check_exact=True)
    columns=['date','code','next_date','half','board','decision_shares','known5','known15','known_no_trade',
             'sensitive_known5','sensitive_known15','opportunity5','opportunity15','one_percent5','one_percent15','adverse_return5','adverse_return15']
    full=labels[columns].merge(d,on=KEYS,how='left',validate='one_to_one')
    for bps in [5,15]:
        assert full.loc[~full[f'known{bps}'],[name+str(bps) for name in NAMES]].isna().all().all()
        assert full[f'positive_before_bad3{bps}'].le(full[f'opportunity{bps}']).loc[full[f'known{bps}']].all()
        assert full[f'space_before_bad3{bps}'].le(full[f'one_percent{bps}']).loc[full[f'known{bps}']].all()
    full.to_parquet(ROOT/'ordered_labels.parquet',index=False,compression='zstd')
    save_json(ROOT/'label_report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),labels_sha256=sha(ROOT/'ordered_labels.parquet'),
        parent_labels_sha256=sha(Path(p['parent_labels'])),rows=len(full),parts=checks,original_statuses_and_complete_keys_preserved=True,
        independent_SQL_all_new_events_prefix_marks_and_original_aggregates=True,new_2026_prices_read=False,no_exit_rules=True))
    return dict(label_report_sha256=sha(ROOT/'label_report.json'),rows=len(full),new_economic_groups_read=False)


def analyze():
    p,_=checked();r=json.loads((ROOT/'label_report.json').read_text());assert r['passed']
    assert r['protocol_sha256']==sha(PROTOCOL) and r['labels_sha256']==sha(ROOT/'ordered_labels.parquet')
    assert sha(ROOT/'label_report.json') in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert not (ROOT/'analysis_report.json').exists()
    labels=pd.read_parquet(ROOT/'ordered_labels.parquet');parts=[]
    counts=['rows','known','unknown','no_trade','positive','space','positive_before_bad3','space_before_bad3',
            'positive_then_bad3','bad3_before_positive','risky_positive']
    metrics=['positive_rate','space_rate','positive_before_bad3_rate','space_before_bad3_rate',
             'bad_first_among_positive','late_bad_among_risky_positive','opportunity_lower','opportunity_upper']
    for year,root in p['selections'].items():
        selection=pd.read_parquet(Path(root)/'selection.parquet');q=labels.merge(selection[['date','code','selected']],on=['date','code'],validate='one_to_one')
        chosen=q.loc[q.selected];assert chosen.date.str.startswith(year).all()
        for arm,frame in [('formula',chosen),('base_same_dates',q.loc[q.date.isin(chosen.date.unique())]),('base_all_dates',q.loc[q.date.str.startswith(year)])]:
            for bps in [5,15]:
                for sensitive in [False,True]:
                    f=frame.copy();f['known']=f[f'sensitive_known{bps}' if sensitive else f'known{bps}']
                    f['unknown']=~f.known & ~f.known_no_trade;f['no_trade']=f.known_no_trade
                    f['positive']=f.known & f[f'opportunity{bps}'].eq(1);f['space']=f.known & f[f'one_percent{bps}'].eq(1)
                    for name in ['positive_before_bad3','space_before_bad3','positive_then_bad3','bad3_before_positive']:f[name]=f.known & f[name+str(bps)].eq(1)
                    f['risky_positive']=f.positive & f[f'first_bad3{bps}'].ge(0)
                    d=f.groupby(['date','half']).agg(rows=('code','size'),**{n:(n,'sum') for n in counts[1:]}).reset_index()
                    for name in ['positive','space','positive_before_bad3','space_before_bad3']:d[name+'_rate']=d[name]/d.known.replace(0,np.nan)
                    d['bad_first_among_positive']=d.bad3_before_positive/d.positive.replace(0,np.nan)
                    d['late_bad_among_risky_positive']=d.positive_then_bad3/d.risky_positive.replace(0,np.nan)
                    d['opportunity_lower']=d.positive_before_bad3/d.rows;d['opportunity_upper']=(d.positive_before_bad3+d.unknown)/d.rows
                    c=conn();c.register('label_rows',f)
                    terms=','.join(f'count(*) FILTER(WHERE "{n}") AS "{n}"' for n in counts[1:])
                    c.sql('SELECT date,half,count(*) AS rows,'+terms+' FROM label_rows GROUP BY date,half').create_view('daily_counts')
                    ex=c.sql('''SELECT *,positive::DOUBLE/nullif(known,0) AS positive_rate,
                        space::DOUBLE/nullif(known,0) AS space_rate,
                        positive_before_bad3::DOUBLE/nullif(known,0) AS positive_before_bad3_rate,
                        space_before_bad3::DOUBLE/nullif(known,0) AS space_before_bad3_rate,
                        bad3_before_positive::DOUBLE/nullif(positive,0) AS bad_first_among_positive,
                        positive_then_bad3::DOUBLE/nullif(risky_positive,0) AS late_bad_among_risky_positive,
                        positive_before_bad3::DOUBLE/rows AS opportunity_lower,
                        (positive_before_bad3+unknown)::DOUBLE/rows AS opportunity_upper
                        FROM daily_counts ORDER BY date,half''').df();c.close()
                    pd.testing.assert_frame_equal(d[['date','half',*counts]],ex[['date','half',*counts]],check_exact=True,check_dtype=False)
                    np.testing.assert_allclose(d[metrics].to_numpy(dtype=float),
                        ex[metrics].to_numpy(dtype=float,na_value=np.nan),rtol=0,atol=2e-12,equal_nan=True)
                    d['arm']=arm;d['year']=year;d['bps']=bps;d['sensitive']=sensitive;parts.append(d)
    daily=pd.concat(parts,ignore_index=True);summaries=[]
    for (year,arm,bps,sensitive),d in daily.groupby(['year','arm','bps','sensitive']):
        for name in [year+'H1',year+'H2',year]:
            q=period(d,name).set_index('date');s=dict(year=year,arm=arm,bps=int(bps),sensitive=bool(sensitive),period=name,days=len(q))
            s.update({n:int(q[n].sum()) for n in counts});s.update({n:number(q[n].mean()) for n in metrics})
            s['positive_before_bad3_ci']=weekly_interval(q.positive_before_bad3_rate)
            s['opportunity_lower_ci']=weekly_interval(q.opportunity_lower);s['opportunity_upper_ci']=weekly_interval(q.opportunity_upper);summaries.append(s)
    daily.to_parquet(ROOT/'daily_summary.parquet',index=False,compression='zstd')
    save_json(ROOT/'analysis_report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),label_report_sha256=sha(ROOT/'label_report.json'),
        daily_summary_sha256=sha(ROOT/'daily_summary.parquet'),summaries=summaries,all_daily_numerators_denominators_SQL_rebuilt=True,
        original_lists_unchanged=True,not_realized_profit=True,no_formula_quality_or_exit_claim=True,new_2026_prices_read=False))
    return dict(analysis_report_sha256=sha(ROOT/'analysis_report.json'),summaries=summaries)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['produce','analyze'])
    print(json.dumps(globals()[a.parse_args().stage](),ensure_ascii=False),flush=True)
