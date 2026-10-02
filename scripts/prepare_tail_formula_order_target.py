"""Rebuild historical training events; do not fit or evaluate stock selectors."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.research_io import check_runtime, check_sources, save_json, sha
from trade_research.tail_formula_additive import conn
from trade_research.tail_formula_baseline import original

ROOT = Path('data/research/tail_formula_order_target')
PROTOCOL = Path('config/tail_formula_order_target_input.json')
KEYS = ['date', 'code', 'next_date']
PARENT = KEYS + ['half', 'board', 'decision_shares', 'known15', 'known_no_trade',
                 'opportunity15', 'one_percent15', 'adverse_return15', 'sustained_return15', 'buy_cash15']
EVENTS = ['first_positive_end15', 'first_space_end15', 'first_bad315',
          'ordered_utility', 'whole_window_utility']


def first(a):
    return np.where(a.any(axis=1), a.argmax(axis=1), -1)


def main():
    check_runtime()
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    check_sources(p['source_hashes'])
    assert p['fits_allowed_at_this_stage'] == 0 and not (ROOT/'source_gate.json').exists()
    labels = pd.read_parquet(p['historical_labels'], columns=PARENT)
    assert len(labels) == 557044 and labels.date.ge('2023-01-01').all() and labels.date.lt('2024-01-01').all()
    eligible = labels.loc[labels.known15].copy()
    parts, checks = [], []
    columns = ['date','code','source_date','clock','timestamp','open','high','low','close','volume','amount']
    for path in p['raw_parts']:
        raw = pd.read_parquet(path, columns=columns, filters=[('date','>=','2023-01-01'),
            ('date','<','2024-01-01'),('kind','=','morning'),('clock','>=','09:31'),('clock','<=','09:59')])
        assert not raw.duplicated(['date','code','timestamp']).any()
        assert raw.clock.eq(raw.timestamp.dt.strftime('%H:%M')).all()
        assert raw.timestamp.eq(raw.timestamp.dt.floor('min')).all()
        assert raw.source_date.eq(raw.timestamp.dt.strftime('%Y-%m-%d')).all()
        assert raw.source_date.gt(raw.date).all() and raw.source_date.le('2024-01-02').all()
        q = eligible.merge(raw,on=['date','code'],validate='one_to_many').sort_values(['date','code','clock']).reset_index(drop=True)
        if not len(q):
            continue
        assert q.source_date.eq(q.next_date).all()
        assert q.groupby(KEYS).size().eq(29).all() and q.groupby(KEYS).clock.nunique().eq(29).all()
        expected = [f'09:{i:02d}' for i in range(31,60)]
        assert (q.clock.to_numpy().reshape(-1,29) == np.array(expected)[None,:]).all()
        prices=q[['open','high','low','close','volume','amount']].to_numpy(dtype=float)
        o,h,l,c,v,a=prices.T
        assert np.isfinite(prices).all() and (prices[:,:4]>0).all()
        assert (h+.0001>=np.maximum.reduce([o,l,c])).all() and (l-.0001<=np.minimum(o,c)).all()
        assert (v>=0).all() and (a>=0).all() and ((v==0)==(a==0)).all()
        vw=a/np.where(v>0,v,np.nan)
        assert ((v==0)|((vw>=l-.0101)&(vw<=h+.0101))).all()
        assert (np.abs(prices[:,:4]-np.rint(prices[:,:4]*100)/100)<=.0001).all()
        keys=q.drop_duplicates(KEYS).reset_index(drop=True)
        active=(v>0).reshape(-1,29);closes=c.reshape(-1,29);lows=l.reshape(-1,29)
        tax=np.where(keys.next_date.lt('2023-08-28'),.001,.0005)+.00001
        def mark(price):
            value=keys.decision_shares.to_numpy()[:,None]*(price-np.maximum(.005,price*.0015))
            return (value-np.maximum(5.,value*.0003)-value*tax[:,None])/keys.buy_cash15.to_numpy()[:,None]-1
        cm,lm=mark(closes),mark(lows)
        positive=active&(cm>0);space=active&(cm>=.01)
        triple=lambda z:z[:,:-2]&z[:,1:-1]&z[:,2:]
        pe=first(triple(positive));pe=np.where(pe>=0,pe+2,-1)
        se=first(triple(space));se=np.where(se>=0,se+2,-1)
        be=first(active&(lm<=-.03))
        ordered=(se>=0)&((be<0)|(se<be));whole=(se>=0)&(be<0)
        sustained=np.where(triple(active),np.minimum.reduce([cm[:,:-2],cm[:,1:-1],cm[:,2:]]),-np.inf).max(axis=1)
        adverse=np.where(active,lm,np.inf).min(axis=1)
        sustained[~np.isfinite(sustained)]=np.nan;adverse[~np.isfinite(adverse)]=np.nan
        np.testing.assert_allclose(sustained,keys.sustained_return15,rtol=0,atol=2e-12,equal_nan=True)
        np.testing.assert_allclose(adverse,keys.adverse_return15,rtol=0,atol=2e-12,equal_nan=True)
        np.testing.assert_array_equal(pe>=0,keys.opportunity15.eq(1))
        np.testing.assert_array_equal(se>=0,keys.one_percent15.eq(1))
        out=keys[KEYS].copy()
        for name,value in zip(EVENTS,[pe,se,be,ordered,whole]):out[name]=np.asarray(value,dtype=float)
        sql=conn();sql.register('quotes',q)
        ex=sql.sql('''WITH cash AS(SELECT *,date_diff('minute',date_trunc('day',timestamp)+INTERVAL 9 HOUR+INTERVAL 31 MINUTE,timestamp) AS pos,
            decision_shares*(close-greatest(.005,close*.0015)) AS cv,
            decision_shares*(low-greatest(.005,low*.0015)) AS lv,
            .00001+CASE WHEN next_date<'2023-08-28' THEN .001 ELSE .0005 END AS tax FROM quotes),
            marks AS(SELECT *,(cv-greatest(5.,cv*.0003)-cv*tax)/buy_cash15-1 AS cm,
                (lv-greatest(5.,lv*.0003)-lv*tax)/buy_cash15-1 AS lm FROM cash),
            windows AS(SELECT *,count(*) OVER w AS n,
                count(*) FILTER(WHERE volume>0 AND cm>0) OVER w AS positive,
                count(*) FILTER(WHERE volume>0 AND cm>=.01) OVER w AS space FROM marks
                WINDOW w AS(PARTITION BY date,code ORDER BY pos ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
            events AS(SELECT date,code,next_date,
                coalesce(min(pos) FILTER(WHERE n=3 AND positive=3),-1) AS pe,
                coalesce(min(pos) FILTER(WHERE n=3 AND space=3),-1) AS se,
                coalesce(min(pos) FILTER(WHERE volume>0 AND lm<=-.03),-1) AS be
                FROM windows GROUP BY date,code,next_date)
            SELECT date,code,next_date,pe AS first_positive_end15,se AS first_space_end15,be AS first_bad315,
                (se>=0 AND (be<0 OR se<be))::INT AS ordered_utility,
                (se>=0 AND be<0)::INT AS whole_window_utility FROM events ORDER BY date,code''').df();sql.close()
        pd.testing.assert_frame_equal(out[KEYS],ex[KEYS],check_exact=True)
        np.testing.assert_array_equal(out[EVENTS].to_numpy(dtype=float),ex[EVENTS].to_numpy(dtype=float,na_value=np.nan))
        parts.append(out);checks.append(dict(path=path,known_rows=len(out),dated_costs_and_events_SQL_rebuilt=True))
        print(json.dumps(dict(parts_verified=len(checks),known_training_rows=sum(x['known_rows'] for x in checks))),flush=True)
    d=pd.concat(parts,ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(d[KEYS],eligible[KEYS].sort_values(KEYS).reset_index(drop=True),check_exact=True)
    history=labels.drop(columns=['sustained_return15','buy_cash15']).merge(d,on=KEYS,how='left',validate='one_to_one')
    assert history.loc[~history.known15,EVENTS].isna().all().all()
    current=pd.read_parquet(p['current_ordered_labels'],columns=[n for n in history.columns if n not in EVENTS]+
        ['first_positive_end15','first_space_end15','first_bad315','space_before_bad315'])
    current['ordered_utility']=current.pop('space_before_bad315')
    current['whole_window_utility']=(current.one_percent15.eq(1)&current.adverse_return15.gt(-.03)).astype(float).where(current.known15)
    targets=pd.concat([history,current[history.columns]],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    f=original();pd.testing.assert_frame_equal(targets[['date','code']],f[['date','code']],check_exact=True)
    pd.testing.assert_frame_equal(targets[['half','board','decision_shares']],f[['half','board','decision_shares']],check_exact=True)
    assert len(f)==1815129 and int(f.formula_input_valid.sum())==1602413
    for utility in ['ordered_utility','whole_window_utility']:
        assert targets.loc[targets.known15,utility].isin([0,1]).all() and targets.loc[~targets.known15,utility].isna().all()
    assert targets.loc[targets.known15,'ordered_utility'].ge(targets.loc[targets.known15,'whole_window_utility']).all()
    targets.to_parquet(ROOT/'targets.parquet',index=False,compression='zstd')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    sql=conn();names=','.join('"'+n+'"' for n in f.columns)
    sql.sql(f'''SELECT {names} FROM read_parquet('data/research/tail_formula_stock_2024/inputs/features.parquet')
        WHERE date<'2024-01-01' UNION ALL SELECT {names}
        FROM read_parquet('data/research/tail_formula_morning_range/inputs/features.parquet')''').create_view('original_features')
    sql.read_parquet(str(ROOT/'features.parquet')).create_view('assembled_features')
    for left,right in [('original_features','assembled_features'),('assembled_features','original_features')]:
        assert sql.sql(f'SELECT count(*) FROM ((SELECT * FROM {left}) EXCEPT ALL (SELECT * FROM {right}))').fetchone()[0]==0
    sql.close()
    save_json(ROOT/'feature_report.json',dict(features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        source_hashes={name:digest for name,digest in p['source_hashes'].items() if name.endswith('/features.parquet')},new_input_features=0))
    save_json(ROOT/'feature_verification.json',dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),
        original_fifty_values_and_all_metadata_SQL_EXCEPT_ALL_equal=True))
    save_json(ROOT/'source_gate.json',dict(passed=True,input_protocol_sha256=sha(PROTOCOL),
        targets_sha256=sha(ROOT/'targets.parquet'),rows=len(targets),historical_parts=checks,
        feature_report_sha256=sha(ROOT/'feature_report.json'),feature_verification_sha256=sha(ROOT/'feature_verification.json'),
        historical_original_aggregates_and_binary_labels_exact=True,dated_fees_and_ordered_events_SQL_verified=True,
        all_original_feature_keys_and_validity_retained=True,original_unknown_and_no_trade_states_retained=True,
        new_selector_fits=0,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(source_gate_sha256=sha(ROOT/'source_gate.json'),rows=len(targets),new_selector_fits=0)),flush=True)


if __name__=='__main__':
    main()
