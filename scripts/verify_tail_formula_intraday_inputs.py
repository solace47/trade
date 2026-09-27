"""Rebuild afternoon inputs in SQL and fixed raw-file windows with pandas."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trade_research.corporate_cash import save_json, sha

ROOT=Path('data/research/tail_formula_intraday')
SOURCE=Path('data/research/tail_formula_1000')
MINUTES=Path('data/hf/pilot/data/stock_1m')


def rebuild(raw):
    r=raw.sort_values('timestamp').copy()
    prices=r[['open','high','low','close']]
    finite=np.isfinite(r[['open','high','low','close','volume','turnover']]).all(axis=1)
    good=(finite & prices.gt(0).all(axis=1) & (prices-prices.round(2)).abs().le(.0001).all(axis=1)
        & r.timestamp.eq(r.timestamp.dt.floor('min'))
        & r.high.add(.0001).ge(prices[['open','low','close']].max(axis=1))
        & r.low.sub(.0001).le(prices[['open','close']].min(axis=1))
        & r.volume.ge(0) & r.turnover.ge(0) & r.volume.eq(0).eq(r.turnover.eq(0))
        & (r.volume.eq(0) | (r.turnover/r.volume).between(r.low-.0101,r.high+.0101)))
    r[['open','high','low','close']]=prices.round(2)
    r['clock']=r.timestamp.dt.strftime('%H%M')
    prior=r.close.shift()
    r['signed']=np.sign(r.close-prior).fillna(0)*r.volume
    r['up']=r.close.gt(prior)
    r['path']=np.abs(np.log(r.close/prior))
    last29=r.loc[r.clock.ge('1421')]
    last4=r.loc[r.clock.ge('1446')]
    at=lambda clock:r.loc[r.clock.eq(clock),'close'].max()
    return dict(bars=len(r),clocks=r.clock.nunique(),good_bars=int(good.sum()),
        window_valid=len(r)==109 and r.clock.nunique()==109 and bool(good.all()),
        p49=at('1449'),p35=at('1435'),p20=at('1420'),p01=at('1301'),
        v29=last29.volume.sum(min_count=1),a29=last29.turnover.sum(min_count=1),
        vprev29=r.loc[r.clock.between('1352','1420'),'volume'].sum(min_count=1),
        v14=r.loc[r.clock.ge('1436'),'volume'].sum(min_count=1),
        vprev14=r.loc[r.clock.between('1422','1435'),'volume'].sum(min_count=1),
        v4=last4.volume.sum(min_count=1),a4=last4.turnover.sum(min_count=1),
        lo29=last29.low.min(),hi29=last29.high.max(),hi109=r.high.max(),lo109=r.low.min(),
        vmax29=last29.volume.max(),signed_v29=last29.signed.sum(min_count=1),
        up29=int(last29.up.sum()),path29=last29.path.sum(min_count=1))


def check():
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert report['protocol_sha256']==sha(Path('config/tail_formula_intraday_protocol.json'))
    assert report['features_sha256']==sha(ROOT/'features.parquet')
    assert report['source_feature_report_sha256']==sha(SOURCE/'feature_report.json')
    manifest=Path('data/research/economic_winner/input_manifest.json')
    assert report['source_manifest_sha256']==sha(manifest)
    hashes=json.loads(manifest.read_text())['source_sha256']
    for path,digest in report['parts_sha256'].items():
        assert sha(Path(path))==digest
    f=pd.read_parquet(ROOT/'features.parquet')
    u=pd.read_parquet(SOURCE/'universe.parquet')
    pd.testing.assert_frame_equal(f[u.columns],u,check_exact=True)
    assert len(f)==report['rows'] and int(f.formula_input_valid.sum())==report['valid']
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all() and f.board.eq('main').all()
    assert not f.code.str[3:].str.startswith(('92','688','300','301')).any()
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.register('f',f)
    e=c.sql('''SELECT date,code,100*(p49/preclose-1) AS A01,100*(daily_open/preclose-1) AS A02,
        100*(p49/daily_open-1) AS A03,p49 AS A04,100*(p49/p20-1) AS A05,
        100*(p49/p35-1) AS A06,100*(p20/p01-1) AS A07,v29/vprev29 AS A08,v14/vprev14 AS A09,
        (v4/4)/((v29-v4)/25) AS A10,100*(p49-lo29)/greatest(hi29-lo29,.01) AS A11,a29/1e8 AS A12,
        100*(p49/(a29/v29)-1) AS A13,100*(p49/(a4/v4)-1) AS A14,100*signed_v29/v29 AS A15,
        100*up29/29 AS A16,100*vmax29/v29 AS A17,100*(p49/hi109-1) AS A18,
        100*(hi109-lo109)/preclose AS A19,
        100*abs(ln(p49/p20))/greatest(path29,.000001) AS A20,
        coalesce(window_valid,false) AND p49=price_1449 AS initial_valid FROM f ORDER BY date,code''').df()
    names=[f'A{i:02d}' for i in range(1,21)]
    pd.testing.assert_frame_equal(f[['date','code']],e[['date','code']],check_exact=True)
    np.testing.assert_allclose(f[names],e[names],rtol=1e-12,atol=2e-10,equal_nan=True)
    expected=e.initial_valid.fillna(False)&np.isfinite(e[names]).all(axis=1)
    assert f.formula_input_valid.equals(expected.rename('formula_input_valid'))
    # Hash sampling is fixed before labels; cover every half and invalid inputs.
    h=f[['date','code','half','formula_input_valid']].copy()
    h['digest']=[hashlib.sha256(f'{d}|{code}|intraday-input-v1'.encode()).hexdigest()
                 for d,code in zip(h.date,h.code)]
    h=h.sort_values('digest')
    chosen=pd.concat([h.groupby('half').head(24),h.loc[~h.formula_input_valid].groupby('half').head(4)])
    chosen=chosen.drop_duplicates(['date','code']).sort_values(['code','date'])
    indexed=f.set_index(['date','code'])
    checks=0
    for code,group in chosen.groupby('code',sort=True):
        path=MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        assert sha(path)==hashes[str(path)]
        filters=[[('timestamp','>=',pd.Timestamp(day+' 13:01').to_pydatetime()),
                  ('timestamp','<=',pd.Timestamp(day+' 14:49:59.999999').to_pydatetime())] for day in group.date]
        raw=pq.read_table(path,filters=filters).to_pandas()
        raw['date']=raw.timestamp.dt.strftime('%Y-%m-%d')
        for day in group.date:
            rebuilt=rebuild(raw.loc[raw.date.eq(day)])
            actual=indexed.loc[(day,code)]
            for key,value in rebuilt.items():
                np.testing.assert_allclose(value,actual[key],rtol=1e-12,atol=2e-8,equal_nan=True,
                    err_msg=f'{day} {code} {key}')
                checks+=1
    coverage=c.sql('''SELECT half,count(*) AS rows,sum(window_valid)::INT AS valid_raw_windows,
        sum(formula_input_valid)::INT AS usable_inputs,sum(bars!=109)::INT AS incomplete_windows,
        sum(bars=109 AND good_bars!=109)::INT AS invalid_values,
        sum(window_valid AND NOT formula_input_valid)::INT AS invalid_denominators
        FROM f GROUP BY half ORDER BY half''').df().to_dict('records')
    result=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),
        all_keys_and_hard_exclusions_checked=True,scalar_features_rebuilt=len(f)*20,
        fixed_raw_windows=len(chosen),raw_aggregate_checks=checks,coverage=coverage,
        sampled_keys=chosen[['date','code']].to_dict('records'),outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',result)
    return {k:v for k,v in result.items() if k!='sampled_keys'}


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
