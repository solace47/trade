"""Independent dense-calendar history, integer cash and source-window checks."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import MINUTES, save_json, sha

ROOT=Path('data/research/opening_cash_history')
BASE=Path('data/research/next_day_winner/visible_base.parquet')


def source_window(p):
    if p.empty:
        return None
    clock=p.timestamp.dt.strftime('%H:%M')
    good=0
    cash=0
    for r in p.itertuples():
        vals=[r.open,r.high,r.low,r.close,r.volume,r.turnover]
        finite=all(math.isfinite(x) for x in vals)
        prices=vals[:4]
        cent=all(abs(x-round(x,2))<=.0001 for x in prices)
        valid=(finite and cent and min(prices)>0 and r.high>=max(prices)-.0001 and r.low<=min(prices)+.0001
            and r.volume>=0 and r.turnover>=0 and (r.volume==0)==(r.turnover==0)
            and (r.volume==0 or r.low-.0101<=r.turnover/r.volume<=r.high+.0101)
            and r.timestamp.second==0 and r.timestamp.microsecond==0)
        good+=int(valid)
        if math.isfinite(r.turnover) and r.turnover>=0:
            cash+=int(Decimal(str(r.turnover)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)*100)
    last=p.loc[clock.eq('10:00')]
    return dict(bars=len(p),labels=clock.nunique(),valid_bars=good,first_clock=clock.min(),last_clock=clock.max(),
        opening_amount_cents=cash,price_1000=last.close.max(),volume_1000=last.volume.max(),
        window_valid=len(p)==30 and clock.nunique()==30 and good==30 and last.volume.max()>0)


def features():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    raw=json.loads((ROOT/'raw_report.json').read_text())
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert manifest['protocol_sha256']==sha(Path('config/opening_cash_history_protocol.json'))
    assert manifest['base_sha256']==sha(BASE)
    assert raw['manifest_sha256']==sha(ROOT/'manifest.json') and report['raw_report_sha256']==sha(ROOT/'raw_report.json')
    assert report['features_sha256']==sha(ROOT/'features.parquet')
    for name,h in manifest['output_sha256'].items():
        assert sha(ROOT/name)==h
    for path,h in raw['parts_sha256'].items():
        assert sha(Path(path))==h
    c=duckdb.connect()
    base=c.execute("""SELECT * FROM read_parquet(?) WHERE board='main' AND necessary_tradeable
        AND high_1449<upper_limit-.005 ORDER BY date,code""",[str(BASE)]).df()
    frozen=pd.read_parquet(ROOT/'frozen_universe.parquet')
    pd.testing.assert_frame_equal(base,frozen,check_dtype=False,check_exact=True)
    f=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[base.columns],base,check_dtype=False,check_exact=True)
    del base,frozen
    schedule=pd.read_parquet(ROOT/'schedule.parquet').date.tolist()
    codes=sorted(f.code.unique())
    windows=pd.concat([pd.read_parquet(p) for p in raw['parts_sha256']],ignore_index=True)
    assert not windows.duplicated(['date','code']).any() and windows.date.isin(schedule).all()
    assert windows.code.isin(codes).all()
    windows['observed']=windows.opening_amount_cents.where(windows.window_valid)
    wide=windows.pivot(index='code',columns='date',values='observed').reindex(index=codes,columns=schedule)
    values=wide.to_numpy(dtype=float,na_value=np.nan)
    counts=np.zeros(values.shape,dtype='int64')
    medians=np.full(values.shape,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=RuntimeWarning)
        for j in range(1,len(schedule)):
            history=values[:,max(0,j-20):j]
            counts[:,j]=np.isfinite(history).sum(axis=1)
            medians[:,j]=np.nanmedian(history,axis=1)
    code_pos={code:i for i,code in enumerate(codes)}
    date_pos={date:i for i,date in enumerate(schedule)}
    i=f.code.map(code_pos).to_numpy()
    j=f.date.map(date_pos).to_numpy()
    assert (j>=20).all()
    np.testing.assert_array_equal(f.history_count,counts[i,j])
    np.testing.assert_allclose(f.prior_median_cents,medians[i,j],atol=0,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(f.history_first_date,np.array(schedule)[j-20])
    np.testing.assert_array_equal(f.history_last_date,np.array(schedule)[j-1])
    originals=windows.drop(columns='observed').set_index(['date','code']).reindex(pd.MultiIndex.from_frame(f[['date','code']])).reset_index()
    pd.testing.assert_frame_equal(f[originals.columns],originals,check_dtype=False,check_exact=True)
    valid=(f.window_valid.fillna(False) & f.history_count.eq(20) & f.prior_median_cents.gt(0)
        & (f.daily_open-f.daily_open.round(2)).abs().le(.0001))
    high=f.opening_amount_cents.ge(2*medians[i,j]).fillna(False)
    rising=(f.price_1000.round(2).mul(100).round().mul(100)>=f.daily_open.mul(100).round().mul(101)).fillna(False)
    held=(f.price_1449.mul(100).round()>=f.price_1000.mul(100).round()).fillna(False)
    for name,value in [('source_valid',valid),('cash_high',high),('opening_up',rising),('held',held)]:
        np.testing.assert_array_equal(f[name].to_numpy(dtype=bool),value.to_numpy(dtype=bool))
    want_group=np.where(~valid,'unknown_source',np.where(high,
        np.where(rising,np.where(held,'high_held','high_lost'),'high_not_up'),
        np.where(rising,np.where(held,'normal_held','normal_lost'),'normal_not_up')))
    np.testing.assert_array_equal(f.group,want_group)
    np.testing.assert_array_equal(f.primary,want_group=='high_held')
    np.testing.assert_allclose(f.opening_return,f.price_1000/f.daily_open-1,atol=0,rtol=0)
    # Read only the whitelisted old prefix endpoint, not its entry-window fields.
    old=Path('data/research/next_day_winner/raw_report.json')
    old_report=json.loads(old.read_text())
    paths=[]
    for path,h in old_report['batch_manifests_sha256'].items():
        assert sha(Path(path))==h
        meta=json.loads(Path(path).read_text())
        data=Path(path).with_suffix('.parquet')
        assert sha(data)==meta['sha256']
        paths.append(str(data))
    c.read_parquet(paths).create_view('old_prefix')
    c.register('keys',f[['date','code']])
    endpoint=c.sql('SELECT p.date,p.code,p.price_1000 FROM old_prefix p JOIN keys k USING(date,code) ORDER BY p.date,p.code').df()
    assert len(endpoint)==len(f)
    np.testing.assert_allclose(endpoint.price_1000,f.price_1000.round(2),atol=1e-12,rtol=0)
    # Select source checks by input-only category and a deterministic hash.
    sample=f[['date','code','half','group']].copy()
    sample['hash']=[hashlib.sha256(('opening-history|'+d+'|'+code).encode()).hexdigest() for d,code in zip(sample.date,sample.code)]
    sample=sample.sort_values(['half','group','hash']).groupby(['half','group']).head(3)
    indexed=windows.set_index(['date','code'])
    expected_clocks=set(pd.date_range('2024-01-02 09:31','2024-01-02 10:00',freq='min').strftime('%H:%M'))
    checks=nonempty=raw_bars=0
    for r in sample.itertuples():
        position=date_pos[r.date]
        days=schedule[position-20:position+1]
        file=MINUTES/r.code[:2].upper()/(r.code[3:]+'.parquet')
        original=pd.read_parquet(file,columns=['timestamp','open','high','low','close','volume','turnover'],
            filters=[('timestamp','>=',pd.Timestamp(days[0]+' 09:31:00')),('timestamp','<=',pd.Timestamp(r.date+' 10:00:00'))])
        original=original.loc[original.timestamp.dt.strftime('%H:%M').isin(expected_clocks)].copy()
        original['date']=original.timestamp.dt.strftime('%Y-%m-%d')
        assert original.date.isin(days).all()
        for day in days:
            p=original.loc[original.date.eq(day)]
            expected=source_window(p)
            checks+=1
            raw_bars+=len(p)
            if expected is None:
                assert (day,r.code) not in indexed.index
                continue
            nonempty+=1
            saved=indexed.loc[(day,r.code)]
            for name,value in expected.items():
                if pd.isna(value):
                    assert pd.isna(saved[name])
                elif isinstance(value,(int,float,np.number)):
                    assert math.isclose(saved[name],value,rel_tol=0,abs_tol=1e-12),(day,r.code,name,saved[name],value)
                else:
                    assert saved[name]==value,(day,r.code,name)
    result=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),
        window_rows=len(windows),calendar_history_fields=len(f)*4,old_endpoint_rows=len(endpoint),
        original_current_prefixes=len(sample),original_windows_checked=checks,original_nonempty_windows=nonempty,
        original_bars=raw_bars,primary=int(f.primary.sum()),unknown=int((~f.source_valid).sum()),
        outcomes_read=False,new_2026_prices_read=False,returns_2023_computed=False)
    save_json(ROOT/'feature_verification.json',result)
    return result


def pairs():
    report=json.loads((ROOT/'input_report.json').read_text())
    assert report['feature_verification_sha256']==sha(ROOT/'feature_verification.json')
    assert json.loads((ROOT/'feature_verification.json').read_text())['passed']
    for name,h in report['output_sha256'].items():
        assert sha(ROOT/name)==h
    c=duckdb.connect()
    c.read_parquet(str(ROOT/'features.parquet')).create_view('f')
    c.execute("""CREATE VIEW distances AS SELECT a.date,a.code,b.code AS control_code,
        abs(a.return_1449-b.return_1449)/.01+abs(a.opening_return-b.opening_return)/.01
        +abs(a.return20_prior_adjusted-b.return20_prior_adjusted)/.05
        +abs(log2(b.price_1449/a.price_1449))+abs(log2(b.amount_1449/a.amount_1449)) AS distance,
        a.return_1449-b.return_1449 AS return_1449_difference,a.opening_return-b.opening_return AS opening_return_difference,
        a.return20_prior_adjusted-b.return20_prior_adjusted AS return20_prior_adjusted_difference,
        a.price_1449-b.price_1449 AS price_1449_difference,a.amount_1449-b.amount_1449 AS amount_1449_difference
        FROM f a JOIN f b ON a.date=b.date AND substr(a.code,1,2)=substr(b.code,1,2) AND b."group"='normal_held'
        AND abs(a.return_1449-b.return_1449)<=.01 AND abs(a.opening_return-b.opening_return)<=.01
        AND abs(a.return20_prior_adjusted-b.return20_prior_adjusted)<=.05
        AND b.price_1449/a.price_1449 BETWEEN .5 AND 2 AND b.amount_1449/a.amount_1449 BETWEEN .5 AND 2
        WHERE a.primary""")
    expected=c.sql("""WITH nearest AS (SELECT * FROM distances QUALIFY row_number() OVER(PARTITION BY date,code ORDER BY distance,control_code)=1)
        SELECT a.date,a.code,a.half,n.* EXCLUDE(date,code) FROM f a LEFT JOIN nearest n USING(date,code)
        WHERE a.primary ORDER BY a.date,a.code""").df()
    actual=pd.read_parquet(ROOT/'primary_pairs.parquet')
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,atol=2e-12,rtol=0)
    c.register('p',expected)
    strategy=c.sql("""WITH keys AS (SELECT date,code FROM p UNION SELECT date,control_code AS code FROM p WHERE control_code IS NOT NULL)
        SELECT f.*,f.primary AS event FROM keys k JOIN f USING(date,code) ORDER BY date,code""").df()
    saved=pd.read_parquet(ROOT/'strategy_features.parquet')
    pd.testing.assert_frame_equal(saved,strategy,check_dtype=False,check_exact=True)
    assert len(expected)==report['primary'] and len(strategy)==report['strategy_rows']
    assert expected.control_code.notna().sum()==report['paired']
    result=dict(passed=True,input_report_sha256=sha(ROOT/'input_report.json'),primary=len(expected),
        paired=int(expected.control_code.notna().sum()),strategy_rows=len(strategy),universe=report['rows'],
        outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'input_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','pairs'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
