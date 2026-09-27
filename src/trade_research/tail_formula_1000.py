"""Native daily-formula inputs and next-morning opportunity observations."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import DAILY, MINUTES, save_json, sha
from .turnover_reference import CALENDAR

ROOT=Path('data/research/tail_formula_1000')
PROTOCOL=Path('config/tail_formula_1000_protocol.json')
BASE=Path('data/research/next_day_winner/visible_base.parquet')
OLD=Path('data/research/economic_winner/period_quality')
EXPRESSIONS={
    'F01':'100*(C/REF(C,1)-1)',
    'F02':'100*(O/REF(C,1)-1)',
    'F03':'100*(C/O-1)',
    'F04':'100*(C-L)/MAX(H-L,0.01)',
    'F05':'100*(H-L)/REF(C,1)',
    'F06':'100*(H-C)/REF(C,1)',
    'F07':'100*(MIN(O,C)-L)/REF(C,1)',
    'F08':'V/REF(MA(V,5),1)',
    'F09':'AMOUNT/100000000',
    'F10':'C',
    'F11':'100*(REF(C,1)/REF(C,6)-1)',
    'F12':'100*(REF(C,1)/REF(C,21)-1)',
    'F13':'100*(C/REF(MA(C,5),1)-1)',
    'F14':'100*(C/REF(MA(C,20),1)-1)',
    'F15':'100*(C/REF(HHV(H,20),1)-1)',
    'F16':'100*(C/REF(LLV(L,20),1)-1)',
}


def connection():
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='5GB'")
    return c


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen native-formula inputs')
    ROOT.mkdir(parents=True,exist_ok=True)
    base_report=json.loads(BASE.with_name('base_report.json').read_text())
    assert sha(BASE)==base_report['base_sha256']
    manifest=BASE.with_name('source_manifest.json')
    assert sha(manifest)==base_report['source_manifest_sha256']
    old_sources=json.loads(manifest.read_text())['sha256']
    base=pd.read_parquet(BASE,columns=['date','code','half','board','necessary_tradeable','price_1449','high_1449','low_1449',
        'daily_open','volume_1449','amount_1449','preclose','listing_age_sessions','reference_gap','known_delisting',
        'decision_shares','upper_limit','isST','tradestatus'])
    base=base.loc[base.board.eq('main') & base.necessary_tradeable & base.listing_age_sessions.ge(60)
        & base.price_1449.le(200)].sort_values(['date','code']).reset_index(drop=True)
    base.to_parquet(ROOT/'universe.parquet',index=False,compression='zstd')
    results=[]
    sources={}
    for i,(code,p) in enumerate(base.groupby('code',sort=True),1):
        source=DAILY/(code.replace('.','_')+'.parquet')
        assert sha(source)==old_sources[str(source)]
        sources[str(source)]=old_sources[str(source)]
        d=pd.read_parquet(source,columns=['date','open','high','low','close','volume','tradestatus','adjustflag'],
            filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        assert not d.date.duplicated().any()
        prices=d[['open','high','low','close']]
        good=(np.isfinite(prices).all(axis=1)&prices.gt(0).all(axis=1)&d.adjustflag.eq(3)
            & d.high.ge(prices.max(axis=1)-.0001)&d.low.le(prices.min(axis=1)+.0001)
            & (prices-prices.round(2)).abs().le(.0001).all(axis=1)&d.volume.gt(0))
        past=pd.DataFrame({'date':d.date,'p1':d.close.shift(1),'p6':d.close.shift(6),'p21':d.close.shift(21),
            'ma5':d.close.rolling(5,min_periods=5).mean().shift(1),
            'ma20':d.close.rolling(20,min_periods=20).mean().shift(1),
            'v5':d.volume.astype(float).rolling(5,min_periods=5).mean().shift(1),
            'h20':d.high.rolling(20,min_periods=20).max().shift(1),
            'l20':d.low.rolling(20,min_periods=20).min().shift(1),
            'valid_history':good.astype(int).rolling(21,min_periods=21).sum().shift(1).eq(21)})
        f=p.merge(past,on='date',how='left',validate='one_to_one')
        f['formula_input_valid']=f.valid_history.fillna(False)&(f.p1-f.preclose).abs().le(.005)
        price=f.price_1449;op=f.daily_open;hi=f.high_1449;lo=f.low_1449
        values=[100*(price/f.p1-1),100*(op/f.p1-1),100*(price/op-1),100*(price-lo)/(hi-lo).clip(lower=.01),
            100*(hi-lo)/f.p1,100*(hi-price)/f.p1,100*(np.minimum(op,price)-lo)/f.p1,f.volume_1449/f.v5,
            f.amount_1449/1e8,price,100*(f.p1/f.p6-1),100*(f.p1/f.p21-1),100*(price/f.ma5-1),
            100*(price/f.ma20-1),100*(price/f.h20-1),100*(price/f.l20-1)]
        for name,value in zip(EXPRESSIONS,values):
            f[name]=value
        f['formula_input_valid'] &= np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
        results.append(f)
        if i%200==0:
            print(json.dumps(dict(codes=i,rows=sum(len(x) for x in results))),flush=True)
    f=pd.concat(results,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),base_sha256=sha(BASE),source_manifest_sha256=sha(manifest),
        source_sha256=sources,rows=len(f),valid=int(f.formula_input_valid.sum()),codes=f.code.nunique(),
        by_half=f.groupby('half').agg(rows=('code','size'),valid=('formula_input_valid','sum')).reset_index().to_dict('records'),
        expressions=EXPRESSIONS,output_sha256={n:sha(ROOT/n) for n in ['universe.parquet','features.parquet']},
        history_start='2023-06-01',outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',report)
    return {k:v for k,v in report.items() if k not in ['source_sha256','expressions']}


def observation_keys():
    if (ROOT/'observation_manifest.json').exists():
        raise ValueError('Do not replace frozen observation identities')
    gate=json.loads((ROOT/'feature_verification.json').read_text())
    assert gate['passed'] and gate['feature_report_sha256']==sha(ROOT/'feature_report.json')
    f=pd.read_parquet(ROOT/'features.parquet',columns=['date','code','half','decision_shares'])
    cal=pd.read_parquet(CALENDAR)
    days=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between('2024-01-01','2025-12-31'),'calendar_date'])
    forward=dict(zip(days[:-1],days[1:]))
    f['next_date']=f.date.map(forward)
    assert f.next_date.gt(f.date).all()
    f.to_parquet(ROOT/'observation_keys.parquet',index=False,compression='zstd')
    raw_manifest=Path('data/research/economic_winner/input_manifest.json')
    result=dict(protocol_sha256=sha(PROTOCOL),feature_verification_sha256=sha(ROOT/'feature_verification.json'),
        keys_sha256=sha(ROOT/'observation_keys.parquet'),source_manifest_sha256=sha(raw_manifest),
        calendar_sha256=sha(CALENDAR),rows=len(f),last_observation=f.next_date.max(),new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'observation_manifest.json',result)
    return result


def observations():
    if (ROOT/'observation_report.json').exists():
        raise ValueError('Do not replace morning observations')
    m=json.loads((ROOT/'observation_manifest.json').read_text())
    assert m['protocol_sha256']==sha(PROTOCOL) and m['keys_sha256']==sha(ROOT/'observation_keys.parquet')
    source=Path('data/research/economic_winner/input_manifest.json')
    assert m['source_manifest_sha256']==sha(source)
    hashes=json.loads(source.read_text())['source_sha256']
    keys=pd.read_parquet(ROOT/'observation_keys.parquet')
    codes=sorted(keys.code.unique())
    folder=ROOT/'observation_parts'
    folder.mkdir(exist_ok=True)
    parts={}
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64]
        path=folder/f'part_{offset//64:03d}.parquet'
        receipt=path.with_suffix('.json')
        if receipt.exists():
            meta=json.loads(receipt.read_text())
            assert meta['manifest_sha256']==sha(ROOT/'observation_manifest.json') and meta['codes']==subset
            assert meta['sha256']==sha(path) and meta['extractor_sha256']==sha(Path(__file__))
        else:
            files=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
            for file in files:
                assert sha(file)==hashes[str(file)]
            c=connection()
            c.read_parquet([str(x) for x in files]).create_view('raw')
            c.register('keys',keys.loc[keys.code.isin(subset),['date','code','next_date']])
            bars=c.sql('''WITH s AS (SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS next_date,timestamp,strftime(timestamp,'%H:%M') AS clock,
                open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
                volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM raw
                WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00'),
                b AS (SELECT k.date,s.*,coalesce(timestamp=date_trunc('minute',timestamp)
                    AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
                    AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
                    AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
                    AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
                    AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
                    AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
                    AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid
                    FROM s JOIN keys k USING(next_date,code)),
                rolling AS (SELECT *,min(close) OVER three AS low_three,
                    count(*) OVER three AS n_three,count(*) FILTER(WHERE valid AND volume>0) OVER three AS active_three,
                    min(timestamp) OVER three AS first_three
                    FROM b WINDOW three AS(PARTITION BY date,code ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))
                SELECT date,code,next_date,count(*) AS bars,count(DISTINCT clock) AS labels,
                    count(*) FILTER(WHERE valid) AS valid_bars,count(*) FILTER(WHERE valid AND volume>0) AS active_minutes,
                    max(close) FILTER(WHERE valid AND volume>0) AS max_close,
                    max(low_three) FILTER(WHERE n_three=3 AND active_three=3
                        AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close,
                    min(low) FILTER(WHERE valid AND volume>0) AS min_low,
                    max(close) FILTER(WHERE clock='10:00' AND valid AND volume>0) AS price_1000,
                    bars=30 AND labels=30 AND valid_bars=30 AS source_valid
                FROM rolling GROUP BY date,code,next_date ORDER BY date,code''').df()
            assert not bars.duplicated(['date','code']).any()
            bars.to_parquet(path,index=False,compression='zstd')
            c.close()
            meta=dict(manifest_sha256=sha(ROOT/'observation_manifest.json'),codes=subset,rows=len(bars),
                sha256=sha(path),extractor_sha256=sha(Path(__file__)))
            save_json(receipt,meta)
        parts[str(path)]=meta['sha256']
        print(json.dumps(dict(codes=offset+len(subset),total_codes=len(codes))),flush=True)
    c=connection()
    c.read_parquet(list(parts)).create_view('parts')
    c.register('keys',keys)
    combined=c.sql('''SELECT k.*,p.* EXCLUDE(date,code,next_date) FROM keys k LEFT JOIN parts p USING(date,code,next_date)
        ORDER BY date,code''').df()
    combined['source_valid']=combined.source_valid.fillna(False)
    combined.to_parquet(ROOT/'observations.parquet',index=False,compression='zstd')
    report=dict(manifest_sha256=sha(ROOT/'observation_manifest.json'),parts_sha256=parts,
        observations_sha256=sha(ROOT/'observations.parquet'),rows=len(combined),
        complete=int(combined.source_valid.sum()),new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'observation_report.json',report)
    return {k:v for k,v in report.items() if k!='parts_sha256'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','observation_keys','observations'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
