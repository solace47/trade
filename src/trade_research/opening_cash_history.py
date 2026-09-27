"""Opening cash relative to twenty prior matching market-day windows."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import MINUTES, save_json, sha
from .touch_sequence import BASE
from .turnover_reference import CALENDAR

ROOT = Path('data/research/opening_cash_history')
PROTOCOL = Path('config/opening_cash_history_protocol.json')
GROUPS = ['high_held','normal_held','high_lost','normal_lost','high_not_up','normal_not_up','unknown_source']
BALANCE = ['return_1449','opening_return','return20_prior_adjusted','price_1449','amount_1449']


def connection():
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='5GB'")
    return c


def freeze():
    if (ROOT/'manifest.json').exists():
        raise ValueError('Do not replace the fixed opening study')
    assert sha(BASE)==json.loads(BASE.with_name('base_report.json').read_text())['base_sha256']
    base=pd.read_parquet(BASE)
    eligible=base.loc[base.board.eq('main') & base.necessary_tradeable].copy()
    universe=eligible.loc[eligible.high_1449.lt(eligible.upper_limit-.005)].sort_values(['date','code']).reset_index(drop=True)
    assert universe.date.between('2024-01-01','2025-12-30').all()
    cal=pd.read_parquet(CALENDAR)
    all_days=sorted(cal.loc[cal.is_trading_day.eq('1'),'calendar_date'])
    first=all_days.index(universe.date.min())
    schedule=pd.DataFrame({'date':all_days[first-20:all_days.index(universe.date.max())+1]})
    assert len(schedule.loc[schedule.date.lt(universe.date.min())])==20
    ROOT.mkdir(parents=True,exist_ok=True)
    universe.to_parquet(ROOT/'frozen_universe.parquet',index=False,compression='zstd')
    schedule.to_parquet(ROOT/'schedule.parquet',index=False,compression='zstd')
    source_manifest=Path('data/research/economic_winner/input_manifest.json')
    sources=json.loads(source_manifest.read_text())['source_sha256']
    for code in universe.code.unique():
        assert str(MINUTES/code[:2].upper()/(code[3:]+'.parquet')) in sources
    report=dict(protocol_sha256=sha(PROTOCOL),base_sha256=sha(BASE),calendar_sha256=sha(CALENDAR),
        source_manifest_sha256=sha(source_manifest),source_start=schedule.date.min(),source_last=schedule.date.max(),
        first_signal=universe.date.min(),last_signal=universe.date.max(),rows=len(universe),codes=universe.code.nunique(),
        prior_touched_excluded=len(eligible)-len(universe),warmup_market_days=20,
        output_sha256={n:sha(ROOT/n) for n in ['frozen_universe.parquet','schedule.parquet']},
        outcomes_read=False,new_2026_prices_read=False,returns_2023_computed=False)
    save_json(ROOT/'manifest.json',report)
    return report


def extract():
    if (ROOT/'raw_report.json').exists():
        raise ValueError('Do not replace recorded opening windows')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert manifest['protocol_sha256']==sha(PROTOCOL)
    for name,digest in manifest['output_sha256'].items():
        assert sha(ROOT/name)==digest
    source_path=Path('data/research/economic_winner/input_manifest.json')
    assert sha(source_path)==manifest['source_manifest_sha256']
    sources=json.loads(source_path.read_text())['source_sha256']
    universe=pd.read_parquet(ROOT/'frozen_universe.parquet',columns=['date','code'])
    codes=sorted(universe.code.unique())
    folder=ROOT/'window_parts'
    folder.mkdir(exist_ok=True)
    summaries={}
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64]
        dest=folder/f'part_{offset//64:03d}.parquet'
        meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists():
            saved=json.loads(meta.read_text())
            assert saved['codes']==subset and saved['manifest_sha256']==sha(ROOT/'manifest.json')
            assert saved['code_sha256']==sha(Path(__file__)) and saved['sha256']==sha(dest)
        else:
            paths=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
            for path in paths:
                assert sha(path)==sources[str(path)]
            c=connection()
            c.read_parquet([str(path) for path in paths]).create_view('original')
            raw=c.execute("""WITH s AS (SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,timestamp,strftime(timestamp,'%H:%M') AS clock,
                open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
                volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM original
                WHERE timestamp>=?::DATE AND timestamp<?::DATE+INTERVAL 1 DAY
                AND strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00'),
                b AS (SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
                    AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
                    AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
                    AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
                    AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
                    AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
                    AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
                    AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid,
                    CASE WHEN isfinite(amount) AND amount>=0 THEN (CAST(amount AS DECIMAL(20,2))*100)::BIGINT END AS cash_cents
                    FROM s)
                SELECT date,code,count(*) AS bars,count(DISTINCT clock) AS labels,
                    count(*) FILTER(WHERE valid) AS valid_bars,
                    min(clock) AS first_clock,max(clock) AS last_clock,
                    sum(cash_cents)::BIGINT AS opening_amount_cents,
                    max(close) FILTER(WHERE clock='10:00') AS price_1000,
                    max(volume) FILTER(WHERE clock='10:00') AS volume_1000,
                    bars=30 AND labels=30 AND valid_bars=30 AND volume_1000>0 AS window_valid
                FROM b GROUP BY date,code ORDER BY date,code""",[manifest['source_start'],manifest['source_last']]).df()
            assert not raw.duplicated(['date','code']).any()
            raw.to_parquet(dest,index=False,compression='zstd')
            c.close()
            saved=dict(codes=subset,rows=len(raw),sha256=sha(dest),manifest_sha256=sha(ROOT/'manifest.json'),
                       code_sha256=sha(Path(__file__)))
            save_json(meta,saved)
        summaries[str(dest)]=saved['sha256']
        print(json.dumps(dict(codes=offset+len(subset),total_codes=len(codes),windows=saved['rows'])),flush=True)
    result=dict(manifest_sha256=sha(ROOT/'manifest.json'),parts_sha256=summaries,
        source_start=manifest['source_start'],source_last=manifest['source_last'],
        new_2026_prices_read=False,returns_2023_computed=False)
    save_json(ROOT/'raw_report.json',result)
    return {k:v for k,v in result.items() if k!='parts_sha256'}


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace opening path features')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    raw=json.loads((ROOT/'raw_report.json').read_text())
    assert raw['manifest_sha256']==sha(ROOT/'manifest.json')
    for path,h in raw['parts_sha256'].items():
        assert sha(Path(path))==h
    c=connection()
    c.read_parquet(list(raw['parts_sha256'])).create_view('windows')
    c.read_parquet(str(ROOT/'frozen_universe.parquet')).create_view('universe')
    c.read_parquet(str(ROOT/'schedule.parquet')).create_view('schedule')
    f=c.sql("""WITH codes AS (SELECT DISTINCT code FROM universe),
        dense AS (SELECT d.date,k.code,w.* EXCLUDE(date,code),
            CASE WHEN w.window_valid THEN w.opening_amount_cents END AS observed_cash
            FROM codes k CROSS JOIN schedule d LEFT JOIN windows w ON k.code=w.code AND d.date=w.date),
        prior AS (SELECT *,count(observed_cash) OVER history AS history_count,
            median(observed_cash) OVER history AS prior_median_cents,
            min(date) OVER history AS history_first_date,max(date) OVER history AS history_last_date
            FROM dense WINDOW history AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT f.*,p.* EXCLUDE(date,code,observed_cash),
            coalesce(p.window_valid AND p.history_count=20 AND p.prior_median_cents>0
                AND abs(f.daily_open-round(f.daily_open,2))<=.0001,false) AS source_valid
            FROM universe f JOIN prior p USING(date,code) ORDER BY date,code""").df()
    assert len(f)==manifest['rows'] and not f.duplicated(['date','code']).any()
    assert f.history_last_date.lt(f.date).all()
    f['opening_return']=f.price_1000/f.daily_open-1
    f['cash_high']=(f.opening_amount_cents>=2*f.prior_median_cents).fillna(False)
    f['opening_up']=(np.rint(f.price_1000*100)*100>=np.rint(f.daily_open*100)*101).fillna(False)
    f['held']=(np.rint(f.price_1449*100)>=np.rint(f.price_1000*100)).fillna(False)
    path=np.where(f.opening_up,np.where(f.held,'held','lost'),'not_up')
    cash=np.where(f.cash_high,'high_','normal_')
    f['group']=np.where(f.source_valid,np.char.add(cash,path),'unknown_source')
    f['primary']=f.group.eq('high_held')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    result=dict(raw_report_sha256=sha(ROOT/'raw_report.json'),features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),primary=int(f.primary.sum()),unknown=int((~f.source_valid).sum()),
        counts=f.groupby(['half','group']).size().rename('rows').reset_index().to_dict('records'),
        outcomes_read=False,new_2026_prices_read=False,returns_2023_computed=False)
    save_json(ROOT/'feature_report.json',result)
    return result


def pair():
    if (ROOT/'input_report.json').exists():
        raise ValueError('Do not replace frozen opening pairs')
    gate=json.loads((ROOT/'feature_verification.json').read_text())
    assert gate['passed'] and gate['feature_report_sha256']==sha(ROOT/'feature_report.json')
    assert sha(ROOT/'features.parquet')==json.loads((ROOT/'feature_report.json').read_text())['features_sha256']
    f=pd.read_parquet(ROOT/'features.parquet')
    candidates=f.loc[f.primary]
    controls={d:p.sort_values('code') for d,p in f.loc[f.group.eq('normal_held')].groupby('date')}
    pairs=[]
    for r in candidates.itertuples():
        choices=controls.get(r.date,f.iloc[:0])
        choices=choices.loc[choices.code.str[:2].eq(r.code[:2])
            & (choices.return_1449-r.return_1449).abs().le(.01)
            & (choices.opening_return-r.opening_return).abs().le(.01)
            & (choices.return20_prior_adjusted-r.return20_prior_adjusted).abs().le(.05)
            & (choices.price_1449/r.price_1449).between(.5,2)
            & (choices.amount_1449/r.amount_1449).between(.5,2)]
        item=dict(date=r.date,code=r.code,half=r.half,control_code=None,distance=np.nan,
                  **{k+'_difference':np.nan for k in BALANCE})
        if not choices.empty:
            distance=((choices.return_1449-r.return_1449).abs()/.01+(choices.opening_return-r.opening_return).abs()/.01
                +(choices.return20_prior_adjusted-r.return20_prior_adjusted).abs()/.05
                +np.abs(np.log2(choices.price_1449/r.price_1449))+np.abs(np.log2(choices.amount_1449/r.amount_1449)))
            peer=choices.loc[distance.idxmin()]
            item.update(control_code=peer.code,distance=float(distance.min()),
                **{k+'_difference':float(getattr(r,k)-peer[k]) for k in BALANCE})
        pairs.append(item)
    pairs=pd.DataFrame(pairs).sort_values(['date','code']).reset_index(drop=True)
    pairs.to_parquet(ROOT/'primary_pairs.parquet',index=False,compression='zstd')
    controls=pairs.loc[pairs.control_code.notna(),['date','control_code']].rename(columns={'control_code':'code'})
    keys=pd.concat([candidates[['date','code']],controls]).drop_duplicates()
    strategy=keys.merge(f,on=['date','code'],validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    strategy['event']=strategy.primary
    strategy.to_parquet(ROOT/'strategy_features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),feature_verification_sha256=sha(ROOT/'feature_verification.json'),
        rows=len(f),primary=len(candidates),paired=int(pairs.control_code.notna().sum()),strategy_rows=len(strategy),
        output_sha256={n:sha(ROOT/n) for n in ['features.parquet','primary_pairs.parquet','strategy_features.parquet']},
        balance={k:pairs[k+'_difference'].describe().to_dict() for k in BALANCE},
        outcomes_read=False,new_2026_prices_read=False,returns_2023_computed=False)
    save_json(ROOT/'input_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','extract','features','pair'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
