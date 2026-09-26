"""Frozen first intraday impulse, subsequent retest and recovery before 14:49."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import MINUTES, save_json, sha

ROOT=Path('data/research/impulse_retest')
BASE=Path('data/research/next_day_winner/visible_base.parquet')
PROTOCOL=Path('config/impulse_retest_protocol.json')
CATEGORIES=['unknown_source','no_impulse','impulse_without_half_retest','retested_not_recovered',
            'recovered_but_lost','held_without_volume_confirmation','held_with_volume_confirmation']


def connection():
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
    c.execute('SET preserve_insertion_order=false');return c


def freeze():
    if (ROOT/'manifest.json').exists():raise ValueError('Do not replace the fixed path protocol')
    ROOT.mkdir(parents=True,exist_ok=True)
    prior=json.loads(BASE.with_name('base_report.json').read_text())
    assert sha(BASE)==prior['base_sha256']
    sources={}
    raw_report=json.loads(BASE.with_name('raw_report.json').read_text())
    for path,digest in raw_report['batch_manifests_sha256'].items():
        assert sha(Path(path))==digest
        for name,value in json.loads(Path(path).read_text())['source_sha256'].items():
            assert name not in sources;sources[name]=value
    base=pd.read_parquet(BASE,columns=['date','code'])
    for code in base.code.unique():assert str(MINUTES/code[:2].upper()/(code[3:]+'.parquet')) in sources
    result={'protocol_sha256':sha(PROTOCOL),'base_sha256':sha(BASE),'code_sha256':sha(Path(__file__)),
        'stock_days':len(base),'codes':base.code.nunique(),'first':base.date.min(),'last':base.date.max(),
        'source_sha256':sources,'new_2026_prices_read':False}
    save_json(ROOT/'manifest.json',result)
    return {k:v for k,v in result.items() if k!='source_sha256'}


def extract():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert sha(PROTOCOL)==manifest['protocol_sha256'] and sha(BASE)==manifest['base_sha256']
    assert sha(Path(__file__))==manifest['code_sha256']
    base=pd.read_parquet(BASE,columns=['date','code'])
    codes=sorted(base.code.unique());folder=ROOT/'parts';folder.mkdir(exist_ok=True)
    summaries=[]
    for offset in range(0,len(codes),32):
        subset=codes[offset:offset+32];path=folder/f'part_{offset//32:03d}.parquet';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            saved=json.loads(meta.read_text())
            assert saved['codes']==subset and saved['sha256']==sha(path) and saved['manifest_sha256']==sha(ROOT/'manifest.json')
            summaries.append(saved);continue
        sources=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
        for source in sources:assert sha(source)==manifest['source_sha256'][str(source)]
        c=connection();c.read_parquet([str(p) for p in sources]).create_view('original')
        c.register('keys',base.loc[base.code.isin(subset)])
        c.execute('''CREATE TABLE bars AS WITH raw AS (
            SELECT lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS date,
              hour(timestamp)*60+minute(timestamp) AS minute,timestamp,
              open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
              volume::DOUBLE AS volume,turnover::DOUBLE AS amount
            FROM original WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01')
            SELECT r.*,round(close*100)::BIGINT AS cents,
              minute-CASE WHEN minute>=781 THEN 90 ELSE 0 END AS trading_minute,
              timestamp=date_trunc('minute',timestamp) AND isfinite(open) AND isfinite(high) AND isfinite(low)
              AND isfinite(close) AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
              AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
              AND abs(close-round(close,2))<=.0001 AND volume>=0 AND amount>=0
              AND (volume=0)=(amount=0) AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101) AS valid
            FROM raw r JOIN keys k USING(date,code) WHERE minute BETWEEN 571 AND 690 OR minute BETWEEN 781 AND 889''')
        c.execute('''CREATE TABLE rolling AS SELECT *,lag(cents,5) OVER w AS start_cents,
            sum(volume) OVER(PARTITION BY date,code ORDER BY minute ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS pulse_volume,
            sum(volume) OVER(PARTITION BY date,code ORDER BY minute ROWS BETWEEN 34 PRECEDING AND 5 PRECEDING) AS prior_volume,
            count(*) OVER(PARTITION BY date,code ORDER BY minute ROWS BETWEEN 34 PRECEDING AND CURRENT ROW) AS rolling_count,
            sum(volume) OVER w AS cumulative_volume,row_number() OVER w AS ordinal
            FROM bars WINDOW w AS(PARTITION BY date,code ORDER BY minute)''')
        c.execute('''CREATE TABLE pulses AS SELECT date,code,minute AS pulse_minute,trading_minute AS pulse_trading_minute,
            ordinal AS pulse_ordinal,start_cents,cents AS end_cents,pulse_volume,prior_volume,cumulative_volume AS pulse_cumulative
            FROM rolling WHERE ((minute BETWEEN 605 AND 660) OR (minute BETWEEN 815 AND 860))
            AND rolling_count=35 AND start_cents>0 AND cents*100>=start_cents*101 AND prior_volume>0
            AND pulse_volume*6>=prior_volume*2
            QUALIFY row_number() OVER(PARTITION BY date,code ORDER BY minute)=1''')
        c.execute('''CREATE TABLE retests AS SELECT p.*,r.minute AS retest_minute,r.ordinal AS retest_ordinal,
            r.cents AS retest_cents,r.cumulative_volume AS retest_cumulative
            FROM pulses p LEFT JOIN rolling r ON p.date=r.date AND p.code=r.code AND r.minute>p.pulse_minute
              AND r.cents*2<=p.start_cents+p.end_cents
            QUALIFY row_number() OVER(PARTITION BY p.date,p.code ORDER BY r.minute NULLS LAST)=1''')
        c.execute('''CREATE TABLE recoveries AS SELECT p.*,r.minute AS recovery_minute,r.ordinal AS recovery_ordinal,
            r.cents AS recovery_cents,r.cumulative_volume AS recovery_cumulative
            FROM retests p LEFT JOIN rolling r ON p.date=r.date AND p.code=r.code AND r.minute>p.retest_minute
              AND r.cents>=p.end_cents
            QUALIFY row_number() OVER(PARTITION BY p.date,p.code ORDER BY r.minute NULLS LAST)=1''')
        frame=c.sql('''WITH quality AS (SELECT date,code,count(*) AS bars,count(DISTINCT minute) AS labels,
            bool_and(valid) AS valid_bars,max(cents) FILTER(WHERE minute=889) AS last_cents,
            sum(volume) AS prefix_volume,sum(amount) AS prefix_amount FROM bars GROUP BY date,code)
            SELECT k.date,k.code,q.*,p.*,
              (p.retest_cumulative-p.pulse_cumulative)/(p.retest_ordinal-p.pulse_ordinal) AS retest_mean_volume,
              (p.recovery_cumulative-p.retest_cumulative)/(p.recovery_ordinal-p.retest_ordinal) AS recovery_mean_volume
            FROM keys k LEFT JOIN quality q USING(date,code) LEFT JOIN recoveries p USING(date,code)''').df()
        # DuckDB returns repeated join identity columns when selecting stars.
        frame=frame[[col for col in frame.columns if not col.startswith(('date_','code_'))]]
        valid=frame.bars.eq(229)&frame.labels.eq(229)&frame.valid_bars.fillna(False)
        frame['volume_confirmed']=(frame.retest_mean_volume.lt(frame.pulse_volume/5)&
            frame.recovery_mean_volume.gt(frame.retest_mean_volume))
        frame['category']=np.select([~valid,frame.pulse_minute.isna(),frame.retest_minute.isna(),
            frame.recovery_minute.isna(),frame.last_cents.lt(frame.end_cents),~frame.volume_confirmed],CATEGORIES[:-1],default=CATEGORIES[-1])
        frame['pulse_gain']=frame.end_cents/frame.start_cents-1
        frame=frame.sort_values(['date','code']).reset_index(drop=True)
        assert not frame.duplicated(['date','code']).any() and len(frame)==base.code.isin(subset).sum()
        frame.to_parquet(path,index=False,compression='zstd');c.close()
        saved={'codes':subset,'rows':len(frame),'sha256':sha(path),'manifest_sha256':sha(ROOT/'manifest.json')}
        save_json(meta,saved);summaries.append(saved)
        print(json.dumps({'codes':offset+len(subset),'total':len(codes),'rows':len(frame)}),flush=True)
    result={'manifest_sha256':sha(ROOT/'manifest.json'),'rows':sum(s['rows'] for s in summaries),
        'parts_sha256':{str(p):sha(p) for p in sorted(folder.glob('*.parquet'))},'new_2026_prices_read':False}
    save_json(ROOT/'raw_report.json',result);return result


def pair():
    if (ROOT/'input_report.json').exists():raise ValueError('Do not replace the fixed comparison')
    manifest=json.loads((ROOT/'manifest.json').read_text());raw=json.loads((ROOT/'raw_report.json').read_text())
    assert raw['manifest_sha256']==sha(ROOT/'manifest.json') and manifest['base_sha256']==sha(BASE)
    parts=[]
    for path,digest in raw['parts_sha256'].items():
        assert sha(Path(path))==digest;parts.append(pd.read_parquet(path))
    paths=pd.concat(parts,ignore_index=True)
    base=pd.read_parquet(BASE)
    frame=base.merge(paths,on=['date','code'],how='left',validate='one_to_one')
    assert len(frame)==len(base)==raw['rows'] and frame.category.notna().all()
    assert np.allclose(frame.last_cents/100,frame.price_1449,atol=1e-10,rtol=0)
    frame['day_range']=(frame.high_1449-frame.low_1449)/frame.preclose
    frame['path_comparison_available']=frame.return20_prior_adjusted.notna()
    columns=['return_1449','return20_prior_adjusted','day_range','price_1449','amount_1449','pulse_gain','pulse_trading_minute']
    def coordinates(p):
        return np.column_stack([p.return_1449/.01,p.return20_prior_adjusted/.1,p.day_range/.02,
            np.log2(p.price_1449),np.log2(p.amount_1449),p.pulse_gain/.01,p.pulse_trading_minute/60])
    pair_rows=[]
    for (date,board),p in frame.loc[frame.necessary_tradeable&frame.path_comparison_available].groupby(['date','board'],sort=True):
        selected=p.loc[p.category.eq(CATEGORIES[-1])].sort_values('code')
        reference=p.loc[p.category.eq(CATEGORIES[-2])].sort_values('code')
        if selected.empty or reference.empty:continue
        a,b=coordinates(selected),coordinates(reference)
        # Exact input-only nearest neighbour; control reuse is explicit.
        distance=np.abs(a[:,None,:]-b[None,:,:]).sum(axis=2)
        for i,(_,row) in enumerate(selected.iterrows()):
            j=int(distance[i].argmin());control=reference.iloc[j]
            item={'date':date,'board':board,'half':row.half,'code':row.code,'control_code':control.code,
                'distance':float(distance[i,j])}
            for col in columns:item[col+'_difference']=float(row[col]-control[col])
            pair_rows.append(item)
    pairs=pd.DataFrame(pair_rows)
    frame.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    pairs.to_parquet(ROOT/'pairs.parquet',index=False,compression='zstd')
    grouped=frame.groupby(['half','board','category']).agg(rows=('code','size'),necessary=('necessary_tradeable','sum'),days=('date','nunique')).reset_index()
    result={'manifest_sha256':sha(ROOT/'manifest.json'),'raw_report_sha256':sha(ROOT/'raw_report.json'),
        'features_sha256':sha(ROOT/'features.parquet'),'pairs_sha256':sha(ROOT/'pairs.parquet'),
        'rows':len(frame),'categories':grouped.to_dict('records'),'pairs':len(pairs),
        'outcome_labels_accessed':False,'new_2026_prices_read':False}
    save_json(ROOT/'input_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','extract','pair'])
    args=parser.parse_args();result=globals()[args.stage]()
    print(json.dumps({k:v for k,v in result.items() if k not in ['parts_sha256','categories']},ensure_ascii=False,indent=2))
