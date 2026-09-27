"""Afternoon minute paths that can be represented by native formula functions."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import MINUTES, save_json, sha
from .tail_formula_1000 import ROOT as SOURCE

ROOT=Path('data/research/tail_formula_intraday')
PROTOCOL=Path('config/tail_formula_intraday_protocol.json')
EXPRESSIONS={
    'A01':'100*(Q/DYNAINFO(3)-1)',
    'A02':'100*(DYNAINFO(4)/DYNAINFO(3)-1)',
    'A03':'100*(Q/DYNAINFO(4)-1)',
    'A04':'Q',
    'A05':'100*(Q/P20-1)',
    'A06':'100*(Q/P35-1)',
    'A07':'100*(P20/P01-1)',
    'A08':'VALUEWHEN(TIME=1449,SUM(V,29)/REF(SUM(V,29),29))',
    'A09':'VALUEWHEN(TIME=1449,SUM(V,14)/REF(SUM(V,14),14))',
    'A10':'VALUEWHEN(TIME=1449,(SUM(V,4)/4)/((SUM(V,29)-SUM(V,4))/25))',
    'A11':'VALUEWHEN(TIME=1449,100*(C-LLV(L,29))/MAX(HHV(H,29)-LLV(L,29),0.01))',
    'A12':'VALUEWHEN(TIME=1449,SUM(AMOUNT,29)/100000000)',
    'A13':'VALUEWHEN(TIME=1449,100*(C/(SUM(AMOUNT,29)/(100*SUM(V,29)))-1))',
    'A14':'VALUEWHEN(TIME=1449,100*(C/(SUM(AMOUNT,4)/(100*SUM(V,4)))-1))',
    'A15':'VALUEWHEN(TIME=1449,100*SUM(IF(C>REF(C,1),V,IF(C<REF(C,1),-V,0)),29)/SUM(V,29))',
    'A16':'VALUEWHEN(TIME=1449,100*COUNT(C>REF(C,1),29)/29)',
    'A17':'VALUEWHEN(TIME=1449,100*HHV(V,29)/SUM(V,29))',
    'A18':'VALUEWHEN(TIME=1449,100*(C/HHV(H,109)-1))',
    'A19':'VALUEWHEN(TIME=1449,100*(HHV(H,109)-LLV(L,109))/DYNAINFO(3))',
    'A20':'VALUEWHEN(TIME=1449,100*ABS(LN(C/REF(C,29)))/MAX(SUM(ABS(LN(C/REF(C,1))),29),0.000001))',
}
HEADER='Q:=VALUEWHEN(TIME=1449,C);\nP20:=VALUEWHEN(TIME=1420,C);\nP35:=VALUEWHEN(TIME=1435,C);\nP01:=VALUEWHEN(TIME=1301,C);\n'


def conn():
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='5GB'")
    return c


def compute(f):
    f=f.copy()
    q=f.p49
    values=[100*(q/f.preclose-1),100*(f.daily_open/f.preclose-1),100*(q/f.daily_open-1),q,
        100*(q/f.p20-1),100*(q/f.p35-1),100*(f.p20/f.p01-1),f.v29/f.vprev29,f.v14/f.vprev14,
        (f.v4/4)/((f.v29-f.v4)/25),100*(q-f.lo29)/(f.hi29-f.lo29).clip(lower=.01),f.a29/1e8,
        100*(q/(f.a29/f.v29)-1),100*(q/(f.a4/f.v4)-1),100*f.signed_v29/f.v29,
        100*f.up29/29,100*f.vmax29/f.v29,100*(q/f.hi109-1),100*(f.hi109-f.lo109)/f.preclose,
        100*np.abs(np.log(q/f.p20))/f.path29.clip(lower=.000001)]
    for name,value in zip(EXPRESSIONS,values):
        f[name]=value
    f['formula_input_valid']=f.window_valid.fillna(False)&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)&q.eq(f.price_1449)
    return f


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen intraday features')
    ROOT.mkdir(parents=True,exist_ok=True)
    proof=json.loads((SOURCE/'feature_verification.json').read_text())
    source_report=json.loads((SOURCE/'feature_report.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(SOURCE/'feature_report.json')
    assert source_report['output_sha256']['universe.parquet']==sha(SOURCE/'universe.parquet')
    keys=pd.read_parquet(SOURCE/'universe.parquet')
    source_manifest=Path('data/research/economic_winner/input_manifest.json')
    source_hashes=json.loads(source_manifest.read_text())['source_sha256']
    codes=sorted(keys.code.unique())
    folder=ROOT/'parts'
    folder.mkdir(exist_ok=True)
    parts={}
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64]
        path=folder/f'part_{offset//64:03d}.parquet'
        meta_path=path.with_suffix('.json')
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            assert meta['protocol_sha256']==sha(PROTOCOL) and meta['codes']==subset and meta['sha256']==sha(path)
            assert meta['extractor_sha256']==sha(Path(__file__))
        else:
            paths=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
            for source in paths:
                assert sha(source)==source_hashes[str(source)]
            c=conn()
            c.read_parquet([str(p) for p in paths]).create_view('raw')
            c.register('keys',keys.loc[keys.code.isin(subset),['date','code']])
            agg=c.sql('''WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS date,
                strftime(timestamp,'%H%M') AS clock,timestamp,open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,
                close::DOUBLE AS close,volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM raw
                WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND strftime(timestamp,'%H%M') BETWEEN '1301' AND '1449'),
                b AS(SELECT s.* EXCLUDE(open,high,low,close),round(open,2) AS open,round(high,2) AS high,
                    round(low,2) AS low,round(close,2) AS close,
                    coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(open) AND isfinite(high) AND isfinite(low)
                    AND isfinite(close) AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
                    AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
                    AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
                    AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
                    AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
                    AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS good
                    FROM s JOIN keys USING(date,code)),
                p AS(SELECT *,lag(close) OVER(PARTITION BY date,code ORDER BY timestamp) AS prior FROM b)
                SELECT date,code,count(*) AS bars,count(DISTINCT clock) AS clocks,count(*) FILTER(WHERE good) AS good_bars,
                    bars=109 AND clocks=109 AND good_bars=109 AS window_valid,
                    max(close) FILTER(WHERE clock='1449') AS p49,max(close) FILTER(WHERE clock='1435') AS p35,
                    max(close) FILTER(WHERE clock='1420') AS p20,max(close) FILTER(WHERE clock='1301') AS p01,
                    sum(volume) FILTER(WHERE clock>='1421') AS v29,sum(amount) FILTER(WHERE clock>='1421') AS a29,
                    sum(volume) FILTER(WHERE clock BETWEEN '1352' AND '1420') AS vprev29,
                    sum(volume) FILTER(WHERE clock>='1436') AS v14,
                    sum(volume) FILTER(WHERE clock BETWEEN '1422' AND '1435') AS vprev14,
                    sum(volume) FILTER(WHERE clock>='1446') AS v4,sum(amount) FILTER(WHERE clock>='1446') AS a4,
                    min(low) FILTER(WHERE clock>='1421') AS lo29,max(high) FILTER(WHERE clock>='1421') AS hi29,
                    max(high) AS hi109,min(low) AS lo109,max(volume) FILTER(WHERE clock>='1421') AS vmax29,
                    sum(CASE WHEN close>prior THEN volume WHEN close<prior THEN -volume ELSE 0 END) FILTER(WHERE clock>='1421') AS signed_v29,
                    count(*) FILTER(WHERE clock>='1421' AND close>prior) AS up29,
                    sum(abs(ln(close/nullif(prior,0)))) FILTER(WHERE clock>='1421') AS path29
                FROM p GROUP BY date,code ORDER BY date,code''').df()
            c.close()
            agg.to_parquet(path,index=False,compression='zstd')
            meta=dict(codes=subset,protocol_sha256=sha(PROTOCOL),sha256=sha(path),extractor_sha256=sha(Path(__file__)))
            save_json(meta_path,meta)
        parts[str(path)]=meta['sha256']
        print(json.dumps(dict(codes=offset+len(subset),total_codes=len(codes))),flush=True)
    agg=pd.concat([pd.read_parquet(p) for p in parts],ignore_index=True)
    f=compute(keys.merge(agg,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True))
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),source_feature_report_sha256=sha(SOURCE/'feature_report.json'),
        source_manifest_sha256=sha(source_manifest),parts_sha256=parts,features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),expressions=EXPRESSIONS,native_header=HEADER,
        by_half=f.groupby('half').agg(rows=('code','size'),valid=('formula_input_valid','sum')).reset_index().to_dict('records'),
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',report)
    return {k:v for k,v in report.items() if k not in ['parts_sha256','expressions']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
