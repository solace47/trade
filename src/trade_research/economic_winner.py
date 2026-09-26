"""Freeze the full winner cohort and obtain strict next-session raw trade windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .corporate_cash import MINUTES,save_json,sha

ROOT=Path('data/research/economic_winner')
PORTRAIT=Path('data/research/next_day_winner')
BASE=PORTRAIT/'visible_base.parquet'
LABELS=PORTRAIT/'labels.parquet'
PROTOCOL=Path('config/economic_winner_protocol.json')
OLD_COVERAGE=Path('data/research/winner_direction/catalog/combined_coverage.parquet')
OLD_EVENTS=OLD_COVERAGE.with_name('events_reconciled.parquet')


def freeze():
    if (ROOT/'input_manifest.json').exists():raise ValueError('Do not replace the economic label protocol')
    ROOT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((PORTRAIT/'base_report.json').read_text())
    assert sha(BASE)==previous['base_sha256'] and sha(LABELS)==previous['labels_sha256']
    base=pd.read_parquet(BASE,columns=['date','code'])
    schedule=pd.read_parquet(LABELS,columns=['date','code','next_date'])
    keys=base.merge(schedule,on=['date','code'],validate='one_to_one')
    assert len(keys)==previous['rows'] and (keys.next_date>keys.date).all() and keys.next_date.le('2025-12-31').all()
    keys.to_parquet(ROOT/'keys.parquet',index=False)
    windows=pd.concat([keys[['date','code']],keys[['next_date','code']].rename(columns={'next_date':'date'})],ignore_index=True).drop_duplicates().sort_values(['date','code'])
    windows.to_parquet(ROOT/'window_keys.parquet',index=False)
    sources={}
    raw_report=json.loads((PORTRAIT/'raw_report.json').read_text())
    for path,digest in raw_report['batch_manifests_sha256'].items():
        assert sha(Path(path))==digest
        for name,value in json.loads(Path(path).read_text())['source_sha256'].items():
            assert name not in sources;sources[name]=value
    needed=windows.assign(year=windows.date.str[:4])[['code','year']].drop_duplicates().sort_values(['code','year'])
    old=pd.read_parquet(OLD_COVERAGE)
    missing=needed.merge(old[['code','year']],on=['code','year'],how='left',indicator=True)
    missing=missing.loc[missing._merge.eq('left_only'),['code','year']].reset_index(drop=True)
    folder=ROOT/'catalog';folder.mkdir(exist_ok=True)
    needed.to_parquet(folder/'needed.parquet',index=False);missing.to_parquet(folder/'jobs.parquet',index=False)
    save_json(folder/'manifest.json',{'rule_commit':'economic_winner_protocol','jobs_sha256':sha(folder/'jobs.parquet'),
        'code_years':len(missing),'new_2026_prices_read':False})
    result={'protocol_sha256':sha(PROTOCOL),'code_sha256':sha(Path(__file__)),
        'base_sha256':sha(BASE),'labels_sha256':sha(LABELS),'keys_sha256':sha(ROOT/'keys.parquet'),
        'window_keys_sha256':sha(ROOT/'window_keys.parquet'),'old_events_sha256':sha(OLD_EVENTS),
        'old_coverage_sha256':sha(OLD_COVERAGE),'source_sha256':sources,
        'stock_days':len(keys),'windows':len(windows),'codes':windows.code.nunique(),
        'needed_code_years':len(needed),'missing_code_years':len(missing),'economic_labels_computed':False,'new_2026_prices_read':False}
    save_json(ROOT/'input_manifest.json',result)
    return {k:v for k,v in result.items() if k!='source_sha256'}


def raw():
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    assert sha(Path(__file__))==manifest['code_sha256'] and sha(PROTOCOL)==manifest['protocol_sha256']
    assert sha(ROOT/'window_keys.parquet')==manifest['window_keys_sha256']
    keys=pd.read_parquet(ROOT/'window_keys.parquet');codes=sorted(keys.code.unique())
    folder=ROOT/'raw_parts';folder.mkdir(exist_ok=True)
    total=0;parts={}
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64];path=folder/f'part_{offset//64:03d}.parquet';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            saved=json.loads(meta.read_text())
            assert saved['codes']==subset and saved['input_manifest_sha256']==sha(ROOT/'input_manifest.json') and saved['sha256']==sha(path)
            total+=saved['rows'];parts[str(path)]=saved['sha256'];continue
        files=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
        for f in files:assert sha(f)==manifest['source_sha256'][str(f)]
        c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
        c.read_parquet([str(f) for f in files]).create_view('original')
        c.register('keys',keys.loc[keys.code.isin(subset)])
        p=c.sql('''WITH source AS (SELECT lower(exchange)||'.'||symbol AS code,
            strftime(timestamp,'%Y-%m-%d') AS date,timestamp,
            hour(timestamp)*60+minute(timestamp) AS minute,open::DOUBLE AS open,high::DOUBLE AS high,
            low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume,turnover::DOUBLE AS amount
            FROM original WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
            AND hour(timestamp)=14 AND minute(timestamp) BETWEEN 52 AND 55)
            SELECT s.* FROM source s JOIN keys k USING(date,code) ORDER BY s.date,s.code,s.minute''').df()
        assert not p.duplicated(['date','code','timestamp']).any()
        p.to_parquet(path,index=False,compression='zstd');c.close()
        saved={'codes':subset,'rows':len(p),'sha256':sha(path),'input_manifest_sha256':sha(ROOT/'input_manifest.json')}
        save_json(meta,saved);parts[str(path)]=saved['sha256'];total+=len(p)
        print(json.dumps({'codes':offset+len(subset),'total':len(codes),'raw_rows':total}),flush=True)
    result={'input_manifest_sha256':sha(ROOT/'input_manifest.json'),'raw_rows':total,'parts_sha256':parts,
        'economic_labels_computed':False,'new_2026_prices_read':False}
    save_json(ROOT/'raw_report.json',result)
    return {k:v for k,v in result.items() if k!='parts_sha256'}


def catalog():
    from .cash_dividend_catalog import fetch,assemble
    folder=ROOT/'catalog';inputs=json.loads((ROOT/'input_manifest.json').read_text())
    assert sha(OLD_COVERAGE)==inputs['old_coverage_sha256'] and sha(OLD_EVENTS)==inputs['old_events_sha256']
    result=fetch(folder)
    if not result['complete']:return result
    assemble(folder)
    old=pd.read_parquet(OLD_EVENTS);new=pd.read_parquet(folder/'events.parquet')
    events=pd.concat([old,new],ignore_index=True).sort_values(['code','dividOperateDate'])
    coverage=pd.concat([pd.read_parquet(OLD_COVERAGE),pd.read_parquet(folder/'query_coverage.parquet')],ignore_index=True)
    assert not events.duplicated(['code','dividOperateDate']).any() and not coverage.duplicated(['code','year']).any()
    required=pd.read_parquet(folder/'needed.parquet')
    assert required.merge(coverage[['code','year']],on=['code','year'],how='left',indicator=True)._merge.eq('both').all()
    events.to_parquet(folder/'combined_events.parquet',index=False);coverage.to_parquet(folder/'combined_coverage.parquet',index=False)
    report={'complete':True,'needed_code_years':len(required),'added_code_years':inputs['missing_code_years'],
        'events_sha256':sha(folder/'combined_events.parquet'),'coverage_sha256':sha(folder/'combined_coverage.parquet'),
        'catalog_report_sha256':sha(folder/'catalog_report.json'),'economic_labels_computed':False,'new_2026_prices_read':False}
    save_json(folder/'coverage_report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','raw','catalog'])
    args=parser.parse_args();print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
