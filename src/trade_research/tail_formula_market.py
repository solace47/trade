"""Prior-day index context aligned to each stock's completed trading dates."""
import argparse
import json
import socket
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from . import tail_formula_recent as study
from . import tail_formula_volatility as stock
from .corporate_cash import save_json,sha
from .ingest import _login,_rows
from .turnover_reference import CALENDAR

ROOT=Path('data/research/tail_formula_market')
PROTOCOL=Path('config/tail_formula_market_protocol.json')
CODES=['sh.000001','sz.399001']
HEADER=stock.HEADER+'\n'.join(f'ICP{n}:=REF(INDEXC,B{n-1});' for n in [1,2,6,21])+'\n'
HEADER+='IM20:=('+ '+'.join(f'REF(INDEXC,B{i})' for i in range(20))+')/20;\n'
EXPRESSIONS={**stock.EXPRESSIONS,'I01':'100*(ICP1/ICP2-1)','I02':'100*(ICP1/ICP6-1)',
    'I03':'100*(ICP1/ICP21-1)','I04':'100*(ICP1/IM20-1)'}


def download():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'index_source_report.json').exists():
        raise ValueError('Do not replace frozen index sources')
    socket.setdefaulttimeout(20)
    folder=ROOT/'index_raw';folder.mkdir(exist_ok=True)
    fields='date,code,open,high,low,close'
    raw_paths={};frames=[]
    _login()
    try:
        for code in CODES:
            path=folder/(code.replace('.','_')+'.json')
            if not path.exists():
                result=bs.query_history_k_data_plus(code,fields,start_date='2023-06-01',end_date='2025-12-30',frequency='d',adjustflag='3')
                data=_rows(result)
                save_json(path,dict(code=code,fields=fields.split(','),start='2023-06-01',end='2025-12-30',
                    frequency='d',adjustflag='3',records=data.to_dict('records')))
            r=json.loads(path.read_text());f=pd.DataFrame(r['records'])
            assert r['code']==code and r['start']=='2023-06-01' and r['end']=='2025-12-30'
            assert len(f) and f.code.eq(code).all() and f.date.between('2023-06-01','2025-12-30').all()
            for n in ['open','high','low','close']:
                f[n]=pd.to_numeric(f[n],errors='raise')
            raw_paths[str(path)]=sha(path);frames.append(f)
    finally:
        bs.logout()
    f=pd.concat(frames).sort_values(['code','date']).reset_index(drop=True)
    calendar=pd.read_parquet(CALENDAR)
    days=sorted(calendar.loc[calendar.is_trading_day.eq('1')&calendar.calendar_date.between('2023-06-01','2025-12-30'),'calendar_date'])
    for code in CODES:
        q=f.loc[f.code.eq(code)];assert q.date.tolist()==days
        prices=q[['open','high','low','close']]
        assert np.isfinite(prices).all().all() and prices.gt(0).all().all()
        assert q.high.add(.0001).ge(prices.max(axis=1)).all() and q.low.sub(.0001).le(prices.min(axis=1)).all()
    f.to_parquet(ROOT/'indices.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),raw_files_sha256=raw_paths,calendar_sha256=sha(CALENDAR),
        indices_sha256=sha(ROOT/'indices.parquet'),rows=len(f),days_per_index=len(days),
        first_date=f.date.min(),last_date=f.date.max(),outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'index_source_report.json',r)
    return r


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen index-context inputs')
    s=json.loads((ROOT/'index_source_report.json').read_text())
    assert s['protocol_sha256']==sha(PROTOCOL) and s['indices_sha256']==sha(ROOT/'indices.parquet')
    p=json.loads((stock.ROOT/'feature_verification.json').read_text())
    r=json.loads((stock.ROOT/'feature_report.json').read_text())
    assert p['passed'] and p['feature_report_sha256']==sha(stock.ROOT/'feature_report.json')
    assert r['features_sha256']==sha(stock.ROOT/'features.parquet')
    files=stock.source_files();c=base.conn();c.read_parquet(files).create_view('stock_daily')
    c.read_parquet(str(ROOT/'indices.parquet')).create_view('indices')
    aligned=c.sql('''WITH a AS(SELECT s.date,s.code,i.close AS ic FROM stock_daily s LEFT JOIN indices i
        ON i.date=s.date AND i.code=CASE WHEN s.code LIKE 'sh.%' THEN 'sh.000001' ELSE 'sz.399001' END
        WHERE s.tradestatus=1 AND s.date BETWEEN '2023-06-01' AND '2025-12-30'),
        h AS(SELECT date,code,lag(ic,1) OVER w AS p1,lag(ic,2) OVER w AS p2,lag(ic,6) OVER w AS p6,
            lag(ic,21) OVER w AS p21,avg(ic) OVER v AS mean20,count(ic) OVER v AS n20
            FROM a WINDOW w AS(PARTITION BY code ORDER BY date),
            v AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT date,code,100*(p1/p2-1) AS I01,100*(p1/p6-1) AS I02,100*(p1/p21-1) AS I03,
            100*(p1/mean20-1) AS I04,n20=20 AND p21 IS NOT NULL AS index_history_valid
        FROM h WHERE date>='2024-01-01' ORDER BY date,code''').df()
    aligned.to_parquet(ROOT/'index_features.parquet',index=False,compression='zstd')
    f=pd.read_parquet(stock.ROOT/'features.parquet').merge(aligned,on=['date','code'],how='left',validate='one_to_one')
    f['formula_input_valid'] &= f.index_history_valid.fillna(False)&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f=f.sort_values(['date','code']).reset_index(drop=True)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),stock_report_sha256=sha(stock.ROOT/'feature_report.json'),
        index_source_report_sha256=sha(ROOT/'index_source_report.json'),index_features_sha256=sha(ROOT/'index_features.parquet'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),expressions=EXPRESSIONS,
        native_header=HEADER,software_compilation_verified=False,native_history_alignment_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',report)
    return {k:v for k,v in report.items() if k not in ['expressions','native_header']}


def verify_features():
    r=json.loads((ROOT/'feature_report.json').read_text());s=json.loads((ROOT/'index_source_report.json').read_text())
    for key,path in [('protocol_sha256',PROTOCOL),('stock_report_sha256',stock.ROOT/'feature_report.json'),
                     ('index_source_report_sha256',ROOT/'index_source_report.json'),('index_features_sha256',ROOT/'index_features.parquet'),
                     ('features_sha256',ROOT/'features.parquet')]:
        assert r[key]==sha(path)
    assert s['indices_sha256']==sha(ROOT/'indices.parquet')
    indices={}
    for name,digest in s['raw_files_sha256'].items():
        assert sha(Path(name))==digest
        item=json.loads(Path(name).read_text());d=pd.DataFrame(item['records'])
        indices[item['code']]=d.set_index('date').close.astype(float)
    rebuilt=[]
    for path in stock.source_files():
        d=pd.read_parquet(path,columns=['date','code','tradestatus'],filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').copy()
        code=d.code.iloc[0];idx=indices['sh.000001' if code.startswith('sh.') else 'sz.399001']
        ic=d.date.map(idx);p1=ic.shift()
        for field,values in [('I01',100*(p1/ic.shift(2)-1)),('I02',100*(p1/ic.shift(6)-1)),
            ('I03',100*(p1/ic.shift(21)-1)),('I04',100*(p1/ic.rolling(20,min_periods=20).mean().shift()-1))]:
            d[field]=values
        d['index_history_valid']=ic.rolling(20,min_periods=20).count().shift().eq(20)&ic.shift(21).notna()
        rebuilt.append(d.loc[d.date.ge('2024-01-01'),['date','code','I01','I02','I03','I04','index_history_valid']])
    expected=pd.concat(rebuilt).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT/'index_features.parquet'),expected,check_dtype=False,rtol=0,atol=2e-12)
    old=pd.read_parquet(stock.ROOT/'features.parquet');actual=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    joined=old[['date','code','formula_input_valid']].merge(expected,on=['date','code'],how='left',validate='one_to_one')
    expected_valid=joined.formula_input_valid&joined.index_history_valid.fillna(False)&np.isfinite(actual[list(EXPRESSIONS)]).all(axis=1)
    assert actual.formula_input_valid.equals(expected_valid) and len(actual)==r['rows']
    result=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(actual),
        all_raw_index_close_alignments_and_four_features_rebuilt=True,previous_33_fields_unchanged=True,
        new_2026_prices_read=False,outcomes_read=False,no_exit_rules=True,software_compilation_verified=False)
    save_json(ROOT/'feature_verification.json',result)
    return result


def setup():
    study.ROOT=ROOT;study.PROTOCOL=PROTOCOL;study.setup()
    base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['download','features','verify_features','model','verify_model','scores','freeze','verify','analyze','diagnose'])
    a=p.parse_args()
    if a.stage in ['download','features','verify_features']:
        r=globals()[a.stage]()
    else:
        setup()
        if a.stage in ['model','verify_model']:
            r=getattr(relative,a.stage)('relative')
        elif a.stage in ['freeze','verify']:
            r=getattr(study,a.stage)()
        else:
            r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
