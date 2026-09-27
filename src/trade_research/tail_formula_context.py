"""Historical intraday index context and two predeclared opportunity targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_market as market
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_context')
PROTOCOL=Path('config/tail_formula_context_protocol.json')
HEADER=market.HEADER+'\n'.join(f'IX{t}:=VALUEWHEN(TIME={t},INDEXC);' for t in [1448,1420,1435,1301])+'\n'
NEW_EXPRESSIONS={'J01':'100*(IX1448/ICP1-1)','J02':'100*(IX1448/IX1420-1)',
    'J03':'100*(IX1448/IX1435-1)','J04':'100*(IX1420/IX1301-1)',
    'R01':'(A01-J01)/V01','R02':'(A05-J02)/V01','R03':'(A06-J03)/V01','R04':'(A07-J04)/V01'}
EXPRESSIONS={**market.EXPRESSIONS,**NEW_EXPRESSIONS}
POINTS={'ip48':227,'ip20':199,'ip35':214,'ip01':120}


def points(rows):
    prefix=rows[:228]
    good=len(prefix)==228 and all(q['sequence']==i and q['price_raw']>0 for i,q in enumerate(prefix))
    return {**{name:rows[pos]['price_raw']/100 if good else np.nan for name,pos in POINTS.items()},'prefix_valid':good}


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen index context inputs')
    s=json.loads((ROOT/'source_report.json').read_text())
    assert s['protocol_sha256']==sha(PROTOCOL)
    assert s['source_feature_sha256']==sha(market.ROOT/'features.parquet')
    proof=json.loads((market.ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(market.ROOT/'feature_report.json')
    previous=pd.read_parquet(market.ROOT/'features.parquet')
    expected={(code,date) for code in market.CODES for date in previous.date.unique()}
    assert {(x['symbol'],x['date']) for x in s['sessions']}==expected
    indices=pd.read_parquet(market.ROOT/'indices.parquet').set_index(['code','date'])
    point_rows=[];audit=[]
    for item in s['sessions']:
        code,date=item['symbol'],item['date']
        if 'path' in item:
            path=Path(item['path']);assert sha(path)==item['sha256']
            rows=json.loads(path.read_text())
        else:
            rows=[]
        point_rows.append(dict(date=date,index_code=code,**points(rows)))
        values=np.array([r['price_raw'] for r in rows])/100;day=indices.loc[(code,date)]
        full_valid=bool(len(values)==240 and (values>0).all() and
            (values>=day.low-.0051).all() and (values<=day.high+.0051).all() and abs(values[-1]-day.close)<=.0051)
        audit.append(dict(date=date,index_code=code,rows=len(values),full_day_source_valid=full_valid,
            last_minus_daily_close=float(values[-1]-day.close) if len(values) else None))
    source_points=pd.DataFrame(point_rows).sort_values(['date','index_code']).reset_index(drop=True)
    source_points.to_parquet(ROOT/'index_points.parquet',index=False,compression='zstd')
    pd.DataFrame(audit).to_parquet(ROOT/'source_audit.parquet',index=False,compression='zstd')
    c=base.conn();c.read_parquet(market.stock.source_files()).create_view('daily')
    c.read_parquet(str(market.ROOT/'indices.parquet')).create_view('indices')
    prior=c.sql('''WITH a AS(SELECT s.date,s.code,i.close FROM daily s LEFT JOIN indices i
        ON i.date=s.date AND i.code=CASE WHEN s.code LIKE 'sh.%' THEN 'sh.000001' ELSE 'sz.399001' END
        WHERE s.tradestatus=1 AND s.date BETWEEN '2023-06-01' AND '2025-12-30'),
        b AS(SELECT date,code,lag(close) OVER(PARTITION BY code ORDER BY date) AS index_prior_close FROM a)
        SELECT * FROM b WHERE date>='2024-01-01' ''').df()
    f=previous.merge(prior,on=['date','code'],how='left',validate='one_to_one')
    f['index_code']=np.where(f.code.str.startswith('sh.'),'sh.000001','sz.399001')
    f=f.merge(source_points,on=['date','index_code'],how='left',validate='many_to_one')
    f['J01']=100*(f.ip48/f.index_prior_close-1);f['J02']=100*(f.ip48/f.ip20-1)
    f['J03']=100*(f.ip48/f.ip35-1);f['J04']=100*(f.ip20/f.ip01-1)
    for output,stock,index in [('R01','A01','J01'),('R02','A05','J02'),('R03','A06','J03'),('R04','A07','J04')]:
        f[output]=(f[stock]-f[index])/f.V01
    f['formula_input_valid'] &= f.prefix_valid.fillna(False)&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f=f.sort_values(['date','code']).reset_index(drop=True)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(market.ROOT/'feature_report.json'),
        source_report_sha256=sha(ROOT/'source_report.json'),index_points_sha256=sha(ROOT/'index_points.parquet'),
        source_audit_sha256=sha(ROOT/'source_audit.parquet'),features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),expressions=EXPRESSIONS,native_header=HEADER,
        index_sessions=len(source_points),invalid_prefix_sessions=int((~source_points.prefix_valid).sum()),
        full_day_source_conflicts=sum(not r['full_day_source_valid'] for r in audit),
        full_day_source_audit_used_for_selection=False,software_compilation_verified=False,native_INDEXC_parity_verified=False,
        index_cutoff='14:48',stock_cutoff='14:49',outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def setup(variant):
    assert variant in ['relative','absolute']
    study.ROOT=Path('data/research/tail_formula_context_'+variant);study.PROTOCOL=PROTOCOL;study.setup()
    base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','freeze','verify','analyze','diagnose'])
    p.add_argument('--variant',choices=['relative','absolute'],default='relative')
    a=p.parse_args()
    if a.stage=='features':
        r=features()
    else:
        setup(a.variant)
        if a.stage in ['model','verify_model']:
            r=getattr(relative,a.stage)(a.variant)
        elif a.stage in ['freeze','verify']:
            r=getattr(study,a.stage)()
        else:
            r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
