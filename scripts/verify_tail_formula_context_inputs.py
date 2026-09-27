"""Rebuild index context from archived wire records and independent date joins."""
import json
from pathlib import Path
import struct

import numpy as np
import pandas as pd

from verify_index_minute_probe import replay
from trade_research import tail_formula_context as study
from trade_research.corporate_cash import save_json,sha

ROOT=study.ROOT


def main():
    r=json.loads((ROOT/'feature_report.json').read_text());s=json.loads((ROOT/'source_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('previous_feature_report_sha256',study.market.ROOT/'feature_report.json'),
        ('source_report_sha256',ROOT/'source_report.json'),('index_points_sha256',ROOT/'index_points.parquet'),
        ('source_audit_sha256',ROOT/'source_audit.parquet'),('features_sha256',ROOT/'features.parquet')]:
        assert r[key]==sha(path)
    indices=[]
    source=json.loads((study.market.ROOT/'index_source_report.json').read_text())
    for filename,digest in source['raw_files_sha256'].items():
        p=Path(filename);assert sha(p)==digest
        a=pd.DataFrame(json.loads(p.read_text())['records'])
        for name in ['close','high','low']:
            a[name]=a[name].astype(float)
        indices.append(a)
    indices=pd.concat(indices).set_index(['code','date'])
    points=[];audits=[];wire_hashes={};perturbations=0
    for item in s['sessions']:
        symbol,date=item['symbol'],item['date']
        if 'path' in item:
            p=Path(item['path']);assert sha(p)==item['sha256']
            original=json.loads(p.read_text());found=False
            for response in sorted((ROOT/'wire'/f'{symbol}_{date}').glob('*/history.response.bin')):
                request=response.with_name('history.request.bin').read_bytes()
                assert request[:12]==bytes.fromhex('0c01300001010d000d00b40f')
                d,m,c=struct.unpack('<IB6s',request[12:])
                assert (d,m,c.decode())==(int(date.replace('-','')),int(symbol.startswith('sh.')),symbol[3:])
                try:
                    raw_rows=replay(response.read_bytes())
                except (ValueError,AssertionError,IndexError,struct.error):
                    continue
                if raw_rows==original:
                    found=True;wire_hashes[str(response)]=sha(response);break
            assert found
        else:
            original=[]
        valid=len(original)>=228 and all(x['sequence']==i and x['price_raw']>0 for i,x in enumerate(original[:228]))
        points.append(dict(date=date,index_code=symbol,ip48=original[227]['price_raw']/100 if valid else np.nan,
            ip20=original[199]['price_raw']/100 if valid else np.nan,ip35=original[214]['price_raw']/100 if valid else np.nan,
            ip01=original[120]['price_raw']/100 if valid else np.nan,prefix_valid=valid))
        values=np.array([v['price_raw']/100 for v in original]);daily=indices.loc[(symbol,date)]
        ok=len(values)==240 and all(daily.low-.0051<=v<=daily.high+.0051 and v>0 for v in values) and abs(values[-1]-daily.close)<=.0051
        audits.append(dict(date=date,index_code=symbol,rows=len(values),full_day_source_valid=bool(ok),
            last_minus_daily_close=float(values[-1]-daily.close) if len(values) else None))
        if valid and perturbations<32:
            changed=[dict(x) for x in original]
            for x in changed[228:]:
                x['price_raw']=-999999
            assert study.points(original)==study.points(changed)
            perturbations+=1
    points=pd.DataFrame(points).sort_values(['date','index_code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(points,pd.read_parquet(ROOT/'index_points.parquet'),check_exact=True)
    pd.testing.assert_frame_equal(pd.DataFrame(audits),pd.read_parquet(ROOT/'source_audit.parquet'),check_exact=True)
    # ASOF chooses the previous stock trading date, independently of the producer's lag window.
    previous=pd.read_parquet(study.market.ROOT/'features.parquet');actual=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(actual[previous.columns.drop('formula_input_valid')],previous.drop(columns='formula_input_valid'),check_exact=True)
    c=study.base.conn();c.register('keys',previous[['date','code']]);c.register('indices',indices.reset_index())
    c.read_parquet(study.market.stock.source_files()).create_view('daily')
    prior=c.sql('''WITH traded AS(SELECT date,code FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'),
        p AS(SELECT k.date,k.code,s.date AS prior_date FROM keys k ASOF LEFT JOIN traded s ON k.code=s.code AND k.date>s.date)
        SELECT p.date,p.code,i.close AS index_prior_close FROM p LEFT JOIN indices i ON i.date=p.prior_date
        AND i.code=CASE WHEN p.code LIKE 'sh.%' THEN 'sh.000001' ELSE 'sz.399001' END ORDER BY p.date,p.code''').df()
    f=previous.merge(prior,on=['date','code'],validate='one_to_one')
    f['index_code']=f.code.map(lambda code:'sh.000001' if code.startswith('sh.') else 'sz.399001')
    f=f.merge(points,on=['date','index_code'],how='left',validate='many_to_one')
    for name,numerator,denominator in [('J01','ip48','index_prior_close'),('J02','ip48','ip20'),('J03','ip48','ip35'),('J04','ip20','ip01')]:
        f[name]=100*(f[numerator]/f[denominator]-1)
    for i,name in enumerate(['A01','A05','A06','A07'],1):
        f[f'R{i:02d}']=(f[name]-f[f'J{i:02d}'])/f.V01
    new=['index_prior_close','index_code','ip48','ip20','ip35','ip01','prefix_valid',*study.NEW_EXPRESSIONS]
    pd.testing.assert_frame_equal(actual[new],f[new],check_dtype=False,rtol=0,atol=2e-12)
    valid=f.formula_input_valid&f.prefix_valid.fillna(False)&np.isfinite(f[list(study.EXPRESSIONS)]).all(axis=1)
    assert actual.formula_input_valid.equals(valid) and len(actual)==r['rows'] and int(valid.sum())==r['valid']
    assert r['full_day_source_conflicts']==sum(not x['full_day_source_valid'] for x in audits)
    assert r['full_day_source_conflicts']==0,'Resolve source conflicts before interpreting results; do not drop dates'
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(actual),valid=int(valid.sum()),
        sessions=len(points),wire_sha256=wire_hashes,all_eight_context_fields_and_previous_dates_rebuilt=True,
        previous_37_fields_unchanged=True,after_cutoff_perturbation_checks=perturbations,
        full_day_source_audit_used_for_selection=False,native_INDEXC_parity_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='wire_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
