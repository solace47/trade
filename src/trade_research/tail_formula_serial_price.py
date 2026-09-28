"""Adjacent late-minute price-change ordering, using verified cached prices."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_path_variance as prices
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_serial_price'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM+'_protocol.json')
EXPRESSIONS = {**previous.EXPRESSIONS,
    'SP01':'VALUEWHEN(TIME=1449,100*SSN/MAX(SSD,0.000000000001))'}
HEADER = previous.HEADER + ('SSR:=100*LN(C/REF(C,1));\nSSN:=SUM(SSR*REF(SSR,1),28);\n'
    'SSD:=SQRT(SUM(SSR*SSR,28)*REF(SUM(SSR*SSR,28),1));\n')


def serial_moments(price):
    assert price.ndim==2 and price.shape[1]==30
    price=np.where(np.isfinite(price)&(price>0),price,np.nan)
    with np.errstate(divide='ignore',invalid='ignore'):
        changes=100*np.log(price[:,1:]/price[:,:-1])
    a,b=changes[:,1:],changes[:,:-1]
    return np.sum(a*b,axis=1),np.sum(a*a,axis=1),np.sum(b*b,axis=1)


def checked_sources():
    p=json.loads(PROTOCOL.read_text())
    assert p['signal_first']=='2024-01-01' and p['signal_last']=='2025-12-30'
    assert p['window_first']=='1420' and p['window_last']=='1449'
    for file,digest in p['source_hashes'].items(): assert sha(Path(file))==digest
    for root in [previous.ROOT,prices.ROOT]:
        r=json.loads((root/'feature_report.json').read_text())
        v=json.loads((root/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(root/'feature_report.json')
        assert r['features_sha256']==sha(root/'features.parquet')
    return p


def features():
    checked_sources();assert not (ROOT/'feature_report.json').exists()
    old=pd.read_parquet(previous.ROOT/'features.parquet')
    raw=pd.read_parquet(prices.ROOT/'features.parquet',columns=['date','code','path_variance_valid',*prices.PRICE_COLUMNS])
    f=old.merge(raw,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    price=f[prices.PRICE_COLUMNS].to_numpy(float)
    valid=f.path_variance_valid.eq(True)&np.isfinite(price).all(axis=1)&(price>0).all(axis=1)
    num,a2,b2=serial_moments(price)
    f['sp_product']=num;f['sp_later_sq']=a2;f['sp_earlier_sq']=b2
    f['SP01']=pd.Series(100*num/np.maximum(np.sqrt(a2*b2),1e-12)).where(valid)
    f['serial_input_valid']=valid&np.isfinite(f.SP01)
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= f.serial_input_valid
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(ROOT/'features.parquet'),
        source_hashes=json.loads(PROTOCOL.read_text())['source_hashes'],rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(old.formula_input_valid.sum()),newly_invalid=int((old.formula_input_valid&~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    checked_sources();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(ROOT/'features.parquet')
    old=pd.read_parquet(previous.ROOT/'features.parquet');got=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    raw=pd.read_parquet(prices.ROOT/'features.parquet',columns=['date','code','path_variance_valid',*prices.PRICE_COLUMNS])
    pd.testing.assert_frame_equal(got[raw.columns],raw,check_exact=True)
    c=base.conn();c.register('raw_prices',raw)
    rr=[f'(CASE WHEN isfinite(pv_c{i:02d}) AND pv_c{i:02d}>0 AND isfinite(pv_c{i-1:02d}) AND pv_c{i-1:02d}>0 '
        f'THEN 100*ln(pv_c{i:02d}/pv_c{i-1:02d}) END)' for i in range(21,50)]
    product='+'.join(f'{a}*{b}' for a,b in zip(rr[1:],rr[:-1]))
    a2='+'.join(f'{a}*{a}' for a in rr[1:]);b2='+'.join(f'{b}*{b}' for b in rr[:-1])
    good=' AND '.join(f'isfinite({n}) AND {n}>0' for n in prices.PRICE_COLUMNS)
    ex=c.sql(f'''SELECT date,code,coalesce(path_variance_valid AND {good},false) AS valid,
        {product} AS sp_product,{a2} AS sp_later_sq,{b2} AS sp_earlier_sq,
        CASE WHEN valid THEN 100*sp_product/greatest(sqrt(sp_later_sq*sp_earlier_sq),1e-12) END AS SP01
        FROM raw_prices ORDER BY date,code''').df();c.close()
    for name in ['sp_product','sp_later_sq','sp_earlier_sq','SP01']:
        np.testing.assert_allclose(got[name],ex[name],rtol=0,atol=2e-9,equal_nan=True)
    np.testing.assert_array_equal(got.serial_input_valid,ex.valid & np.isfinite(ex.SP01))
    np.testing.assert_array_equal(got.formula_input_valid,old.formula_input_valid & got.serial_input_valid)
    finite=np.isfinite(ex.SP01)
    np.testing.assert_array_equal(np.floor(np.clip(100*got.loc[finite,'SP01']+10000+.000001,0,999999)),
                                  np.floor(np.clip(100*ex.loc[finite,'SP01']+10000+.000001,0,999999)))
    assert len(got)==r['rows'] and got.formula_input_valid.sum()==r['valid']
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(got),
        all_source_prices_original_inputs_moments_validity_and_encoding_rebuilt=True,
        effective_input_intersection_unchanged=bool(got.formula_input_valid.equals(old.formula_input_valid)),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof);return proof


def native():
    checked_sources();report=json.loads((ROOT/'feature_report.json').read_text())
    proof=json.loads((ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(ROOT/'feature_report.json')
    f=pd.read_parquet(ROOT/'features.parquet');f=f.loc[f.formula_input_valid].copy()
    f['fixed_hash']=[hashlib.sha256((d+'|'+c+'|serial-price-v1').encode()).hexdigest() for d,c in zip(f.date,f.code)]
    sample=f.sort_values('fixed_hash').groupby('half',sort=True).head(8)
    assert len(sample)==32
    manifest=json.loads((prices.ROOT/'window_report.json').read_text())['source_sha256'];checks=[]
    for row in sample.itertuples():
        file=MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet');assert sha(file)==manifest[str(file)]
        start,end=[pd.Timestamp(row.date+' '+t) for t in ['14:20','14:49']]
        d=pd.read_parquet(file,columns=['timestamp','close'],filters=[('timestamp','>=',start),('timestamp','<=',end)]).sort_values('timestamp')
        assert d.timestamp.tolist()==pd.date_range(start,end,freq='min').tolist()
        cents=[int(round(x*100)) for x in d.close];assert all(abs(x-c/100)<=.0001 for x,c in zip(d.close,cents))
        np.testing.assert_allclose([c/100 for c in cents],[getattr(row,n) for n in prices.PRICE_COLUMNS],rtol=0,atol=1e-12)
        r=[100*math.log(cents[i]/cents[i-1]) for i in range(1,30)]
        num=sum(r[i]*r[i-1] for i in range(1,29));a=sum(v*v for v in r[1:]);b=sum(v*v for v in r[:-1])
        value=100*num/max(math.sqrt(a*b),1e-12)
        np.testing.assert_allclose(value,row.SP01,rtol=0,atol=2e-9)
        assert math.floor(min(max(100*value+10000+.000001,0),999999))==math.floor(min(max(100*row.SP01+10000+.000001,0),999999))
        checks.append(dict(date=row.date,code=row.code,source_sha256=manifest[str(file)],bars=30,value=value))
    r=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),fixed_samples=32,raw_bars=960,checks=checks,
        all_raw_prices_fixed_pairs_native_sums_and_encoding_rebuilt=True,software_compilation_verified=False,
        native_source_parity_verified=False,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',r);return {k:v for k,v in r.items() if k!='checks'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
