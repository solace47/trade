"""Prior-close volume-weighted mid-CDF at two already visible tail prices."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_history_weight as source
from .corporate_cash import MINUTES, save_json, sha

STEM='tail_formula_history_cdf'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_protocol.json')
EXPRESSIONS={**previous.EXPRESSIONS,'HF01':'HFC49','HF02':'HFC49-HFC20'}
HEADER=source.HEADER.split('HW20:=')[0]
for suffix,point in [('49','Q'),('20','P20')]:
    HEADER+=f'HFC{suffix}:=50*('+ '+'.join(
        f'HDV{i:02d}*(2*(INTPART(DCP{i}*100+0.5)<INTPART({point}*100+0.5))+(INTPART(DCP{i}*100+0.5)=INTPART({point}*100+0.5)))'
        for i in range(1,21))+')/HV20;\n'


def midcdf(point_cents,history_cents,volume):
    point=np.asarray(point_cents)[:,None]
    weights=2*(history_cents<point)+(history_cents==point).astype(float)
    return 50*np.sum(volume*weights,axis=1)/np.sum(volume,axis=1)


def checked_sources():
    p=json.loads(PROTOCOL.read_text())
    assert p['history_stock_days']==20 and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    for root in [source.ROOT,previous.ROOT]:
        r=json.loads((root/'feature_report.json').read_text())
        v=json.loads((root/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(root/'feature_report.json')
        assert r['features_sha256']==sha(root/'features.parquet')
    return p


def features():
    checked_sources();assert not (ROOT/'feature_report.json').exists()
    old=pd.read_parquet(previous.ROOT/'features.parquet')
    hist=pd.read_parquet(source.ROOT/'features.parquet')
    pd.testing.assert_frame_equal(old[['date','code']],hist[['date','code']],check_exact=True)
    f=old.copy();prices=hist[source.PRICE_COLUMNS].to_numpy(dtype=float)
    volume=hist[source.VOLUME_COLUMNS].to_numpy(dtype=float)
    cents=np.floor(prices*100+.5);p49=np.floor(f.price_1449*100+.5);p20=np.floor(f.p20*100+.5)
    valid=hist.history_weight_valid.copy()
    valid &= np.isfinite(f[['price_1449','p20']]).all(axis=1)&f.price_1449.gt(0)&f.p20.gt(0)
    valid &= f.price_1449.sub(p49/100).abs().le(.0001)&f.p20.sub(p20/100).abs().le(.0001)
    valid &= np.isfinite(prices).all(axis=1)&(np.abs(prices-cents/100)<=.0001).all(axis=1)
    c49=midcdf(p49,cents,volume);c20=midcdf(p20,cents,volume)
    f['HF01']=pd.Series(c49).where(valid);f['HF02']=pd.Series(c49-c20).where(valid)
    f['history_cdf_valid']=valid;f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(ROOT/'features.parquet'),
        source_hashes=json.loads(PROTOCOL.read_text())['source_hashes'],rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid&~f.formula_input_valid).sum()),expressions=EXPRESSIONS,native_header=HEADER,
        valid_with_reference_breaks=int((f.formula_input_valid&hist.hc_reference_breaks.gt(0)).sum()),
        valid_with_history_gaps=int((f.formula_input_valid&hist.hc_market_span.gt(20)).sum()),
        raw_unadjusted_close_distribution_not_holder_cost=True,software_compilation_verified=False,
        native_source_parity_verified=False,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    checked_sources();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(ROOT/'features.parquet')
    old=pd.read_parquet(previous.ROOT/'features.parquet');h=pd.read_parquet(source.ROOT/'features.parquet')
    f=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    c=base.conn();c.register('hist',h[['date','code','price_1449','p20','history_weight_valid',*source.PRICE_COLUMNS,*source.VOLUME_COLUMNS]])
    c.sql('SELECT date,code,price_1449,p20,history_weight_valid,unnest(['+','.join(source.PRICE_COLUMNS)+']) AS price,unnest(['+
        ','.join(source.VOLUME_COLUMNS)+']) AS volume FROM hist').create_view('long_prices')
    ex=c.sql('''WITH a AS(SELECT *,floor(price*100+.5) AS pc,floor(price_1449*100+.5) AS q49,
        floor(p20*100+.5) AS q20 FROM long_prices),
        b AS(SELECT date,code,bool_and(history_weight_valid AND isfinite(price_1449) AND price_1449>0
          AND isfinite(p20) AND p20>0 AND abs(price_1449-q49/100.)<=.0001 AND abs(p20-q20/100.)<=.0001
          AND isfinite(price) AND abs(price-pc/100.)<=.0001) AS valid,
          50*sum(volume*(CASE WHEN pc<q49 THEN 2 WHEN pc=q49 THEN 1 ELSE 0 END))/sum(volume) AS c49,
          50*sum(volume*(CASE WHEN pc<q20 THEN 2 WHEN pc=q20 THEN 1 ELSE 0 END))/sum(volume) AS c20
          FROM a GROUP BY date,code)
        SELECT date,code,valid,CASE WHEN valid THEN c49 END AS HF01,
          CASE WHEN valid THEN c49-c20 END AS HF02 FROM b ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[['date','code']],ex[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.history_cdf_valid,ex.valid)
    np.testing.assert_allclose(f[['HF01','HF02']],ex[['HF01','HF02']],rtol=0,atol=2e-12,equal_nan=True)
    valid=old.formula_input_valid & ex.valid & np.isfinite(ex[['HF01','HF02']]).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid,valid)
    enc=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    np.testing.assert_array_equal(enc(f.loc[valid,['HF01','HF02']]),enc(ex.loc[valid,['HF01','HF02']]))
    assert f.loc[valid,'HF01'].between(0,100).all() and f.loc[valid,'HF02'].between(-100,100).all()
    assert r['expressions']==EXPRESSIONS and r['native_header']==HEADER
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS);assert len(names)==len(set(names))
    assert len(f)==r['rows'] and int(valid.sum())==r['valid']
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),
        all_twenty_price_volume_pairs_integer_ties_and_features_rebuilt=True,
        all_encodings_and_original48_values_and_keys_verified=True,effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof);return proof


def native():
    p=checked_sources();v=json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(ROOT/'feature_report.json')
    old=json.loads((source.ROOT/'feature_verification.json').read_text());assert old['native_probe_date_mismatches']==old['native_probe_field_mismatches']==0
    f=pd.read_parquet(ROOT/'features.parquet').set_index(['date','code'])
    h=pd.read_parquet(source.ROOT/'features.parquet').set_index(['date','code']);cases=[]
    hashes=json.loads(Path(p['minute_manifest']).read_text())['source_sha256']
    for sample in old['fixed_native_cases']:
        day,code=sample['date'],sample['code'];row=f.loc[(day,code)];hist=h.loc[(day,code)]
        path=MINUTES/code[:2].upper()/(code[3:]+'.parquet');assert sha(path)==hashes[str(path)]
        bars=pd.read_parquet(path,columns=['timestamp','close'],filters=[
            [('timestamp','==',pd.Timestamp(day+' 14:20'))],
            [('timestamp','==',pd.Timestamp(day+' 14:49'))]])
        assert len(bars)==2
        q=[]
        for clock in ['14:20','14:49']:
            x=bars.loc[bars.timestamp.eq(pd.Timestamp(day+' '+clock)),'close'];assert len(x)==1
            cents=int(np.floor(float(x.iloc[0])*100+.5));assert abs(float(x.iloc[0])-cents/100)<=.0001;q.append(cents)
        assert q==[int(round(row.p20*100)),int(round(row.price_1449*100))]
        prices=[int(round(hist[name]*100)) for name in source.PRICE_COLUMNS]
        volumes=[int(hist[name]) for name in source.VOLUME_COLUMNS]
        values=[50*sum(vol*(2*int(price<point)+int(price==point)) for price,vol in zip(prices,volumes))/sum(volumes) for point in q]
        actual=[values[1],values[1]-values[0]];expected=[row.HF01,row.HF02]
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12)
        enc=lambda a:np.floor(np.clip(100*np.asarray(a)+10000+.000001,0,999999))
        np.testing.assert_array_equal(enc(actual),enc(expected))
        cases.append(dict(date=day,code=code,source_sha256=sha(path),prior_daily_points_reused=20,new_point_minutes=2,values=actual))
    assert len(cases)==32
    proof=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),reused_history_probe_sha256=sha(source.ROOT/'feature_verification.json'),
        fixed_samples=32,new_point_minutes=64,prior_daily_points_reused=640,samples=cases,
        all_integer_ties_native_sums_and_encodings_verified=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',proof);return {k:v for k,v in proof.items() if k!='samples'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
