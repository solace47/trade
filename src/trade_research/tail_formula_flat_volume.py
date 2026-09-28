"""Volume share on late minutes with unchanged adjacent cent close quotes."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_volume_response as cached
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_flat_volume'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {'FV01': 'VALUEWHEN(TIME=1449,100*SUM(IF(INTPART(C*100+0.5)=INTPART(REF(C,1)*100+0.5),V,0),29)/SUM(V,29))'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    assert p['expressions'] == NEW_EXPRESSIONS and p['return_bars'] == 29
    assert not p['new_2026_prices_allowed']
    cached.checked_sources()
    r = json.loads((cached.ROOT/'feature_report.json').read_text())
    v = json.loads((cached.ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(cached.ROOT/'feature_report.json')
    assert r['features_sha256'] == sha(cached.ROOT/'features.parquet')
    return p


def measure(prices, volume):
    prices = np.asarray(prices, dtype=float); volume = np.asarray(volume, dtype=float)
    assert prices.ndim == volume.ndim == 2 and prices.shape == (len(volume),30) and volume.shape[1] == 29
    cents = np.rint(prices*100)
    valid = np.isfinite(prices).all(axis=1) & (prices>0).all(axis=1)
    valid &= (np.abs(prices*100-cents)<=1e-6).all(axis=1)
    valid &= np.isfinite(volume).all(axis=1) & (volume>=0).all(axis=1) & (volume==np.floor(volume)).all(axis=1)
    total = volume.sum(axis=1)
    flat = np.where(cents[:,1:]==cents[:,:-1],volume,0).sum(axis=1)
    valid &= total>0
    value = np.divide(100*flat,total,out=np.full(len(prices),np.nan),where=valid)
    return dict(valid=valid,fv_volume=total,fv_flat_volume=flat,FV01=value)


def features():
    p = checked_sources(); assert not (ROOT/'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT/'features.parquet')
    d = cached.cached_windows(old[['date','code']])
    prices = d[cached.price_source.PRICE_COLUMNS].to_numpy(float)
    volume = d[cached.volume_source.COLUMNS['v']].to_numpy(float)
    np.testing.assert_allclose(prices[:,1:],d[cached.volume_source.COLUMNS['c']],rtol=0,atol=0,equal_nan=True)
    values = measure(prices,volume); valid = values.pop('valid')
    for prefix,count in [('pv',30),('mp',29)]:
        for field in ['bars','clocks','good_bars']:
            valid &= d[prefix+'_'+field].eq(count).to_numpy()
    f = old.copy()
    for name,value in values.items(): f[name]=value
    f['FV01']=f.FV01.where(valid)
    f['flat_volume_valid']=valid; f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    active=f.formula_input_valid
    np.testing.assert_allclose(f.loc[active,'fv_volume'],old.loc[active,'v29'],rtol=0,atol=0)
    assert f.loc[active,'FV01'].between(0,100).all()
    assert f.date.between(p['signal_first'],p['signal_last']).all()
    ROOT.mkdir(parents=True,exist_ok=True)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),source_hashes=p['source_hashes'],features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(active.sum()),newly_invalid=int((old.formula_input_valid & ~active).sum()),
        valid_all_flat_volume=int((active & f.FV01.eq(100)).sum()),valid_no_flat_volume=int((active & f.FV01.eq(0)).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['source_hashes','expressions','native_header']}


def verify_features():
    checked_sources();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(ROOT/'features.parquet')
    old=pd.read_parquet(previous.ROOT/'features.parquet');f=pd.read_parquet(ROOT/'features.parquet')
    d=cached.cached_windows(old[['date','code']]);c=base.conn();c.register('cached',d)
    atoms=[f'struct_pack(prior:=pv_c{i-1:02d},current:=pv_c{i:02d},other:=mp_c{i:02d},v:=mp_v{i:02d})' for i in range(21,50)]
    c.sql('SELECT date,code,unnest(['+','.join(atoms)+']) AS z FROM cached').create_view('atoms')
    stats=c.sql('''SELECT date,code,sum(z.v) AS fv_volume,
        sum(CASE WHEN floor(z.current*100+.5)=floor(z.prior*100+.5) THEN z.v ELSE 0 END) AS fv_flat_volume,
        bool_and(coalesce(isfinite(z.current) AND z.current>0 AND isfinite(z.prior) AND z.prior>0
        AND abs(z.current*100-floor(z.current*100+.5))<=.000001 AND abs(z.prior*100-floor(z.prior*100+.5))<=.000001
        AND z.current=z.other AND isfinite(z.v) AND z.v>=0 AND z.v=floor(z.v),false)) AS good
        FROM atoms GROUP BY date,code ORDER BY date,code''').df()
    c.register('stats',stats)
    e=c.sql('''SELECT date,code,coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30
        AND mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29 AND good AND fv_volume>0,false) AS valid,
        CASE WHEN valid THEN 100*fv_flat_volume/fv_volume END AS FV01
        FROM stats JOIN cached USING(date,code) ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid,old.formula_input_valid,check_names=False,check_exact=True)
    np.testing.assert_array_equal(f.flat_volume_valid,e.valid)
    np.testing.assert_allclose(f.FV01,e.FV01,rtol=0,atol=2e-12,equal_nan=True)
    for field in ['fv_volume','fv_flat_volume']:
        np.testing.assert_array_equal(f.loc[e.valid,field],stats.loc[e.valid,field])
    final=old.formula_input_valid & e.valid
    np.testing.assert_array_equal(f.formula_input_valid,final)
    np.testing.assert_array_equal(np.floor(100*f.loc[final,'FV01']+10000+.000001),np.floor(100*e.loc[final,'FV01']+10000+.000001))
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names)==len({n.casefold() for n in names})
    assert r['expressions']==EXPRESSIONS and r['native_header']==HEADER
    assert int(final.sum())==r['valid'] and int((old.formula_input_valid & ~final).sum())==r['newly_invalid']
    v=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),valid=int(final.sum()),
        all_29_pairs_integer_price_equalities_volume_weights_encodings_and_validities_sql_rebuilt=True,
        all_original_48_values_and_keys_unchanged=True,effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',v);return v


def native_value(prices,lots,outside=1000000.):
    assert len(prices)==len(lots)==30
    env=dict(C=np.r_[outside,outside,prices,outside,outside],V=np.r_[9999.,9999.,lots,9999.,9999.],
        TIME=np.r_[1418,1419,np.arange(1420,1450),1450,1451])
    env.update(REF=lambda x,n:pd.Series(x).shift(int(n)).to_numpy(),
        SUM=lambda x,n:pd.Series(x).rolling(int(n),min_periods=int(n)).apply(math.fsum,raw=True).to_numpy(),
        INTPART=np.floor,IF=np.where,VALUEWHEN=lambda condition,values:np.asarray(values)[np.flatnonzero(condition)[-1]])
    expr=NEW_EXPRESSIONS['FV01'].replace('TIME=1449','TIME==1449').replace(')=INTPART',')==INTPART')
    with np.errstate(divide='ignore',invalid='ignore'):
        return float(eval(expr,{'__builtins__':{}},env))


def native():
    checked_sources();v=json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(ROOT/'feature_report.json')
    old=json.loads((cached.ROOT/'native_input_verification.json').read_text())
    assert old['passed'] and old['feature_report_sha256']==sha(cached.ROOT/'feature_report.json')
    f=pd.read_parquet(ROOT/'features.parquet').set_index(['date','code']);cases=[]
    for sample in old['samples']:
        date,code=sample['date'],sample['code'];path=MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        assert '2024-01-01'<=date<='2025-12-30' and sha(path)==sample['source_sha256']
        d=pd.read_parquet(path,columns=['timestamp','close','volume'],filters=[('timestamp','>=',pd.Timestamp(date+' 14:20')),('timestamp','<=',pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert d.timestamp.dt.strftime('%H:%M').tolist()==[f'14:{i}' for i in range(20,50)]
        prices=d.close.to_numpy(float);assert np.abs(prices-prices.round(2)).max()<=.0001
        prices=prices.round(2);lots=d.volume.to_numpy(float)/100
        value=native_value(prices,lots);row=f.loc[(date,code)];assert row.formula_input_valid
        np.testing.assert_allclose(value,row.FV01,rtol=0,atol=2e-12)
        np.testing.assert_array_equal(np.floor(100*value+10000+.000001),np.floor(100*row.FV01+10000+.000001))
        np.testing.assert_allclose(value,native_value(prices,lots*100,.01),rtol=0,atol=2e-12)
        cases.append(dict(date=date,code=code,source_sha256=sample['source_sha256'],minutes=30,FV01=value))
    assert len(cases)==32
    result=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),samples=cases,raw_minutes=960,
        actual_expression_price_rounding_ref_sum_time_boundary_and_volume_units_rebuilt=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',result);return {k:v for k,v in result.items() if k!='samples'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
