"""Mean late close premium to the cumulative VWAP actually visible at each minute."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_price_impact as cached
from .corporate_cash import MINUTES,save_json,sha

STEM='tail_formula_vwap_path'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_protocol.json')
NEW_EXPRESSIONS={'VP01':'VALUEWHEN(TIME=1449,SUM(PVG,29)/29)/V01'}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
EXTRA_HEADER='''PVSV:=SUM(V,B0);
PVSA:=SUM(AMOUNT,B0);
PVWP:=PVSA/MAX(100*PVSV,0.01);
PVG:=100*(C/MAX(PVWP,0.00000001)-1);
PVT:=VALUEWHEN(TIME=1449,COUNT(PVSV>0 AND PVSA>0,29));
'''
HEADER=previous.HEADER+EXTRA_HEADER
ORIGINAL_NATIVE_CORE=base.native_core
CLOCKS=[900+i for i in range(30,60)]+[1000+i for i in range(60)]+[1100+i for i in range(31)]+[1300+i for i in range(1,60)]+[1400+i for i in range(50)]


def native_core(*args,**kwargs):
    text=ORIGINAL_NATIVE_CORE(*args,**kwargs);assert text.count('CORE:SC>')==1
    return text.replace('CORE:SC>','CORE:PVT=29 AND SC>')


def checked_sources():
    p=json.loads(PROTOCOL.read_text())
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    cached.checked_sources()
    r=json.loads((cached.ROOT/'feature_report.json').read_text());v=json.loads((cached.ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(cached.ROOT/'feature_report.json')
    assert r['features_sha256']==sha(cached.ROOT/'features.parquet')
    assert p['return_bars']==29 and not p['new_2026_prices_allowed']
    return p


def measure(prices,volume,amount,total_volume,total_amount,atr):
    prices=np.asarray(prices,float);volume=np.asarray(volume,float);amount=np.asarray(amount,float)
    total_volume=np.asarray(total_volume,float);total_amount=np.asarray(total_amount,float);atr=np.asarray(atr,float)
    assert prices.shape==(len(volume),30) and volume.shape==amount.shape==(len(prices),29)
    early_v=total_volume-volume.sum(axis=1);early_a=total_amount-amount.sum(axis=1)
    cv=early_v[:,None]+np.cumsum(volume,axis=1);ca=early_a[:,None]+np.cumsum(amount,axis=1)
    good=np.isfinite(prices).all(axis=1)&(prices>0).all(axis=1)
    good&=np.isfinite(volume).all(axis=1)&(volume>=0).all(axis=1)&(volume==np.floor(volume)).all(axis=1)
    good&=np.isfinite(amount).all(axis=1)&(amount>=0).all(axis=1)&((volume==0)==(amount==0)).all(axis=1)
    good&=np.isfinite(early_v)&(early_v>=0)&np.isfinite(early_a)&(early_a>=0)
    good&=(cv>0).all(axis=1)&(ca>0).all(axis=1)&np.isfinite(atr)&(atr>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        vw=ca/cv;premium=100*(prices[:,1:]/vw-1);value=premium.mean(axis=1)/atr
    good&=np.isfinite(value)&(vw>=1e-8).all(axis=1)
    return dict(valid=good,early_volume=early_v,early_amount=early_a,cumulative_volume=cv,cumulative_amount=ca,
        premium=premium,VP01=np.where(good,value,np.nan))


def source_arrays(d,old):
    prices=d[cached.cache.price_source.PRICE_COLUMNS].to_numpy(float)
    volume=d[cached.cache.volume_source.COLUMNS['v']].to_numpy(float)
    amount=d[cached.amounts.COLUMNS].to_numpy(float)
    np.testing.assert_allclose(prices[:,1:],d[cached.cache.volume_source.COLUMNS['c']],rtol=0,atol=0,equal_nan=True)
    return prices,volume,amount,old.volume_1449.to_numpy(float),old.amount_1449.to_numpy(float),old.V01.to_numpy(float)


def window_valid(d):
    good=np.ones(len(d),dtype=bool)
    for prefix,count in [('pv',30),('mp',29),('il',29)]:
        for name in ['bars','clocks','good_bars']:good&=d[prefix+'_'+name].eq(count).to_numpy()
    return good


def features():
    p=checked_sources();assert not (ROOT/'feature_report.json').exists()
    old=pd.read_parquet(previous.ROOT/'features.parquet');d=cached.cached_windows(old[['date','code']])
    arrays=source_arrays(d,old);m=measure(*arrays);good=m['valid']&window_valid(d)
    f=old.copy();f['vp_early_volume']=m['early_volume'];f['vp_early_amount']=m['early_amount']
    f['VP01']=pd.Series(m['VP01']).where(good);f['vwap_path_valid']=good
    f['prior_formula_input_valid']=f.formula_input_valid;f['formula_input_valid']&=good
    active=f.formula_input_valid
    np.testing.assert_allclose(arrays[1][active].sum(axis=1),old.loc[active,'v29'],rtol=0,atol=0)
    np.testing.assert_allclose(arrays[2][active].sum(axis=1),old.loc[active,'a29'],rtol=2e-13,atol=2e-6)
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),source_hashes=p['source_hashes'],features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(active.sum()),newly_invalid=int((old.formula_input_valid & ~active).sum()),
        minimum_vwap=float(np.nanmin((m['cumulative_amount']/m['cumulative_volume'])[active])),
        expressions=EXPRESSIONS,native_header=HEADER,native_core_gate='PVT=29',
        software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['source_hashes','expressions','native_header']}


def verify_features():
    checked_sources();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(ROOT/'features.parquet')
    old=pd.read_parquet(previous.ROOT/'features.parquet');f=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid,old.formula_input_valid,check_names=False,check_exact=True)
    d=cached.cached_windows(old[['date','code']]);all_good=[];all_values=[];checked=0
    for start in range(0,len(d),25000):
        part=d.iloc[start:start+25000];src=old.iloc[start:start+25000];p,v,a,tv,ta,atr=source_arrays(part,src)
        n=len(part);long=pd.DataFrame(dict(key=np.repeat(np.arange(n),29),i=np.tile(np.arange(29),n),
            price=p[:,1:].ravel(),prior=p[:,:-1].ravel(),v=v.ravel(),a=a.ravel(),
            total_v=np.repeat(tv,29),total_a=np.repeat(ta,29),atr=np.repeat(atr,29)))
        c=base.conn();c.register('long_rows',long)
        c.sql('''SELECT *,total_v-sum(v) OVER(PARTITION BY key) AS early_v,
            total_a-sum(a) OVER(PARTITION BY key) AS early_a,
            sum(v) OVER(PARTITION BY key ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS sv,
            sum(a) OVER(PARTITION BY key ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS sa
            FROM long_rows''').create_view('sums')
        e=c.sql('''SELECT *,early_v+sv AS cv,early_a+sa AS ca,
            100*(price/((early_a+sa)/(early_v+sv))-1) AS premium,
            coalesce(isfinite(price) AND price>0 AND isfinite(prior) AND prior>0
            AND isfinite(v) AND v>=0 AND v=floor(v) AND isfinite(a) AND a>=0 AND ((v=0)=(a=0))
            AND isfinite(early_v) AND early_v>=0 AND isfinite(early_a) AND early_a>=0
            AND cv>0 AND ca>0 AND ca/cv>=.00000001 AND isfinite(atr) AND atr>0,false) AS good
            FROM sums ORDER BY key,i''').df()
        c.register('expected',e)
        agg=c.sql('SELECT key,bool_and(good) AS good,avg(premium)/first(atr) AS value FROM expected GROUP BY key ORDER BY key').df();c.close()
        source_good=window_valid(part);valid=agg.good.to_numpy()&source_good&np.isfinite(agg.value)
        all_good.extend(valid.tolist());all_values.extend(np.where(valid,agg.value,np.nan).tolist())
        original=measure(p,v,a,tv,ta,atr)
        for field,expected in [('cumulative_volume',e.cv),('cumulative_amount',e.ca),('premium',e.premium)]:
            x=expected.to_numpy(float).reshape(n,29)
            np.testing.assert_allclose(original[field][valid],x[valid],rtol=2e-12,atol=2e-6 if field=='cumulative_amount' else 2e-10)
        checked+=len(e)
    values=np.array(all_values);valid=np.array(all_good)
    np.testing.assert_array_equal(f.vwap_path_valid,valid)
    np.testing.assert_allclose(f.VP01,values,rtol=0,atol=2e-10,equal_nan=True)
    final=old.formula_input_valid&valid
    np.testing.assert_array_equal(f.formula_input_valid,final)
    enc=lambda x:np.floor(np.clip(100*np.asarray(x)+10000+.000001,0,999999))
    np.testing.assert_array_equal(enc(f.loc[final,'VP01']),enc(values[final]))
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS);assert len(names)==len({n.casefold() for n in names})
    assert r['expressions']==EXPRESSIONS and r['native_header']==HEADER
    assert int(final.sum())==r['valid'] and int((old.formula_input_valid&~final).sum())==r['newly_invalid']
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),valid=int(final.sum()),minute_checks=checked,
        all_29_cumulative_amounts_volumes_premiums_mean_validity_and_encodings_sql_rebuilt=True,
        all_original_48_values_and_keys_unchanged=True,effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof);return proof


def native_value(prices,lots,amounts,atr,outside=1000000.):
    n=len(prices);assert n==len(lots)==len(amounts)==230
    times=np.r_[1459,1500,CLOCKS,1450,1451]
    env=dict(C=np.r_[outside,outside,prices,outside,outside],V=np.r_[9999.,9999.,lots,9999.,9999.],
        AMOUNT=np.r_[9e9,9e9,amounts,9e9,9e9],TIME=times,B0=np.r_[1,2,np.arange(1,n+1),n+1,n+2],V01=float(atr))
    def sum_(x,length):
        lens=np.repeat(length,len(x)) if np.ndim(length)==0 else np.asarray(length)
        return np.array([math.fsum(x[i-int(k)+1:i+1]) if int(k)<=i+1 else np.nan for i,k in enumerate(lens)])
    env.update(SUM=sum_,COUNT=sum_,MAX=np.maximum,VALUEWHEN=lambda condition,values:np.asarray(values)[np.flatnonzero(condition)[-1]])
    for statement in EXTRA_HEADER.strip().split(';'):
        if statement.strip():
            key,expr=statement.strip().split(':=');expr=expr.replace('TIME=1449','TIME==1449').replace('PVSV>0 AND PVSA>0','((PVSV>0) & (PVSA>0))')
            env[key]=eval(expr,{'__builtins__':{}},env)
    value=eval(NEW_EXPRESSIONS['VP01'].replace('TIME=1449','TIME==1449'),{'__builtins__':{}},env)
    return float(value),float(env['PVT'])


def native():
    checked_sources();v=json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(ROOT/'feature_report.json')
    source=json.loads((cached.ROOT/'native_input_verification.json').read_text())
    assert source['passed'] and len(source['samples'])==32
    f=pd.read_parquet(ROOT/'features.parquet').set_index(['date','code']);cases=[]
    for sample in source['samples']:
        date,code=sample['date'],sample['code'];path=MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        assert sha(path)==sample['source_sha256']
        q=pd.read_parquet(path,columns=['timestamp','close','volume','turnover'],filters=[('timestamp','>=',pd.Timestamp(date+' 09:30')),('timestamp','<=',pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert len(q)==230 and q.timestamp.is_unique and q.timestamp.dt.strftime('%H%M').astype(int).tolist()==CLOCKS
        p=q.close.to_numpy(float);assert np.abs(p-p.round(2)).max()<=.0001
        p=p.round(2);vol=q.volume.to_numpy(float);amount=q.turnover.to_numpy(float);row=f.loc[(date,code)]
        np.testing.assert_allclose([math.fsum(vol),math.fsum(amount)],[row.volume_1449,row.amount_1449],rtol=2e-13,atol=2e-6)
        value,guard=native_value(p,vol/100,amount,row.V01)
        if row.vwap_path_valid:
            assert guard==29
            cv=np.cumsum(vol);ca=np.cumsum(amount)
            direct=math.fsum(100*(float(p[i])*float(cv[i])/float(ca[i])-1) for i in range(201,230))/29/row.V01
            np.testing.assert_allclose([value,direct,native_value(p,vol/100,amount,row.V01,.01)[0]],row.VP01,rtol=0,atol=2e-10)
            enc=lambda x:np.floor(np.clip(100*np.asarray(x)+10000+.000001,0,999999))
            np.testing.assert_array_equal(enc([value,direct]),enc([row.VP01,row.VP01]))
        else:assert pd.isna(row.VP01)
        cases.append(dict(date=date,code=code,source_sha256=sample['source_sha256'],minutes=len(q),valid=bool(row.vwap_path_valid)))
    result=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),samples=cases,raw_minutes=7360,
        complete_open_to_1449_prefix_independently_rebuilt=True,actual_expressions_time_boundary_volume_units_and_encodings_verified=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',result);return {k:v for k,v in result.items() if k!='samples'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
