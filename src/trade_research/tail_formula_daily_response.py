"""Prior complete-day return magnitudes interacting with complete daily volume."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_daily_efficiency as prices
from . import tail_formula_history_direction as volumes
from .corporate_cash import save_json, sha

STEM='tail_formula_daily_response'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_protocol.json')
VOLUMES=[f'dr_v{i:02d}' for i in range(1,21)]
HISTORY=['hd_rows20','hd_good20','hd_first_date','hd_last_date','hd_reference_date','hd_volume20',
    'hd_reference_breaks','hd_market_span','hd_last_gap','history_direction_valid']
NEW_EXPRESSIONS={'DR01':'(20*DRPV/MAX(DRV,0.000000000001)-DRR)/V01'}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER=volumes.HEADER+''.join(f'DRR{i:02d}:=100*(DCP{i}/DCP{i+1}-1);\n' for i in range(1,21))
HEADER+='DRV:='+ '+'.join(f'HDV{i:02d}' for i in range(1,21))+';\n'
HEADER+='DRR:='+ '+'.join(f'DRR{i:02d}' for i in range(1,21))+';\n'
HEADER+='DRPV:='+ '+'.join(f'DRR{i:02d}*HDV{i:02d}' for i in range(1,21))+';\n'
ORIGINAL_NATIVE_CORE=base.native_core


def native_core(*args,**kwargs):
    text=ORIGINAL_NATIVE_CORE(*args,**kwargs);assert text.count('CORE:SC>')==1
    return text.replace('CORE:SC>','CORE:DRV>0 AND SC>')


def checked_sources():
    p=json.loads(PROTOCOL.read_text())
    for file,digest in p['source_hashes'].items(): assert sha(Path(file))==digest
    for root in [previous.ROOT,prices.ROOT,volumes.ROOT]:
        r=json.loads((root/'feature_report.json').read_text());v=json.loads((root/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(root/'feature_report.json')
        assert r['features_sha256']==sha(root/'features.parquet')
    sources=json.loads(prices.DAILY_REPORT.read_text())['source_sha256']
    for file,digest in sources.items():assert sha(Path(file))==digest
    assert p['history_days']==20 and not p['new_2026_prices_allowed']
    return p,list(sources)


def response(closes,volume,atr):
    closes=np.asarray(closes,float);volume=np.asarray(volume,float);atr=np.asarray(atr,float)
    assert closes.shape==(len(volume),21) and volume.shape[1]==20
    cents=np.rint(closes*100)
    good=np.isfinite(closes).all(axis=1)&(closes>0).all(axis=1)&(np.abs(closes*100-cents)<=1e-6).all(axis=1)
    good&=np.isfinite(volume).all(axis=1)&(volume>=0).all(axis=1)&(volume==np.floor(volume)).all(axis=1)
    total=volume.sum(axis=1);good&=(total>0)&np.isfinite(atr)&(atr>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        r=100*(cents[:,:-1]/cents[:,1:]-1)
        value=(20*(r*volume).sum(axis=1)/total-r.sum(axis=1))/atr
    return np.where(good,value,np.nan)


def features():
    p,files=checked_sources();assert not (ROOT/'feature_report.json').exists()
    old=pd.read_parquet(previous.ROOT/'features.parquet')
    price=pd.read_parquet(prices.ROOT/'features.parquet',columns=['date','code',*prices.PRICE_COLUMNS,'daily_efficiency_valid'])
    history=pd.read_parquet(volumes.ROOT/'features.parquet',columns=['date','code',*HISTORY])
    c=base.conn();c.read_parquet(files).create_view('daily')
    fields=','.join(f'lag(volume::DOUBLE,{i}) OVER w AS dr_v{i:02d}' for i in range(1,21))
    d=c.sql(f"""WITH h AS(SELECT date,code,{fields} FROM daily
        WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'
        WINDOW w AS(PARTITION BY code ORDER BY date)) SELECT * FROM h WHERE date>='2024-01-01' ORDER BY date,code""").df();c.close()
    f=old.merge(price,on=['date','code'],validate='one_to_one',how='left').merge(history,on=['date','code'],validate='one_to_one',how='left').merge(d,on=['date','code'],validate='one_to_one',how='left').sort_values(['date','code']).reset_index(drop=True)
    value=response(f[prices.PRICE_COLUMNS].to_numpy(float),f[VOLUMES].to_numpy(float),f.V01.to_numpy(float))
    valid=f.daily_efficiency_valid & f.history_direction_valid & np.isfinite(value)
    f['DR01']=pd.Series(value).where(valid);f['daily_response_valid']=valid
    f['prior_formula_input_valid']=f.formula_input_valid;f['formula_input_valid']&=valid
    np.testing.assert_array_equal(f.loc[valid,VOLUMES].sum(axis=1),f.loc[valid,'hd_volume20'])
    assert (f.loc[valid,'hd_reference_date']<f.loc[valid,'hd_first_date']).all() and (f.loc[valid,'hd_last_date']<f.loc[valid,'date']).all()
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),source_hashes=p['source_hashes'],features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.hd_reference_breaks.gt(0)).sum()),
        valid_with_market_gaps=int((f.formula_input_valid & (f.hd_market_span.gt(20)|f.hd_last_gap.gt(1))).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,native_core_gate='DRV>0',
        native_source_parity_verified=False,software_compilation_verified=False,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['source_hashes','expressions','native_header']}


def verify_features():
    p,files=checked_sources();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(ROOT/'features.parquet')
    old=pd.read_parquet(previous.ROOT/'features.parquet');f=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid,old.formula_input_valid,check_names=False,check_exact=True)
    c=base.conn();c.read_parquet(files).create_view('daily')
    c.sql("""SELECT date,code,close::DOUBLE AS raw,round(close::DOUBLE,2) AS cl,
        volume::DOUBLE AS vol,adjustflag::DOUBLE AS adj FROM daily
        WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'""").create_view('active')
    c.sql('''SELECT *,lag(cl) OVER w AS pc,lag(raw) OVER w AS pr,lag(adj) OVER w AS pa,
        lag(date) OVER w AS pd FROM active WINDOW w AS(PARTITION BY code ORDER BY date)''').create_view('lagged')
    c.sql('''SELECT *,100*(round(cl*100)/round(pc*100)-1) AS ret,
        coalesce(isfinite(raw) AND raw>0 AND abs(raw-cl)<=.0001 AND isfinite(pr) AND pr>0 AND abs(pr-pc)<=.0001
        AND adj=3 AND pa=3 AND isfinite(vol) AND vol>=0 AND vol=floor(vol),false) AS good FROM lagged''').create_view('atoms')
    fields=','.join(f'lag(vol,{i}) OVER seq AS dr_v{i:02d}' for i in range(1,21))
    h=c.sql(f'''SELECT date,code,{fields},count(*) OVER w AS n,sum(good::INT) OVER w AS good,
        sum(ret) OVER w AS sr,sum(ret*vol) OVER w AS srv,sum(vol) OVER w AS sv,sum(vol) OVER w5 AS sv5,
        min(date) OVER w AS first,max(date) OVER w AS last,first_value(pd) OVER w AS reference
        FROM atoms WINDOW seq AS(PARTITION BY code ORDER BY date),
        w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        w5 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)''').df();c.close()
    e=old[['date','code','V01','formula_input_valid']].merge(h,on=['date','code'],validate='one_to_one',how='left')
    pd.testing.assert_frame_equal(f[VOLUMES],e[VOLUMES],check_dtype=False,check_exact=True)
    good=e.n.eq(20)&e.good.eq(20)&e.sv.gt(0)&e.sv5.gt(0)&e.reference.lt(e['first'])&e['last'].lt(e.date)&np.isfinite(e.V01)&e.V01.gt(0)
    expected=((20*e.srv/e.sv-e.sr)/e.V01).where(good)
    np.testing.assert_array_equal(f.daily_response_valid,good)
    np.testing.assert_allclose(f.DR01,expected,rtol=0,atol=2e-10,equal_nan=True)
    final=old.formula_input_valid & good
    np.testing.assert_array_equal(f.formula_input_valid,final)
    np.testing.assert_array_equal(np.floor(np.clip(100*f.loc[final,'DR01']+10000+.000001,0,999999)),np.floor(np.clip(100*expected.loc[final]+10000+.000001,0,999999)))
    pd.testing.assert_series_equal(f.loc[final,'hd_first_date'],e.loc[final,'first'],check_names=False)
    pd.testing.assert_series_equal(f.loc[final,'hd_last_date'],e.loc[final,'last'],check_names=False)
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS);assert len(names)==len({n.casefold() for n in names})
    assert r['expressions']==EXPRESSIONS and r['native_header']==HEADER
    assert int(final.sum())==r['valid'] and int((old.formula_input_valid & ~final).sum())==r['newly_invalid']
    v=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),valid=int(final.sum()),
        all_previous_day_volume_columns_and_weighted_returns_sql_rebuilt=True,strict_history_dates_verified=True,
        all_original_48_values_keys_and_new_encodings_checked=True,effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',v);return v


def native_value(closes,lots,atr):
    env={'MAX':max,'V01':float(atr)}
    for i,x in enumerate(closes,1):env[f'DCP{i}']=float(x)
    for i,x in enumerate(lots,1):env[f'HDV{i:02d}']=float(x)
    for line in HEADER[len(volumes.HEADER):].strip().split(';'):
        if line.strip():
            key,expr=line.strip().split(':=');env[key]=eval(expr,{'__builtins__':{}},env)
    return float(eval(NEW_EXPRESSIONS['DR01'],{'__builtins__':{}},env))


def native():
    checked_sources();v=json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(ROOT/'feature_report.json')
    proof=json.loads((volumes.ROOT/'native_input_verification.json').read_text())
    assert proof['passed'] and proof['samples']==32 and proof['feature_verification_sha256']==sha(volumes.ROOT/'feature_verification.json')
    f=pd.read_parquet(ROOT/'features.parquet').set_index(['date','code']);cases=[]
    for case in proof['cases']:
        row=f.loc[(case['date'],case['code'])];valid=bool(row.daily_response_valid)
        if valid:
            cl=row[prices.PRICE_COLUMNS].to_numpy(float);vol=row[VOLUMES].to_numpy(float)
            value=native_value(cl,vol/100,row.V01)
            rates=[100*(round(cl[i]*100)/round(cl[i+1]*100)-1) for i in range(20)]
            scalar=(20*math.fsum(a*b for a,b in zip(rates,vol))/math.fsum(vol)-math.fsum(rates))/row.V01
            np.testing.assert_allclose([value,scalar,native_value(cl,vol,row.V01)],row.DR01,rtol=0,atol=2e-10)
            np.testing.assert_array_equal(np.floor(np.clip(100*np.array([value,scalar])+10000+.000001,0,999999)),np.repeat(np.floor(np.clip(100*row.DR01+10000+.000001,0,999999)),2))
        else:
            assert pd.isna(row.DR01)
        cases.append(dict(date=case['date'],code=case['code'],valid=valid,prior_day_source_proof_reused=True))
    result=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),prior_day_native_proof_sha256=sha(volumes.ROOT/'native_input_verification.json'),
        cases=cases,source_cases_reused=32,actual_new_expressions_and_integer_encodings_rebuilt=True,
        no_new_minute_extraction=True,full_native_source_parity_verified=False,software_compilation_verified=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',result);return {k:v for k,v in result.items() if k!='cases'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
