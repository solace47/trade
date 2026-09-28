"""Current main-board cross-sectional dispersion as relative-strength scale."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_equal_weight as equal
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM='tail_formula_cross_dispersion'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_protocol.json')
EXPRESSIONS={**previous.EXPRESSIONS,'CZ01':'(A01-CZD)/MAX(CZDS,0.01)','CZ02':'(A05-CZT)/MAX(CZTS,0.01)'}
HEADER=previous.HEADER+"""CZN:=INSUM('沪深Ａ股','YJCS20',1,0);
CZD:=INSUM('沪深Ａ股','YJCS20',2,0)/MAX(CZN,1);
CZT:=INSUM('沪深Ａ股','YJCS20',3,0)/MAX(CZN,1);
CZDS:=SQRT(MAX(INSUM('沪深Ａ股','YJCS20',4,0)/MAX(CZN,1)-CZD*CZD,0));
CZTS:=SQRT(MAX(INSUM('沪深Ａ股','YJCS20',5,0)/MAX(CZN,1)-CZT*CZT,0));
"""
HELPER=equal.members_source.HELPER.split('NB:')[0]+"""ZDR:=100*(P49/MAX(PC,1)-1);
ZTR:=100*(P49/MAX(P20,1)-1);
NB:IF(OK,1,0);
DR:IF(OK,ZDR,0);
TR:IF(OK,ZTR,0);
DQ:IF(OK,ZDR*ZDR,0);
TQ:IF(OK,ZTR*ZTR,0);
"""
GATE='CZN>=2000'
ORIGINAL_NATIVE_CORE=base.native_core


def native_core(*args,**kwargs):
    text=ORIGINAL_NATIVE_CORE(*args,**kwargs)
    assert text.count('CORE:SC>')==1
    return text.replace('CORE:SC>','CORE:'+GATE+' AND SC>')


def standardized(value,mean,std):
    return (value-mean)/np.maximum(std,.01)


def checked_sources():
    p=json.loads(PROTOCOL.read_text())
    assert p['minimum_members']==2000 and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    equal.checked_sources()
    r=json.loads((equal.ROOT/'feature_report.json').read_text())
    assert r['features_sha256']==sha(equal.ROOT/'features.parquet')
    assert r['market_reference_sha256']==sha(equal.ROOT/'market_reference.parquet')
    return p


def features():
    checked_sources();assert not (ROOT/'feature_report.json').exists()
    m=pd.read_parquet(equal.members_source.ROOT/'members.parquet')
    m['day_atom']=100*(m.p49-m.pc)/m.pc;m['tail_atom']=100*(m.p49-m.p20)/m.p20
    assert np.isfinite(m[['day_atom','tail_atom']]).all().all()
    daily=m.groupby('date',sort=True).agg(cz_members=('code','size'),cz_day_mean=('day_atom','mean'),
        cz_tail_mean=('tail_atom','mean'),cz_day_std=('day_atom',lambda x:x.std(ddof=0)),
        cz_tail_std=('tail_atom',lambda x:x.std(ddof=0))).reset_index()
    coverage=pd.read_parquet(equal.ROOT/'market_reference.parquet',columns=['date','expected_local_members','missing_local_members'])
    daily=daily.merge(coverage,on='date',validate='one_to_one')
    old=pd.read_parquet(previous.ROOT/'features.parquet')
    f=old.merge(daily,on='date',how='left',validate='many_to_one').sort_values(['date','code']).reset_index(drop=True)
    valid=f.cz_members.ge(2000)&np.isfinite(f[['cz_day_mean','cz_tail_mean','cz_day_std','cz_tail_std']]).all(axis=1)
    f['CZ01']=standardized(f.A01,f.cz_day_mean,f.cz_day_std).where(valid)
    f['CZ02']=standardized(f.A05,f.cz_tail_mean,f.cz_tail_std).where(valid)
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);daily.to_parquet(ROOT/'market_reference.parquet',index=False,compression='zstd')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd');(ROOT/'YJCS20.tdx').write_text(HELPER)
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(ROOT/'features.parquet'),
        market_reference_sha256=sha(ROOT/'market_reference.parquet'),helper_sha256=sha(ROOT/'YJCS20.tdx'),
        source_hashes=json.loads(PROTOCOL.read_text())['source_hashes'],members_sha256=sha(equal.members_source.ROOT/'members.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),newly_invalid=int((old.formula_input_valid&~f.formula_input_valid).sum()),
        member_rows=len(m),days=len(daily),min_members=int(daily.cz_members.min()),
        day_scale_floor_days=int(daily.cz_day_std.lt(.01).sum()),tail_scale_floor_days=int(daily.cz_tail_std.lt(.01).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,native_core_gate=GATE,native_auxiliary_indicator_required=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_group_outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    checked_sources();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    for key,file in [('features','features.parquet'),('market_reference','market_reference.parquet'),('helper','YJCS20.tdx')]:
        assert r[key+'_sha256']==sha(ROOT/file)
    members=pd.read_parquet(equal.members_source.ROOT/'members.parquet')
    c=base.conn();c.register('members',members[['date','code','p49','p20','pc']])
    c.sql('''SELECT date,code,100*(p49::DOUBLE/pc-1) AS d,100*(p49::DOUBLE/p20-1) AS t FROM members''').create_view('atoms')
    stats=c.sql('''SELECT date,count(*) AS cz_members,avg(d) AS cz_day_mean,avg(t) AS cz_tail_mean,
        sqrt(greatest(avg(d*d)-avg(d)*avg(d),0)) AS cz_day_std,
        sqrt(greatest(avg(t*t)-avg(t)*avg(t),0)) AS cz_tail_std FROM atoms GROUP BY date ORDER BY date''').df()
    daily=pd.read_parquet(ROOT/'market_reference.parquet')
    pd.testing.assert_frame_equal(daily[stats.columns],stats,check_dtype=False,rtol=0,atol=2e-11)
    old_eq=pd.read_parquet(equal.ROOT/'market_reference.parquet')
    pd.testing.assert_frame_equal(daily[['date','expected_local_members','missing_local_members']],
        old_eq[['date','expected_local_members','missing_local_members']],check_exact=True)
    for new,old in [('cz_members','ew_members'),('cz_day_mean','ew_day_mean'),('cz_tail_mean','ew_tail_mean')]:
        np.testing.assert_allclose(daily[new],old_eq[old],rtol=0,atol=2e-12)
    old=pd.read_parquet(previous.ROOT/'features.parquet');got=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    c.register('stats',stats);c.register('old',old[['date','code']])
    ex=c.sql('''SELECT old.date,old.code,
        CASE WHEN cz_members>=2000 THEN (d-cz_day_mean)/greatest(cz_day_std,.01) END AS CZ01,
        CASE WHEN cz_members>=2000 THEN (t-cz_tail_mean)/greatest(cz_tail_std,.01) END AS CZ02
        FROM old LEFT JOIN atoms USING(date,code) LEFT JOIN stats USING(date) ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(got[['date','code']],ex[['date','code']],check_exact=True)
    for name in ['CZ01','CZ02']:
        np.testing.assert_allclose(got[name],ex[name],rtol=0,atol=2e-9,equal_nan=True)
    valid=old.formula_input_valid & np.isfinite(ex[['CZ01','CZ02']]).all(axis=1)
    np.testing.assert_array_equal(got.formula_input_valid,valid)
    eq_valid=pd.read_parquet(equal.ROOT/'features.parquet',columns=['date','code','formula_input_valid'])
    pd.testing.assert_frame_equal(got[eq_valid.columns],eq_valid,check_exact=True)
    encode=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    np.testing.assert_array_equal(encode(got.loc[valid,['CZ01','CZ02']]),encode(ex.loc[valid,['CZ01','CZ02']]))
    assert (ROOT/'YJCS20.tdx').read_text()==HELPER
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names)==len(set(names)) and r['expressions']==EXPRESSIONS and r['native_header']==HEADER
    assert len(got)==r['rows'] and valid.sum()==r['valid']
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(got),
        all_member_prices_population_variances_stock_components_and_encodings_independently_rebuilt=True,
        all_previous_inputs_keys_and_coverage_unchanged=True,equal_weight_control_effective_intersection_unchanged=True,
        original48_effective_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        native_helper_arithmetic_verified=True,source_coverage_proof_reused=True,new_group_outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof);return proof


def native():
    checked_sources();v=json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(ROOT/'feature_report.json')
    old=json.loads((equal.ROOT/'native_input_verification.json').read_text())
    assert old['passed'] and old['feature_report_sha256']==sha(equal.ROOT/'feature_report.json')
    f=pd.read_parquet(ROOT/'features.parquet').set_index(['date','code']);receipts=[]
    for sample in old['samples']:
        row=f.loc[(sample['date'],sample['code'])];n,d,t=sample['helper_outputs'];assert n==1
        value=np.array([(d-row.cz_day_mean)/max(row.cz_day_std,.01),(t-row.cz_tail_mean)/max(row.cz_tail_std,.01)])
        np.testing.assert_allclose(value,[row.CZ01,row.CZ02],rtol=0,atol=2e-9)
        enc=lambda x:np.floor(np.clip(100*np.asarray(x)+10000+.000001,0,999999))
        np.testing.assert_array_equal(enc(value),enc([row.CZ01,row.CZ02]))
        receipts.append(dict(date=sample['date'],code=sample['code'],source_sha256=sample['source_sha256'],
            helper_outputs=[1,d,t,d*d,t*t],new_raw_extraction=False))
    assert len(receipts)==32
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),samples=receipts,
        reused_raw_sample_report_sha256=sha(equal.ROOT/'native_input_verification.json'),
        fixed_samples=32,original_raw_minutes_verified_in_source=old['raw_minutes'],new_raw_minutes_extracted=0,
        all_new_helper_square_outputs_scales_and_encoding_verified=True,
        full_client_member_and_history_guards_verified=False,software_compilation_verified=False,
        native_source_parity_verified=False,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',proof);return {k:v for k,v in proof.items() if k!='samples'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
