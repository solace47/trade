"""Verify direct market dispersion from cached Q1 members before joint selection."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import freeze_tail_formula_q1_candidates as freezing
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_market_dispersion_q1 as study
from trade_research import tail_formula_path_relative_q1 as previous
from trade_research import tail_formula_q1_candidate_inputs as source_inputs
from trade_research.corporate_cash import save_json, sha

inputs = SimpleNamespace(ROOT=study.ROOT/'inputs')
NAMES = list(study.inputs.EXPRESSIONS)
NEW = ['CD01','CD02']


def checked_sources():
    p, gate = study.checked_models()
    r = json.loads((source_inputs.ROOT/'feature_report.json').read_text())
    for file, digest in r['artifacts_sha256'].items():
        assert sha(source_inputs.ROOT/file) == digest
    proof = json.loads((source_inputs.ROOT/'feature_verification.json').read_text())
    native = json.loads((source_inputs.ROOT/'native_input_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(source_inputs.ROOT/'feature_report.json')
    assert native['passed'] and native['feature_verification_sha256'] == sha(source_inputs.ROOT/'feature_verification.json')
    assert native['feature_report_sha256'] == sha(source_inputs.ROOT/'feature_report.json')
    return p, gate


def features():
    p, gate = checked_sources(); assert not (inputs.ROOT/'feature_report.json').exists()
    old = pd.read_parquet(source_inputs.ROOT/'features.parquet')
    members = pd.read_parquet(source_inputs.ROOT/'members.parquet')
    assert not members.duplicated(['date','code']).any()
    assert members.date.between(p['signal_first'],p['signal_last']).all()
    for field in ['p49','p20','pc']:
        assert members[field].gt(0).all() and np.equal(members[field],np.floor(members[field])).all()
    atoms = pd.DataFrame(dict(date=members.date,CD01=100*(members.p49/members.pc-1),CD02=100*(members.p49/members.p20-1)))
    group = atoms.groupby('date',sort=True)
    daily = group[NEW].std(ddof=0).reset_index()
    daily['members'] = group.size().to_numpy()
    f = old.merge(daily,on='date',how='left',validate='many_to_one').sort_values(['date','code']).reset_index(drop=True)
    good = f.members.ge(p['benchmark_minimum_members']) & np.isfinite(f[NEW]).all(axis=1)
    f[NEW] = f[NEW].where(good)
    f['market_dispersion_input_valid'] = f.formula_input_valid & good & np.isfinite(f[NAMES]).all(axis=1)
    old_names = list(previous.inputs.EXPRESSIONS)
    f['path_relative_input_valid'] = f.formula_input_valid & f.path_input_valid & f.equal_weight_input_valid & np.isfinite(f[old_names]).all(axis=1)
    assert len(f) == p['original_rows'] and f.date.nunique() == p['original_days']
    assert int(f.formula_input_valid.sum()) == p['original_valid']
    inputs.ROOT.mkdir(parents=True,exist_ok=True)
    f.to_parquet(inputs.ROOT/'features.parquet',index=False,compression='zstd')
    daily.to_parquet(inputs.ROOT/'market_reference.parquet',index=False,compression='zstd')
    (inputs.ROOT/'YJCS20.tdx').write_text(study.inputs.HELPER)
    r = dict(protocol_sha256=sha(study.PROTOCOL),models_gate_sha256=sha(study.ROOT/'models_gate.json'),
        source_feature_report_sha256=sha(source_inputs.ROOT/'feature_report.json'),
        source_feature_verification_sha256=sha(source_inputs.ROOT/'feature_verification.json'),
        source_native_verification_sha256=sha(source_inputs.ROOT/'native_input_verification.json'),
        artifacts_sha256={n:sha(inputs.ROOT/n) for n in ['features.parquet','market_reference.parquet','YJCS20.tdx']},
        rows=len(f),valid=int(f.market_dispersion_input_valid.sum()),days=len(daily),minimum_members=int(daily.members.min()),
        newly_invalid=int((f.formula_input_valid & ~f.market_dispersion_input_valid).sum()),
        expressions=study.inputs.EXPRESSIONS,native_header=study.inputs.HEADER,native_core_gate=study.inputs.parent.GATE,
        q1_new_group_outcomes_read=False,no_q2_signal_prices_read=True,no_exit_rules=True)
    save_json(inputs.ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def checked_features():
    p,gate=checked_sources();r=json.loads((inputs.ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(study.PROTOCOL) and r['models_gate_sha256']==sha(study.ROOT/'models_gate.json')
    assert r['source_feature_report_sha256']==sha(source_inputs.ROOT/'feature_report.json')
    for file,digest in r['artifacts_sha256'].items():assert sha(inputs.ROOT/file)==digest
    return p,gate,r,pd.read_parquet(inputs.ROOT/'features.parquet')


def verify_features():
    p,gate,r,f=checked_features();old=pd.read_parquet(source_inputs.ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns],old,check_exact=True)
    c=base.conn();c.read_parquet(str(source_inputs.ROOT/'members.parquet')).create_view('members')
    expected=c.sql("""WITH atoms AS(SELECT date,100*(p49::DOUBLE/pc-1) AS d,
        100*(p49::DOUBLE/p20-1) AS t FROM members)
        SELECT date,count(*) AS members,sqrt(greatest(avg(d*d)-avg(d)*avg(d),0)) AS CD01,
        sqrt(greatest(avg(t*t)-avg(t)*avg(t),0)) AS CD02 FROM atoms GROUP BY date ORDER BY date""").df()
    daily=pd.read_parquet(inputs.ROOT/'market_reference.parquet')
    pd.testing.assert_frame_equal(daily,expected[daily.columns],check_dtype=False,rtol=0,atol=2e-11)
    e=old[['date','code']].merge(expected,on='date',how='left',validate='many_to_one')
    good=e.members.ge(p['benchmark_minimum_members']) & np.isfinite(e[NEW]).all(axis=1)
    for name in NEW:e[name]=e[name].where(good)
    np.testing.assert_allclose(f[NEW],e[NEW],rtol=0,atol=2e-11,equal_nan=True)
    np.testing.assert_array_equal(f.market_dispersion_input_valid,old.formula_input_valid & good)
    oldnames=list(previous.inputs.EXPRESSIONS)
    flags=['formula_input_valid','path_input_valid','equal_weight_input_valid']
    c.register('f',f[['date','code',*flags,*oldnames]])
    finite=' AND '.join(f'isfinite("{n}")' for n in oldnames)
    joint=c.sql('SELECT coalesce('+ ' AND '.join(flags)+' AND '+finite+',false) AS valid FROM f ORDER BY date,code').df()
    np.testing.assert_array_equal(f.path_relative_input_valid,joint.valid);c.close()
    enc=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    valid=f.market_dispersion_input_valid
    np.testing.assert_array_equal(enc(f.loc[valid,NEW]),enc(e.loc[valid,NEW]))
    assert len(expected)==p['original_days']==r['days']
    assert int(valid.sum())==r['valid'] and int((old.formula_input_valid & ~valid).sum())==r['newly_invalid']
    assert r['expressions']==study.inputs.EXPRESSIONS and r['native_header']==study.inputs.HEADER
    assert (inputs.ROOT/'YJCS20.tdx').read_text()==study.inputs.HELPER
    v=dict(passed=True,feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),rows=len(f),valid=int(valid.sum()),
        all_population_variances_members_mappings_validities_and_new_encodings_rebuilt=True,
        original_52_values_and_keys_unchanged=True,effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        q1_new_group_outcomes_read=False,no_q2_signal_prices_read=True,no_exit_rules=True)
    save_json(inputs.ROOT/'feature_verification.json',v);return v


def native():
    p,gate,r,f=checked_features();v=json.loads((inputs.ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')
    source=json.loads((source_inputs.ROOT/'native_input_verification.json').read_text());assert len(source['samples'])==24
    members=pd.read_parquet(source_inputs.ROOT/'members.parquet',columns=['date','code','p49','p20','pc'])
    f=f.set_index(['date','code']);cases=[]
    for sample in source['samples']:
        d=members.loc[members.date.eq(sample['date'])];values=[]
        for col in ['pc','p20']:
            atoms=[100*(a/b-1) for a,b in zip(d.p49,d[col])];mean=math.fsum(atoms)/len(atoms)
            values.append(math.sqrt(max(math.fsum(a*a for a in atoms)/len(atoms)-mean*mean,0)))
        actual=f.loc[(sample['date'],sample['code']),NEW].to_numpy(float)
        np.testing.assert_allclose(values,actual,rtol=0,atol=2e-11)
        np.testing.assert_array_equal(np.floor(100*np.array(values)+10000+.000001),np.floor(100*actual+10000+.000001))
        cases.append(dict(date=sample['date'],code=sample['code'],members=len(d),CD01=values[0],CD02=values[1]))
    result=dict(passed=True,feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),
        feature_verification_sha256=sha(inputs.ROOT/'feature_verification.json'),
        source_native_proof_sha256=sha(source_inputs.ROOT/'native_input_verification.json'),samples=cases,
        all_native_aggregates_and_encodings_rebuilt=True,source_raw_quote_proofs_reused=True,
        native_source_parity_verified=False,software_compilation_verified=False,
        q1_new_group_outcomes_read=False,no_q2_signal_prices_read=True,no_exit_rules=True)
    save_json(inputs.ROOT/'native_input_verification.json',result);return {k:v for k,v in result.items() if k!='samples'}


def checked_inputs():
    p,gate,r,f=checked_features()
    v=json.loads((inputs.ROOT/'feature_verification.json').read_text());n=json.loads((inputs.ROOT/'native_input_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')
    assert n['passed'] and n['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')
    assert n['feature_verification_sha256']==sha(inputs.ROOT/'feature_verification.json')
    return p,gate,f


def scores():
    p, gate, f = checked_inputs(); assert not (study.ROOT/'score_report.json').exists()
    old = json.loads((previous.ROOT/'score_report.json').read_text())
    proof = json.loads((previous.ROOT/'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256'] == sha(previous.ROOT/'score_report.json')
    records = {}
    for variant in gate['active_variants']:
        folder = study.ROOT/variant; folder.mkdir(exist_ok=True)
        if variant != 'market_dispersion':
            source = previous.ROOT/variant/'scores.parquet'
            assert old['variants'][variant]['scores_sha256'] == sha(source)
            (folder/'scores.parquet').symlink_to(source.resolve())
            records[variant] = dict(old['variants'][variant], reused=True)
            continue
        m = json.loads((Path(gate['models'][variant]['root'])/'model_report.json').read_text())
        valid = f.market_dispersion_input_valid
        x = np.floor(np.clip(100*f.loc[valid,NAMES].to_numpy(float)+10000+.000001,0,999999)).astype('int32')
        out = f[freezing.COLUMNS].copy(); out['formula_input_valid'] = valid; out['score'] = np.nan
        out.loc[valid,'score'] = base.predict(x,m)
        out.to_parquet(folder/'scores.parquet',index=False,compression='zstd')
        records[variant] = dict(scores_sha256=sha(folder/'scores.parquet'), valid=int(valid.sum()),reused=False)
    r = dict(protocol_sha256=sha(study.PROTOCOL), models_gate_sha256=sha(study.ROOT/'models_gate.json'),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),
        native_verification_sha256=sha(inputs.ROOT/'native_input_verification.json'),
        joint_input_verification_sha256=sha(inputs.ROOT/'feature_verification.json'),
        variants=records,rows=len(f),q1_new_group_outcomes_read=False,no_q2_signal_prices_read=True,no_exit_rules=True)
    save_json(study.ROOT/'score_report.json',r);return r


def setup_freezing():
    freezing.study = study; freezing.ROOT = study.ROOT; freezing.checked_inputs = checked_inputs; freezing.inputs = inputs


def verify_selections():
    setup_freezing(); r = freezing.verify()
    for variant in ['control','path','equal_weight','path_relative']:
        pd.testing.assert_frame_equal(pd.read_parquet(study.ROOT/variant/'selection.parquet'),
            pd.read_parquet(previous.ROOT/variant/'selection.parquet'),check_exact=True)
    r['all_four_original_selections_exactly_reused'] = True
    save_json(study.ROOT/'selection_verification.json',r); return r


def coverage():
    p, gate = study.checked_models(); out = study.ROOT/'label_coverage_verification.json'; assert not out.exists()
    sr = json.loads((study.ROOT/'selection_report.json').read_text())
    sv = json.loads((study.ROOT/'selection_verification.json').read_text())
    assert sv['passed'] and sv['selection_report_sha256'] == sha(study.ROOT/'selection_report.json')
    source = previous.ROOT/'before1000'
    lr = json.loads((source/'full_label_report.json').read_text())
    lv = json.loads((source/'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256'] == sha(source/'full_label_report.json')
    assert lr['labels_sha256'] == sha(source/'full_labels.parquet')
    labels = pd.read_parquet(source/'full_labels.parquet',columns=['date','code'])
    universe = pd.read_parquet(source_inputs.OLD/'universe.parquet',columns=['date','code'])
    needed = universe.loc[universe.date.isin(sr['union_selected_dates'])].sort_values(['date','code']).reset_index(drop=True)
    seen = needed.merge(labels.assign(cached=True),on=['date','code'],how='left',validate='one_to_one')
    missing = seen.loc[~seen.cached.eq(True),['date','code']]
    r = dict(passed=True,protocol_sha256=sha(study.PROTOCOL),selection_verification_sha256=sha(study.ROOT/'selection_verification.json'),
        cached_label_report_sha256=sha(source/'full_label_report.json'),needed_rows=len(needed),needed_days=needed.date.nunique(),
        cached_rows=len(labels),cached_days=labels.date.nunique(),missing_rows=len(missing),missing_dates=sorted(missing.date.unique()),
        full_base_not_only_selected_stocks=True,all_needed_labels_cached=missing.empty,
        no_rows_or_dates_dropped=True,q1_new_group_outcomes_read=False,no_q2_signal_prices_read=True,no_exit_rules=True)
    save_json(out,r)
    if missing.empty:
        dest = study.ROOT/'before1000';dest.mkdir(exist_ok=True)
        for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
            (dest/name).symlink_to((source/name).resolve())
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','verify_features','native','scores','verify_scores','freeze','verify','coverage'])
    stage = parser.parse_args().stage
    if stage in ['verify_scores','freeze']:
        setup_freezing(); result = getattr(freezing,stage)()
    else:
        result = verify_selections() if stage == 'verify' else globals()[stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
