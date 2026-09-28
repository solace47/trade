"""Reuse audited Q1 inputs and controls for the separately frozen joint model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import freeze_tail_formula_q1_candidates as freezing
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_path_relative_q1 as study
from trade_research import tail_formula_q1_candidates as previous
from trade_research import tail_formula_q1_candidate_inputs as inputs
from trade_research.corporate_cash import save_json, sha

NAMES = list(study.inputs.EXPRESSIONS)


def cached_inputs():
    p, gate = study.checked_models()
    r = json.loads((inputs.ROOT/'feature_report.json').read_text())
    for file, digest in r['artifacts_sha256'].items():
        assert sha(inputs.ROOT/file) == digest
    proof = json.loads((inputs.ROOT/'feature_verification.json').read_text())
    native = json.loads((inputs.ROOT/'native_input_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
    assert native['passed'] and native['feature_verification_sha256'] == sha(inputs.ROOT/'feature_verification.json')
    assert native['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
    f = pd.read_parquet(inputs.ROOT/'features.parquet')
    assert len(f) == p['original_rows'] and f.date.nunique() == p['original_days']
    assert int(f.formula_input_valid.sum()) == p['original_valid']
    assert f.date.between(p['signal_first'],p['signal_last']).all()
    f['path_relative_input_valid'] = (f.formula_input_valid & f.path_input_valid & f.equal_weight_input_valid
        & np.isfinite(f[NAMES]).all(axis=1))
    return p, gate, f


def verify_inputs():
    p, gate, f = cached_inputs(); out = study.ROOT/'input_reuse_verification.json'; assert not out.exists()
    c = base.conn()
    flags = ['formula_input_valid','path_input_valid','equal_weight_input_valid']
    c.register('f', f[['date','code',*flags,*NAMES]])
    finite = ' AND '.join(f'isfinite("{n}")' for n in NAMES)
    expected = c.sql('SELECT date,code,coalesce('+ ' AND '.join(flags)+' AND '+finite+
        ',false) AS valid FROM f ORDER BY date,code').df()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.path_relative_input_valid, expected.valid)
    fields = ','.join(f'floor(least(greatest(100*"{n}"+10000+.000001,0),999999))::INT AS "{n}"' for n in NAMES)
    actual = c.sql('SELECT '+fields+' FROM f WHERE '+ ' AND '.join(flags)+' AND '+finite+' ORDER BY date,code').df()
    c.close()
    np.testing.assert_array_equal(actual.to_numpy(), np.floor(np.clip(
        100*f.loc[f.path_relative_input_valid,NAMES].to_numpy(float)+10000+.000001,0,999999)).astype('int32'))
    r = dict(passed=True, protocol_sha256=sha(study.PROTOCOL),models_gate_sha256=sha(study.ROOT/'models_gate.json'),
        source_feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),
        source_feature_verification_sha256=sha(inputs.ROOT/'feature_verification.json'),
        source_native_verification_sha256=sha(inputs.ROOT/'native_input_verification.json'),
        rows=len(f), valid=int(f.path_relative_input_valid.sum()),
        newly_invalid=int((f.formula_input_valid & ~f.path_relative_input_valid).sum()),
        all_52_encodings_and_joint_validity_sql_rebuilt=True, cached_values_unchanged=True,
        fixed_24_source_samples_reused=True, no_new_raw_extraction=True,
        q1_new_group_outcomes_read=False, native_source_parity_verified=False,
        no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(out,r); return r


def checked_inputs():
    p, gate, f = cached_inputs()
    v = json.loads((study.ROOT/'input_reuse_verification.json').read_text())
    assert v['passed'] and v['models_gate_sha256'] == sha(study.ROOT/'models_gate.json')
    assert v['source_feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
    assert v['valid'] == int(f.path_relative_input_valid.sum())
    return p, gate, f


def scores():
    p, gate, f = checked_inputs(); assert not (study.ROOT/'score_report.json').exists()
    old = json.loads((previous.ROOT/'score_report.json').read_text())
    proof = json.loads((previous.ROOT/'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256'] == sha(previous.ROOT/'score_report.json')
    records = {}
    for variant in gate['active_variants']:
        folder = study.ROOT/variant; folder.mkdir(exist_ok=True)
        if variant != 'path_relative':
            source = previous.ROOT/variant/'scores.parquet'
            assert old['variants'][variant]['scores_sha256'] == sha(source)
            (folder/'scores.parquet').symlink_to(source.resolve())
            records[variant] = dict(old['variants'][variant], reused=True)
            continue
        m = json.loads((Path(gate['models'][variant]['root'])/'model_report.json').read_text())
        valid = f.path_relative_input_valid
        x = np.floor(np.clip(100*f.loc[valid,NAMES].to_numpy(float)+10000+.000001,0,999999)).astype('int32')
        out = f[freezing.COLUMNS].copy(); out['formula_input_valid'] = valid; out['score'] = np.nan
        out.loc[valid,'score'] = base.predict(x,m)
        out.to_parquet(folder/'scores.parquet',index=False,compression='zstd')
        records[variant] = dict(scores_sha256=sha(folder/'scores.parquet'), valid=int(valid.sum()),reused=False)
    r = dict(protocol_sha256=sha(study.PROTOCOL), models_gate_sha256=sha(study.ROOT/'models_gate.json'),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),
        native_verification_sha256=sha(inputs.ROOT/'native_input_verification.json'),
        joint_input_verification_sha256=sha(study.ROOT/'input_reuse_verification.json'),
        variants=records,rows=len(f),q1_new_group_outcomes_read=False,no_q2_signal_prices_read=True,no_exit_rules=True)
    save_json(study.ROOT/'score_report.json',r);return r


def setup_freezing():
    freezing.study = study; freezing.ROOT = study.ROOT; freezing.checked_inputs = checked_inputs


def verify_selections():
    setup_freezing(); r = freezing.verify()
    for variant in ['control','path','equal_weight']:
        pd.testing.assert_frame_equal(pd.read_parquet(study.ROOT/variant/'selection.parquet'),
            pd.read_parquet(previous.ROOT/variant/'selection.parquet'),check_exact=True)
    r['all_three_original_selections_exactly_reused'] = True
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
    universe = pd.read_parquet(inputs.OLD/'universe.parquet',columns=['date','code'])
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
    parser.add_argument('stage',choices=['verify_inputs','scores','verify_scores','freeze','verify','coverage'])
    stage = parser.parse_args().stage
    if stage in ['verify_scores','freeze']:
        setup_freezing(); result = getattr(freezing,stage)()
    else:
        result = verify_selections() if stage == 'verify' else globals()[stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
