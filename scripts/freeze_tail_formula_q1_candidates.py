"""Freeze every predeclared Q1 candidate and its matched input-quality control."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_tail_formula_additive import tree_sql
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_q1_candidates as study
from trade_research import tail_formula_q1_candidate_inputs as inputs
from trade_research.corporate_cash import save_json, sha

ROOT = study.ROOT
COLUMNS = ['date', 'code', 'half', 'board', 'decision_shares']


def checked_inputs():
    p, gate = study.checked_models()
    r = json.loads((inputs.ROOT/'feature_report.json').read_text())
    for file, digest in r['artifacts_sha256'].items():
        assert sha(inputs.ROOT/file) == digest
    proof = json.loads((inputs.ROOT/'feature_verification.json').read_text())
    native = json.loads((inputs.ROOT/'native_input_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
    assert native['passed'] and native['feature_verification_sha256'] == sha(inputs.ROOT/'feature_verification.json')
    assert native['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
    return p, gate, pd.read_parquet(inputs.ROOT/'features.parquet')


def valid_column(variant):
    return 'formula_input_valid' if variant == 'control' else variant+'_input_valid'


def scores():
    p, gate, f = checked_inputs(); assert not (ROOT/'score_report.json').exists()
    records = {}
    for variant in gate['active_variants']:
        folder = ROOT/variant; folder.mkdir(exist_ok=True)
        m = json.loads((Path(gate['models'][variant]['root'])/'model_report.json').read_text())
        valid = f[valid_column(variant)]
        x = np.floor(np.clip(100*f.loc[valid, m['feature_names']].to_numpy(float)+10000+.000001, 0, 999999)).astype('int32')
        out = f[COLUMNS].copy(); out['formula_input_valid'] = valid; out['score'] = np.nan
        out.loc[valid, 'score'] = base.predict(x, m)
        out.to_parquet(folder/'scores.parquet', index=False, compression='zstd')
        records[variant] = dict(scores_sha256=sha(folder/'scores.parquet'), valid=int(valid.sum()))
    result = dict(protocol_sha256=sha(study.PROTOCOL), models_gate_sha256=sha(ROOT/'models_gate.json'),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'), native_verification_sha256=sha(inputs.ROOT/'native_input_verification.json'),
        variants=records, rows=len(f), q1_new_group_outcomes_read=False, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'score_report.json', result); return result


def verify_scores():
    p, gate, f = checked_inputs(); report = json.loads((ROOT/'score_report.json').read_text())
    assert report['protocol_sha256'] == sha(study.PROTOCOL) and report['models_gate_sha256'] == sha(ROOT/'models_gate.json')
    assert report['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
    checked = {}
    for variant in gate['active_variants']:
        folder = ROOT/variant; meta = gate['models'][variant]
        assert report['variants'][variant]['scores_sha256'] == sha(folder/'scores.parquet')
        m = json.loads((Path(meta['root'])/'model_report.json').read_text())
        names = m['feature_names']; valid = valid_column(variant)
        # DuckDB identifiers ignore case: project exactly the required names.
        c = base.conn(); c.register('f', f[COLUMNS+[valid]+names])
        encode = ','.join(f'floor(least(greatest(100*"{name}"+10000+.000001,0),999999))::INT AS X{i:02d}' for i,name in enumerate(names,1))
        c.sql('SELECT date,code,'+encode+' FROM f WHERE '+valid).create_view('encoded')
        formula = format(m['bias'], '.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
        c.sql('SELECT date,code,'+formula+' AS score FROM encoded').create_view('rebuilt')
        expected = c.sql('SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.'+valid+
            ' AS formula_input_valid,r.score FROM f LEFT JOIN rebuilt r USING(date,code) ORDER BY date,code').df(); c.close()
        actual = pd.read_parquet(folder/'scores.parquet')
        pd.testing.assert_frame_equal(actual.drop(columns='score'), expected.drop(columns='score'), check_exact=True)
        np.testing.assert_allclose(actual.score, expected.score, rtol=0, atol=2e-11, equal_nan=True)
        np.testing.assert_array_equal(actual.score.gt(meta['chosen_threshold']['threshold']), expected.score.gt(meta['chosen_threshold']['threshold']))
        checked[variant] = len(actual)
    proof = dict(passed=True, score_report_sha256=sha(ROOT/'score_report.json'), checked_rows=checked,
        all_integer_encodings_tree_scores_and_thresholds_rebuilt=True, exact_feature_projection=True,
        q1_new_group_outcomes_read=False, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'score_verification.json', proof); return proof


def freeze():
    p, gate, f = checked_inputs(); assert not (ROOT/'selection_report.json').exists()
    assert not (ROOT/'labels/input_manifest.json').exists()
    report = json.loads((ROOT/'score_report.json').read_text())
    proof = json.loads((ROOT/'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256'] == sha(ROOT/'score_report.json')
    flags = {}; control = None
    for variant in gate['active_variants']:
        folder = ROOT/variant
        assert report['variants'][variant]['scores_sha256'] == sha(folder/'scores.parquet')
        s = pd.read_parquet(folder/'scores.parquet')
        flags[variant] = s.formula_input_valid & s.score.gt(gate['models'][variant]['chosen_threshold']['threshold'])
        if variant == 'control':
            control = flags[variant]
        else:
            flags['control_'+variant] = control & f[valid_column(variant)]
    records = {}; dates = set()
    for name, selected in flags.items():
        folder = ROOT/name; folder.mkdir(exist_ok=True)
        out = f[COLUMNS].copy(); out['selected'] = selected
        out.to_parquet(folder/'selection.parquet', index=False, compression='zstd')
        count = out.loc[selected].groupby('date').size(); dates.update(count.index)
        r = dict(protocol_sha256=sha(study.PROTOCOL), score_report_sha256=sha(ROOT/'score_report.json'),
            models_gate_sha256=sha(ROOT/'models_gate.json'), selection_sha256=sha(folder/'selection.parquet'),
            arm=name, selected=int(selected.sum()), days=len(count), max_daily=int(count.max()) if len(count) else 0,
            by_month={month:int((selected & out.date.str.startswith(month)).sum()) for month in p['periods'][:3]},
            identical_to_full_control=bool(np.array_equal(selected, control)),
            q1_new_group_outcomes_read=False, no_q2_signal_prices_read=True, strict_blind=False,
            native_source_parity_verified=False, no_exit_rules=True)
        save_json(folder/'selection_report.json', r); records[name] = sha(folder/'selection_report.json')
    r = dict(protocol_sha256=sha(study.PROTOCOL), score_report_sha256=sha(ROOT/'score_report.json'),
        selection_report_sha256=records, union_selected_dates=sorted(dates), all_arms_frozen_before_new_group_outcomes=True,
        q1_new_group_outcomes_read=False, no_q2_signal_prices_read=True, strict_blind=False, no_exit_rules=True)
    save_json(ROOT/'selection_report.json', r); return r


def verify():
    p, gate, f = checked_inputs(); report = json.loads((ROOT/'selection_report.json').read_text())
    assert report['protocol_sha256'] == sha(study.PROTOCOL)
    assert report['score_report_sha256'] == sha(ROOT/'score_report.json')
    verified = {}; dates = set()
    for arm, digest in report['selection_report_sha256'].items():
        variant = 'control' if arm.startswith('control') else arm
        folder = ROOT/arm; r = json.loads((folder/'selection_report.json').read_text())
        assert digest == sha(folder/'selection_report.json') and r['selection_sha256'] == sha(folder/'selection.parquet')
        s = pd.read_parquet(ROOT/variant/'scores.parquet')
        c = base.conn(); c.register('scores', s)
        cut = format(gate['models'][variant]['chosen_threshold']['threshold'], '.17e')
        extra = ''
        if arm.startswith('control_'):
            c.register('quality', f[['date', 'code', valid_column(arm[8:])]].rename(columns={valid_column(arm[8:]):'additional_valid'}))
            extra = ' AND q.additional_valid'
        query = 'SELECT s.date,s.code,s.half,s.board,s.decision_shares,s.formula_input_valid AND s.score>'+cut+extra+' AS selected FROM scores s'
        if extra:
            query += ' JOIN quality q USING(date,code)'
        expected = c.sql(query+' ORDER BY s.date,s.code').df(); c.close()
        expected['selected'] = expected.selected.fillna(False).astype(bool)
        actual = pd.read_parquet(folder/'selection.parquet')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert r['selected'] == int(actual.selected.sum())
        dates.update(actual.loc[actual.selected, 'date'])
        proof = dict(passed=True, selection_report_sha256=sha(folder/'selection_report.json'), rows=len(actual),
            selected=r['selected'], all_full_selection_flags_rebuilt=True, q1_new_group_outcomes_read=False,
            no_q2_signal_prices_read=True, no_exit_rules=True)
        save_json(folder/'selection_verification.json', proof); verified[arm] = sha(folder/'selection_verification.json')
    assert sorted(dates) == report['union_selected_dates']
    r = dict(passed=True, selection_report_sha256=sha(ROOT/'selection_report.json'),
        selection_verification_sha256=verified, union_dates=len(dates), q1_new_group_outcomes_read=False,
        no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'selection_verification.json', r); return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['scores','verify_scores','freeze','verify'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
