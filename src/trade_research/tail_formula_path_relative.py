"""Join already-verified daily path and equal-weight relative-strength inputs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_daily_efficiency as path_inputs
from . import tail_formula_equal_weight as relative_inputs
from .corporate_cash import save_json, sha

STEM = 'tail_formula_path_relative'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {**path_inputs.NEW_EXPRESSIONS, **relative_inputs.NEW_EXPRESSIONS}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
assert path_inputs.HEADER.startswith(previous.HEADER) and relative_inputs.HEADER.startswith(previous.HEADER)
HEADER = previous.HEADER + path_inputs.HEADER[len(previous.HEADER):] + relative_inputs.HEADER[len(previous.HEADER):]
native_core = relative_inputs.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for module in [previous, path_inputs, relative_inputs]:
        r = json.loads((module.ROOT / 'feature_report.json').read_text())
        v = json.loads((module.ROOT / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
        assert r['features_sha256'] == sha(module.ROOT / 'features.parquet')
        if module is not previous:
            n = json.loads((module.ROOT / 'native_input_verification.json').read_text())
            assert n['passed'] and n['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
    assert list(NEW_EXPRESSIONS) == p['new_fields'] and len(EXPRESSIONS) == p['expected_features'] == 52
    return p


def features():
    p = checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = old.copy()
    f['prior_formula_input_valid'] = old.formula_input_valid
    for module, name in [(path_inputs, 'path_parent_valid'), (relative_inputs, 'relative_parent_valid')]:
        part = pd.read_parquet(module.ROOT / 'features.parquet', columns=['date','code','formula_input_valid',*module.NEW_EXPRESSIONS])
        pd.testing.assert_frame_equal(part[['date','code']], old[['date','code']], check_exact=True)
        f = f.merge(part.rename(columns={'formula_input_valid':name}), on=['date','code'], how='left', validate='one_to_one')
        f['formula_input_valid'] &= f[name].fillna(False)
    f['formula_input_valid'] &= np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f = f.sort_values(['date','code']).reset_index(drop=True)
    ROOT.mkdir(parents=True, exist_ok=True); f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'], features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=relative_inputs.GATE,
        all_existing_features_reused=True, no_new_raw_extraction=True, native_source_parity_verified=False,
        software_compilation_verified=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    assert len(f) == p['all_keys_expected']
    save_json(ROOT / 'feature_report.json', r)
    return {k:v for k,v in r.items() if k not in ['source_hashes','expressions','native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    c = base.conn()
    c.register('original', old[['date','code','formula_input_valid',*previous.EXPRESSIONS]])
    c.read_parquet(str(path_inputs.ROOT / 'features.parquet')).create_view('daily')
    c.read_parquet(str(relative_inputs.ROOT / 'features.parquet')).create_view('relative_strength')
    expected = c.sql('''SELECT o.date,o.code,coalesce(o.formula_input_valid AND d.formula_input_valid AND e.formula_input_valid,false) AS valid,
        d.formula_input_valid AS path_parent_valid,e.formula_input_valid AS relative_parent_valid,d.DE05,d.DE20,e.EW01,e.EW02
        FROM original o LEFT JOIN daily d USING(date,code) LEFT JOIN relative_strength e USING(date,code) ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(f[['date','code']], expected[['date','code']], check_exact=True)
    pd.testing.assert_frame_equal(f[['path_parent_valid','relative_parent_valid',*NEW_EXPRESSIONS]],
        expected[['path_parent_valid','relative_parent_valid',*NEW_EXPRESSIONS]], check_exact=True)
    np.testing.assert_array_equal(f.formula_input_valid, expected.valid)
    c.register('joined', f[['date','code','formula_input_valid',*EXPRESSIONS]])
    fields = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in EXPRESSIONS)
    encoded = c.sql(f'SELECT {fields} FROM joined WHERE formula_input_valid ORDER BY date,code').df(); c.close()
    np.testing.assert_array_equal(encoded.to_numpy(), np.floor(np.clip(100*f.loc[f.formula_input_valid,list(EXPRESSIONS)].to_numpy()+10000+.000001,0,999999)).astype('int32'))
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER)+list(EXPRESSIONS)
    assert len(names) == len(set(names)) and r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert r['native_core_gate'] == relative_inputs.GATE
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        original_48_fields_and_keys_unchanged=True, both_parent_values_and_validity_sql_joined=True,
        all_52_encodings_rebuilt=True, shared_native_header_once_and_no_variable_collisions=True,
        effective_input_intersection_unchanged=bool(np.array_equal(f.formula_input_valid,old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    proofs = {str(module.ROOT / 'native_input_verification.json'):sha(module.ROOT / 'native_input_verification.json')
        for module in [path_inputs,relative_inputs]}
    assert all(EXPRESSIONS[n] == expr for module in [path_inputs,relative_inputs] for n,expr in module.NEW_EXPRESSIONS.items())
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), reused_parent_native_proofs=proofs,
        each_expression_and_header_fragment_preserved=True, equal_weight_membership_gate_preserved=True,
        raw_sample_extraction_not_repeated=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof); return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features','verify_features','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
