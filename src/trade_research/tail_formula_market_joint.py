"""Joint, previously verified size and prior-breadth inputs for the same model."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_prior_breadth as breadth
from . import tail_formula_prior_day as adapter
from . import tail_formula_size_context as size
from .corporate_cash import save_json, sha

STEM = 'tail_formula_market_joint'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
EXPRESSIONS = {**size.EXPRESSIONS, **breadth.NEW_EXPRESSIONS}
assert size.HEADER.startswith(previous.HEADER) and breadth.HEADER.startswith(previous.HEADER)
HEADER = size.HEADER + breadth.HEADER[len(previous.HEADER):]


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['previous_feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert p['daily_feature_report_sha256'] == sha(base.SOURCE / 'feature_report.json')
    assert p['features'] == list(size.NEW_EXPRESSIONS) + list(breadth.NEW_EXPRESSIONS)
    for spec, module in zip(p['components'], [size, breadth]):
        assert spec['root'] == str(module.ROOT)
        for name in ['feature_report', 'feature_verification']:
            assert spec[name + '_sha256'] == sha(module.ROOT / (name + '.json'))
        report = json.loads((module.ROOT / 'feature_report.json').read_text())
        proof = json.loads((module.ROOT / 'feature_verification.json').read_text())
        assert proof['passed'] and proof['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
        assert report['features_sha256'] == sha(module.ROOT / 'features.parquet')
        assert proof['native_expression_arithmetic_verified'] and proof['original_48_inputs_unchanged']
    return p


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen joint inputs'
    p = checked_sources(); f = pd.read_parquet(size.ROOT / 'features.parquet')
    g = pd.read_parquet(breadth.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', *breadth.NEW_EXPRESSIONS])
    pd.testing.assert_frame_equal(f[['date', 'code']], g[['date', 'code']], check_exact=True)
    f['joint_size_valid'] = f.formula_input_valid
    f = f.merge(g.rename(columns={'formula_input_valid': 'joint_breadth_valid'}), on=['date', 'code'], validate='one_to_one')
    f['formula_input_valid'] &= f.joint_breadth_valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(exist_ok=True, parents=True); f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), components=p['components'], features_sha256=sha(ROOT / 'features.parquet'),
        previous_feature_report_sha256=sha(previous.ROOT / 'feature_report.json'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(f.prior_formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, original_48_inputs_unchanged=True,
        original_source_audits_reused_without_recollection=True, native_source_parity_verified=False,
        software_compilation_verified=False, new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['components'] == p['components']
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    c = base.conn()
    c.read_parquet(str(size.ROOT / 'features.parquet')).create_view('a')
    c.read_parquet(str(breadth.ROOT / 'features.parquet')).create_view('b')
    expected = c.sql('''SELECT a.* EXCLUDE(formula_input_valid),
        a.formula_input_valid AND b.formula_input_valid AS formula_input_valid,
        a.formula_input_valid AS joint_size_valid,b.formula_input_valid AS joint_breadth_valid,b.BR01,b.BR02
        FROM a JOIN b USING(date,code) ORDER BY date,code''').df()
    actual = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(actual, expected[actual.columns], check_exact=True)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    assert actual.prior_formula_input_valid.equals(old.formula_input_valid)
    assert r['rows'] == len(actual) and r['valid'] == int(actual.formula_input_valid.sum())
    assert r['previous_valid'] == int(old.formula_input_valid.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~actual.formula_input_valid).sum())
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert list(EXPRESSIONS.items())[:48] == list(previous.EXPRESSIONS.items())
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len(set(names))
    assert HEADER == previous.HEADER + size.HEADER[len(previous.HEADER):] + breadth.HEADER[len(previous.HEADER):]
    c.register('f', expected); fields = list(EXPRESSIONS); checked = 0
    for start in range(0, len(fields), 13):
        columns = fields[start:start + 13]
        expressions = ','.join(f'floor(greatest(0,least(999999,100*{name}+10000+0.000001)))::INT AS {name}' for name in columns)
        sql = c.sql(f'SELECT {expressions} FROM f WHERE formula_input_valid ORDER BY date,code').df()
        values = actual.loc[actual.formula_input_valid, columns].to_numpy()
        np.testing.assert_array_equal(np.floor(np.clip(values * 100 + 10000 + .000001, 0, 999999)).astype('int32'), sql.to_numpy())
        checked += values.size
    c.close()
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(actual), valid=r['valid'],
        all_52_inputs_and_quality_intersection_independently_joined=True, original_48_inputs_unchanged=True,
        integer_encoding_checks=checked, all_native_declarations_preserved=True, original_source_audits_reused_without_recollection=True,
        native_source_parity_verified=False, software_compilation_verified=False, new_selection_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def configure():
    adapter.STEM = STEM; adapter.ROOT = ROOT; adapter.PROTOCOL = PROTOCOL
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.CONTROL = Path('data/research') / (STEM + '_control')
    adapter.EXPRESSIONS = EXPRESSIONS; adapter.HEADER = HEADER
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert v['all_52_inputs_and_quality_intersection_independently_joined'] and v['all_native_declarations_preserved']
    for fold in ['2024', 'recent', 'combined']:
        p = json.loads((Path('config') / (STEM + '_' + fold + '_protocol.json')).read_text())
        assert p['inputs_protocol_sha256'] == sha(PROTOCOL)


if __name__ == '__main__':
    from . import tail_formula_context_2024 as linkage
    from . import tail_formula_recent as study
    from . import tail_formula_relative as relative
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined', 'control'], default='2024')
    a = p.parse_args()
    if a.stage in ['features', 'verify_features']:
        result = globals()[a.stage]()
    else:
        configure()
        if a.fold == 'control':
            assert a.stage in ['freeze', 'verify', 'analyze']
            result = (linkage.common_analysis(adapter.CONTROL, adapter.COMBINED_PROTOCOL) if a.stage == 'analyze'
                else adapter.control(a.stage + '_control'))
        else:
            adapter.setup(a.fold)
            if a.stage == 'analyze':
                assert (Path('data/research') / (STEM + '_2025') / 'selection_verification.json').exists()
                assert (adapter.CONTROL / 'selection_verification.json').exists()
            if a.fold == 'combined':
                assert a.stage in ['freeze', 'verify', 'analyze']
                result = (linkage.common_analysis(linkage.COMBINED, linkage.PROTOCOL) if a.stage == 'analyze'
                    else getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')())
            elif a.stage in ['model', 'verify_model']:
                result = getattr(relative, a.stage)('relative')
            elif a.stage == 'verify_scores':
                result = verify_scores()
            elif a.stage in ['freeze', 'verify']:
                result = getattr(study, a.stage)()
            else:
                result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
