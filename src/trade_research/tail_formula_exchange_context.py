"""One known trading-venue flag and a same-dimension constant control."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_exchange_context'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {'EX01': "IF(CODELIKE('60'),1,0)", 'EX00': '0'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    assert p['new_fields'] == list(NEW_EXPRESSIONS) and p['native_expressions'] == NEW_EXPRESSIONS
    return p


def venue_flag(codes):
    codes = pd.Series(codes, copy=False)
    sh = codes.str.fullmatch(r'sh\.60[0-9]{4}', na=False)
    sz = codes.str.fullmatch(r'sz\.00[0-9]{4}', na=False)
    if not (sh | sz).all():
        raise ValueError('Unknown or excluded exchange/code mapping; do not impute zero')
    return sh.astype(float)


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.copy()
    f['EX01'] = venue_flag(f.code)
    f['EX00'] = 0.0
    assert len(f) == 1258085 and int(f.formula_input_valid.sum()) == 1117397
    ROOT.mkdir(exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=0, expressions={**previous.EXPRESSIONS, **NEW_EXPRESSIONS}, native_header=HEADER,
        venue_rows=f.groupby('EX01').size().to_dict(), new_feature_count_per_arm=1,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True,
        software_compilation_verified=False)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header', 'source_hashes']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    c = base.conn(); c.register('identity', old[['date', 'code']])
    bad = c.sql("""SELECT count(*) FROM identity WHERE NOT
        (regexp_full_match(code,'sh[.]60[0-9]{4}') OR regexp_full_match(code,'sz[.]00[0-9]{4}'))""").fetchone()[0]
    assert bad == 0
    rebuilt = c.sql("""SELECT date,code,CASE WHEN split_part(code,'.',1)='sh' THEN 1e0 ELSE 0e0 END AS EX01,
        0e0 AS EX00 FROM identity ORDER BY date,code""").df()
    pd.testing.assert_frame_equal(f[['date', 'code', 'EX01', 'EX00']], rebuilt, check_exact=True)
    # Replay the native prefix semantics independently on the six-digit symbol.
    native = c.sql("""SELECT CASE WHEN starts_with(split_part(code,'.',2),'60') THEN 1e0 ELSE 0e0 END AS flag
        FROM identity ORDER BY date,code""").df().flag.to_numpy()
    np.testing.assert_array_equal(native, f.EX01)
    for field, values in [('EX01', native), ('EX00', np.zeros(len(native)))]:
        production = np.floor(np.clip(100*f[field].to_numpy()+10000+1e-6, 0, 999999)).astype(np.int32)
        expected = np.where(values == 1, 10100, 10000).astype(np.int32)
        np.testing.assert_array_equal(production, expected)
    c.close()
    assert (f.formula_input_valid == old.formula_input_valid).all()
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), all_original_columns_bitwise_unchanged=True,
        effective_input_intersection_unchanged=True, all_new_fields_and_encodings_independently_rebuilt=True,
        unknown_mapping_count=bad, no_outcomes_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def verify_native():
    checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    p = json.loads(PROTOCOL.read_text())
    assert HEADER == previous.HEADER and p['native_expressions'] == NEW_EXPRESSIONS
    old_definitions = re.findall(r'^([A-Z][A-Z0-9]*):=', HEADER, re.M) + list(previous.EXPRESSIONS)
    assert not set(n.casefold() for n in old_definitions) & set(n.casefold() for n in NEW_EXPRESSIONS)
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        all_native_prefix_semantics_replayed_on_entire_population=True,
        original_48_expressions_and_header_unchanged=True,
        original_source_proof_reused=True, no_new_raw_price_extraction=True,
        native_reference_url=p['native_reference_url'], software_compilation_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r)
    return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'verify_native'])
    a = p.parse_args()
    print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
