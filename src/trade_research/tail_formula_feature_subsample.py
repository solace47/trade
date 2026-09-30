"""Reuse 2023–2025 inputs for a fixed six-feature split-search experiment."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_dual_units as current
from . import tail_formula_units_history2024 as history
from .corporate_cash import save_json, sha

STEM = 'tail_formula_feature_subsample'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
ARMS = {'norm': current.previous.EXPRESSIONS, 'raw': current.raw.EXPRESSIONS}
EXPRESSIONS = {**ARMS['norm'], **current.raw.RAW_FIELDS}
HEADER = current.HEADER
META = history.META
FIELDS = [*META, *EXPRESSIONS]
LABEL_FIELDS = ['date', 'code', 'next_date', 'known15', 'opportunity15', 'known_no_trade', 'adverse_return15']


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['arms'] == ARMS and p['native_header'] == HEADER
    assert p['parameters']['max_features'] == 6 and p['threshold'] == .995
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for root in [history.INPUTS, current.ROOT]:
        r = json.loads((root / 'feature_report.json').read_text())
        v = json.loads((root / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    for root in [history.HISTORY, labels.ROOT]:
        r = json.loads((root / 'full_label_report.json').read_text())
        v = json.loads((root / 'full_label_verification.json').read_text())
        assert v['passed'] and v['label_report_sha256'] == sha(root / 'full_label_report.json')
        assert r['labels_sha256'] == sha(root / 'full_labels.parquet')
    return p


def projected_sources(kind):
    fields = FIELDS if kind == 'features' else LABEL_FIELDS
    roots = [history.INPUTS, current.ROOT] if kind == 'features' else [history.HISTORY, labels.ROOT]
    filename = 'features.parquet' if kind == 'features' else 'full_labels.parquet'
    frames = [pd.read_parquet(root / filename, columns=fields) for root in roots]
    overlap = [f.loc[f.date.ge('2024-01-01') & f.date.lt('2025-01-01')].reset_index(drop=True) for f in frames]
    pd.testing.assert_frame_equal(*overlap, check_exact=True)
    out = pd.concat([frames[0].loc[frames[0].date.lt('2024-01-01')], frames[1]], ignore_index=True)
    out = out.sort_values(['date', 'code']).reset_index(drop=True)
    assert not out.duplicated(['date', 'code']).any()
    assert out.date.ge('2023-01-01').all() and out.date.lt('2026-01-01').all()
    return out


def prepare():
    p = checked(); assert not (INPUTS / 'feature_report.json').exists()
    INPUTS.mkdir(parents=True, exist_ok=True)
    f = projected_sources('features'); l = projected_sources('labels')
    assert len(f) == p['expected_keys'] == 1815129
    pd.testing.assert_frame_equal(f[['date', 'code']], l[['date', 'code']], check_exact=True)
    for name, frame in [('features.parquet', f), ('full_labels.parquet', l)]:
        frame.to_parquet(INPUTS / name, index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(INPUTS / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        expressions=EXPRESSIONS, native_header=HEADER, no_new_raw_price_extraction=True,
        no_new_feature_math_or_validity_filter=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_report.json', r)
    save_json(INPUTS / 'full_label_report.json', dict(protocol_sha256=sha(PROTOCOL),
        labels_sha256=sha(INPUTS / 'full_labels.parquet'), rows=len(l), training_only_projection=True,
        original_costs_and_all_training_unknowns_preserved=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(rows=len(f), valid=r['valid'], feature_report_sha256=sha(INPUTS / 'feature_report.json'))


def verify():
    checked()
    for kind, filename, report, key in [
        ('features', 'features.parquet', 'feature_report.json', 'features_sha256'),
        ('labels', 'full_labels.parquet', 'full_label_report.json', 'labels_sha256')]:
        r = json.loads((INPUTS / report).read_text())
        assert r['protocol_sha256'] == sha(PROTOCOL) and r[key] == sha(INPUTS / filename)
        expected = projected_sources(kind); got = pd.read_parquet(INPUTS / filename)
        pd.testing.assert_frame_equal(got, expected, check_exact=True)
        c = base.conn()
        # SQL reconstructs the exact projection with no raw feature transformations.
        roots = [history.INPUTS, current.ROOT] if kind == 'features' else [history.HISTORY, labels.ROOT]
        fields = FIELDS if kind == 'features' else LABEL_FIELDS
        a = pd.read_parquet(roots[0] / filename, columns=fields)
        b = pd.read_parquet(roots[1] / filename, columns=fields)
        c.register('a', a); c.register('b', b)
        rebuilt = c.sql("SELECT * FROM (SELECT * FROM a WHERE date<'2024-01-01' UNION ALL SELECT * FROM b) ORDER BY date,code").df()
        c.close(); pd.testing.assert_frame_equal(got, rebuilt, check_exact=True)
        proof = dict(passed=True, rows=len(got), entire_source_projection_and_2024_overlap_exact=True,
            all_metadata_values_and_validity_unchanged=True, source_mathematics_and_encoding_proofs_reused=True,
            new_2026_prices_read=False, no_exit_rules=True)
        proof['feature_report_sha256' if kind == 'features' else 'label_report_sha256'] = sha(INPUTS / report)
        save_json(INPUTS / ('feature_verification.json' if kind == 'features' else 'full_label_verification.json'), proof)
    native = dict(passed=True, protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'),
        original_headers_and_both_48_column_native_definitions_unchanged=True,
        all_previous_normalized_and_percentage_input_proofs_reused=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'native_input_verification.json', native)
    return dict(passed=True, native=native)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['prepare', 'verify'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
