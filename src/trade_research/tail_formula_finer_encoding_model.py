"""Ten subdivisions of the same clipped encoding, with all other choices fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as old
from . import tail_formula_float as inputs
from .corporate_cash import save_json, sha
from .tail_formula_offset_logit48 import verify_scores as score_verifier

STEM = 'tail_formula_finer_encoding'
MASTER = Path('config') / (STEM + '_protocol.json')
ROOT = Path('data/research') / STEM
ORIGINAL_NATIVE = base.native_core


def checked():
    p = json.loads(MASTER.read_text())
    for path, digest in p['references'].items():
        assert sha(Path(path)) == digest
    assert p['encoding_multiplier'] == 10 and p['expected_features'] == 48
    return p


def encode_values(values):
    assert np.isfinite(values).all()
    return np.floor(10 * np.clip(100 * values + 10000 + .000001, 0, 999999)).astype('int32')


def encode(frame):
    return encode_values(frame[list(inputs.EXPRESSIONS)].to_numpy(dtype=float))


def native_core(m, cut, expressions=None, header=None):
    expressions = base.EXPRESSIONS if expressions is None else expressions
    header = base.HEADER if header is None else header
    core = ORIGINAL_NATIVE(m, cut, expressions, header)
    for i, name in enumerate(expressions, 1):
        old_line = f'X{i:02d}:=INTPART(MIN(MAX(100*{name}+10000+0.000001,0),999999));'
        new_line = f'X{i:02d}:=INTPART(10*MIN(MAX(100*{name}+10000+0.000001,0),999999));'
        assert core.count(old_line) == 1
        core = core.replace(old_line, new_line)
    return core


def setup(fold):
    checked()
    old.STEM = STEM
    old.setup(fold)
    base.encode = encode
    base.native_core = native_core
    for name in ['2024', 'recent', 'combined']:
        p = json.loads((Path('config') / f'{STEM}_{name}_protocol.json').read_text())
        assert p['inputs_protocol_sha256'] == sha(MASTER) and p['encoding_multiplier'] == 10


def audit_encoding():
    checked()
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'encoding_report.json').exists()
    source = json.loads((inputs.ROOT / 'feature_report.json').read_text())
    proof = json.loads((inputs.ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert source['features_sha256'] == sha(inputs.ROOT / 'features.parquet')
    names = list(inputs.EXPRESSIONS)
    frame = pd.read_parquet(inputs.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', *names])
    valid = frame.loc[frame.formula_input_valid].reset_index(drop=True)
    fine = encode(valid)
    coarse = np.floor(np.clip(100 * valid[names].to_numpy() + 10000 + .000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(fine // 10, coarse)
    np.testing.assert_array_equal(fine.astype('float32').astype('int32'), fine)
    assert fine.min() >= 0 and fine.max() <= 9999990 < 2**24
    c = base.conn()
    c.register('features', pq.read_table(inputs.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', *names]))
    sql = ','.join(f'floor(10*least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    rebuilt = c.sql('SELECT date,code,' + sql + ' FROM features WHERE formula_input_valid ORDER BY date,code').df()
    c.close()
    pd.testing.assert_frame_equal(valid[['date', 'code']], rebuilt[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(fine, rebuilt[names].to_numpy())
    records = []
    for i, name in enumerate(names):
        records.append(dict(feature=name, coarse_unique=int(np.unique(coarse[:, i]).size),
                            finer_unique=int(np.unique(fine[:, i]).size),
                            nonzero_subdivision_rows=int((fine[:, i] % 10 != 0).sum()),
                            minimum=int(fine[:, i].min()), maximum=int(fine[:, i].max())))
    report = dict(protocol_sha256=sha(MASTER), feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
                  feature_verification_sha256=sha(inputs.ROOT / 'feature_verification.json'),
                  rows=len(frame), valid=len(valid), scalar_encodings=fine.size, encoding_multiplier=10,
                  fine_int32_little_endian_sha256=hashlib.sha256(fine.astype('<i4').tobytes()).hexdigest(),
                  by_feature=records, unchanged_feature_values_keys_and_validity=True,
                  all_encodings_sql_rebuilt=True, all_coarse_bins_recovered=True,
                  all_integer_float32_roundtrips_exact=True, new_group_outcomes_read=False,
                  new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'encoding_report.json', report)
    save_json(ROOT / 'encoding_verification.json', dict(passed=True,
        encoding_report_sha256=sha(ROOT / 'encoding_report.json'), all_scalar_encodings_compared=True,
        all_coarse_bin_and_float32_roundtrips_checked=True, rows=len(frame), valid=len(valid)))
    return {k: v for k, v in report.items() if k != 'by_feature'}


def checked_encoding():
    r = json.loads((ROOT / 'encoding_report.json').read_text())
    v = json.loads((ROOT / 'encoding_verification.json').read_text())
    assert v['passed'] and v['encoding_report_sha256'] == sha(ROOT / 'encoding_report.json')
    assert r['protocol_sha256'] == sha(MASTER) and r['valid'] == 1117397 and r['rows'] == 1258085
    assert r['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')


def model():
    checked_encoding()
    old.relative.model('relative')
    root = base.ROOT
    m = json.loads((root / 'model_report.json').read_text())
    m.update(encoding_multiplier=10, encoding_report_sha256=sha(ROOT / 'encoding_report.json'))
    save_json(root / 'model_report.json', m)
    return dict(rows=m['rows'], days=m['days'], model_report_sha256=sha(root / 'model_report.json'))


def verify_model(fold):
    checked_encoding()
    m = json.loads((base.ROOT / 'model_report.json').read_text())
    assert m['encoding_multiplier'] == 10 and m['encoding_report_sha256'] == sha(ROOT / 'encoding_report.json')
    prior = json.loads((Path('data/research') / f'tail_formula_before1000_model_{fold}' / 'model_report.json').read_text())
    for name in ['parameters', 'rows', 'days', 'feature_names', 'last_observation', 'training_start', 'training_end']:
        assert m[name] == prior[name]
    return old.relative.verify_model('relative', encoding_multiplier=10)


def verify():
    checked_encoding()
    proof = old.study.verify()
    root = base.ROOT
    r = json.loads((root / 'selection_report.json').read_text())
    m = json.loads((root / 'model_report.json').read_text())
    core = (root / 'frozen_numeric_core.tdx').read_text()
    assert core == native_core(m, r['chosen_threshold']['threshold'], base.EXPRESSIONS, base.HEADER)
    lines = re.findall(r'^X(\d{2}):=INTPART\(10\*MIN\(MAX\(100\*([A-Z][A-Z0-9]*)\+10000\+0.000001,0\),999999\)\);$', core, re.M)
    assert lines == [(f'{i:02d}', name) for i, name in enumerate(inputs.EXPRESSIONS, 1)]
    proof.update(encoding_multiplier=10, all_native_encoding_and_tree_expressions_verified=True)
    save_json(root / 'selection_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit_encoding', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args()
    setup(args.fold)
    if args.stage == 'audit_encoding':
        result = audit_encoding()
    elif args.stage == 'analyze':
        joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
        assert joint['passed'] and joint['protocol_sha256'] == sha(MASTER)
        root = old.linkage.COMBINED if args.fold == 'combined' else base.ROOT
        result = old.evaluation.analyze(root, old.linkage.PROTOCOL if args.fold == 'combined' else base.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze', 'verify']
        result = old.linkage.combine() if args.stage == 'freeze' else old.linkage.verify_combined()
    elif args.stage == 'model':
        result = model()
    elif args.stage == 'verify_model':
        result = verify_model(args.fold)
    elif args.stage == 'verify_scores':
        result = score_verifier(encoding_multiplier=10)
    elif args.stage == 'freeze':
        result = old.study.freeze()
    elif args.stage == 'verify':
        result = verify()
    else:
        checked_encoding()
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
