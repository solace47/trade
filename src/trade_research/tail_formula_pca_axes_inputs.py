"""Materialize and independently replay both frozen coordinate systems."""
import argparse
import json
import re

import numpy as np
import pandas as pd

from . import tail_formula_pca_axes as parent
from .corporate_cash import save_json, sha


def native_parts(m, arm):
    header = parent.original.HEADER + ''.join(f'{n}:={e};\n' for n, e in parent.original.EXPRESSIONS.items())
    fragment = ''
    for i, name in enumerate(parent.original.EXPRESSIONS):
        fragment += f'PAE{i:02d}:=INTPART(MIN(MAX(100*{name}+10000+0.000001,0),999999));\n'
        fragment += (f'PAZ{i:02d}:=MIN(MAX((PAE{i:02d}-({m["input_means"][i]:.17g}))/'
            f'({m["input_scales"][i]:.17g}),-5),5)-({m["clip_means"][i]:.17g});\n')
    expressions = {}
    for j, name in enumerate(parent.FIELDS):
        expressions[name] = (f'PAZ{j:02d}' if arm == 'axis' else
            '+'.join(f'({m["loadings"][k][j]:.17g})*PAZ{k:02d}' for k in range(48)))
    return header+fragment, expressions, fragment


def native_values(raw, m, arm):
    _, expressions, fragment = native_parts(m, arm)
    env = dict(INTPART=np.floor, MIN=np.minimum, MAX=np.maximum)
    env.update({name: raw[:, i] for i, name in enumerate(parent.original.EXPRESSIONS)})
    for line in fragment.splitlines():
        name, expr = line.rstrip(';').split(':=')
        env[name] = eval(expr, {'__builtins__': {}}, env)
    return np.column_stack([eval(expr, {'__builtins__': {}}, env) for expr in expressions.values()])


def sql_centered_terms(m):
    # DuckDB greatest/least ignore NULL, so the validity guard must enclose
    # the full clipping expression, not merely its encoded input.
    return ','.join(f'CASE WHEN formula_input_valid THEN least(greatest((e{i}-({m["input_means"][i]:.17g}))/({m["input_scales"][i]:.17g}),-5),5)-({m["clip_means"][i]:.17g}) END AS u{i}' for i in range(48))


def features(fold):
    _, m = parent.checked_transform(fold)
    source = pd.read_parquet(parent.original.ROOT / 'features.parquet', columns=[*parent.KEYS, *parent.original.EXPRESSIONS])
    valid = source.formula_input_valid.to_numpy()
    ids = np.flatnonzero(valid)
    x = parent.base.encode(source.loc[valid])
    outputs = []
    for arm in ['axis', 'pca']:
        root = parent.ROOT / fold / arm
        assert not (root / 'feature_report.json').exists()
        values = np.full((len(source), 48), np.nan)
        for start in range(0, len(ids), parent.BLOCK):
            stop = min(start+parent.BLOCK, len(ids))
            values[ids[start:stop]] = parent.transform(x[start:stop], m, arm)
        f = pd.concat([source, pd.DataFrame(values, columns=parent.FIELDS)], axis=1)
        assert np.isfinite(values[valid]).all()
        header, expressions, _ = native_parts(m, arm)
        all_names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', header)+parent.FIELDS
        assert len(all_names) == len({n.casefold() for n in all_names})
        scaled = 100*values[valid]+10000+.000001
        root.mkdir(parents=True, exist_ok=True)
        f.to_parquet(root / 'features.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(parent.MASTER), preprocessing_report_sha256=sha(parent.ROOT / fold / 'preprocessing_report.json'),
            preprocessing_verification_sha256=sha(parent.ROOT / fold / 'preprocessing_verification.json'),
            original_feature_report_sha256=sha(parent.original.ROOT / 'feature_report.json'),
            features_sha256=sha(root / 'features.parquet'), fold=fold, arm=arm, rows=len(f), valid=int(valid.sum()),
            newly_invalid=0, original_input_names=list(parent.original.EXPRESSIONS), expressions=expressions, native_header=header,
            below_clip=int((scaled < 0).sum()), above_clip=int((scaled > 999999).sum()),
            near_integer_boundary=int((np.abs(scaled-np.rint(scaled)) < 1e-8).sum()),
            software_compilation_verified=False, native_source_parity_verified=False,
            new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'feature_report.json', r)
        outputs.append({k: v for k, v in r.items() if k not in ['original_input_names', 'expressions', 'native_header']})
    return outputs


def verify(fold):
    _, m = parent.checked_transform(fold)
    names = list(parent.original.EXPRESSIONS)
    old = pd.read_parquet(parent.original.ROOT / 'features.parquet', columns=[*parent.KEYS, *names])
    c = parent.base.conn()
    c.register('source', old)
    encoded = ','.join(f'CASE WHEN formula_input_valid THEN floor(least(greatest(100*{n}+10000+.000001,0),999999)) END AS e{i}'
        for i, n in enumerate(names))
    c.sql('SELECT date,code,formula_input_valid,'+encoded+' FROM source').create_view('encoded')
    centered = sql_centered_terms(m)
    c.sql('SELECT date,code,formula_input_valid,'+centered+' FROM encoded').create_view('u')
    ids = np.flatnonzero(old.formula_input_valid.to_numpy())
    raw = old.loc[old.formula_input_valid, names].to_numpy(float)
    results = []
    for arm in ['axis', 'pca']:
        root = parent.ROOT / fold / arm
        r = json.loads((root / 'feature_report.json').read_text())
        assert r['protocol_sha256'] == sha(parent.MASTER)
        assert r['features_sha256'] == sha(root / 'features.parquet')
        assert r['preprocessing_report_sha256'] == sha(parent.ROOT / fold / 'preprocessing_report.json')
        assert r['preprocessing_verification_sha256'] == sha(parent.ROOT / fold / 'preprocessing_verification.json')
        f = pd.read_parquet(root / 'features.parquet')
        pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
        header, expressions, _ = native_parts(m, arm)
        assert r['expressions'] == expressions and r['native_header'] == header
        sql = []
        for j, name in enumerate(parent.FIELDS):
            expr = f'u{j}' if arm == 'axis' else '+'.join(f'({m["loadings"][k][j]:.17g})*u{k}' for k in range(48))
            sql.append(f'({expr}) AS {name}')
        expected = c.sql('SELECT date,code,'+','.join(sql)+' FROM u ORDER BY date,code').df()
        pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
        np.testing.assert_allclose(f[parent.FIELDS], expected[parent.FIELDS], rtol=0, atol=2e-11, equal_nan=True)
        recorded = f.loc[f.formula_input_valid, parent.FIELDS].to_numpy(float)
        np.testing.assert_array_equal(parent.encode_values(recorded), parent.encode_values(expected.loc[f.formula_input_valid, parent.FIELDS].to_numpy()))
        maximum_native_error = 0.
        for start in range(0, len(ids), parent.BLOCK):
            stop = min(start+parent.BLOCK, len(ids))
            native = native_values(raw[start:stop], m, arm)
            wanted = recorded[start:stop]
            np.testing.assert_allclose(native, wanted, rtol=0, atol=2e-11)
            np.testing.assert_array_equal(parent.encode_values(native), parent.encode_values(wanted))
            maximum_native_error = max(maximum_native_error, float(np.max(np.abs(native-wanted))))
        scaled = 100*recorded+10000+.000001
        assert r['rows'] == len(f) and r['valid'] == len(ids) and r['newly_invalid'] == 0
        assert r['below_clip'] == int((scaled < 0).sum()) and r['above_clip'] == int((scaled > 999999).sum())
        assert r['near_integer_boundary'] == int((np.abs(scaled-np.rint(scaled)) < 1e-8).sum())
        proof = dict(passed=True, feature_report_sha256=sha(root / 'feature_report.json'), rows=len(f), valid=len(ids),
            original_48_values_keys_and_validity_unchanged=True, all_transforms_and_encodings_sql_rebuilt=True,
            all_actual_native_numeric_transforms_and_encodings_rebuilt=True, maximum_native_transform_error=maximum_native_error,
            effective_input_intersection_unchanged=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'feature_verification.json', proof)
        save_json(root / 'native_input_verification.json', dict(passed=True, protocol_sha256=sha(parent.MASTER),
            feature_report_sha256=sha(root / 'feature_report.json'), feature_verification_sha256=sha(root / 'feature_verification.json'),
            all_valid_rows_replayed=len(ids), original_48_native_numerical_proofs_reused=True,
            software_compilation_verified=False, native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True))
        results.append(dict(arm=arm, **proof))
    c.close()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify'])
    parser.add_argument('--fold', choices=['2024', 'recent'], required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](args.fold), ensure_ascii=False, indent=2))
