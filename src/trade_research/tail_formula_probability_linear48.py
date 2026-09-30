"""A fixed, date-equal binary linear score on the two existing 48-input tables.

Learn the absolute observed morning-opportunity event. The score is an
uncalibrated model logit, not an established probability of making money.
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from . import tail_formula_additive as base
from . import tail_formula_feature_subsample as cached
from .corporate_cash import save_json, sha

STEM = 'tail_formula_probability_linear48'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
INPUTS = cached.INPUTS
ARMS = cached.ARMS
HEADER = cached.HEADER
META = cached.META
PARAMETERS = dict(C=1., penalty='l2', solver='newton-cholesky', tol=1e-10,
                  max_iter=100, fit_intercept=True, class_weight=None, random_state=20260927)


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['arms'] == ARMS and p['native_header'] == HEADER and p['parameters'] == PARAMETERS
    assert p['threshold'] == .995 and p['standardized_clip'] == 5 and p['scale_floor'] == 1e-12
    assert p['sklearn_version'] == sklearn.__version__ == '1.7.2'
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for kind in ['feature', 'full_label']:
        r = json.loads((INPUTS / (kind + '_report.json')).read_text())
        v = json.loads((INPUTS / (kind + '_verification.json')).read_text())
        key = 'feature_report_sha256' if kind == 'feature' else 'label_report_sha256'
        assert v['passed'] and v[key] == sha(INPUTS / (kind + '_report.json'))
        name = 'features' if kind == 'feature' else 'full_labels'
        assert r['features_sha256' if kind == 'feature' else 'labels_sha256'] == sha(INPUTS / (name + '.parquet'))
    return p


def setup(arm, fold):
    master = checked(); assert arm in ARMS and fold in master['folds']
    base.ROOT = ROOT / arm / fold
    base.PROTOCOL = Path('config') / (STEM + '_' + arm + '_' + fold + '_protocol.json')
    base.FEATURES = base.SOURCE = INPUTS; base.EXPRESSIONS = ARMS[arm]; base.HEADER = HEADER
    p = json.loads(base.PROTOCOL.read_text())
    assert p['master_protocol_sha256'] == sha(PROTOCOL) and p['arm'] == arm and p['fold'] == fold
    assert all(p[k] == v for k, v in master['folds'][fold].items())
    assert p['parameters'] == PARAMETERS and p['feature_names'] == list(ARMS[arm])
    base.ROOT.mkdir(parents=True, exist_ok=True)
    return p


def training(p):
    t = base.training(start=p['training_start'], end=p['training_end'])
    assert len(t) == p['expected_training_rows'] and t.date.nunique() == p['expected_training_days']
    assert t.next_date.max() == p['expected_last_observation'] < p['evaluation_start']
    assert t.date.ge(p['training_start']).all() and t.next_date.lt(p['training_end']).all()
    assert t.opportunity15.isin([0, 1]).all() and t.opportunity15.nunique() == 2
    return t


def predict(x, m):
    out = np.empty(len(x), dtype=float)
    for start in range(0, len(x), 8192):
        z = np.clip((x[start:start + 8192] - np.asarray(m['input_means'])) /
                    np.asarray(m['input_scales']), -5, 5)
        s = np.full(len(z), m['bias'], dtype=float)
        for j, coefficient in enumerate(m['coefficients']):
            s += coefficient * z[:, j]
        out[start:start + len(z)] = s
    return out


def model():
    p = json.loads(base.PROTOCOL.read_text()); root = base.ROOT
    assert not (root / 'model_report.json').exists(), 'Do not refit frozen models'
    t = training(p); x = base.encode(t).astype(float)
    y = t.opportunity15.to_numpy(dtype=float)
    w = 1 / t.groupby('date').code.transform('size').to_numpy(dtype=float)
    # Each date contributes one unit. C=1 then gives a normalized L2 penalty
    # of 1/sum(w), as independently checked against installed sklearn source.
    means = np.average(x, axis=0, weights=w)
    scales = np.sqrt(np.average((x - means) ** 2, axis=0, weights=w))
    constants = scales <= 1e-12; scales[constants] = 1
    z = np.clip((x - means) / scales, -5, 5)
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        estimator = LogisticRegression(**PARAMETERS).fit(z, y, sample_weight=w)
    assert estimator.classes_.tolist() == [0., 1.] and max(estimator.n_iter_) < PARAMETERS['max_iter']
    m = dict(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        label_report_sha256=sha(INPUTS / 'full_label_report.json'), rows=len(t), days=t.date.nunique(),
        last_observation=t.next_date.max(), training_start=p['training_start'], training_end=p['training_end'],
        parameters=PARAMETERS, feature_names=list(base.EXPRESSIONS), variant='absolute_binary_linear_logit',
        input_means=means.tolist(), input_scales=scales.tolist(),
        input_constant_indices=np.flatnonzero(constants).tolist(), standardized_clip=5, scale_floor=1e-12,
        coefficients=estimator.coef_[0].tolist(), bias=float(estimator.intercept_[0]),
        iterations=estimator.n_iter_.tolist(), weights_sum=float(w.sum()), classes=[0., 1.],
        score_is_logit=True, probability_calibration_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    score = predict(x, m)
    np.testing.assert_allclose(score, estimator.decision_function(z), rtol=0, atol=2e-12)
    # Only q995 is used; other legacy indices are not searched or evaluated.
    m['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q)))
                       for i, q in enumerate(base.QUANTILES)]
    save_json(root / 'model_report.json', m)
    return dict(rows=len(t), days=m['days'], iterations=m['iterations'], threshold=m['thresholds'][3])


def checked_model():
    p = json.loads(base.PROTOCOL.read_text()); m = json.loads((base.ROOT / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('feature_report_sha256', INPUTS / 'feature_report.json'),
                      ('label_report_sha256', INPUTS / 'full_label_report.json')]:
        assert m[key] == sha(path)
    assert m['parameters'] == PARAMETERS and m['feature_names'] == list(base.EXPRESSIONS)
    assert m['variant'] == 'absolute_binary_linear_logit' and m['standardized_clip'] == 5 and m['scale_floor'] == 1e-12
    for key in ['input_means', 'input_scales', 'coefficients']:
        assert len(m[key]) == 48 and np.isfinite(m[key]).all()
    assert min(m['input_scales']) > 0 and np.isfinite(m['bias'])
    assert m['rows'] == p['expected_training_rows'] and m['days'] == p['expected_training_days']
    assert m['last_observation'] == p['expected_last_observation'] < p['evaluation_start']
    assert m['training_start'] == p['training_start'] and m['training_end'] == p['training_end']
    return p, m


def register(c):
    # Project exact names first: DuckDB's identifiers are case-insensitive.
    c.register('features', pq.read_table(INPUTS / 'features.parquet', columns=[*META, *base.EXPRESSIONS]))
    c.execute('CREATE VIEW encoded AS SELECT ' + ','.join(META) + ',' + ','.join(
        f'CASE WHEN formula_input_valid THEN floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT '
        f'ELSE NULL END AS {n}' for n in base.EXPRESSIONS) + ' FROM features')


def score_sql(m):
    terms = [f'({b:.17e})*least(greatest(({n}-({mu:.17e}))/({s:.17e}),-5),5)'
             for n, mu, s, b in zip(base.EXPRESSIONS, m['input_means'], m['input_scales'], m['coefficients'])]
    return '(' + f'{m["bias"]:.17e}' + '+' + '+'.join(terms) + ')'


def verify_model():
    p, m = checked_model(); t = training(p); c = base.conn(); register(c)
    c.execute(f'''CREATE VIEW training AS SELECT f.*, l.next_date, l.opportunity15 AS y,
        1./count(*) OVER(PARTITION BY f.date) AS w FROM encoded f
        JOIN read_parquet('{INPUTS}/full_labels.parquet') l USING(date,code)
        WHERE l.known15 AND f.formula_input_valid AND f.date>='{p['training_start']}'
        AND l.next_date<'{p['training_end']}' ''')
    d = c.sql('SELECT * FROM training ORDER BY date,code').df()
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date']], t[['date', 'code', 'next_date']], check_exact=True)
    np.testing.assert_array_equal(d.y, t.opportunity15)
    np.testing.assert_array_equal(d[list(base.EXPRESSIONS)].to_numpy(), base.encode(t))
    np.testing.assert_allclose(d.w, 1 / t.groupby('date').code.transform('size'), rtol=0, atol=2e-15)
    means = np.array(c.sql('SELECT ' + ','.join(f'sum(w*{n})/sum(w)' for n in base.EXPRESSIONS) + ' FROM training').fetchone())
    scales = np.sqrt(c.sql('SELECT ' + ','.join(f'sum(w*({n}-({mu:.17e}))*({n}-({mu:.17e})))/sum(w)'
                     for n, mu in zip(base.EXPRESSIONS, means)) + ' FROM training').fetchone())
    constants = scales <= 1e-12; scales[constants] = 1
    np.testing.assert_allclose(means, m['input_means'], rtol=0, atol=2e-7)
    np.testing.assert_allclose(scales, m['input_scales'], rtol=0, atol=2e-7)
    assert np.flatnonzero(constants).tolist() == m['input_constant_indices']
    weights_sum = float(d.w.sum()); assert abs(weights_sum - m['days']) < 2e-10
    np.testing.assert_allclose(weights_sum, m['weights_sum'], rtol=0, atol=2e-10)
    c.execute('CREATE VIEW scored AS SELECT *, ' + score_sql(m) + ' AS score FROM training')
    c.execute('CREATE VIEW residual AS SELECT *, 1./(1+exp(-score))-y AS residual FROM scored')
    gradients = c.sql('SELECT ' + ','.join(
        f'sum(w*residual*least(greatest(({n}-({mu:.17e}))/({s:.17e}),-5),5))/sum(w)+({b:.17e})/sum(w)'
        for n, mu, s, b in zip(base.EXPRESSIONS, m['input_means'], m['input_scales'], m['coefficients'])) +
        ',sum(w*residual)/sum(w) FROM residual').fetchone()
    gradient_max = float(np.max(np.abs(gradients))); assert gradient_max < 5e-8
    # Convex binary loss + positive L2; unpenalized intercept is checked too.
    # This optimality condition checks fitting without refitting a second model.
    thresholds = c.sql('SELECT ' + ','.join(f'quantile_cont(score,{q})' for q in base.QUANTILES) + ' FROM scored').fetchone()
    np.testing.assert_allclose(thresholds, [v['threshold'] for v in m['thresholds']], rtol=0, atol=2e-10)
    c.close()
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d),
        all_training_keys_targets_weights_encodings_and_preprocessing_sql_verified=True,
        all_penalized_coefficients_and_unpenalized_bias_gradient_verified=True,
        normalized_l2_penalty=1 / weights_sum, max_abs_gradient=gradient_max,
        coefficients_checked=48, absolute_target_not_date_centered=True,
        all_quantiles_independently_verified=True, probability_calibration_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof)
    return proof


def scores():
    checked_model()
    base.predict = predict
    return base.scores()


def native_core(m, threshold, expressions, header):
    core = header + '\n'.join(f'{n}:={v};' for n, v in expressions.items()) + '\n'
    for i, (name, mu, scale) in enumerate(zip(expressions, m['input_means'], m['input_scales']), 1):
        core += f'X{i:02d}:=INTPART(MIN(MAX(100*{name}+10000+0.000001,0),999999));\n'
        core += f'Z{i:02d}:=MIN(MAX((X{i:02d}-({mu:.17g}))/({scale:.17g}),-5),5);\n'
    core += 'SC:=' + format(m['bias'], '.17g') + '+' + '+'.join(
        f'({b:.17g})*Z{i:02d}' for i, b in enumerate(m['coefficients'], 1)) + ';\n'
    return core + 'CORE:SC>' + format(threshold, '.17g') + ';\n'


def verify_scores():
    p, m = checked_model(); root = base.ROOT
    r = json.loads((root / 'score_report.json').read_text())
    assert r['protocol_sha256'] == sha(base.PROTOCOL) and r['model_report_sha256'] == sha(root / 'model_report.json')
    assert r['feature_report_sha256'] == sha(INPUTS / 'feature_report.json') and r['scores_sha256'] == sha(root / 'scores.parquet')
    c = base.conn(); register(c)
    expected = c.sql('SELECT ' + ','.join(META) + ',CASE WHEN formula_input_valid THEN ' + score_sql(m) +
                     ' ELSE NULL END AS score FROM encoded ORDER BY date,code').df(); c.close()
    got = pd.read_parquet(root / 'scores.parquet')
    pd.testing.assert_frame_equal(got[META], expected[META], check_exact=True)
    assert len(got) == r['rows'] == 1815129 and got.formula_input_valid.sum() == r['valid'] == 1602413
    assert got.score.notna().equals(got.formula_input_valid)
    np.testing.assert_allclose(got.score, expected.score, rtol=0, atol=2e-10, equal_nan=True)
    core = root / 'frozen_numeric_core.tdx'
    text = native_core(m, m['thresholds'][3]['threshold'], base.EXPRESSIONS, HEADER)
    if core.exists():
        assert core.read_text() == text
    else:
        core.write_text(text)
    # Round-trip the actual exported decimal constants, then recompute every
    # valid score; header/input-source/client parity is not inferred from this.
    import re
    constants = re.findall(r'Z\d+:=MIN\(MAX\(\(X\d+-\(([^)]+)\)\)/\(([^)]+)\),-5\),5\);', text)
    assert len(constants) == 48
    exported = dict(m, input_means=[float(v[0]) for v in constants], input_scales=[float(v[1]) for v in constants])
    line = next(line for line in text.splitlines() if line.startswith('SC:='))
    exported['bias'] = float(line[4:line.index('+(')])
    exported['coefficients'] = [float(v) for v in re.findall(r'\+\(([^)]+)\)\*Z\d+', line)]
    assert len(exported['coefficients']) == 48
    valid = base.feature_inputs().loc[got.formula_input_valid]
    np.testing.assert_allclose(predict(base.encode(valid), exported), got.loc[got.formula_input_valid, 'score'], rtol=0, atol=2e-12)
    proof = dict(passed=True, score_report_sha256=sha(root / 'score_report.json'), rows=len(got),
        entire_pool_metadata_validity_and_scores_sql_verified=True,
        all_native_exported_decimal_constants_and_valid_scores_verified=True,
        numeric_core_sha256=sha(core), software_compilation_verified=False, native_source_parity_verified=False,
        probability_calibration_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'score_verification.json', proof)
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores'])
    p.add_argument('--arm', choices=list(ARMS), required=True)
    p.add_argument('--fold', choices=['2024h1', '2024h2', '2025h1', '2025h2'], required=True)
    a = p.parse_args(); setup(a.arm, a.fold)
    print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
