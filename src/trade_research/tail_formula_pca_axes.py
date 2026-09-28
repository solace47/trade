"""Paired original and full rotated axes, fitted only on each training window."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_polynomial_ridge as independent
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_pca_axes'
ROOT = Path('data/research') / STEM
MASTER = Path('config') / (STEM + '_protocol.json')
FIELDS = [f'XA{i:02d}' for i in range(1, 49)]
KEYS = ['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid']
BLOCK = 8192


def checked_sources():
    p = json.loads(MASTER.read_text())
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    r = json.loads((original.ROOT / 'feature_report.json').read_text())
    v = json.loads((original.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(original.ROOT / 'features.parquet')
    assert p['arms'] == ['axis', 'pca'] and p['expected_features'] == 48 and not p['new_2026_prices_allowed']
    return p


def source_setup(fold):
    assert fold in ['2024', 'recent']
    checked_sources()
    base.FEATURES = original.ROOT
    base.EXPRESSIONS = original.EXPRESSIONS
    base.HEADER = original.HEADER
    base.SOURCE = Path('data/research/tail_formula_before1000')
    relative.PROTOCOL = Path('config') / ('tail_formula_before1000_model_' + fold + '_protocol.json')
    base.PROTOCOL = relative.PROTOCOL
    return json.loads(relative.PROTOCOL.read_text())


def fit_transform(x, weights):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    means = np.average(x, axis=0, weights=weights)
    scales = np.sqrt(np.average((x-means)**2, axis=0, weights=weights))
    constant = scales <= 1e-12
    scales[constant] = 1
    z = np.clip((x-means)/scales, -5, 5)
    clip_means = np.average(z, axis=0, weights=weights)
    u = z-clip_means
    covariance = u.T @ (weights[:, None]*u)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    for j in range(vectors.shape[1]):
        k = int(np.argmax(np.abs(vectors[:, j])))
        if vectors[k, j] < 0:
            vectors[:, j] *= -1
    assert eigenvalues.min() >= -1e-10
    return dict(input_means=means.tolist(), input_scales=scales.tolist(),
        input_constant_indices=np.flatnonzero(constant).tolist(), clip_means=clip_means.tolist(),
        covariance=covariance.tolist(), eigenvalues=eigenvalues.tolist(), loadings=vectors.tolist(),
        degenerate_adjacent_indices=np.flatnonzero(np.abs(np.diff(eigenvalues)) <= 1e-10).tolist())


def centered(x, m):
    return np.clip((x-np.asarray(m['input_means']))/np.asarray(m['input_scales']), -5, 5)-np.asarray(m['clip_means'])


def transform(x, m, arm):
    u = centered(x, m)
    assert arm in ['axis', 'pca']
    if arm == 'axis':
        return u
    # Preserve the addition order of the exported native arithmetic.
    loadings = np.asarray(m['loadings'])
    out = np.zeros_like(u)
    for k in range(u.shape[1]):
        out += u[:, k:k+1]*loadings[k:k+1]
    return out


def encode_values(x):
    return np.floor(np.clip(100*x+10000+.000001, 0, 999999)).astype('int32')


def prepare(fold):
    p = source_setup(fold)
    root = ROOT / fold
    assert not (root / 'preprocessing_report.json').exists()
    t = relative.training('relative')
    expected = 502747 if fold == '2024' else 522653
    assert len(t) == expected and t.date.nunique() == 241
    assert t.date.ge(p['training_start']).all() and t.next_date.lt(p['training_end']).all()
    x = base.encode(t)
    w = 1/t.groupby('date').code.transform('size').to_numpy()/t.date.nunique()
    with threadpool_limits(limits=2):
        m = fit_transform(x.astype(float), w)
    frame = t[['date', 'code', 'next_date', 'target']].copy()
    frame['w'] = w/w.sum()
    frame[list(original.EXPRESSIONS)] = x
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / 'training.parquet', index=False, compression='zstd')
    m.update(protocol_sha256=sha(MASTER), source_protocol_sha256=sha(relative.PROTOCOL),
        feature_report_sha256=sha(original.ROOT / 'feature_report.json'),
        label_report_sha256=sha(base.SOURCE / 'full_label_report.json'),
        training_sha256=sha(root / 'training.parquet'), fold=fold, rows=len(t), days=t.date.nunique(),
        feature_names=list(original.EXPRESSIONS), training_start=p['training_start'], training_end=p['training_end'],
        last_observation=t.next_date.max(), standardized_clip=5, scale_floor=1e-12,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'preprocessing_report.json', m)
    return dict(fold=fold, rows=len(t), days=241, preprocessing_sha256=sha(root / 'preprocessing_report.json'),
        minimum_eigenvalue=min(m['eigenvalues']), maximum_eigenvalue=max(m['eigenvalues']),
        degenerate_adjacent_indices=m['degenerate_adjacent_indices'])


def checked_transform(fold, require_proof=True):
    p = source_setup(fold)
    root = ROOT / fold
    m = json.loads((root / 'preprocessing_report.json').read_text())
    assert m['protocol_sha256'] == sha(MASTER) and m['source_protocol_sha256'] == sha(relative.PROTOCOL)
    assert m['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert m['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    assert m['training_sha256'] == sha(root / 'training.parquet')
    assert m['training_start'] == p['training_start'] and m['training_end'] == p['training_end']
    assert m['last_observation'] < m['training_end'] and m['feature_names'] == list(original.EXPRESSIONS)
    if require_proof:
        v = json.loads((root / 'preprocessing_verification.json').read_text())
        assert v['passed'] and v['preprocessing_report_sha256'] == sha(root / 'preprocessing_report.json')
    return p, m


def verify_preprocessing(fold):
    p, m = checked_transform(fold, require_proof=False)
    root = ROOT / fold
    t = pd.read_parquet(root / 'training.parquet')
    c = base.conn()
    independent.register_features(c)
    independent.training_sql(c, p)
    expected = c.sql('SELECT * FROM training ORDER BY date,code').df()
    pd.testing.assert_frame_equal(t[['date', 'code', *original.EXPRESSIONS]],
        expected[['date', 'code', *original.EXPRESSIONS]], check_exact=True)
    np.testing.assert_allclose(t.target, expected.target, rtol=0, atol=2e-12)
    np.testing.assert_allclose(t.w, expected.w, rtol=0, atol=2e-15)
    assert len(t) == m['rows'] and t.date.nunique() == m['days'] == 241
    assert t.next_date.lt(p['training_end']).all() and t.date.ge(p['training_start']).all()
    names = list(original.EXPRESSIONS)
    means = c.sql('SELECT '+','.join(f'sum(w*{n}) AS m{i}' for i, n in enumerate(names))+' FROM training').fetchone()
    np.testing.assert_allclose(m['input_means'], means, rtol=0, atol=2e-8)
    variances = c.sql('SELECT '+','.join(f'sum(w*power({n}-({means[i]:.17g}),2)) AS s{i}' for i, n in enumerate(names))+' FROM training').fetchone()
    scales = np.sqrt(variances)
    scales[scales <= 1e-12] = 1
    np.testing.assert_allclose(m['input_scales'], scales, rtol=2e-11, atol=2e-9)
    expressions = [f'least(greatest(({n}-({means[i]:.17g}))/({scales[i]:.17g}),-5),5) AS z{i}' for i, n in enumerate(names)]
    c.sql('SELECT w,'+','.join(expressions)+' FROM training').create_view('clipped')
    clip_means = c.sql('SELECT '+','.join(f'sum(w*z{i}) AS m{i}' for i in range(48))+' FROM clipped').fetchone()
    np.testing.assert_allclose(m['clip_means'], clip_means, rtol=0, atol=2e-9)
    c.sql('SELECT w,'+','.join(f'z{i}-({clip_means[i]:.17g}) AS u{i}' for i in range(48))+' FROM clipped').create_view('centered')
    pairs = [(i, j) for i in range(48) for j in range(i, 48)]
    values = c.sql('SELECT '+','.join(f'sum(w*u{i}*u{j}) AS c{i}_{j}' for i, j in pairs)+' FROM centered').fetchone()
    covariance = np.zeros((48, 48))
    for (i, j), value in zip(pairs, values):
        covariance[i, j] = covariance[j, i] = value
    c.close()
    np.testing.assert_allclose(m['covariance'], covariance, rtol=0, atol=3e-9)
    weights = t.w.to_numpy()
    u = centered(t[names].to_numpy(float), m)
    with threadpool_limits(limits=2):
        _, singular, svd_vt = np.linalg.svd(u*np.sqrt(weights[:, None]), full_matrices=False)
    eigenvalues, vectors = np.asarray(m['eigenvalues']), np.asarray(m['loadings'])
    np.testing.assert_allclose(eigenvalues, singular**2, rtol=0, atol=2e-10)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(48), rtol=0, atol=2e-13)
    np.testing.assert_allclose((vectors*eigenvalues) @ vectors.T, covariance, rtol=0, atol=3e-9)
    assert all(vectors[int(np.argmax(np.abs(vectors[:, j]))), j] > 0 for j in range(48))
    cuts = [0]+[i+1 for i in range(47) if abs(eigenvalues[i]-eigenvalues[i+1]) > 1e-10]+[48]
    maximum_subspace_error = 0.
    for first, last in zip(cuts[:-1], cuts[1:]):
        a, b = vectors[:, first:last], svd_vt[first:last].T
        error = float(np.max(np.abs(a @ a.T - b @ b.T)))
        maximum_subspace_error = max(maximum_subspace_error, error)
        assert error < 1e-6
    proof = dict(passed=True, preprocessing_report_sha256=sha(root / 'preprocessing_report.json'),
        rows=len(t), days=241, original_keys_encodings_targets_and_date_weights_sql_rebuilt=True,
        means_scales_clipped_means_and_covariance_sql_rebuilt=True, independent_weighted_svd_eigenvalues_and_subspaces_checked=True,
        maximum_subspace_projector_error=maximum_subspace_error, all_48_axes_retained_without_whitening=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'preprocessing_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'verify_preprocessing'])
    parser.add_argument('--fold', choices=['2024', 'recent'], required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](args.fold), ensure_ascii=False, indent=2))
