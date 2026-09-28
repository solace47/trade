import numpy as np

from trade_research.tail_formula_pca_axes import centered, fit_transform, transform


def test_all_axes_preserve_centered_geometry_without_whitening_or_labels():
    x = np.random.default_rng(731).normal(size=(91, 4))
    x[:, 2] = 3*x[:, 1]+.2*x[:, 2]
    x[:, 3] = 17
    weights = np.linspace(1, 2, len(x))
    m = fit_transform(x, weights)
    assert m['input_constant_indices'] == [3]
    u, rotated = transform(x, m, 'axis'), transform(x, m, 'pca')
    np.testing.assert_allclose(rotated @ np.asarray(m['loadings']).T, u, rtol=0, atol=2e-13)
    np.testing.assert_allclose(np.sum(rotated**2, axis=1), np.sum(u**2, axis=1), rtol=0, atol=2e-13)
    np.testing.assert_allclose(np.average(u, axis=0, weights=weights), 0, rtol=0, atol=2e-13)
    assert not np.allclose(m['eigenvalues'], 1)


def test_frozen_training_transform_does_not_recompute_on_future_rows():
    x = np.array([[0., 3.], [1., 5.], [2., 7.], [3., 11.]])
    m = fit_transform(x, np.ones(4))
    a = transform(x, m, 'pca')
    b = transform(np.vstack([x, [1e9, -1e9]]), m, 'pca')
    np.testing.assert_array_equal(a, b[:4])
    z = centered(np.array([[1e9, -1e9]]), m)
    np.testing.assert_allclose(z, np.array([[5., -5.]])-m['clip_means'], rtol=0, atol=0)


def test_actual_native_arithmetic_matches_both_encodings():
    from trade_research.tail_formula_pca_axes import encode_values
    from trade_research.tail_formula_pca_axes_inputs import native_values
    raw = np.random.default_rng(81).normal(size=(123, 48))
    x = encode_values(raw)
    m = fit_transform(x.astype(float), np.ones(len(raw)))
    for arm in ['axis', 'pca']:
        expected = transform(x, m, arm)
        actual = native_values(raw, m, arm)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(encode_values(actual), encode_values(expected))


def test_sql_verifier_keeps_invalid_rows_unknown_through_clipping():
    import duckdb
    import pandas as pd
    from trade_research.tail_formula_pca_axes_inputs import sql_centered_terms
    m = dict(input_means=[0.]*48, input_scales=[1.]*48, clip_means=[0.]*48)
    f = pd.DataFrame({'formula_input_valid': [False, True], **{f'e{i}': [np.nan, 2.] for i in range(48)}})
    with duckdb.connect() as c:
        c.register('encoded', f)
        result = c.sql('SELECT '+sql_centered_terms(m)+' FROM encoded').df().to_numpy()
    assert np.isnan(result[0]).all()
    np.testing.assert_array_equal(result[1], np.full(48, 2.))
