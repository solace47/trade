import numpy as np

from trade_research import tail_formula_training_support as study


def test_training_dates_are_equal_weight_despite_stock_counts():
    dates = np.array(['2024-01-02'] * 3 + ['2024-01-03'])
    w = study.date_weights(dates)
    np.testing.assert_allclose(w, [1/6, 1/6, 1/6, .5], atol=1e-15)
    x = np.array([[0., 7], [0., 7], [0., 7], [10., 7]])
    m = study.fit_numeric(x, dates)
    np.testing.assert_allclose(m['means'], [5., 7.], atol=1e-12)
    np.testing.assert_allclose(m['scales'], [5., 1.], atol=1e-12)
    assert m['constant_indices'] == [1]


def test_weighted_quantile_keeps_exact_boundary_and_ties():
    values = np.array([10., 2., 2., 1.])
    weights = np.array([.005, .005, .49, .5])
    assert study.weighted_quantile(values, weights, .99) == 2.
    d = study.distances(np.array([[2., 0.], [2.000001, 0.], [-2., 0.]]), [0, 0], [1, 1])
    np.testing.assert_array_equal(d <= 2, [True, False, True])
    np.testing.assert_array_equal((d <= 2) | (d > 2), [True, True, True])


def test_one_extreme_input_is_not_hidden_by_other_coordinates_or_clipped():
    x = np.zeros((3, 48))
    x[0, 19] = 9
    x[1, 47] = -11
    x[2] = 1
    np.testing.assert_array_equal(study.distances(x, np.zeros(48), np.ones(48)), [9, 11, 1])
