import numpy as np

from trade_research.tail_formula_intraday_beta import measure, native_value


def prices():
    return 3000*np.exp(np.cumsum(np.r_[0, .0001*np.sin(np.arange(28))]))


def test_synchronous_proportional_and_opposite_moves_have_zero_residual():
    ix = prices()
    for exponent in [2., -1.]:
        stock = 10*(ix/ix[0])**exponent
        result = measure(stock[None, :], ix[None, :], [1.])
        assert result['valid'][0]
        np.testing.assert_allclose(result['IC01'][0], 100*np.sign(exponent), atol=2e-8)
        np.testing.assert_allclose(result['IC02'][0], 0., atol=2e-10)


def test_flat_index_keeps_stock_change_and_missing_point_is_not_skipped():
    stock = np.linspace(10, 10.3, 29)[None, :]
    ix = np.full((1, 29), 3000.)
    result = measure(stock, ix, [2.])
    assert result['valid'][0] and result['IC01'][0] == 0
    np.testing.assert_allclose(result['IC02'][0], 100*np.log(10.3/10)/2, atol=2e-12)
    ix[0, 7] = np.nan
    result = measure(stock, ix, [2.])
    assert not result['valid'][0] and np.isnan(result['IC01'][0])


def test_native_expression_uses_same_label_points_and_ignores_1449_onward():
    stock = np.round(10 + .02*np.sin(np.arange(29)) + np.arange(29)*.005, 2)
    ix = prices().round(2)
    expected = measure(stock[None, :], ix[None, :], [2.])
    result = native_value(stock, ix, 2.)
    changed = native_value(stock, ix, 2., .01)
    for name in ['IC01', 'IC02']:
        np.testing.assert_allclose(result[name], expected[name][0], atol=2e-8, rtol=0)
        assert result[name] == changed[name]
