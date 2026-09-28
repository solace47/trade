import numpy as np

from trade_research.tail_formula_market_direction import measure, native_value


def test_flat_stock_resists_index_decline_and_lags_its_rebound():
    index = np.r_[100., 99., np.full(27, 100.)]
    stock = np.full(29, 10.)
    result = measure(stock[None, :], index[None, :], [2.])
    assert result['valid'][0]
    assert result['du_down_count'][0] == result['du_up_count'][0] == 1
    np.testing.assert_allclose(result['DU01'][0], -50*np.log(.99), atol=1e-12)
    np.testing.assert_allclose(result['DU02'][0], -50*np.log(100/99), atol=1e-12)
    np.testing.assert_allclose(result['DU01'][0] + result['DU02'][0], 0., atol=1e-12)


def test_synchronous_moves_and_flat_market_neither_fabricate_excess_nor_skip_missing():
    index = 3000*np.exp(np.cumsum(np.r_[0, .0001*np.sin(np.arange(28))]))
    stock = index/100
    same = measure(stock[None, :], index[None, :], [1.])
    np.testing.assert_allclose([same['DU01'][0], same['DU02'][0]], 0., atol=1e-10)
    flat = np.full((1, 29), 3000.)
    result = measure(stock[None, :], flat, [2.])
    assert result['valid'][0] and result['DU01'][0] == result['DU02'][0] == 0
    flat[0, 7] = np.nan
    result = measure(stock[None, :], flat, [2.])
    assert not result['valid'][0] and np.isnan(result['DU01'][0])


def test_native_timing_and_price_unit_invariance():
    index = np.round(3000 + np.sin(np.arange(29))*2, 2)
    stock = np.round(10 + .02*np.sin(np.arange(29)) + np.arange(29)*.005, 2)
    result = measure(stock[None, :], index[None, :], [2.])
    scaled = measure(stock[None, :]*100, index[None, :]/10, [2.])
    native = native_value(stock, index, 2.)
    changed = native_value(stock, index, 2., .01)
    for name in ['DU01', 'DU02']:
        np.testing.assert_allclose(result[name][0], scaled[name][0], atol=1e-10)
        np.testing.assert_allclose(native[name], result[name][0], atol=1e-10)
        assert changed[name] == native[name]
