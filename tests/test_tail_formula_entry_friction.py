import numpy as np

from trade_research.tail_formula_before1000 import net_mark
from trade_research.tail_formula_entry_friction import binary_zero, measure, native_value


def test_integer_lots_and_fee_inverse_cross_both_slippage_and_commission_branches():
    q = np.array([.8, 1.6, 3.33, 3.34, 100., 100.01, 160., 170., 200.])
    result = measure(q, 2., 100000.)
    assert result['valid'].all()
    np.testing.assert_array_equal(result['fr_shares'][:2], [25000, 12500])
    n, cash, zero = [result[k] for k in ['fr_shares', 'fr_cash', 'fr_breakeven']]
    assert (.0003*result['fr_buy_value'] < 5).any() and (.0003*result['fr_buy_value'] > 5).any()
    np.testing.assert_allclose(zero, binary_zero(q, n, cash), atol=1e-12, rtol=0)
    np.testing.assert_allclose(net_mark(zero, n, cash, 15), 0, atol=1e-14, rtol=0)
    assert (net_mark(zero-1e-5, n, cash, 15) < 0).all()
    assert (net_mark(zero+1e-5, n, cash, 15) > 0).all()


def test_missing_zero_volume_or_unaffordable_lot_is_invalid_not_free_capacity():
    result = measure([10., 10., 10., 201., 10.001, np.nan], [2., 2., 0., 2., 2., 2.], [0., np.nan, 100., 100., 100., 100.])
    assert not result['valid'].any()
    assert np.isnan(result['FR01']).all() and np.isnan(result['FR02']).all()


def test_native_share_lot_conversion_and_future_anchor_change_capacity_correctly():
    shares = np.array([100., 210., 500., 900.])
    expected = measure([1.6], [2.], [shares.sum()])
    value = native_value(1.6, 2., shares/100)
    future = native_value(1.6, 2., shares/100, .01)
    scaled = native_value(1.6, 2., shares)
    for name in ['FR01', 'FR02']:
        np.testing.assert_allclose(value[name], expected[name][0], rtol=0, atol=1e-12)
        assert value[name] == future[name]
    assert scaled['FR01'] == value['FR01'] and scaled['FR02'] < value['FR02']
