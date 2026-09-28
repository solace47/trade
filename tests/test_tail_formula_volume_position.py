import numpy as np

from trade_research.tail_formula_volume_position import measure, native_value


def test_volume_is_compared_to_final_quote_and_ties_are_separate():
    prices = np.array([[9.]*14+[11.]*14+[10.], [10.]*29])
    volume = np.ones((2, 29))
    volume[0, -1] = 2
    result = measure(prices, volume)
    assert result['valid'].all()
    np.testing.assert_allclose(result['VC01'], [100*14/30, 0])
    np.testing.assert_allclose(result['VC02'], [100*2/30, 100])
    volume[0, 14:] = 0
    result = measure(prices, volume)
    assert result['VC01'][0] == 100 and result['VC02'][0] == 0


def test_missing_quote_is_invalid_even_when_its_volume_is_zero():
    prices = np.full((4, 29), 10.)
    volume = np.ones((4, 29))
    prices[0, 3] = np.nan
    volume[0, 3] = 0
    volume[1] = 0
    prices[2, 0] = 10.001
    volume[3, 0] = .5
    result = measure(prices, volume)
    assert not result['valid'].any()
    assert np.isnan(result['VC01']).all() and np.isnan(result['VC02']).all()


def test_actual_native_expression_anchors_every_term_before_future_quotes():
    prices = np.array([9.99, 10.01, 10.]*9+[9.98, 10.])
    volume = np.arange(1., 30.)
    expected = measure(prices[None, :], volume[None, :])
    native = native_value(prices, volume/100)
    changed = native_value(prices, volume, .01)
    for name in ['VC01', 'VC02']:
        np.testing.assert_allclose(native[name], expected[name][0], atol=2e-12, rtol=0)
        np.testing.assert_allclose(changed[name], native[name], atol=2e-12, rtol=0)
