import numpy as np

from trade_research.tail_formula_financial_context import native_values, prefix_points


def test_prior_stock_bar_and_fixed_anchors_survive_future_quotes():
    clocks = [1500, 930, 1420, 1448, 1449]
    broker = [100, np.nan, 110, 121, 999]
    bank = [200, np.nan, 210, 220.5, .01]
    result = native_values(broker, bank, clocks, 4)
    np.testing.assert_allclose([result['FI01'], result['FI02']], [10.75, 5.0], rtol=0, atol=1e-12)
    # Advancing the stock minute grid moves B0, preserving its prior completed bar.
    future = native_values(broker+[1e6, .01], bank+[.01, 1e6], clocks+[1450, 1451], 6)
    assert result == future
    scaled = native_values(np.array(broker)*7, np.array(bank)*3, clocks, 4)
    np.testing.assert_allclose(list(result.values()), list(scaled.values()), rtol=0, atol=1e-12)


def test_index_prefix_rejects_missing_sequence_without_reading_future():
    rows = [dict(sequence=i, price_raw=10000+i) for i in range(240)]
    expected = prefix_points(rows)
    assert expected['p20'] == 101.99 and expected['p48'] == 102.27
    rows[228]['price_raw'] = -9999
    assert prefix_points(rows) == expected
    rows[227]['sequence'] = 226
    invalid = prefix_points(rows)
    assert not invalid['prefix_valid'] and np.isnan(invalid['p48'])
