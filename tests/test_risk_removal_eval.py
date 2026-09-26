import numpy as np
import pandas as pd
import pytest

from trade_research.risk_removal_eval import add_cost_scenarios, cost_price


def test_tick_floor_and_percentage_cost_hand_calculation():
    np.testing.assert_allclose(cost_price([1., 10.], 15, 'buy'), [1.005, 10.015])
    np.testing.assert_allclose(cost_price([1., 10.], 15, 'sell'), [.995, 9.985])
    np.testing.assert_allclose(cost_price([1., 20.], 5, 'buy'), [1.005, 20.01])
    with pytest.raises(ValueError):
        cost_price([np.nan], 15, 'sell')


def test_unknown_stays_unknown_and_original_ledger_is_preserved():
    source = pd.DataFrame({
        'entry_status': ['filled', 'filled', 'volume_cap'],
        'entry_price': [1.0005, 1.0005, np.nan], 'exit_price': [1.999, np.nan, np.nan],
        'entry_window_status': ['valid'] * 3, 'exit_window_status': ['valid', 'missing', 'valid'],
        'entry_window_low': [1., 1., 1.], 'entry_window_high': [1., 1., 1.],
        'exit_window_low': [2., np.nan, 2.], 'exit_window_high': [2., np.nan, 2.],
        'shares': [100, 100, 100], 'catalog_sold_shares': [100, 100, 100],
        'catalog_dividend_gross': [1., 0., 0.], 'catalog_dividend_tax': [.2, 0., 0.],
        'known_return5': [np.nan, np.nan, 0.], 'known_return15': [np.nan, np.nan, 0.],
    })
    result = add_cost_scenarios(source)
    # 100 shares at 1.005 / 1.995; both commissions hit the five-yuan minimum.
    expected = (199.5 - 5 - 199.5 * .00051 + .8) / (100.5 + 5 + 100.5 * .00001) - 1
    assert result.loc[0, 'tick_return15'] == pytest.approx(expected)
    assert np.isnan(result.loc[1, 'tick_return15'])
    assert result.loc[2, 'tick_return15'] == 0
    pd.testing.assert_frame_equal(result[source.columns], source)
