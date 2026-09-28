import numpy as np
import pandas as pd

from trade_research import tail_formula_daily_pressure as study


def test_flat_and_invalid_candles_are_distinguished():
    good, position = study.daily_parts([10, 10, 10], [10, 9, 9], [10, 9.5, 10.01], [100, 100, 100], [3, 3, 3])
    np.testing.assert_array_equal(good, [True, True, False])
    np.testing.assert_array_equal(position[:2], [0, 0])


def test_high_close_volume_and_lot_units_affect_only_the_intended_ratio():
    high, low = np.full(20, 11.), np.full(20, 9.)
    close = np.array([11.] * 10 + [9.] * 10)
    volume = np.array([300.] * 10 + [100.] * 10)
    assert study.native_value(high, low, close, volume) == 50
    assert study.native_value(high, low, close, volume / 100) == 50
    assert study.native_value(high, low, close[::-1], volume) == -50


def test_current_future_values_do_not_enter_history_and_bad_days_are_not_skipped():
    d = pd.DataFrame(dict(date=pd.date_range('2024-01-01', periods=24).strftime('%Y-%m-%d'), code='test',
        high=11., low=9., close=10., preclose=10., volume=100., adjustflag=3))
    original = study.history_for_stock(d)
    changed = d.copy()
    changed.loc[20:, ['high', 'low', 'close', 'volume']] = [21., 19., 21., 900.]
    result = study.history_for_stock(changed)
    pd.testing.assert_frame_equal(original.iloc[:21], result.iloc[:21], check_exact=True)
    assert result.cp_weighted20.iloc[21] == 900
    broken = d.copy()
    broken.loc[10, 'close'] = 99
    result = study.history_for_stock(broken)
    assert result.cp_rows20.iloc[20] == 20 and result.cp_good20.iloc[20] == 19
    assert result.cp_first_date.iloc[20] == '2024-01-01'
