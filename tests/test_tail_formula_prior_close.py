import numpy as np
import pandas as pd

from trade_research.tail_formula_prior_close import aggregate, native_values


def window():
    t = pd.date_range('2024-01-02 14:32', periods=29, freq='min')
    return pd.DataFrame(dict(code='sh.600000', date='2024-01-02', timestamp=t,
        clock=t.strftime('%H%M'), open=10., high=10., low=10., close=10., volume=1000.))


def test_missing_and_duplicate_minutes_are_not_replaced_by_nearby_bars():
    f = window()
    assert aggregate(f).window_valid.all()
    assert not aggregate(f.iloc[:-1]).window_valid.any()
    duplicate = pd.concat([f.iloc[:-1], f.iloc[[-2]]], ignore_index=True)
    assert not aggregate(duplicate).window_valid.any()


def test_zero_closing_volume_is_valid_but_zero_previous_volume_is_not():
    f = window()
    f.loc[f.clock.ge('1457'), 'volume'] = 0
    a = aggregate(f).iloc[0]
    assert a.window_valid and a.volume4 == 0
    f.loc[f.clock.le('1456'), 'volume'] = 0
    assert not aggregate(f).window_valid.any()


def test_actual_native_ref_boundary_units_and_current_day_invariance():
    # 29 source bars, followed by a partial current day. Only prior bars matter.
    prior = window()
    close = np.r_[prior.close.to_numpy(), [123., 456., 789.]]
    close[28] = 10.10
    lots = np.r_[np.repeat(10., 25), np.repeat(20., 4), [1234., 5678., 9876.]]
    clocks = np.r_[prior.clock.astype(int), [930, 931, 932]]
    values, gate = native_values(close, lots, clocks, 3, 2.)
    np.testing.assert_allclose(values, [2., .5], rtol=0, atol=1e-12)
    assert gate
    close[-3:] = 1e6
    lots[-3:] = 0
    again, gate2 = native_values(close, lots, clocks, 3, 2.)
    np.testing.assert_array_equal(values, again)
    assert gate2
