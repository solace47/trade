"""Chronology and missing-opportunity semantics of the proposed input proxy."""
import numpy as np
import pandas as pd

from probe_tail_formula_cost_history import history


def panel():
    dates = pd.date_range('2024-01-02',periods=24).strftime('%Y-%m-%d')
    frame = pd.DataFrame(dict(date=dates,code='sh.600000',q_count=1,q_rows=1,q=10.,
        entry_rows=4,entry_clocks=4,entry_good=4,entry_high=10.,entry_capacity=20000.,
        morning_rows=29,morning_clocks=29,morning_good=29,morning_close=10.1,sustained=np.nan))
    return frame,frame[['date','code']]


def test_complete_morning_with_no_three_active_bars_is_known_zero_frequency():
    windows,calendar = panel()
    result = history(windows,calendar)
    # Hand-built two-sided cash calculation; no sustainable three-bar quote is
    # a known zero opportunity, while the active closing quote remains usable.
    buy_cash = 20030 + 6.009 + .2003
    sell_cash = 20169.7 - 6.05091 - 10.286547
    expected = 100*(sell_cash/buy_cash-1)
    assert result.atom_valid.iloc[1:].all()
    assert result.history_valid.iloc[-1]
    assert result.frequency20.iloc[-1] == 0
    np.testing.assert_allclose(result.mean20.iloc[-1],expected,atol=1e-12,rtol=0)
    assert not result.history_valid.iloc[19]


def test_current_tail_and_future_mornings_cannot_change_today_history():
    windows,calendar = panel()
    point = windows.date.iloc[21]
    before = history(windows,calendar)
    changed = windows.copy()
    changed.loc[changed.date.ge(point),['q','entry_high','entry_capacity']] = 999999
    changed.loc[changed.date.gt(point),['morning_close','sustained']] = .01
    after = history(changed,calendar)
    cols = ['date','code','proxy_mark','proxy_positive','atom_valid','mean20','frequency20','history_valid']
    pd.testing.assert_frame_equal(before.loc[before.date.le(point),cols],after.loc[after.date.le(point),cols],check_exact=True)
