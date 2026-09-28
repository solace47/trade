import numpy as np
import pandas as pd

from trade_research.tail_formula_reversal_change import native_values, rebuild_history


def test_generated_expressions_exclude_today_and_separate_recent_history():
    # One reference day, 240 earlier days, 20 recent days, current day.
    daily_open = np.full(262,10.)
    daily_close = np.full(262,10.)
    daily_open[1:241] = 9.
    daily_open[241:261] = 11.
    sizes = 2+np.arange(262)%3
    dates = np.repeat(np.arange(262),sizes)
    opens,closes = np.repeat(daily_open,sizes),np.repeat(daily_close,sizes)
    values,offsets = native_values(opens,closes,dates)
    np.testing.assert_allclose(values,[100,-100],atol=1e-12)
    np.testing.assert_array_equal(offsets,np.cumsum(sizes[::-1])[:260])
    opens[dates==261] = 1000.;closes[dates==261] = .01
    assert native_values(opens,closes,dates)[0] == values
    # The oldest included event contributes exactly 1/240, not 1/20.
    opens[dates==1] = 10.
    np.testing.assert_allclose(native_values(opens,closes,dates)[0],[100,-100+100/240],atol=1e-12)


def test_missing_history_is_not_refilled_or_counted_as_observed_zero():
    d = pd.DataFrame(dict(date=pd.date_range('2022-01-01',periods=600).strftime('%Y-%m-%d'),
        code='sh.600000',open=11.,close=10.,preclose=10.,tradestatus=1,adjustflag=3))
    clean = rebuild_history(d)
    assert clean.rc_good.iloc[260] == 259 and clean.rc_good.iloc[261] == 260
    assert clean.rc_nr20.iloc[261] == 20 and clean.rc_nr240.iloc[261] == 240
    d.loc[300,'open'] = np.nan
    got = rebuild_history(d)
    assert got.rc_good.iloc[300] == 260
    assert got.rc_good.iloc[301:561].eq(259).all()
    assert got.rc_good.iloc[561] == 260
    assert got.rc_rows.iloc[301:561].eq(260).all()
