import numpy as np
import pandas as pd

from trade_research import tail_formula_quarter_history2024_inputs as study


def minute_sample():
    times = pd.date_range('2023-08-25 09:31', periods=30, freq='min')
    close = np.ones(30) * 10
    close[-3:] = [11, 11, 11]
    return pd.DataFrame(dict(date='2023-08-24', code='sh.600000', timestamp=times,
        clock=times.strftime('%H:%M'), open=close, high=close, low=close, close=close,
        volume=100., amount=100*close))


def test_three_active_minutes_must_finish_before_ten_and_may_not_bridge_a_gap():
    raw = minute_sample()
    assert study.pandas_morning(raw, '10:00').sustained_close.iloc[0] == 11
    assert study.pandas_morning(raw, '09:59').sustained_close.iloc[0] == 10
    # Missing the middle minute cannot form a three-minute run, even with three observations.
    broken = raw.iloc[-4:].drop(raw.index[-2])
    assert pd.isna(study.pandas_morning(broken, '10:00').sustained_close.iloc[0])


def test_historical_mark_uses_sale_date_tax_and_preserves_minimum_commission():
    next_date = pd.Series(['2023-08-25', '2023-08-28'])
    got = study.historical_net_mark(pd.Series([10., 10.]), pd.Series([100., 100.]),
                                   pd.Series([1005., 1005.]), 15, next_date)
    amount = 100*(10-.015)
    expected = np.array([(amount-5-amount*rate)/1005-1 for rate in [.00101, .00051]])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
    assert got.iloc[1] > got.iloc[0]
