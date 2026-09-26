import numpy as np
import pandas as pd

from trade_research.reference_gain_accounting import complete_mean, weekly_interval


def test_unknown_trade_prevents_complete_mean():
    frame = pd.DataFrame({"date": ["2024-01-02", "2024-01-02", "2024-01-03"],
                          "value": [.1, np.nan, .2]})
    assert complete_mean(frame, "value") is None
    frame.loc[1, "value"] = 0.
    assert np.isclose(complete_mean(frame, "value"), .125)


def test_week_blocks_preserve_constant_and_reject_unknown_day():
    daily = pd.Series([.01] * 8, index=pd.bdate_range("2024-01-02", periods=8).strftime("%Y-%m-%d"))
    assert np.allclose(weekly_interval(daily), [.01, .01])
    daily.iloc[2] = np.nan
    assert weekly_interval(daily) is None
