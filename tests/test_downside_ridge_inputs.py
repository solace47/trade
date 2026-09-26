import duckdb
import numpy as np
import pandas as pd
import pytest

from trade_research.downside_ridge_inputs import window_summary
from trade_research.round_number_entry import validate_window


@pytest.mark.parametrize("change", ["none", "zero", "outside", "nan", "duplicate", "seconds", "wrong_label", "negative"])
def test_vector_windows_match_scalar_source_validation(change):
    bars = pd.DataFrame({"timestamp": pd.date_range("2024-05-06 14:52", periods=4, freq="min"),
        "open": 10., "high": 10.1, "low": 9.9, "close": 10., "volume": 1000., "turnover": 10000.})
    if change == "zero":
        bars[["volume", "turnover"]] = 0.
    elif change == "outside":
        bars.loc[2, "turnover"] = 11000.
    elif change == "nan":
        bars.loc[2, "open"] = np.nan
    elif change == "duplicate":
        bars.loc[2, "timestamp"] = bars.loc[1, "timestamp"]
    elif change == "seconds":
        bars.loc[2, "timestamp"] += pd.Timedelta(seconds=1)
    elif change == "wrong_label":
        bars.loc[3, "timestamp"] += pd.Timedelta(minutes=1)
    elif change == "negative":
        bars.loc[2, "volume"] = -1.
    connection = duckdb.connect()
    connection.register("selected_bars", bars)
    actual = window_summary(connection).iloc[0]
    connection.close()
    expected, quote = validate_window(bars)
    assert actual.window_status == expected
    if quote is not None:
        assert actual.volume == quote["volume"]
        assert actual.turnover == quote["turnover"]
