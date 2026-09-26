import duckdb
import pandas as pd
import pytest

from trade_research.cash_ex_inputs import register_history


def history(rows):
    connection = duckdb.connect()
    try:
        connection.register("daily", pd.DataFrame(rows))
        register_history(connection)
        return connection.execute("SELECT * FROM history ORDER BY date").df()
    finally:
        connection.close()


def test_history_excludes_2023_and_current_close_from_the_signal():
    rows = [{"date": d.strftime("%Y-%m-%d"), "code": "sh.600001", "isST": 0,
             "tradestatus": 1, "preclose": 10., "close": 10.1}
            for d in pd.date_range("2024-01-01", periods=22)]
    rows.insert(0, dict(rows[0], date="2023-12-31", close=10000.))
    result = history(rows)
    assert len(result) == 22
    assert result.iloc[0].prior_sessions == 0
    assert result.iloc[20].prior20_return == pytest.approx(1.01 ** 20 - 1)
    rows[-1]["close"] = 99999.
    changed = history(rows)
    pd.testing.assert_series_equal(result.iloc[-1], changed.iloc[-1])
    assert changed.iloc[-1].previous_traded_date < changed.iloc[-1].date


def test_adjustment_reference_and_suspended_days_are_handled_before_lagging():
    rows = [{"date": "2024-01-01", "code": "sh.600001", "isST": 0,
             "tradestatus": 1, "preclose": 10., "close": 10.},
            {"date": "2024-01-02", "code": "sh.600001", "isST": 0,
             "tradestatus": 0, "preclose": 10., "close": 99.},
            {"date": "2024-01-03", "code": "sh.600001", "isST": 0,
             "tradestatus": 1, "preclose": 9., "close": 9.},
            {"date": "2024-01-04", "code": "sh.600001", "isST": 0,
             "tradestatus": 1, "preclose": 9., "close": 9.}]
    result = history(rows)
    assert result.iloc[-1].prior_sessions == 2
    assert result.iloc[-1].prior20_return == 0
    assert result.iloc[-1].previous_close == 9
