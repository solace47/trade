"""Boundary checks that keep future exit prices out of strategy selection."""

import duckdb
import pandas as pd
import pytest

from trade_research.strategy_scan import _last_safe_entry


def test_last_safe_entry_reserves_ten_market_sessions() -> None:
    dates = pd.bdate_range("2023-12-15", periods=11).strftime("%Y-%m-%d")
    connection = duckdb.connect()
    connection.register("available", pd.DataFrame({"date": dates}))
    connection.execute("CREATE VIEW snapshots AS SELECT date FROM available")

    assert _last_safe_entry(connection, "2023-01-01", "2024-01-01") == dates[0]
    with pytest.raises(ValueError, match="Too few trading dates"):
        _last_safe_entry(connection, dates[1], "2024-01-01")
