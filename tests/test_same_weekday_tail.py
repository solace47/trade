import json

import duckdb
import pandas as pd
import pytest

from trade_research.same_weekday_tail import build_history, gross


def _signal_score(today_pct_change: float) -> tuple[int, float, str]:
    mondays = pd.date_range("2023-01-02", periods=55, freq="W-MON")
    rows = [{"date": date.strftime("%Y-%m-%d"), "code": "sh.600000",
             "pctChg": float(offset % 5 - 2), "tradestatus": 1, "isST": 0}
            for offset, date in enumerate(mondays)]
    rows[-1]["pctChg"] = today_pct_change
    connection = duckdb.connect()
    try:
        connection.register("daily_source", pd.DataFrame(rows))
        build_history(connection)
        return connection.execute("""
            SELECT prior_count, weekday_mean, last_prior_date
            FROM weekday_history WHERE date = ?
        """, [rows[-1]["date"]]).fetchone()
    finally:
        connection.close()


def test_signal_day_close_cannot_change_same_weekday_score() -> None:
    calm = _signal_score(0.0)
    extreme = _signal_score(20.0)
    assert calm[0] == 52
    assert calm[2] == "2024-01-08"
    assert extreme == calm


def test_gross_cannot_read_future_prices_after_failed_input_gate(tmp_path) -> None:
    (tmp_path / "input_audit.json").write_text(
        json.dumps({"outcome_gate_passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="Input gate failed"):
        gross(input_dir=tmp_path)
