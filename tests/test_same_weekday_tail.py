import json

import duckdb
import pandas as pd
import pytest

from trade_research.same_weekday_tail import build_history, freeze, gross


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


def test_freeze_saves_weekday_count_instead_of_upstream_beta_count(tmp_path) -> None:
    signal_date = "2024-01-15"
    source_rows, daily_rows = [], []
    for index in range(30):
        code = f"sh.{600000 + index}"
        source_rows.append({"date": signal_date, "code": code,
                            "price_1449": 10.0, "amount_1449": 200_000_000.0,
                            "return_1449": 0.0, "return_last29": 0.0,
                            "return20_prior_adjusted": 0.0,
                            "prior_count": 999})
        for date in pd.date_range("2023-01-02", periods=55, freq="W-MON"):
            daily_rows.append({"date": date.strftime("%Y-%m-%d"), "code": code,
                               "pctChg": float(index - 15) / 10,
                               "tradestatus": 1, "isST": 0})
    source = tmp_path / "source.parquet"
    pd.DataFrame(source_rows).to_parquet(source)
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    pd.DataFrame(daily_rows).to_parquet(daily_dir / "sample.parquet")
    prefix_audit = tmp_path / "prefix_audit.json"
    prefix_audit.write_text(json.dumps({"input_gate_passed": True}), encoding="utf-8")

    freeze(source_file=source, prefix_audit_file=prefix_audit,
           daily_dir=daily_dir, output_dir=tmp_path / "output")
    inputs = pd.read_parquet(tmp_path / "output" / "inputs.parquet")
    assert len(inputs) == 12
    assert set(inputs.weekday_prior_count) == {52}
    assert set(inputs.last_prior_date) == {"2024-01-08"}
    assert "prior_count" not in inputs


def test_gross_cannot_read_future_prices_after_failed_input_gate(tmp_path) -> None:
    (tmp_path / "input_audit.json").write_text(
        json.dumps({"outcome_gate_passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="Input gate failed"):
        gross(input_dir=tmp_path)
