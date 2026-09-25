import duckdb
import pandas as pd
import pytest

from trade_research.overnight_composition import build_history, build_candidates


def _candidate_mean(signal_day_daily_open: float) -> tuple[str, int, float]:
    dates = pd.bdate_range("2023-12-01", periods=30).strftime("%Y-%m-%d")
    signal_date = dates[-1]
    daily_rows = []
    codes = ["sh.600000", *(f"sh.{600100 + n}" for n in range(24))]
    for code in codes:
        gap = .002 if code == "sh.600000" else -.001
        previous_close = 10.0
        for date in dates:
            opening = previous_close * (1 + gap)
            if date == signal_date and code == "sh.600000":
                opening = signal_day_daily_open
            daily_rows.append((date, code, opening, opening,
                               previous_close, 1, 0))
            previous_close = opening
    daily = pd.DataFrame(daily_rows, columns=[
        "date", "code", "open", "close", "preclose", "tradestatus", "isST"
    ])
    prefix = pd.DataFrame([{
        "date": signal_date, "code": code, "price_1449": 10.0,
        "amount_1449": 200_000_000.0, "return_last29": 0.0,
        "low_1449": 9.0, "high_1449": 11.0,
        "quote_outside_traded_range": False,
    } for code in codes])
    snapshots = pd.DataFrame([{
        "date": signal_date, "code": code, "preclose": 10.0,
        "open_1450": 10.0, "return20_prior_adjusted": .01,
        "isST": 0, "reference_gap": False,
        "listing_age_sessions": 100,
    } for code in codes])
    connection = duckdb.connect()
    try:
        connection.register("daily_raw", daily)
        connection.register("prefix", prefix)
        connection.register("snapshots", snapshots)
        build_history(connection)
        build_candidates(connection)
        row = connection.execute("""
            SELECT history_end, valid_days, mean_overnight_gap
            FROM candidates WHERE code = 'sh.600000'
        """).fetchone()
        assert row is not None
        return row
    finally:
        connection.close()


def test_signal_day_daily_open_cannot_change_historical_mean() -> None:
    calm = _candidate_mean(10.0)
    strong = _candidate_mean(10.8)
    assert calm[0] == "2024-01-10"
    assert calm[1] == 20
    assert strong[0] == calm[0]
    assert strong[2] == pytest.approx(calm[2], rel=0, abs=1e-12)
