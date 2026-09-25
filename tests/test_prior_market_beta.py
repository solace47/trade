import duckdb
import pandas as pd
import pytest

from trade_research.prior_market_beta import build_beta_history, build_candidates


def _candidate_beta(signal_day_return: float) -> tuple[str, int, float]:
    dates = pd.bdate_range("2023-10-02", periods=80).strftime("%Y-%m-%d")
    signal_date = dates[-1]
    daily_rows = []
    for n, date in enumerate(dates):
        market_move = (n % 9 - 4) * .12
        for code, multiplier in (
            ("sh.600000", .8), ("sh.600001", 1.5), ("sz.000001", 1.0)
        ):
            move = multiplier * market_move + (n % 3 - 1) * .03
            if date == signal_date and code == "sh.600000":
                move = signal_day_return
            daily_rows.append((date, code, move, 1, 0))
    daily = pd.DataFrame(daily_rows, columns=[
        "date", "code", "pctChg", "tradestatus", "isST"
    ])
    prefix = pd.DataFrame([{
        "date": signal_date, "code": "sh.600000", "price_1449": 10.0,
        "amount_1449": 200_000_000.0, "return_last29": 0.0,
        "low_1449": 9.0, "high_1449": 11.0,
        "quote_outside_traded_range": False,
    }])
    snapshots = pd.DataFrame([{
        "date": signal_date, "code": "sh.600000", "preclose": 10.0,
        "return20_prior_adjusted": .01, "isST": 0,
        "reference_gap": False, "listing_age_sessions": 100,
    }])
    connection = duckdb.connect()
    try:
        connection.register("daily_raw", daily)
        connection.register("prefix", prefix)
        connection.register("snapshots", snapshots)
        build_beta_history(connection)
        build_candidates(connection)
        row = connection.execute("""
            SELECT beta_date, prior_count, market_beta FROM candidates
        """).fetchone()
        assert row is not None
        return row
    finally:
        connection.close()


def test_signal_day_close_cannot_change_historical_beta() -> None:
    calm = _candidate_beta(0.0)
    extreme = _candidate_beta(20.0)
    assert calm[0] == "2024-01-18"
    assert calm[1] == 60
    assert extreme[0] == calm[0]
    assert extreme[2] == pytest.approx(calm[2], rel=0, abs=1e-12)
