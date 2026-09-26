import numpy as np
import pandas as pd
import pytest

from trade_research.round_number_entry import _stock, evaluate_day, validate_window


def bars(volume=20000, price=10.):
    return pd.DataFrame({"timestamp": pd.date_range("2024-05-06 14:52", periods=4, freq="min"),
        "open": price, "high": price + .02, "low": price - .02, "close": price,
        "volume": volume, "turnover": volume * price})


def signal():
    return {"date": "2024-05-06", "code": "sh.600001", "half": "2024H1", "side": "above",
            "price_1449": 10.01, "quote_cents": 1001, "preclose": 10., "round_yuan": 10}


def test_fixed_quote_size_does_not_follow_execution_price():
    rows = evaluate_day(signal(), bars())
    assert [r["target_shares"] for r in rows] == [1900, 9900]
    assert rows[0]["aggregate_filled"] and rows[0]["participation_filled"]
    assert rows[0]["aggregate_price"] == pytest.approx(10.005)
    assert rows[0]["decision_drift_bps"] == pytest.approx((10/10.01-1)*10000)
    assert rows[0]["original_side_retained"] is False
    assert not rows[1]["aggregate_filled"]
    assert rows[1]["participation_shares"] == 8000


def test_source_unknown_is_not_a_known_zero_fill():
    for broken in [bars().iloc[:3], pd.concat([bars(), bars().iloc[:1]])]:
        rows = evaluate_day(signal(), broken)
        assert all(r["aggregate_status"] == "source_unknown" for r in rows)
        assert rows[0]["participation_shares"] is None and rows[0]["raw_vwap"] is None
    assert evaluate_day(signal(), bars(), False)[0]["window_status"] == "missing_source"
    zero = evaluate_day(signal(), bars(volume=0))[0]
    assert zero["bars_valid"] and zero["aggregate_status"] == "no_liquidity"
    assert zero["participation_shares"] == 0


@pytest.mark.parametrize("column,value", [("volume", -1), ("turnover", np.nan),
    ("high", 9.9), ("low", 10.1), ("open", 0), ("turnover", 0)])
def test_bad_bar_stays_unknown(column, value):
    frame = bars()
    frame.loc[0, column] = value
    assert validate_window(frame)[0] == "invalid_numeric_bar"


def test_vwap_range_tolerance_and_minute_timestamp_are_explicit():
    frame = bars()
    frame.loc[0, "turnover"] = frame.loc[0, "volume"] * 10.031
    assert validate_window(frame)[0] == "vwap_outside_bar_range"
    frame.loc[0, "turnover"] = frame.loc[0, "volume"] * 10.030
    assert validate_window(frame)[0] == "valid"
    frame.loc[0, "timestamp"] += pd.Timedelta(seconds=1)
    assert validate_window(frame)[0] == "incomplete_or_duplicate_window"


def test_per_minute_lot_capacity_and_upper_limit_are_not_aggregate_capacity():
    frame = bars(volume=4999)
    row = evaluate_day(signal(), frame)[0]
    assert row["aggregate_filled"]
    assert row["participation_shares"] == 1600
    assert not row["participation_filled"]
    row = evaluate_day(signal(), bars(price=11.))[0]
    assert row["aggregate_status"] == "estimated_upper_limit"
    assert row["participation_shares"] == 0


def test_cash_need_can_exceed_decision_target_and_is_not_clipped():
    row = evaluate_day(signal(), bars(price=10.70))[0]
    assert row["aggregate_filled"] and row["target_shares"] == 1900
    assert row["aggregate_buy_cost"] > 20000
    assert row["excess_cash_yuan"] == pytest.approx(row["aggregate_buy_cost"] - 20000)


def test_raw_extraction_returns_only_frozen_dates_and_four_entry_minutes(tmp_path):
    original = bars()
    other_time, other_date, holdout = original.copy(), original.copy(), original.copy()
    other_time["timestamp"] += pd.Timedelta(minutes=30)
    other_date["timestamp"] += pd.Timedelta(days=1)
    holdout["timestamp"] += pd.DateOffset(years=2)
    # Artificial bars only: the extraction must never return these future rows.
    for extra in (other_time, other_date, holdout):
        extra["turnover"] = -1
    destination = tmp_path / "SH"
    destination.mkdir()
    pd.concat([original, other_time, other_date, holdout]).to_parquet(destination / "600001.parquet", index=False)
    rows, raw, source = _stock(("sh.600001", pd.DataFrame([signal()])), tmp_path)
    assert len(raw) == source["extracted_rows"] == 4
    assert raw.date.eq("2024-05-06").all()
    assert rows[0]["bars_valid"] and rows[0]["aggregate_filled"]
