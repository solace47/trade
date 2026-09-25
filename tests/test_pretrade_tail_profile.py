"""The historical closing-volume profile must remain causal and complete."""

import pandas as pd

from trade_research.pretrade_tail_profile import build_history, forecast_ratios


def test_profile_ignores_same_day_and_future_volume() -> None:
    days = pd.bdate_range("2023-12-15", periods=12)
    history = pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"), "code": "sh.600001",
        "ratio": [1.0] * 10 + [50.0, 100.0],
    })
    signals = pd.DataFrame({"date": [days[10].strftime("%Y-%m-%d")],
                            "code": ["sh.600001"]})
    original = forecast_ratios(signals, history)
    changed = history.copy()
    changed.loc[10:, "ratio"] = [0.0, 0.0]
    assert original.history_days.tolist() == [10]
    assert original.historical_ratio_p20.tolist() == [1.0]
    pd.testing.assert_frame_equal(original, forecast_ratios(signals, changed))


def test_profile_requires_ten_recent_sessions() -> None:
    signals = pd.DataFrame({"date": ["2024-03-01"], "code": ["sz.000001"]})
    old = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10).strftime("%Y-%m-%d"),
        "code": "sz.000001", "ratio": [1.0] * 10,
    })
    result = forecast_ratios(signals, old)
    assert result.history_days.tolist() == [0]
    assert result.historical_ratio_p20.isna().all()


def test_history_rejects_incomplete_and_duplicate_minute_windows(tmp_path) -> None:
    root = tmp_path / "raw"
    for exchange in ("SH", "SZ"):
        (root / exchange).mkdir(parents=True)
    valid = [46, 47, 48, 49, 50, 52, 53, 54, 55]
    rows = []
    for date, minutes in (("2023-12-29", valid),
                          ("2024-01-02", valid[:-1]),
                          ("2024-01-03", valid + [55])):
        rows.extend({"timestamp": pd.Timestamp(f"{date} 14:{minute:02d}"),
                     "exchange": "SH", "symbol": "600001", "volume": 100}
                    for minute in minutes)
    pd.DataFrame(rows).to_parquet(root / "SH" / "stock.parquet")
    pd.DataFrame([{"timestamp": pd.Timestamp("2025-01-02 14:50"),
                   "exchange": "SZ", "symbol": "000001", "volume": 100}]
                 ).to_parquet(root / "SZ" / "stock.parquet")
    result = pd.read_parquet(build_history(root, tmp_path / "history.parquet"))
    assert result.date.astype(str).tolist() == ["2023-12-29"]
    assert result.ratio.tolist() == [.8]
