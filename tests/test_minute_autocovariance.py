"""Check the sealed input definition and deterministic daily matching."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from trade_research.minute_autocovariance import build_year, select_inputs


def test_autocorrelation_uses_only_completed_bars(tmp_path: Path) -> None:
    labels = pd.date_range("2024-01-02 14:20", periods=32, freq="min")
    returns = np.array([.001, -.002] * 15)
    closes = 10 * np.exp(np.r_[0, np.cumsum(returns)])
    source = tmp_path / "minutes.parquet"
    output = tmp_path / "acf.parquet"
    pd.DataFrame({
        "symbol": "600000", "exchange": "SH", "timestamp": labels,
        "close": np.r_[closes, 100.0],
    }).to_parquet(source, index=False)

    build_year([str(source)], 2024, output, 1)

    row = pd.read_parquet(output).iloc[0]
    centered = returns - returns.mean()
    expected = np.sum(centered[1:] * centered[:-1]) / np.sum(centered ** 2)
    assert row.price_1450 == pytest.approx(closes[-1])
    assert row.last_minute_log_return == pytest.approx(returns[-1])
    assert row.rho1 == pytest.approx(expected)
    assert row.nonzero_returns == 30


def test_autocorrelation_excludes_sparse_and_pre_2024(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2024-2025"):
        build_year(["unused.parquet"], 2023, tmp_path / "old.parquet")
    labels = pd.date_range("2024-01-02 14:20", periods=32, freq="min")
    source = tmp_path / "sparse.parquet"
    pd.DataFrame({
        "symbol": "600000", "exchange": "SH", "timestamp": labels,
        "close": [10.0] * 30 + [10.01, 99.0],
    }).to_parquet(source, index=False)
    build_year([str(source)], 2024, tmp_path / "sparse_out.parquet", 1)
    assert pd.read_parquet(tmp_path / "sparse_out.parquet").empty


def test_daily_pairs_use_strong_and_weak_quintiles_only() -> None:
    date = "2024-01-03"
    rows = []
    for index in range(40):
        code = f"sh.{600001 + index:06d}"
        rows.append({
            "date": date, "code": code, "isST": 0,
            "listing_age_sessions": 100, "reference_gap": False,
            "quote_outside_traded_range": False, "price_1450": 10.0,
            "amount_1450": 200_000_000,
            "return20_prior_adjusted": 0.0, "return_1450": -.01,
            "return_last30": -.004, "last_minute_log_return": -.001,
            "realized_variance_last30": 1e-5,
            "rho1": -.8 + index * .04,
        })
    frame = pd.DataFrame(rows)
    connection = duckdb.connect()
    for name, columns in (
        ("snapshots", ["date", "code", "isST", "listing_age_sessions",
                        "reference_gap", "quote_outside_traded_range",
                        "price_1450", "amount_1450", "return20_prior_adjusted",
                        "return_1450"]),
        ("intraday", ["date", "code", "price_1450", "return_last30"]),
        ("autocovariance", ["date", "code", "price_1450",
                            "last_minute_log_return",
                            "realized_variance_last30", "rho1"]),
        ("variance", ["date", "code", "price_1450",
                      "realized_variance_last30"]),
    ):
        connection.register(name, frame[columns])

    signals, audit = select_inputs(connection)

    assert audit["base_pool"] == 40
    assert audit["strong_pool"] == audit["weak_pool"] == 8
    assert audit["by_half"][0]["attempts"] == 5
    assert audit["pairs"] == 5
    assert audit["stage_reach"]["2024H1"]["variance"] == 5
    assert audit["outcome_gate_passed"] is False
    assert signals.loc[signals.candidate.eq("strong_negative"), "rho1"].max() < -.5
    assert signals.loc[signals.candidate.eq("weak_negative"), "rho1"].min() > .4
    assert signals.duplicated(["date", "code"]).sum() == 0
