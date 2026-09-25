"""Market subtraction uses only completed contemporaneous minute returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_research import late_residual_variance as study
from trade_research import late_variance_risk_eval as evaluator


def test_common_minute_move_is_removed_without_changing_raw_variance(
        tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    (tmp_path / "late_variance").mkdir()
    bars = []
    original = []
    timestamps = pd.date_range("2024-03-01 14:20:00", periods=31, freq="min")
    for symbol, minute_move in (("600001", .001), ("600002", .001),
                                ("600003", .03)):
        prices = 10 * np.exp(np.arange(31) * minute_move)
        code = "sh." + symbol
        for timestamp, close in zip(timestamps, prices):
            bars.append({"exchange": "SH", "symbol": symbol,
                         "timestamp": timestamp, "close": close})
        original.append({"date": "2024-03-01", "code": code,
                         "price_1450": prices[-1],
                         "realized_variance_last30": 30 * minute_move**2})
    minute_path = tmp_path / "minutes.parquet"
    pd.DataFrame(bars).to_parquet(minute_path, index=False)
    pd.DataFrame(original).to_parquet(
        tmp_path / "late_variance" / "2024.parquet", index=False)

    audit = study.build_year([str(minute_path)], 2024,
                             tmp_path / "residual", min_market_stocks=3)

    residual = pd.read_parquet(tmp_path / "residual" / "2024_variance.parquet")
    residual = residual.set_index("code")
    assert audit["days"] == 3
    assert audit["min_market_stocks"] == 3
    assert audit["mismatch_original"] == 0
    assert residual.loc["sh.600001", "residual_variance_last30"] == pytest.approx(0)
    assert residual.loc["sh.600002", "residual_variance_last30"] == pytest.approx(0)
    assert residual.loc["sh.600003", "residual_variance_last30"] == pytest.approx(
        30 * .029**2)

    mean_audit = study.build_mean_year(2024, tmp_path / "residual",
                                       min_market_stocks=3)
    mean_residual = pd.read_parquet(
        tmp_path / "residual" / "2024_variance_mean.parquet"
    ).set_index("code")
    assert mean_audit["market_zero_share"] == 0
    assert mean_audit["mismatch_original"] == 0
    assert mean_residual.loc[
        "sh.600001", "residual_variance_last30"] == pytest.approx(
            30 * (.019 / 3)**2)
    assert mean_residual.loc[
        "sh.600003", "residual_variance_last30"] == pytest.approx(
            30 * (.068 / 3)**2)


def test_earlier_years_cannot_enter_extraction(tmp_path):
    with pytest.raises(ValueError, match="2024-2025"):
        study.build_year([str(tmp_path / "minutes.parquet")], 2023,
                         tmp_path / "residual")


def test_full_market_check_rejects_missing_archived_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    empty_keys = pd.DataFrame({"code": pd.Series(dtype=str),
                               "date": pd.Series(dtype=str)})
    monkeypatch.setattr(evaluator, "_quality_keys", lambda path: empty_keys)
    monkeypatch.setattr(evaluator, "_quality_symbols",
                        lambda path: empty_keys[["code"]])
    output = tmp_path / "study"
    output.mkdir()
    pd.DataFrame({"date": ["2024-03-01", "2024-03-01"],
                  "code": ["sh.600001", "sh.600002"],
                  "surprise": [1.6, .8]}).to_parquet(
        output / "all_candidates.parquet", index=False)
    archive = tmp_path / "market_outcomes_ci"
    archive.mkdir()
    pd.DataFrame({"date": ["2024-03-01"], "code": ["sh.600001"],
                  "horizon": [1]}).to_parquet(archive / "part.parquet", index=False)
    with pytest.raises(ValueError, match="lacks archived T\\+1 outcomes"):
        evaluator._full_market_diagnostic(output)
