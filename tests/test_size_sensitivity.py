"""Repriced outcomes retain exit-period source quality masks."""

from pathlib import Path

import pandas as pd

from trade_research import size_sensitivity


def test_quality_mask_covers_entry_through_exit(tmp_path) -> None:
    pd.DataFrame([{
        "date": "2025-01-03", "code": "sh.600001",
        "kind": "ohlc_disagreement",
    }]).to_csv(tmp_path / "shard_00.csv", index=False)
    pd.DataFrame([{"code": "sh.600002", "invalid_rows": 1}]).to_csv(
        tmp_path / "stocks.csv", index=False
    )
    outcomes = pd.DataFrame([
        {"date": "2025-01-02", "code": "sh.600001",
         "exit_date": "2025-01-03", "exit_status": "filled"},
        {"date": "2025-01-02", "code": "sh.600002",
         "exit_date": "2025-01-03", "exit_status": "filled"},
        {"date": "2025-01-02", "code": "sh.600003",
         "exit_date": None, "exit_status": "entry_not_filled"},
        {"date": "2025-01-02", "code": "sh.600004",
         "exit_date": "2025-01-03", "exit_status": "filled"},
    ])
    result = size_sensitivity._attach_quality(outcomes, tmp_path)
    assert result["quality_clean_exit"].tolist() == [False, False, False, True]


def test_corporate_action_exit_is_not_quality_clean(monkeypatch) -> None:
    monkeypatch.setattr(size_sensitivity, "_quality_keys",
                        lambda _: pd.DataFrame(columns=["date", "code"]))
    monkeypatch.setattr(size_sensitivity, "_quality_symbols",
                        lambda _: pd.DataFrame(columns=["code"]))
    frame = pd.DataFrame([
        {"date": "2024-06-04", "code": "sh.600001", "exit_date": "2024-06-05",
         "exit_status": "filled"},
        {"date": "2024-06-04", "code": "sh.600002", "exit_date": "2024-06-05",
         "exit_status": "corporate_action_unadjusted"},
    ])
    result = size_sensitivity._attach_quality(frame, Path("unused"))
    assert result.quality_clean_exit.tolist() == [True, False]
