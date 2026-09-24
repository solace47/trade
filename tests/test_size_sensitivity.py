"""Repriced outcomes retain exit-period source quality masks."""

import pandas as pd

from trade_research.size_sensitivity import _attach_quality


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

    result = _attach_quality(outcomes, tmp_path)

    assert result["quality_clean_exit"].tolist() == [False, False, False, True]
