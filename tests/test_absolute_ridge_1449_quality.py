"""A bad-symbol table must not be reduced to its column names."""

import json

import pandas as pd
import pytest

from trade_research import absolute_ridge_1449_eval as evaluation


@pytest.mark.parametrize("bad_codes", [["sh.600001"], []])
def test_period_quality_keeps_symbol_and_holding_day_exclusions(
        tmp_path, monkeypatch, bad_codes):
    report = tmp_path / "quality.json"
    report.write_text(json.dumps({
        "first_date": "2024-01-01", "last_date": "2025-12-31",
        "period_bad_symbols": bad_codes,
    }))
    monkeypatch.setattr(evaluation, "PERIOD_QUALITY", report)
    monkeypatch.setattr(evaluation, "_quality_keys", lambda _: pd.DataFrame({
        "code": ["sh.600002", "sh.600003"],
        "date": ["2025-01-03", "2025-01-02"],
    }))
    rows = pd.DataFrame([{
        "date": "2025-01-02", "code": f"sh.60000{i}",
        "exit_date": "2025-01-06", "exit_status": "filled",
        "target_notional": 20000, "entry_window": "baseline",
        "exit_window": "close", "horizon": 1,
        "quality_clean_exit": False,
    } for i in range(1, 5)])
    result, released = evaluation.apply_period_quality(rows)
    actual = result.set_index("code").quality_clean_exit.to_dict()
    assert actual == {
        "sh.600001": not bool(bad_codes), "sh.600002": False,
        "sh.600003": False, "sh.600004": True,
    }
    assert released == (1 if bad_codes else 2)
