"""Keep calibration date weighted and outcomes behind the input gate."""

import json

import pandas as pd
import pytest

from trade_research.absolute_ridge_calibration import evaluate, summarize_daily


def test_quintile_spreads_use_same_dates_and_equal_day_weight() -> None:
    rows = []
    for date, top in (("2024-07-01", .10), ("2024-07-08", 0.0)):
        for quintile in range(1, 6):
            cash = top if quintile == 5 else 0.0
            rows.append({
                "date": date, "quintile": quintile, "cash": cash,
                "stress15": cash - .002, "score": quintile * .001,
                "entry": 1.0, "clean_exit": 1.0,
            })

    result = summarize_daily(pd.DataFrame(rows))

    assert result["days"] == 2
    assert result["top_minus_middle"] == pytest.approx(.05)
    assert result["top_minus_low"] == pytest.approx(.05)
    assert result["quintiles"]["5"]["cash"] == pytest.approx(.05)


def test_failed_calibration_gate_does_not_read_outcomes(tmp_path) -> None:
    (tmp_path / "input_audit.json").write_text(
        json.dumps({"outcome_gate_passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="input gate failed"):
        evaluate(tmp_path)
