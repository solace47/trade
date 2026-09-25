"""Check the frozen residual adjustment and outcome gate boundary."""

import json

import numpy as np
import pandas as pd
import pytest

from trade_research.morning_path_continuous_eval import (
    _adjusted_edge, _period_quality, evaluate,
)


def test_date_balanced_adjustment_recovers_zero_imbalance_edge():
    rng = np.random.default_rng(81)
    columns = ("gap_pp", "day_pp", "prior20_pp", "tail_pp",
               "max5", "log_price", "log_amount")
    values = rng.normal(size=(24, len(columns)))
    frame = pd.DataFrame(values, columns=[f"delta_{x}" for x in columns])
    frame["date"] = ["2024-02-01"] * 18 + ["2024-02-02"] * 6
    frame["edge_cash"] = .002 + values @ np.arange(1, 8) / 1000
    adjusted = _adjusted_edge(frame)
    assert adjusted["adjusted_edge_at_zero_input_difference"] == pytest.approx(.002)


def test_failed_input_gate_does_not_read_outcomes(tmp_path):
    (tmp_path / "input_audit.json").write_text(
        json.dumps({"outcome_gate_passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="Input gate failed"):
        evaluate(tmp_path)


def test_period_quality_keeps_other_year_source_faults_out(tmp_path):
    issue_dir = tmp_path / "issues"
    issue_dir.mkdir()
    pd.DataFrame([{"date": "2024-03-13", "code": "sh.600002",
                   "kind": "ohlc_disagreement"}]).to_csv(
        issue_dir / "shard_0.csv", index=False)
    period_file = tmp_path / "period.json"
    period_file.write_text(json.dumps({
        "first_date": "2024-01-01", "last_date": "2025-12-31",
        "period_bad_symbols": ["sh.600003"],
    }), encoding="utf-8")
    trades = pd.DataFrame([
        {"date": "2024-03-12", "code": "sh.600001",
         "exit_date": "2024-03-13", "exit_status": "filled",
         "quality_clean_exit": False},
        {"date": "2024-03-12", "code": "sh.600002",
         "exit_date": "2024-03-13", "exit_status": "filled",
         "quality_clean_exit": True},
        {"date": "2024-03-12", "code": "sh.600003",
         "exit_date": "2024-03-13", "exit_status": "filled",
         "quality_clean_exit": True},
    ])
    result = _period_quality(trades, issue_dir, period_file)
    assert result.quality_clean_exit.tolist() == [True, False, False]
