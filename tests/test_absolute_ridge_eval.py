"""Keep unmatched absolute returns separate from matched relative returns."""

import json

import pandas as pd
import pytest

from trade_research.absolute_ridge_eval import evaluate, summarize


def test_unmatched_selection_counts_in_own_cash_only() -> None:
    model = pd.DataFrame([
        {"date": "2024-07-01", "pair_id": "sh.600001", "cash": .02,
         "stress15": .018, "entry_status": "filled", "valid": True},
        {"date": "2024-07-01", "pair_id": "sh.600002", "cash": -.04,
         "stress15": -.042, "entry_status": "filled", "valid": True},
    ])
    control = pd.DataFrame([
        {"date": "2024-07-01", "pair_id": "sh.600001", "cash": .01,
         "stress15": .008, "entry_status": "filled", "valid": True},
    ])

    result = summarize(model, control, primary=False)

    assert result["selected"] == 2
    assert result["matched_pairs"] == 1
    assert result["own_all_cash"] == pytest.approx(-.01)
    assert result["own_matched_cash"] == pytest.approx(.02)
    assert result["matched_edge_cash"] == pytest.approx(.01)


def test_failed_absolute_input_gate_does_not_read_returns(tmp_path) -> None:
    (tmp_path / "input_audit.json").write_text(
        json.dumps({"outcome_gate_passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="input gate failed"):
        evaluate(tmp_path)
