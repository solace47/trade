"""Protect the 09:34 decision boundary and its outcome gate."""

import json

import pandas as pd
import pytest

from trade_research import next_morning_exit
from trade_research.next_morning_exit_eval import _pair_policy, evaluate


def test_0934_decisions_use_only_completed_unique_active_bar(
    tmp_path, monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(next_morning_exit, "SOURCE", source)
    codes = [f"sh.60000{x}" for x in range(4)]
    signals = pd.DataFrame({
        "date": ["2024-01-02"] * 4, "code": codes,
        "candidate": ["late_decline", "late_rally_control"] * 2,
        "pair_id": [1, 1, 2, 2], "price_1450": [10.0] * 4,
    })
    signals.to_parquet(source / "selections.parquet", index=False)
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame({
        "calendar_date": ["2024-01-02", "2024-01-03"],
        "is_trading_day": ["1", "1"],
    }).to_parquet(calendar, index=False)
    minutes = tmp_path / "minutes"
    (minutes / "SH").mkdir(parents=True)
    bars = [
        [("09:34", 9.9, 100)],
        [("09:34", 10.1, 100)],
        [("09:33", 9.8, 100)],
        [("09:34", 9.8, 100), ("09:34", 9.8, 100)],
    ]
    for code, rows in zip(codes, bars):
        pd.DataFrame({
            "timestamp": [pd.Timestamp(f"2024-01-03 {time}")
                          for time, _, _ in rows],
            "close": [price for _, price, _ in rows],
            "volume": [volume for _, _, volume in rows],
        }).to_parquet(minutes / "SH" / f"{code.split('.')[1]}.parquet",
                      index=False)
    output = tmp_path / "output"
    audit = next_morning_exit.build_inputs(output, minutes, calendar, workers=1)
    decisions = pd.read_parquet(output / "decisions.parquet").set_index("code")
    assert decisions.loc[codes[0], "exit_choice"] == "close"
    assert decisions.loc[codes[1], "exit_choice"] == "morning"
    assert decisions.loc[codes[2], "exit_choice"] == "morning"
    assert decisions.loc[codes[3], "exit_choice"] == "morning"
    assert decisions.observed_0934.tolist() == [True, True, False, False]
    assert audit["missing_or_inactive_0934"] == 2
    assert not audit["outcome_gate_passed"]
    with pytest.raises(ValueError, match="input gate failed"):
        evaluate(output, source)
    assert json.loads((output / "input_audit.json").read_text())["pairs"] == 2


def test_dynamic_policy_selects_each_paired_legs_own_exit() -> None:
    rows = pd.DataFrame([
        {"date": "2024-01-02", "pair_id": 1, "code": "sh.600001",
         "candidate": "late_decline", "exit_choice": "close",
         "exit_window": window, "cash": cash}
        for window, cash in (("morning", 0.01), ("close", 0.02))
    ] + [
        {"date": "2024-01-02", "pair_id": 1, "code": "sh.600002",
         "candidate": "late_rally_control", "exit_choice": "morning",
         "exit_window": window, "cash": cash}
        for window, cash in (("morning", 0.03), ("close", 0.04))
    ])
    dynamic = _pair_policy(rows, "dynamic", 1)
    assert dynamic.cash_down.iloc[0] == .02
    assert dynamic.cash_up.iloc[0] == .03
