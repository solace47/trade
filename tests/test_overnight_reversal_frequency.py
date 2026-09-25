import json

import pandas as pd
import pytest

from trade_research.overnight_reversal_frequency import (
    build_inputs, evaluate_gross, select,
)


def test_joint_event_history_excludes_signal_day(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=22).strftime("%Y-%m-%d")
    source = tmp_path / "candidates.parquet"
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    pd.DataFrame([{
        "date": dates[-1], "code": "sh.600000",
        "mean_overnight_gap": .01,
    }]).to_parquet(source, index=False)
    rows = [{
        "date": date, "code": "sh.600000", "open": 10.1,
        "close": 10.0, "preclose": 10.0, "isST": 0,
        "tradestatus": 1,
    } for date in dates]
    for signal_close in (10.0, 11.0):
        rows[-1]["close"] = signal_close
        pd.DataFrame(rows).to_parquet(daily_dir / "part.parquet", index=False)
        inputs = build_inputs(source, daily_dir)
        assert len(inputs) == 1
        assert inputs.event_count.iloc[0] == 20
        assert inputs.event_history_end.iloc[0] == dates[-2]


def test_same_day_quartiles_drop_future_fields_and_reject_2026():
    rows = []
    for date in ("2024-05-06", "2024-10-08", "2025-05-06", "2025-10-08"):
        for index in range(8):
            rows.append({
                "date": date, "code": f"sh.600{index:03d}",
                "board": "main", "price_1449": 10.0,
                "amount_1449": 200_000_000.0, "return_1449": .005,
                "open_gap_today": .001, "mean_overnight_gap": .001,
                "return20_prior_adjusted": .01,
                "event_count": index, "history_end": "2024-01-01",
                "event_history_end": "2024-01-01",
                "future_return": 99.0,
            })
    inputs = pd.DataFrame(rows)
    selected, audit = select(inputs)
    assert "future_return" not in selected
    assert selected.groupby(["half", "arm"]).size().eq(2).all()
    assert set(selected.loc[selected.arm.eq("high"), "event_count"]) == {6, 7}
    assert set(selected.loc[selected.arm.eq("low"), "event_count"]) == {0, 1}
    assert not audit["outcome_gate_passed"]

    inputs.loc[0, "date"] = "2026-01-05"
    with pytest.raises(ValueError, match="future-leaking input"):
        select(inputs)


def test_failed_input_gate_cannot_read_gross_prices(tmp_path):
    (tmp_path / "input_audit.json").write_text(
        json.dumps({"outcome_gate_passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="Input gate failed"):
        evaluate_gross(tmp_path)
