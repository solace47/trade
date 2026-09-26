import pandas as pd
import pytest

from trade_research.pretrade_upper_limit import evaluate, freeze_inputs


def test_limit_screen_uses_known_preclose_and_stops_before_outcomes(tmp_path):
    prefix_dir = tmp_path / "prefix" / "2024"
    prefix_dir.mkdir(parents=True)
    pd.DataFrame([
        {"date": "2024-03-04", "code": "sh.600001", "price_1449": 11.06,
         "amount_1449": 200_000_000, "quote_outside_traded_range": False},
        {"date": "2024-03-05", "code": "sh.600002", "price_1449": 11.05,
         "amount_1449": 200_000_000, "quote_outside_traded_range": False},
        {"date": "2026-03-05", "code": "sh.600003", "price_1449": 11.06,
         "amount_1449": 200_000_000, "quote_outside_traded_range": False},
    ]).to_parquet(prefix_dir / "part_000.parquet")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    pd.DataFrame([
        {"date": date, "code": code, "preclose": 10.05, "isST": 0,
         "tradestatus": 1, "listing_age_sessions": 100,
         "reference_gap": False}
        for date, code in [("2024-03-04", "sh.600001"),
                           ("2024-03-05", "sh.600002"),
                           ("2026-03-05", "sh.600003")]
    ]).to_parquet(snapshots / "part.parquet")
    output = tmp_path / "result"
    audit = freeze_inputs(prefix_dir.parent, snapshots, output)
    frame = pd.read_parquet(output / "inputs.parquet").sort_values("date")
    assert frame.date.tolist() == ["2024-03-04", "2024-03-05"]
    assert frame.upper_limit.tolist() == [11.06, 11.06]
    assert frame.at_upper.tolist() == [True, False]
    assert not audit["input_gate_passed"]
    with pytest.raises(ValueError, match="Input gate failed"):
        evaluate(output_dir=output)
