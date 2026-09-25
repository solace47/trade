"""Auction input audit must ignore 14:58 volume, including positive SH bars."""

import pandas as pd
import pytest

from trade_research.closing_auction_entry import build_inputs
from trade_research.closing_auction_entry_eval import evaluate
from trade_research.hf_outcomes import ENTRY_WINDOWS


def test_auction_bar_quality_and_order_capacity(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    codes = [f"sh.60000{x}" for x in range(4)]
    pd.DataFrame({
        "date": ["2024-01-02"] * 4, "code": codes,
        "candidate": ["late_decline", "late_rally_control"] * 2,
        "pair_id": [1, 1, 2, 2],
    }).to_parquet(source / "selections.parquet", index=False)
    minutes = tmp_path / "minutes"
    (minutes / "SH").mkdir(parents=True)
    bars = [
        [("14:58", 10.0, 200_000, 2_000_000),
         ("15:00", 10.0, 30_000, 300_000)],
        [("14:58", 10.0, 200_000, 2_000_000),
         ("15:00", 10.0, 10_000, 100_000)],
        [("15:00", 10.0, 30_000, 300_000)] * 2,
        [("15:00", 10.0, 30_000, 305_000)],
    ]
    for code, rows in zip(codes, bars):
        pd.DataFrame({
            "timestamp": [pd.Timestamp(f"2024-01-02 {time}")
                          for time, _, _, _ in rows],
            "close": [price for _, price, _, _ in rows],
            "volume": [volume for _, _, volume, _ in rows],
            "turnover": [turnover for _, _, _, turnover in rows],
        }).to_parquet(minutes / "SH" / f"{code.split('.')[1]}.parquet",
                      index=False)
    output = tmp_path / "output"
    audit = build_inputs(output, minutes, source, workers=1)
    inputs = pd.read_parquet(output / "auction_inputs.parquet").set_index("code")
    assert inputs.loc[codes[0], "auction_volume"] == 30_000
    assert inputs.loc[codes[1], "auction_volume"] == 10_000
    assert inputs.valid_auction_bar.tolist() == [True, True, False, False]
    assert inputs.capacity_20k.tolist() == [True, False, False, False]
    assert audit["invalid_auction_bars"] == 2
    assert audit["capacity_20k_failures"] == 3
    assert not audit["outcome_gate_passed"]
    assert ENTRY_WINDOWS["auction"] == ("1500",)
    with pytest.raises(ValueError, match="input gate failed"):
        evaluate(output, source)
