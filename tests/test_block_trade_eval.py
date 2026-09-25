"""The block-trade decision statistic weights signal days equally."""

import pandas as pd

from trade_research.block_trade_eval import _summary
from trade_research.block_trade_inputs import CONTROL, EVENT


def test_primary_edge_is_equal_weighted_by_signal_day() -> None:
    rows = []
    for day, event_values in (("2024-06-03", (.10, .00)),
                              ("2024-07-01", (-.10,))):
        for index, value in enumerate(event_values):
            for label, cash in ((EVENT, value), (CONTROL, .00)):
                rows.append({"date": day, "pair_code": f"s{index}",
                             "candidate": label, "cash_return": cash,
                             "cash_stress10": cash - .001,
                             "entry_status": "filled",
                             "quality_clean_exit": True,
                             "exit_delay_sessions": 0})
    result = _summary(pd.DataFrame(rows), "2024", "full", "2024-06")
    assert result["pairs"] == 3
    assert result["days"] == 2
    assert abs(result["edge_mean"] + .025) < 1e-12
    assert result["event_entry_rate"] == 1
