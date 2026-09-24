import pandas as pd

from trade_research.buyback_eval import _summary


def test_buyback_spread_weights_signal_days_equally() -> None:
    rows = []
    for date, event_returns in (
            ("2024-01-02", [0.1, 0.1]),
            ("2024-01-03", [-0.1])):
        for value in event_returns:
            for candidate, cash in (("buyback_plan", value),
                                    ("same_day_nonannouncer", 0.0)):
                rows.append({"date": date, "candidate": candidate,
                             "cash_return": cash, "cash_stress10": cash,
                             "entry_status": "filled",
                             "quality_clean_exit": True,
                             "exit_delay_sessions": 0})
    summary = _summary(pd.DataFrame(rows), "2024", "full")
    assert summary["pairs"] == 3
    assert summary["days"] == 2
    assert abs(summary["edge_mean"]) < 1e-12
