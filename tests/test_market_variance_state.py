"""The market-state interval must preserve shocks shared within a week."""

import pandas as pd

from trade_research.market_variance_state import _interaction_interval


def test_interaction_resamples_both_states_together_within_week() -> None:
    rows = []
    for elevated_date, normal_date, risk in (
        ("2024-01-02", "2024-01-03", 0.0),
        ("2024-01-09", "2024-01-10", .25),
        ("2024-01-16", "2024-01-17", .75),
        ("2024-01-23", "2024-01-24", 1.0),
    ):
        rows.extend((
            {"date": elevated_date, "market_state": "elevated",
             "risk_difference": risk},
            {"date": normal_date, "market_state": "normal",
             "risk_difference": risk},
        ))

    assert _interaction_interval(pd.DataFrame(rows), "week", 1) == [0.0, 0.0]
