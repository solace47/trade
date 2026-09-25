"""Check the frozen minute-bounce comparison's weighting."""

import pandas as pd
import pytest

from trade_research.minute_bounce_cost import _date_equal_drift


def test_drift_comparison_weights_boards_then_dates_equally() -> None:
    rows = []
    for date, board, low, high, repeats in (
        ("2024-01-02", "main", 12.0, 2.0, 10),
        ("2024-01-02", "star", 4.0, 2.0, 1),
        ("2024-01-03", "main", 1.0, 3.0, 2),
    ):
        for _ in range(repeats):
            rows.append({"half": "2024H1", "date": date, "board": board,
                         "rho_third": 1, "drift_bps": low})
            rows.append({"half": "2024H1", "date": date, "board": board,
                         "rho_third": 3, "drift_bps": high})
    daily = _date_equal_drift(pd.DataFrame(rows))

    assert daily.low_minus_high_bps.tolist() == pytest.approx([6.0, -2.0])
    assert daily.low_minus_high_bps.mean() == pytest.approx(2.0)
