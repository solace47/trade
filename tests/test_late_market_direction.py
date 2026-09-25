"""Check market-state contrasts keep dates as the unit of comparison."""

import pandas as pd

from trade_research.late_market_direction import _state_contrast


def test_month_block_contrast_preserves_constant_daily_state_gap() -> None:
    days = pd.DataFrame([
        {"date": f"2024-{month:02d}-{day:02d}",
         "market_state": "up" if day == 2 else "down",
         "edge": .01 if day == 2 else 0.0}
        for month in range(1, 7)
        for day in (2, 3)
    ])
    result = _state_contrast(days, "edge", "month")
    assert abs(result["up_minus_down"] - .01) < 1e-12
    assert all(abs(value - .01) < 1e-12 for value in result["ci"])
