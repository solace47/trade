"""Guard the cash-aware, same-day comparison used by the residual study."""

import pandas as pd

from trade_research.residual_liquidity import _period


def test_period_counts_unusable_selected_trade_as_zero() -> None:
    days = ["2024-04-08", "2024-04-15"]
    shared = {
        "date": days, "entry_status": ["filled", "filled"],
        "exit_status": ["filled", "filled"],
        "exit_date": ["2024-04-09", "2024-04-16"],
        "exit_delay_sessions": [0, 0],
        "entry_price": [10.0, 10.0], "exit_price": [10.0, 10.0],
        "shares": [100, 100],
    }
    candidate = pd.DataFrame({
        **shared, "quality_clean_exit": [True, False],
        "net_return": [0.02, 0.5],
    })
    control = pd.DataFrame({
        **shared, "quality_clean_exit": [True, True],
        "net_return": [0.01, 0.01],
    })
    result = _period(candidate, control)
    assert result["signals"] == 2
    assert result["clean_exits"] == 1
    assert abs(result["cash_mean"] - 0.01) < 1e-12
    assert abs(result["edge_mean"]) < 1e-12
