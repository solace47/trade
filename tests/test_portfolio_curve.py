"""Cash and daily valuation checks for the conditional risk curve."""

import pandas as pd

from trade_research.hf_outcomes import Assumptions, _fees
from trade_research.portfolio_curve import curve


def test_one_completed_trade_reconciles_to_cash_after_sale(tmp_path) -> None:
    code = "sh.600000"
    pd.DataFrame({
        "date": ["2025-01-02", "2025-01-03"],
        "close": [10.2, 11.1], "tradestatus": [1, 1],
    }).to_parquet(tmp_path / "sh_600000.parquet")
    trades = pd.DataFrame([{
        "date": "2025-01-02", "code": code, "daily_rank": 1,
        "entry_status": "filled", "entry_price": 10.0, "shares": 100,
        "exit_status": "filled", "exit_date": "2025-01-03",
        "exit_price": 11.0, "quality_clean_exit": True,
    }])
    frame, summary = curve(trades, tmp_path,
                           ["2025-01-02", "2025-01-03"], 2_000.0)
    terms = Assumptions()
    expected = 2_000.0 - (1_000 + _fees(1_000, "buy", terms, "2025-01-02")) \
        + (1_100 - _fees(1_100, "sell", terms, "2025-01-03"))
    assert abs(summary["final_equity"] - expected) < 1e-9
    assert frame["open_positions"].tolist() == [1, 0]
    assert summary["purchased_after_cash_limit"] == 1
    assert summary["capital_blocked"] == 0
