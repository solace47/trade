"""Verify the historical open-close numerator and future-day exclusion."""

import numpy as np
import pandas as pd

from trade_research.open_close_illiquidity import build


def test_open_close_amihud_uses_only_history_through_margin_date(tmp_path) -> None:
    dates = pd.bdate_range("2023-09-01", "2024-01-05")
    dates_as_text = dates.strftime("%Y-%m-%d")
    daily = pd.DataFrame({
        "date": dates_as_text, "code": "sh.600000", "open": 10.0,
        "close": 10.1, "amount": 100_000_000.0, "tradestatus": 1,
    })
    daily.loc[daily.date.eq("2024-01-05"), "close"] = 12.0
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    daily.to_parquet(daily_dir / "one.parquet", index=False)
    calendar_path = tmp_path / "calendar.parquet"
    pd.DataFrame({"calendar_date": dates_as_text,
                  "is_trading_day": "1"}).to_parquet(calendar_path, index=False)
    universe_path = tmp_path / "universe.parquet"
    pd.DataFrame({"trade_date": ["2024-01-04"],
                  "code": ["sh.600000"]}).to_parquet(universe_path, index=False)
    output_path = tmp_path / "amihud.parquet"
    result = build(daily_dir, calendar_path, universe_path, output_path)
    assert len(result) == 1
    assert result.trade_date.iloc[0] == "2024-01-04"
    assert result.n.iloc[0] == 60
    assert result.span.iloc[0] == 59
    assert np.isclose(result.oc_amihud60.iloc[0], .01)
