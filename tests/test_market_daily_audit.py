"""Historical universe coverage includes delisted stocks through their last listing day."""

import json

import pandas as pd
import pytest

from trade_research.market_daily_audit import audit


@pytest.mark.parametrize("boundary_placeholder", [False, True])
def test_delisting_boundary_row_is_optional(tmp_path, boundary_placeholder):
    metadata = tmp_path / "metadata"
    daily = tmp_path / "daily"
    metadata.mkdir()
    daily.mkdir()
    (metadata / "selection.json").write_text(json.dumps({
        "first_date": "2026-06-08", "last_date": "2026-06-10",
    }), encoding="utf-8")
    pd.DataFrame([{
        "code": "sh.688287", "ipoDate": "2022-05-25", "outDate": "2026-06-10",
    }]).to_parquet(metadata / "symbols.parquet", index=False)
    pd.DataFrame([{"calendar_date": date, "is_trading_day": "1"}
                  for date in ("2026-06-08", "2026-06-09", "2026-06-10")]).to_parquet(
        metadata / "calendar.parquet", index=False,
    )
    rows = [{
        "date": date, "code": "sh.688287", "tradestatus": status,
        "open": .35, "high": .37, "low": .35, "close": .37,
        "volume": 1000 if status else None, "amount": 370 if status else None,
        "isST": 0,
    } for date, status in (("2026-06-08", 1), ("2026-06-09", 0))]
    if boundary_placeholder:
        rows.append({**rows[-1], "date": "2026-06-10"})
    pd.DataFrame(rows).to_parquet(
        daily / "sh_688287.parquet", index=False,
    )
    report = audit(tmp_path)
    assert report["expected_days"] == 2
    assert report["missing_days"] == 0
    assert report["accepted"] is True
