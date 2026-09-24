import json

import pandas as pd
import pytest

from trade_research.annual_cash_direct_eval import (
    _every_fifth_dates, _summarize, evaluate,
)
from trade_research.annual_cash_direct_inputs import _two_cell_coverage


def test_cash_spread_uses_only_same_day_complete_strata() -> None:
    rows = []
    for date, groups in (
            ("2024-05-15", {"cash_supported": 0.03,
                            "low_cash_conversion": -0.01}),
            ("2024-05-16", {"cash_supported": 0.50})):
        for n, (group, value) in enumerate(groups.items()):
            rows.append({
                "date": date, "board": "sh_main", "size_bucket": 1,
                "momentum_bucket": 1, "industry": "C27",
                "cash_group": group, "code": f"sh.{n:06d}",
                "cash_return": value, "entry_filled": True,
                "clean_exit": True,
            })
    report = _summarize(pd.DataFrame(rows))
    section = report["by_year"]["2024"]["full"]
    assert section["two_cell_strata"] == 1
    assert section["two_cell_stock_days"] == 2
    assert section["signal_days"] == 1
    assert section["difference"] == pytest.approx(0.04)
    assert report["by_year"]["2024"]["early"]["difference"] == pytest.approx(
        0.04)
    assert _summarize(pd.DataFrame(rows), exact_industry=True)[
        "by_year"]["2024"]["full"]["difference"] == pytest.approx(0.04)


def test_partial_annual_index_fails_before_outcome_access(tmp_path) -> None:
    rows = []
    for year in (2024, 2025):
        for n, group in enumerate(("cash_supported", "low_cash_conversion")):
            rows.append({
                "date": f"{year}-05-15", "notice_date": f"{year}-04-20",
                "code": f"sh.{n:06d}", "board": "sh_main",
                "size_bucket": 1, "momentum_bucket": 1,
                "industry": "C27",
                "cash_group": group, "float_mv": 100.0,
                "avg20_amount": 50.0,
            })
    inputs = pd.DataFrame(rows)
    input_path = tmp_path / "inputs.parquet"
    audit_path = tmp_path / "audit.json"
    inputs.to_parquet(input_path, index=False)
    audit_path.write_text(json.dumps({
        "stock_days": len(inputs),
        "by_year": _two_cell_coverage(inputs),
        "exact_industry_by_year": _two_cell_coverage(
            inputs, exact_industry=True),
        "source_summary_index_by_year": {"2023": 1, "2024": 1},
        "source_indexed_summary_stock_years": 2,
    }))
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(input_path, audit_path, tmp_path / "nonexistent_outcomes",
                 tmp_path / "nonexistent_issues", tmp_path / "report.json")


def test_every_fifth_session_restarts_each_year() -> None:
    dates = pd.Series([f"2024-05-{day:02d}" for day in range(15, 22)]
                      + [f"2025-05-{day:02d}" for day in range(15, 21)])
    assert _every_fifth_dates(dates) == {
        "2024-05-15", "2024-05-20", "2025-05-15", "2025-05-20",
    }
