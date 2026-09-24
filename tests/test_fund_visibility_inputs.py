import pandas as pd
import pytest

from trade_research.fund_visibility_inputs import (
    _four_cell_coverage, _holding_intervals, _load_sources, _visibility, build,
)


def test_strict_publication_day_and_rejected_replacement() -> None:
    metadata = pd.DataFrame([
        {"uploadInfoId": 1, "fundId": 10, "available_after": "2024-01-10",
         "reportYear": "2023", "report_quarter": 4},
        {"uploadInfoId": 2, "fundId": 10, "available_after": "2024-02-01",
         "reportYear": "2024", "report_quarter": 1},
        {"uploadInfoId": 3, "fundId": 20, "available_after": "2024-01-01",
         "reportYear": "2023", "report_quarter": 4},
    ])
    # Report 2 has no parsed holdings. It still ends report 1's validity.
    holdings = pd.DataFrame([
        {"uploadInfoId": 1, "fundId": 10, "code": "sh.600000"},
        {"uploadInfoId": 3, "fundId": 20, "code": "sh.600000"},
    ])
    intervals = _holding_intervals(metadata, holdings)
    dates = ["2024-01-10", "2024-01-11", "2024-02-01",
             "2024-02-02", "2024-12-31"]
    universe = pd.DataFrame({"date": dates, "code": "sh.600000"})
    result = _visibility(universe, intervals)
    assert result.visible_funds.tolist() == [1, 2, 2, 1, 0]


def test_report_ownership_mismatch_fails() -> None:
    metadata = pd.DataFrame([{
        "uploadInfoId": 1, "fundId": 10, "available_after": "2024-01-10",
        "reportYear": "2023", "report_quarter": 4,
    }])
    holdings = pd.DataFrame([{
        "uploadInfoId": 1, "fundId": 20, "code": "sh.600000",
    }])
    with pytest.raises(ValueError, match="different fund owners"):
        _holding_intervals(metadata, holdings)


def test_late_old_quarter_does_not_replace_newer_disclosed_quarter() -> None:
    metadata = pd.DataFrame([
        {"uploadInfoId": 1, "fundId": 10, "available_after": "2025-04-22",
         "reportYear": "2025", "report_quarter": 1},
        {"uploadInfoId": 2, "fundId": 10, "available_after": "2025-05-24",
         "reportYear": "2024", "report_quarter": 4},
        {"uploadInfoId": 3, "fundId": 10, "available_after": "2025-07-21",
         "reportYear": "2025", "report_quarter": 2},
    ])
    holdings = pd.DataFrame([
        {"uploadInfoId": 1, "fundId": 10, "code": "sh.600000"},
        {"uploadInfoId": 2, "fundId": 10, "code": "sh.600001"},
        {"uploadInfoId": 3, "fundId": 10, "code": "sh.600002"},
    ])
    intervals = _holding_intervals(metadata, holdings)
    universe = pd.DataFrame([
        {"date": "2025-05-25", "code": "sh.600000"},
        {"date": "2025-05-25", "code": "sh.600001"},
        {"date": "2025-07-22", "code": "sh.600000"},
        {"date": "2025-07-22", "code": "sh.600002"},
    ])
    assert _visibility(universe, intervals).visible_funds.tolist() == [1, 0, 0, 1]


def test_partial_quarter_archive_cannot_build_visibility(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="2023q4"):
        _load_sources(tmp_path / "index", tmp_path / "holdings")


def test_four_cell_input_audit_counts_missing_strata() -> None:
    rows = []
    for date, combinations in (
            ("2024-05-15", [(False, 1), (False, 5), (True, 1), (True, 5)]),
            ("2024-05-16", [(False, 1), (False, 5), (True, 1)])):
        for number, (visible, quintile) in enumerate(combinations):
            rows.append({"date": date, "board": "sh_main", "size_bucket": 1,
                         "cash_group": "low_cash_conversion",
                         "fund_visible": visible, "quintile": quintile,
                         "code": f"sh.{number:06d}",
                         "float_mv": 100.0 if visible else 90.0,
                         "avg20_amount": 100.0 if visible else 80.0})
    coverage = _four_cell_coverage(pd.DataFrame(rows))[
        "low_cash_conversion"]["2024"]
    assert coverage["source_strata"] == 2
    assert coverage["four_cell_strata"] == 1
    assert coverage["source_signal_days"] == 2
    assert coverage["four_cell_signal_days"] == 1
    assert coverage["source_extreme_stock_days"] == 7
    assert coverage["four_cell_stock_days"] == 4
    assert coverage["median_stratum_absent_visible_ratio"] == {
        "float_mv": {"q1": 0.9, "q5": 0.9},
        "avg20_amount": {"q1": 0.8, "q5": 0.8},
    }


def test_complete_eight_quarter_sources_build_audited_inputs(tmp_path) -> None:
    quarters = ("2023q4", "2024q1", "2024q2", "2024q3", "2024q4",
                "2025q1", "2025q2", "2025q3")
    index_dir = tmp_path / "index"
    holdings_dir = tmp_path / "holdings"
    index_dir.mkdir()
    holdings_dir.mkdir()
    for report_id, quarter in enumerate(quarters, start=1):
        year, number = int(quarter[:4]), int(quarter[-1])
        available = {1: f"{year}-04-20", 2: f"{year}-07-20",
                     3: f"{year}-10-20", 4: f"{year + 1}-01-20"}[number]
        base = {"uploadInfoId": report_id, "fundId": 10,
                "reportYear": str(year), "report_quarter": number}
        pd.DataFrame([{**base, "available_after": available,
                       "uploadDate": available,
                       "reportSendDate": available}]).to_parquet(
            index_dir / f"{quarter}.parquet", index=False)
        pd.DataFrame([{**base, "code": "sh.600000"}]).to_parquet(
            holdings_dir / f"{quarter}.parquet", index=False)
        pd.DataFrame([{**base, "status": "parsed"}]).to_parquet(
            holdings_dir / f"{quarter}_audit.parquet", index=False)
    universe_path = tmp_path / "universe.parquet"
    pd.DataFrame([
        {"date": day, "code": "sh.600000", "board": "sh_main",
         "size_bucket": 1, "cash_group": "low_cash_conversion",
         "quintile": 1, "float_mv": 1_000_000_000.0,
         "avg20_amount": 40_000_000.0}
        for day in ("2024-05-15", "2025-05-15")
    ]).to_parquet(universe_path, index=False)
    output = tmp_path / "inputs.parquet"
    report_path = tmp_path / "inputs.json"
    report = build(index_dir, holdings_dir, universe_path, output, report_path)
    assert report["source_reports"] == 8
    assert set(report["quarter_source_reports"]) == set(quarters)
    assert report["eligible_stock_days"] == 2
    assert report["visible_stock_days"] == 2
    assert pd.read_parquet(output).visible_funds.tolist() == [1, 1]
    assert report_path.exists()
    corrupt = pd.read_parquet(index_dir / "2023q4.parquet")
    corrupt["available_after"] = "2024-01-19"
    corrupt.to_parquet(index_dir / "2023q4.parquet", index=False)
    with pytest.raises(ValueError, match="public availability"):
        build(index_dir, holdings_dir, universe_path, output, report_path)
