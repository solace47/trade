import pandas as pd
import pytest

from trade_research.fund_visibility_inputs import _holding_intervals, _visibility


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
