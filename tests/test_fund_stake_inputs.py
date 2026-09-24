import pandas as pd
import pytest

from trade_research.fund_stake_inputs import (
    _active_stakes, _quarter_float_shares, _stake_by_stock_day,
)


def test_quarter_end_uses_last_valid_trade_before_publication(tmp_path) -> None:
    daily = tmp_path / "daily"
    daily.mkdir()
    pd.DataFrame([
        {"code": "sh.600000", "date": "2023-12-28", "volume": 90,
         "turn": 9.0, "close": 10.0, "tradestatus": 1, "adjustflag": 3},
        {"code": "sh.600000", "date": "2023-12-29", "volume": 100,
         "turn": 10.0, "close": 10.0, "tradestatus": 1, "adjustflag": 3},
        {"code": "sh.600000", "date": "2024-01-02", "volume": 500,
         "turn": 10.0, "close": 10.0, "tradestatus": 1, "adjustflag": 3},
    ]).to_parquet(daily / "sample.parquet", index=False)
    result = _quarter_float_shares(daily)
    selected = result.loc[result.reportYear.eq(2023)
                          & result.report_quarter.eq(4)].iloc[0]
    assert selected.denominator_date == "2023-12-29"
    assert selected.float_shares == 1000.0


def test_rejected_report_stops_old_stake_and_missing_denominator_is_not_zero(
        ) -> None:
    metadata = pd.DataFrame([
        {"uploadInfoId": 1, "fundId": 10,
         "available_after": "2024-01-10", "reportYear": "2023",
         "report_quarter": 4},
        {"uploadInfoId": 2, "fundId": 10,
         "available_after": "2024-02-01", "reportYear": "2024",
         "report_quarter": 1},
        {"uploadInfoId": 3, "fundId": 20,
         "available_after": "2024-01-10", "reportYear": "2023",
         "report_quarter": 4},
    ])
    positions = pd.DataFrame([
        {"uploadInfoId": 1, "fundId": 10, "code": "sh.600000",
         "shares": 100, "reportYear": 2023, "report_quarter": 4},
        {"uploadInfoId": 3, "fundId": 20, "code": "sh.600000",
         "shares": 50, "reportYear": 2023, "report_quarter": 4},
    ])
    valid_denominator = pd.DataFrame([{
        "reportYear": 2023, "report_quarter": 4, "code": "sh.600000",
        "float_shares": 1000.0,
    }])
    active = _active_stakes(metadata, positions, valid_denominator)
    visible = pd.DataFrame([
        {"date": "2024-01-10", "code": "sh.600000", "visible_funds": 0},
        {"date": "2024-01-11", "code": "sh.600000", "visible_funds": 2},
        {"date": "2024-02-02", "code": "sh.600000", "visible_funds": 1},
    ])
    for col, value in {"board": "sh_main", "size_bucket": 1,
                       "cash_group": "low_cash_conversion"}.items():
        visible[col] = value
    result = _stake_by_stock_day(visible, active)
    assert result.reported_float_fraction.tolist() == pytest.approx(
        [0.0, 0.15, 0.05])
    assert result.ownership_tercile.isna().tolist() == [True, False, False]

    missing = _active_stakes(metadata, positions, valid_denominator.iloc[:0])
    result = _stake_by_stock_day(visible, missing)
    assert result.reported_float_fraction.isna().tolist() == [False, True, True]
    assert result.ownership_tercile.isna().all()


def test_stake_count_disagreement_fails_before_outcome() -> None:
    visible = pd.DataFrame([{
        "date": "2024-05-15", "code": "sh.600000", "visible_funds": 1,
        "board": "sh_main", "size_bucket": 1,
        "cash_group": "low_cash_conversion",
    }])
    active = pd.DataFrame(columns=["code", "fundId", "first_usable",
                                   "last_usable", "reported_float_fraction"])
    with pytest.raises(ValueError, match="disagrees"):
        _stake_by_stock_day(visible, active)
