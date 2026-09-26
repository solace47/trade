from copy import deepcopy

import pytest

from trade_research.analyst_source_probe import compare_record, verify_forecast_table


FORECASTS = {"2024": "0.73", "2025": "0.91", "2026": "1.15"}
TABLE = "2023A 2024E 2025E 2026E\n摊薄每股收益（元） 0.56 0.73 0.91 1.15"


def records():
    row = {"infoCode": "frozen", "stockCode": "003006",
           "publishDate": "2024-03-22 00:00:00.000", "predictThisYearEps": "1.1500000000"}
    review = {"info_code": "frozen", "stock_code": "003006", "printed_date": "2024-03-22",
              "eps_page": 2, "eps_row_label": "摊薄每股收益（元）", "forecast_eps": FORECASTS,
              "chronology_conflict": False, "public_time_verified": False}
    return row, review, ["2024年03月22日\n公司报告", TABLE]


def test_old_report_this_year_does_not_mean_report_year():
    row, review, pages = records()
    result = compare_record(row, review, 2026, pages)
    assert result["numeric_matching_years"] == ["2026"]
    assert result["matches_container_fiscal_year"]
    assert not result["matches_report_calendar_year"]
    assert result["api_and_header_date_equal"]
    assert not result["public_time_verified"]
    assert not result["source_eligible_as_direct_event"]


def test_forecast_order_and_value_mutations_are_rejected():
    for bad in (TABLE.replace("2024E 2025E", "2025E 2024E"),
                TABLE.replace("0.91", "0.92"),
                TABLE.replace("2024E", "2024A"),
                TABLE + "\n摊薄每股收益（元） 0.56 0.73 0.91 1.15"):
        with pytest.raises(ValueError):
            verify_forecast_table(bad, "摊薄每股收益（元）", FORECASTS)


def test_extraction_spaces_do_not_merge_numerical_columns():
    spaced = TABLE.replace("摊薄每股收益（元）", "摊薄 每股 收益（元）")
    assert verify_forecast_table(spaced, "摊薄每股收益（元）", FORECASTS) == FORECASTS
    with pytest.raises(ValueError):
        verify_forecast_table(TABLE.replace("0.91 1.15", "0.911.15"), "摊薄每股收益（元）", FORECASTS)


def test_matching_header_does_not_hide_internal_chronology_conflict():
    row, review, pages = records()
    review = deepcopy(review)
    review.update(chronology_conflict=True, referenced_report_date="2024-03-26")
    pages[0] += "\n2024Q1点评\n此前报告 2024-03-26"
    result = compare_record(row, review, 2026, pages)
    assert result["api_and_header_date_equal"]
    assert result["chronology_conflict"]
    assert not result["source_eligible_as_direct_event"]
    with pytest.raises(ValueError, match="contradiction"):
        compare_record(row, review, 2026, [pages[0].replace("2024-03-26", "2024-03-20"), TABLE])


def test_wrong_report_or_printed_date_fails():
    row, review, pages = records()
    with pytest.raises(ValueError, match="identity"):
        compare_record(dict(row, stockCode="600011"), review, 2026, pages)
    with pytest.raises(ValueError, match="Printed"):
        compare_record(row, review, 2026, [pages[0].replace("22日", "23日"), TABLE])
