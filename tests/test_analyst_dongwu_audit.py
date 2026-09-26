from decimal import Decimal

import pytest

from trade_research.analyst_dongwu_audit import printed_security, profit_table, reported_direction


def test_fiscal_rollover_does_not_shift_columns_or_read_actuals():
    earlier = "2021A 2022A 2023E 2024E 2025E\n归母净利润（百万元） 100 110 198 270 358"
    later = "2023A 2024A 2025E 2026E 2027E\n归母净利润（百万元） (233.07) 20.51 105.09 159.02 234.62"
    assert profit_table(earlier) == {2024: Decimal("270"), 2025: Decimal("358")}
    assert profit_table(later) == {2025: Decimal("105.09"), 2026: Decimal("159.02"), 2027: Decimal("234.62")}
    assert 2026 not in profit_table(earlier)


def test_four_column_alias_and_chart_annotation():
    table = "2022A 2023E 2024E 2025E\n归属母公司净利润（百万元） 1,220 1,545 2,021 2,527 130%"
    assert profit_table(table) == {2024: Decimal("2021"), 2025: Decimal("2527")}


def test_ambiguous_units_or_column_counts_are_not_guessed():
    table = "2023A 2024E 2025E\n归母净利润（百万元） 100 200 300"
    for altered in (table.replace("百万元", "亿元"), table.replace("200 300", "200300"),
                    table + "\n归母净利润（百万元） 100 200 300", table.replace("2023A", "2024A")):
        with pytest.raises(ValueError):
            profit_table(altered)


def test_maintained_without_old_numbers_is_unknown_not_zero_revision():
    review = {"comparison_kind": "stated_maintained", "reported_current_profit_cny_100million": "111.43",
              "reported_prior_profit_cny_100million": None}
    assert reported_direction(review) is None
    # A report may say "basically maintain" while its explicit target-year
    # figures decrease; the reviewed numeric values determine the direction.
    review.update(comparison_kind="numeric_prior_levels", reported_prior_profit_cny_100million="113.21")
    assert reported_direction(review) == "down"
    review.update(reported_current_profit_cny_100million="113.21")
    assert reported_direction(review) == "unchanged"


def test_retrospective_code_alias_is_limited_to_reviewed_documents():
    alias = {"reviewed_document_ids": ["fixed"], "stock_code": "920062",
             "printed_legacy_code": "834062", "entity_name": "科润智控"}
    assert printed_security("科润智控（834062）", "fixed", "920062", alias) == "834062"
    for report, canonical, page in [("other", "920062", "科润智控（834062）"),
                                    ("fixed", "920999", "科润智控（834062）"),
                                    ("fixed", "920062", "其他公司（834062）")]:
        with pytest.raises(ValueError):
            printed_security(page, report, canonical, alias)
