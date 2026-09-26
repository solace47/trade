from copy import deepcopy

import pytest

from trade_research.analyst_catalog import finish_month, project_and_sample, validate_page
from trade_research.analyst_catalog_sources import detail_object


def row(code, date="2024-03-22"):
    return {"infoCode": code, "publishDate": date + " 00:00:00.000", "orgCode": "80000031",
            "stockCode": "600001", "stockName": "公司", "title": "研究报告"}


def page(rows, number=1, hits=None, total=1):
    return {"data": rows, "pageNo": number, "hits": len(rows) if hits is None else hits, "TotalPage": total}


def test_cross_page_duplicates_and_changing_totals_do_not_silently_drop():
    a = page([row("a")], 1, 2, 2)
    b = page([row("a")], 2, 2, 2)
    with pytest.raises(ValueError, match="Duplicate"):
        finish_month([a, b], "2024-03-01", "2024-03-31")
    b["data"] = [row("b")]
    assert len(finish_month([a, b], "2024-03-01", "2024-03-31")) == 2
    b["hits"] = 3
    with pytest.raises(ValueError, match="totals changed"):
        finish_month([a, b], "2024-03-01", "2024-03-31")


def test_ignored_broker_or_month_filter_is_rejected():
    for record in (dict(row("a"), orgCode="other"), row("a", "2026-03-22")):
        with pytest.raises(ValueError):
            validate_page(page([record]), "2024-03-01", "2024-03-31", 1)


def test_sample_is_order_invariant_and_excludes_dynamic_forecasts():
    rows = [row(f"{year}{month:02d}-{i}", f"{year}-{month:02d}-01")
            for year in (2024, 2025) for month in (1, 7) for i in range(10)]
    expected = project_and_sample(rows)
    changed = deepcopy(rows)
    for record in changed:
        record["predictThisYearEps"] = "9999.00"
        record["predictThisYearPe"] = "0.01"
    assert project_and_sample(changed[::-1]) == expected
    assert len(expected[1]) == 32
    assert all("predictThisYearEps" not in record for record in expected[0])


def test_unrelated_current_page_content_is_not_parsed_or_executed():
    html = 'var current = {"year": 2026}; var zwinfo = {"info_code":"fixed","notice_content":"old"}; throw 1;'
    assert detail_object(html, "fixed") == {"info_code": "fixed", "notice_content": "old"}
    with pytest.raises(ValueError, match="identity"):
        detail_object(html, "different")
