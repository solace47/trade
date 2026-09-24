import pytest

from scripts.extract_fund_xbrl_holdings import parse_holdings


HEADER = ("<tr class=\"cc\"><td>序号</td><td>股票代码</td>"
          "<td>股票名称</td><td>数量（股）</td>"
          "<td>公允价值（元）</td><td>占基金资产净值比例（％）</td></tr>")


def _document(rows: str, sent: str = "2024-04-19") -> str:
    return (f"<html><p>报告送出日期：{sent}</p>"
            f"<a name=\"tabItem4_topTenStockDetal\"></a>{HEADER}{rows}"
            "<a name=\"tabItem4_bondCombination\"></a>"
            "<tr class=\"dd\"><td>1</td><td>600000</td></tr></html>")


def test_parses_only_public_fund_a_share_rows() -> None:
    rows = ("<tr class=\"dd\"><td>1</td><td>002025</td><td>航天电器</td>"
            "<td>2,099,227</td><td>79,476,734.22</td><td>3.46</td></tr>"
            "<tr class=\"dd\"><td>2</td><td>00700</td><td>腾讯控股</td>"
            "<td>100</td><td>10,000</td><td>0.50</td></tr>")
    holdings, total = parse_holdings(_document(rows), "2024-04-19")
    assert total == 2
    assert holdings == [{"code": "sz.002025", "rank": 1,
                         "shares": 2099227, "fair_value_yuan": 79476734.22,
                         "nav_weight_pct": 3.46}]


def test_rejects_report_date_mismatch() -> None:
    with pytest.raises(ValueError, match="dates differ"):
        parse_holdings(_document("注：无。"), "2024-04-18")


def test_accepts_explicitly_empty_stock_section() -> None:
    assert parse_holdings(_document("<p>注：无。</p>"),
                          "2024-04-19") == ([], 0)


def test_zero_decimal_share_count_is_still_an_integer() -> None:
    row = ("<tr class=\"dd\"><td>1</td><td>601899</td><td>紫金矿业</td>"
           "<td>903,100.00</td><td>15,190,142.00</td><td>2.44</td></tr>")
    holdings, total = parse_holdings(_document(row), "2024-04-19")
    assert total == 1
    assert holdings[0]["shares"] == 903100
