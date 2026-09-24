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


def test_a_and_h_securities_can_share_one_report_rank() -> None:
    rows = ("<tr class=\"dd\"><td>1</td><td>600036</td><td>招商银行</td>"
            "<td>100</td><td>1,000</td><td>1.0</td></tr>"
            "<tr class=\"dd\"><td>1</td><td>03968</td><td>招商银行</td>"
            "<td>100</td><td>1,000</td><td>1.0</td></tr>"
            "<tr class=\"dd\"><td>2</td><td>000001</td><td>平安银行</td>"
            "<td>100</td><td>1,000</td><td>1.0</td></tr>")
    holdings, total = parse_holdings(_document(rows), "2024-04-19")
    assert total == 3
    assert [row["code"] for row in holdings] == ["sh.600036", "sz.000001"]


def test_unused_ranks_filled_with_dashes_are_not_holdings() -> None:
    rows = ("<tr class=\"dd\"><td>1</td><td>689009</td><td>九号公司</td>"
            "<td>112</td><td>3,354.40</td><td>0.00</td></tr>"
            "<tr class=\"dd\"><td>2</td><td>-</td><td>-</td>"
            "<td>-</td><td>-</td><td>-</td></tr>")
    holdings, total = parse_holdings(_document(rows), "2024-04-19")
    assert total == 1
    assert holdings[0]["code"] == "sh.689009"


def test_blank_stock_table_requires_zero_stock_assets() -> None:
    document = ("<p>报告送出日期：2024-04-19</p>"
                "<a name=\"tabItem4_assetsCircs\"></a>"
                "<tr class=\"dd\"><td></td><td>其中：股票</td>"
                "<td>-</td><td>-</td></tr>"
                "<a name=\"tabItem4_topTenStockDetal\"></a>"
                + HEADER + "<a name=\"tabItem4_bondCombination\"></a>")
    assert parse_holdings(document, "2024-04-19") == ([], 0)
    with pytest.raises(ValueError, match="empty"):
        parse_holdings(_document(""), "2024-04-19")


def test_split_a_h_row_can_omit_second_rank() -> None:
    rows = ("<tr class=\"dd\"><td>1</td><td>00883 HG</td>"
            "<td>中国海洋石油</td><td>100</td><td>1,000</td><td>1.0</td></tr>"
            "<tr class=\"dd\"><td>-</td><td>600938</td>"
            "<td>中国海油</td><td>100</td><td>1,000</td><td>1.0</td></tr>"
            "<tr class=\"dd\"><td>2</td><td>000001</td>"
            "<td>平安银行</td><td>100</td><td>1,000</td><td>1.0</td></tr>")
    holdings, total = parse_holdings(_document(rows), "2024-04-19")
    assert total == 3
    assert [row["rank"] for row in holdings] == [1, 2]


def test_reported_extra_security_after_top_ten_is_retained() -> None:
    rows = "".join(
        f'<tr class="dd"><td>{rank}</td><td>600{rank:03d}</td>'
        f'<td>股票{rank}</td><td>100</td><td>1,000</td><td>1.0</td></tr>'
        for rank in range(1, 12))
    holdings, total = parse_holdings(_document(rows), "2024-04-19")
    assert total == 11
    assert len(holdings) == 11


def test_tied_rank_may_skip_the_next_number() -> None:
    rows = ("<tr class=\"dd\"><td>1</td><td>600001</td><td>一</td>"
            "<td>100</td><td>1,000</td><td>1.0</td></tr>"
            "<tr class=\"dd\"><td>1</td><td>600002</td><td>二</td>"
            "<td>100</td><td>1,000</td><td>1.0</td></tr>"
            "<tr class=\"dd\"><td>3</td><td>600003</td><td>三</td>"
            "<td>100</td><td>1,000</td><td>1.0</td></tr>")
    holdings, total = parse_holdings(_document(rows), "2024-04-19")
    assert total == 3
    assert [row["rank"] for row in holdings] == [1, 1, 3]


def test_cross_border_template_is_zero_only_when_securities_are_offshore() -> None:
    header = ("<tr class=\"cc\"><td>序号</td><td>公司名称（英文）</td>"
              "<td>公司名称（中文）</td><td>证券代码</td><td>所在证券市场</td>"
              "<td>所属国家（地区）</td><td>数量（股）</td>"
              "<td>公允价值</td><td>占基金资产净值比例（%）</td></tr>")
    def document(code: str, market: str) -> str:
        row = ("<tr class=\"dd\"><td>1</td><td>Tencent</td><td>腾讯控股</td>"
               f"<td>{code}</td><td>{market}</td><td>中国香港</td>"
               "<td>100</td><td>1,000</td><td>1.0</td></tr>")
        return ("<p>报告送出日期：2024-04-19</p>"
                "<a name=\"tabItem7_topTenStockDetal\"></a>"
                + header + row + "<a name=\"tabItem7_topTenJJTZStockDetal\"></a>"
                + header + "<a name=\"tabItem7_bondCombination\"></a>")

    assert parse_holdings(document("0700 HK", "香港联合交易所"),
                          "2024-04-19") == ([], 0)
    with pytest.raises(ValueError, match="Missing or duplicated"):
        parse_holdings(document("600519", "上海证券交易所"), "2024-04-19")
