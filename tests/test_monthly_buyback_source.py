from decimal import Decimal

from trade_research.monthly_buyback_source import (
    extract_cumulative_cash, previous_month_end,
)


def test_report_month_and_cumulative_cash_are_not_plan_amount():
    assert previous_month_end("2025-01-02").isoformat() == "2024-12-31"
    text = (
        "回购方案资金总额上限为一亿元。公司通过集中竞价方式回购。"
        "截至2024年12月31日，公司通过回购专用账户累计回购股份"
        "2999974股，成交总金额为人民币56,826,894.55元。"
        "下一期回购价格上限为36元。"
    )
    assert extract_cumulative_cash(text, "2025-01-02") == Decimal(
        "56826894.55")
    assert extract_cumulative_cash(text, "2025-02-03") is None


def test_mixed_or_contradictory_amounts_do_not_create_signal():
    text = (
        "以集中竞价交易方式回购股份。截至2025年11月30日，"
        "公司累计回购股份100万股，支付的金额总额约为人民币20,156.87万元。"
    )
    assert extract_cumulative_cash(text, "2025-12-03") == Decimal(
        "201568700")
    mixed = text + "截至2025年11月30日，公司累计回购股份101万股，" \
            "支付的金额总额约为人民币21,000万元。"
    assert extract_cumulative_cash(mixed, "2025-12-03") is None
