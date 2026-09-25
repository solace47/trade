from trade_research.first_insider_buy_source import (
    confirm_original, strict_first_title,
)


def test_title_requires_controller_first_purchase_of_a_shares() -> None:
    assert strict_first_title("关于控股股东首次增持公司A股股份的公告")
    assert strict_first_title("关于实际控制人首次增持公司股份的公告")
    assert not strict_first_title("关于股东首次增持公司股份的公告")
    assert not strict_first_title("关于控股股东首次增持公司H股股份的公告")
    assert not strict_first_title("控股股东首次增持股份更正公告")


def test_completed_trade_date_and_amount_do_not_come_from_prior_plan() -> None:
    original = (
        "证券代码：600001。控股股东计划自2025年4月1日起通过上海证券交易所"
        "交易系统以集中竞价方式增持公司股份500万股，拟投入3亿元。"
        "首次增持情况：2025年4月16日，控股股东按照增持计划通过上海证券交易所"
        "交易系统以集中竞价方式实施了首次增持，首次增持共计120,000股，"
        "增持金额为人民币1,622,699元。"
    )
    actual = confirm_original(original, "sh.600001", "2025-04-17")
    assert actual["status"] == "ok"
    assert actual["buy_date"] == "2025-04-16"
    assert actual["shares"] == 120_000
    assert actual["parsed_actual_amount_yuan"] == 1_622_699


def test_prospective_plan_is_not_a_completed_purchase() -> None:
    original = (
        "证券代码：600001。控股股东拟于2025年4月16日通过上海证券交易所"
        "交易系统以集中竞价方式增持公司股份120,000股，计划投入100万元。"
    )
    actual = confirm_original(original, "sh.600001", "2025-04-17")
    assert actual["status"] != "ok"


def test_h_share_purchase_is_not_treated_as_a_share_purchase() -> None:
    original = (
        "证券代码：600001。控股股东于2025年4月16日通过香港联合交易所"
        "交易系统以集中竞价方式首次增持H股股份120,000股。"
    )
    actual = confirm_original(original, "sh.600001", "2025-04-17")
    assert actual["status"] != "ok"
