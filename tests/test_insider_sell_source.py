from trade_research.insider_sell_source import (
    confirm_pdf_text, strict_plan_title,
)


def test_new_major_holder_sale_plan():
    assert strict_plan_title("持股5%以上股东减持股份计划预披露公告")
    assert strict_plan_title("关于实际控制人拟减持公司股份的公告")
    assert strict_plan_title("关于董事长集中竞价减持股份计划的公告")
    assert confirm_pdf_text(
        "证券代码：600821 持股5%以上股东拟减持公司股份，减持计划自公告日起实施。",
        "sh.600821")


def test_reject_progress_transfer_or_wrong_company():
    for title in (
        "控股股东减持计划实施进展公告",
        "关于控股股东内部转让股份及减持计划的公告",
        "关于持股5%以上股东减持股份计划调整的公告",
        "关于员工减持股份计划的公告",
    ):
        assert not strict_plan_title(title)
    assert not confirm_pdf_text(
        "证券代码：600820 持股5%以上股东减持股份计划公告", "sh.600821")
