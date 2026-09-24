from trade_research.insider_plan_source import confirm_pdf_text


def test_confirm_original_new_plan():
    assert confirm_pdf_text(
        "证券代码：600821 控股股东拟增持公司股份。增持计划自公告日起实施。",
        "sh.600821")


def test_reject_wrong_company_or_unsupported_text():
    assert not confirm_pdf_text(
        "证券代码：600820 控股股东拟增持公司股份。增持计划自公告日起实施。",
        "sh.600821")
    assert not confirm_pdf_text("证券代码：600821 增持计划已完成。", "sh.600821")
