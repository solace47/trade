from trade_research.insider_plan_source import strict_plan_title


def test_new_controlling_holder_or_chairman_plan():
    assert strict_plan_title("关于控股股东拟增持公司股份计划的公告")
    assert strict_plan_title("关于实际控制人一致行动人股份增持计划的公告")
    assert strict_plan_title("关于董事长以专项贷款增持公司股份计划的公告")


def test_reject_amendments_executions_and_other_actors():
    for title in (
        "控股股东股份增持计划调整公告",
        "董事长增持股份计划实施情况公告",
        "关于控股股东增持股份计划增加增持主体的公告",
        "关于控股股东增持股份计划进展的公告",
        "关于控股股东回购公司股份计划的公告",
        "关于员工增持股份计划的公告",
    ):
        assert not strict_plan_title(title)
