from trade_research.buyback_source import strict_plan_title


def test_buyback_title_requires_a_new_issuer_plan() -> None:
    assert strict_plan_title("关于以集中竞价交易方式回购公司股份方案的公告")
    assert strict_plan_title("关于股份回购方案的公告")
    assert not strict_plan_title("关于回购股份方案实施进展的公告")
    assert not strict_plan_title("关于回购注销限制性股票方案的公告")
    assert not strict_plan_title("保荐机构关于股份回购方案的核查意见")
    assert not strict_plan_title("关于回购股份方案的临时受托管理事务报告")
    assert not strict_plan_title("关于回购股份方案披露前十大股东持股情况的公告")
