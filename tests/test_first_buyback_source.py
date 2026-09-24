from trade_research.first_buyback_source import strict_first_title


def test_first_actual_buyback_title_requires_issuer_a_share_notice() -> None:
    assert strict_first_title("关于以集中竞价交易方式首次回购公司A股股份的公告")
    assert strict_first_title("关于首次回购股份暨回购进展的公告")
    assert not strict_first_title("关于首次回购公司境内上市外资股（B股）的公告")
    assert not strict_first_title("关于首次减持回购股份的公告")
    assert not strict_first_title("关于首次回购股份暨回购股份方案实施完成的公告")
    assert not strict_first_title("关于首次回购公司股份的更正公告")
    assert not strict_first_title("关于限制性股票激励计划首次授予部分回购注销的法律意见书")
