from scripts.probe_buyback_purpose import provisional_purpose


def test_buyback_purpose_prioritizes_stated_use_over_fallback() -> None:
    assert provisional_purpose(
        "回购股份的用途：用于员工持股计划；未使用部分将在三年后注销"
    ) == "employee"
    assert provisional_purpose(
        "回购用途□减少注册资本□员工持股计划√为维护公司价值及股东权益"
    ) == "maintain"
    assert provisional_purpose(
        "回购股份的用途：将全部用于注销并减少公司注册资本"
    ) == "cancel"
