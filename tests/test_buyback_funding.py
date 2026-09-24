"""Funding parser safeguards against converting estimates into commitments."""

from trade_research.buyback_funding import explicit_funding_floor


def test_explicit_bound_and_chinese_unit() -> None:
    value, method, _ = explicit_funding_floor(
        "回购资金总额：不低于人民币1.5亿元且不超过人民币3亿元。")
    assert value == 150_000_000 and method == "explicit_lower"


def test_range_and_upper_only() -> None:
    value, method, _ = explicit_funding_floor(
        "预计回购金额2,500万元~5,000万元。")
    assert value == 25_000_000 and method == "range_lower"
    assert explicit_funding_floor("回购资金总额不超过2亿元。")[:2] == (
        None, "unknown")


def test_do_not_treat_share_estimate_or_malformed_number_as_floor() -> None:
    for example in (
        "按回购金额下限人民币3,000万元和价格上限测算股数。",
        "回购金额约为11,620万元至15,355万元，按价格上限测算。",
        "回购股份金额：不低于人民币8,00万元。",
    ):
        assert explicit_funding_floor(example)[0] is None


def test_conflicting_total_and_component_bounds_remain_unknown() -> None:
    value, method, _ = explicit_funding_floor(
        "回购资金总额不低于6亿元。用于注销的回购资金总额不低于2.35亿元。")
    assert value is None and method == "conflicting_bounds"
