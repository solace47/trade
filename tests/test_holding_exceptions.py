"""A recovered position must retain all shares, dates and other quality gates."""

import pandas as pd
import pytest

from trade_research.hf_outcomes import Assumptions, _fill
from trade_research.holding_exceptions import new_share_count, quality_clean, recover


def distribution():
    return {"bonus_per_share": "0", "reserve_per_share": ".3",
            "new_share_trade_date": "2025-05-29"}


def test_new_shares_are_not_an_optional_part_of_exit_capacity():
    shares = new_share_count(400, distribution(), "2025-05-29")
    assert shares == 520
    quote = pd.Series({"vwap": 5.5, "volume": 5000})
    day = pd.Series({"date": "2025-05-29", "tradestatus": 1, "preclose": 5.51, "isST": 0})
    assert _fill(quote, day, "sz.000715", "sell", 400, Assumptions())[1] == "filled"
    assert _fill(quote, day, "sz.000715", "sell", shares, Assumptions())[1] == "volume_cap"


def test_new_shares_cannot_be_sold_before_listing_or_rounded_down():
    with pytest.raises(ValueError, match="not tradable"):
        new_share_count(400, distribution(), "2025-05-28")
    with pytest.raises(ValueError, match="Fractional"):
        new_share_count(401, distribution(), "2025-05-29")


def test_opening_release_never_removes_another_day_or_bad_security():
    bad = pd.DataFrame({"code": ["sh.600001", "sh.600001"],
                        "date": ["2025-01-03", "2025-01-06"]})
    allowed = {("sh.600001", "2025-01-03")}
    assert quality_clean("sh.600001", "2025-01-02", "2025-01-03", bad, set(), allowed)
    assert not quality_clean("sh.600001", "2025-01-02", "2025-01-06", bad, set(), allowed)
    assert not quality_clean("sh.600001", "2025-01-02", "2025-01-03", bad,
                             {"sh.600001"}, allowed)


def test_recovery_sells_increased_shares_and_keeps_entitlement_on_old_shares():
    row = pd.Series({"entry_status": "filled", "shares": 400, "date": "2025-05-22",
                     "buy_cost5": 2885.0288, "buy_cost15": 2887.9088})
    values = recover(row, 520, 5.5, "2025-05-29", gross_dividend=40, tax=8)
    assert values["verified_proceeds5"] == pytest.approx(2853.5414)
    assert values["known_return5"] == pytest.approx((2853.5414 + 32) / 2885.0288 - 1)
    assert values["sold_shares"] == 520
    assert "shares" not in values
    assert "old_score5" not in values
    with pytest.raises(ValueError, match="Invalid recovered"):
        recover(row, 520, float("nan"), "2025-05-29")
