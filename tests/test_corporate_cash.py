"""Dividend entitlements must not hide unverified bought positions."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trade_research.corporate_cash import (
    accounting_summary, apply_cash, cash_entitlement, reference_price, verify_notice,
)
from trade_research.fill_accounting import KEY, account_rows


def event(code="sh.600001", **changes):
    return dict({
        "code": code, "record_date": "2025-01-02", "action_date": "2025-01-03",
        "pay_date": "2025-01-06", "cash_per_share": ".3",
        "reference_cash_per_share": ".29", "bonus_per_share": "0",
        "reserve_per_share": "0", "pure_cash": True,
    }, **changes)


def trades():
    raw = pd.DataFrame([{
        "date": "2025-01-02", "code": code, "target_notional": 20000,
        "horizon": 1, "shares": 100, "entry_price": 10.0, "entry_status": "filled",
        "target_exit_date": "2025-01-03", "exit_date": "2025-01-03",
        "exit_delay_sessions": 0, "exit_price": 9.7,
        "exit_status": "corporate_action_unadjusted", "net_return": np.nan,
        "quality_clean_exit": False, "candidate": arm, "pair_id": pair,
    } for code, arm, pair in (("sh.600001", "absolute_model", 1),
                              ("sh.600002", "same_day_control", 1),
                              ("sz.000715", "absolute_model", 2))])
    return account_rows(raw, ["2025-01-02", "2025-01-03", "2025-01-06"])


def test_cash_and_ex_reference_are_different_quantities_and_round_half_up():
    assert reference_price(9.6, event()) == Decimal("9.31")
    assert reference_price(2.55, event(reference_cash_per_share=".025")) == Decimal("2.53")
    assert reference_price(7.26, event(reference_cash_per_share=".1",
                                      reserve_per_share=".3")) == Decimal("5.51")
    gross, tax = cash_entitlement(trades().iloc[0], event())
    assert gross == 30  # Never use the .29 virtual ex-price adjustment.
    assert tax == 6


def test_quality_and_share_actions_remain_unknown_and_payment_is_separate():
    original = trades()
    eligible = original[KEY].assign(quality_clean_exit=[True, False, True])
    events = [event("sh.600001"), event("sh.600002"),
              event("sz.000715", reserve_per_share=".3", pure_cash=False)]
    result = apply_cash(original, eligible, events)
    assert result.unknown_after_buy.tolist() == [False, True, True]
    resolved = result.iloc[0]
    assert resolved.dividend_pay_date == "2025-01-06"
    assert resolved.dividend_tax_recognition_date == "2025-01-03"
    assert resolved.verified_proceeds5 == pytest.approx(964.5053)
    assert resolved.known_return5 == pytest.approx((964.5053 + 24) / 1005.01 - 1)
    assert resolved.known_return15 < resolved.known_return5
    assert result.old_score5.eq(0).all()
    for column in ("known_return5", "known_return15", "verified_proceeds5", "dividend_net"):
        assert result.loc[result.unknown_after_buy, column].isna().all()
    summary = accounting_summary(result)
    model = next(c for c in summary["cells"] if c["arm"] == "absolute_model")
    assert model["complete_cohort_mean5"] is None
    assert model["known_contribution5"] == pytest.approx(resolved.known_return5 / 2)


@pytest.mark.parametrize("change", [
    {"code": "sh.600002"}, {"record_date": "2025-01-01"},
    {"record_date": "2025-01-03"}, {"action_date": "2025-01-02"},
    {"pay_date": "2026-01-05"}, {"pay_date": "2025-01-02"},
    {"reserve_per_share": ".3"}, {"bonus_per_share": ".1"},
])
def test_unentitled_out_of_period_and_share_actions_are_rejected(change):
    with pytest.raises(ValueError):
        cash_entitlement(trades().iloc[0], event(**change))


def test_primary_pdf_dates_cannot_be_taken_from_b_share_row():
    notice = """证券代码：600001 每股分配比例 A 股每股现金红利 0.3 元
    相关日期 股权登记日 除权日 发放日
    Ａ股 2025/1/2 － 2025/1/3 2025/1/6
    Ｂ股 2025/1/7 2025/1/3 2025/1/10
    差异化分红：是 除息参考现金 0.29 元"""
    verify_notice(notice, event())
    with pytest.raises(ValueError, match="date table"):
        verify_notice(notice, event(pay_date="2025-01-10"))
    with pytest.raises(ValueError, match="cash in primary"):
        verify_notice(notice, event(cash_per_share=".29"))
