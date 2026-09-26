import json

import pandas as pd
import pytest

from trade_research.cash_dividend_primary import overlay, primary_dates, verify_terms
from trade_research.cash_dividend_catalog import record_hash


def review(**changes):
    return dict({"code": "sh.600001", "disposition": "add_missing_event",
        "notice_date": "2024-05-20", "record_date": "2024-05-29", "action_date": "2024-05-30",
        "pay_date": "2024-05-30", "cash_per_share": ".3", "bonus_per_share": "0",
        "reserve_per_share": "0", "reference_cash_per_share": ".29",
        "source_url": "primary"}, **changes)


def vendor(**changes):
    return dict({"code": "sh.600001", "dividPlanDate": "2024-05-20",
        "dividRegistDate": "2024-05-29", "dividOperateDate": "2024-05-30",
        "dividPayDate": "2024-05-30", "dividCashPsBeforeTax": ".3", "dividStocksPs": "0",
        "dividReserveToStockPs": "0", "action_type": "provisional_pure_cash"}, **changes)


def test_primary_cash_and_reference_are_separate_and_dates_are_not_guessed():
    text = ("证券代码：600001 每股分配比例 A股每股现金红利0.3元 相关日期 "
            "Ａ股 2024/5/29 － 2024/5/30 2024/5/30 差异化是 虚拟分派0.29元")
    verify_terms(text, review())
    with pytest.raises(ValueError, match="cash amount"):
        verify_terms(text, review(cash_per_share=".29"))
    with pytest.raises(ValueError, match="dates disagree"):
        verify_terms(text, review(pay_date="2024-06-01"))
    assert primary_dates("股权登记日：2025年8月20日；除权日：2025年8月21日。", "sz.001323") == (
        "2025-08-20", "2025-08-21", "")


def test_supplements_cannot_double_count_and_later_notices_do_not_move_visibility(tmp_path):
    original = pd.DataFrame([vendor()])
    with pytest.raises(ValueError, match="double-count"):
        overlay(original, [review()], tmp_path)
    repeated = review(disposition="existing_event_notice", notice_date="2024-05-30")
    result, counts = overlay(original, [repeated, review(code="sh.600002")], tmp_path)
    assert len(result) == 2
    old = result[result.code.eq("sh.600001")].iloc[0]
    assert old.dividPlanDate == "2024-05-20"
    assert not old.primary_terms_verified
    assert counts["add_missing_event"] == counts["existing_event_notice"] == 1


def test_wrong_cash_field_is_corrected_only_for_the_exact_reviewed_vendor_event(tmp_path):
    original = vendor(dividCashPsBeforeTax="1", dividReserveToStockPs="")
    raw = {k: v for k, v in original.items() if k != "action_type"}
    cache = tmp_path / "vendor"
    cache.mkdir()
    path = cache / "sh.600001_2024.json"
    path.write_text(json.dumps([raw]))
    patch = review(disposition="correct_vendor_action", cash_per_share="0", reserve_per_share=".1",
                   reference_cash_per_share="0", vendor_event_sha256=record_hash(raw))
    result, _ = overlay(pd.DataFrame([original]), [patch], tmp_path)
    assert result.iloc[0].action_type == "share_distribution"
    assert result.iloc[0].dividCashPsBeforeTax == "0"
    path.write_text(json.dumps([dict(raw, dividCashPsBeforeTax="2")]))
    with pytest.raises(ValueError, match="exact reviewed"):
        overlay(pd.DataFrame([original]), [patch], tmp_path)


def test_future_implementation_is_excluded_by_the_actual_ex_date():
    text = ("证券代码：600001 相关日期 Ａ股 2025/12/31 － 2026/1/5 2026/1/5 差异化否")
    verify_terms(text, review(disposition="outside_implementation_years", action_date="2026-01-05"))
    with pytest.raises(ValueError, match="boundary notice"):
        verify_terms(text, review(disposition="outside_implementation_years", action_date="2025-12-31"))
