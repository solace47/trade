import pandas as pd
import pytest

from trade_research.insider_sell_completion_review import (
    exclude_unverified_announcers, validate_review,
)


def _sample():
    day, notice = "2024-06-04", "2024-06-03"
    originals = pd.DataFrame([
        {"date": day, "code": f"sz.30000{i}", "notice_date": notice,
         "pdf_url": f"https://static.cninfo.com.cn/finalpage/{notice}/{i}.PDF",
         "status": "ok"} for i in (1, 2)
    ])
    decisions = pd.DataFrame([
        {"signal_date": day, "notice_date": notice, "code": row.code,
         "pdf_url": row.pdf_url, "decision": "verified_direct",
         "actor": "股东甲", "reported_sale_period": "2024-05-30",
         "actual_shares": "100", "evidence": "原件证实股东甲完成计划且实际卖出100股",
         "support_url": ""} for row in originals.itertuples()
    ])
    return originals, decisions


def test_review_must_cover_every_eligible_original_before_outcomes() -> None:
    originals, decisions = _sample()
    verified, report = validate_review(originals, decisions, 2024)
    assert len(verified) == 2
    assert report["outcomes_opened"] is False
    with pytest.raises(ValueError, match="Incomplete"):
        validate_review(originals, decisions.iloc[:1], 2024)


def test_review_rejects_late_support_and_pre_2024_sale() -> None:
    originals, decisions = _sample()
    decisions.loc[0, "support_url"] = (
        "https://static.cninfo.com.cn/finalpage/2024-06-05/late.PDF")
    with pytest.raises(ValueError, match="not public"):
        validate_review(originals, decisions, 2024)
    decisions.loc[0, "support_url"] = ""
    decisions.loc[0, "reported_sale_period"] = "2023-12-31"
    with pytest.raises(ValueError, match="2024"):
        validate_review(originals, decisions, 2024)


def test_rejected_title_cannot_use_capacity_or_become_control() -> None:
    originals, decisions = _sample()
    decisions.loc[1, "decision"] = "reject_no_sale"
    verified, report = validate_review(originals, decisions, 2024)
    assert report["verified_originals"] == 1
    universe = pd.DataFrame([
        {"date": "2024-06-04", "code": code}
        for code in ("sz.300001", "sz.300002", "sz.300003")
    ])
    remaining = exclude_unverified_announcers(universe, originals, verified)
    assert set(remaining.code) == {"sz.300001", "sz.300003"}
