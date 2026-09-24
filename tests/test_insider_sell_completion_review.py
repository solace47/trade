import pandas as pd
import pytest

from trade_research.insider_sell_completion_inputs import EVENT
from trade_research.insider_sell_completion_review import validate_review


def _sample():
    date = "2024-06-04"
    notice = "2024-06-03"
    originals = [
        {"date": date, "code": "sz.300001", "pair_code": "sz.300001",
         "candidate": EVENT, "notice_date": notice,
         "pdf_url": "https://static.cninfo.com.cn/finalpage/2024-06-03/1.PDF"},
        {"date": date, "code": "sz.300002", "pair_code": "sz.300002",
         "candidate": EVENT, "notice_date": notice,
         "pdf_url": "https://static.cninfo.com.cn/finalpage/2024-06-03/2.PDF"},
    ]
    controls = [
        {"date": date, "code": "sz.300003", "pair_code": "sz.300001",
         "candidate": "same_day_nonannouncer", "notice_date": None,
         "pdf_url": None},
        {"date": date, "code": "sz.300004", "pair_code": "sz.300002",
         "candidate": "same_day_nonannouncer", "notice_date": None,
         "pdf_url": None},
    ]
    review = [
        {"signal_date": date, "notice_date": notice, "code": row["code"],
         "pdf_url": row["pdf_url"], "decision": "verified_direct",
         "actor": "股东甲", "sale_date": "2024-05-30", "actual_shares": "100",
         "evidence": "原件证实股东甲完成计划且实际卖出100股",
         "support_url": ""} for row in originals
    ]
    return pd.DataFrame(originals + controls), pd.DataFrame(review)


def test_review_must_cover_every_matched_original_before_outcomes() -> None:
    pairs, review = _sample()
    _, report = validate_review(pairs, review, 2024)
    assert report["verified_pairs"] == 2
    assert report["outcomes_opened"] is False
    with pytest.raises(ValueError, match="Incomplete"):
        validate_review(pairs, review.iloc[:1], 2024)


def test_review_rejects_support_published_after_signal() -> None:
    pairs, review = _sample()
    review.loc[0, "support_url"] = (
        "https://static.cninfo.com.cn/finalpage/2024-06-05/late.PDF")
    with pytest.raises(ValueError, match="not public"):
        validate_review(pairs, review, 2024)


def test_review_rejects_reused_same_day_control() -> None:
    pairs, review = _sample()
    pairs.loc[3, "code"] = "sz.300003"
    with pytest.raises(ValueError, match="reused"):
        validate_review(pairs, review, 2024)
