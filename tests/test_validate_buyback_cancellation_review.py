import pandas as pd
import pytest

from scripts.validate_buyback_cancellation_review import validate_rows


def test_review_gate_needs_full_membership_and_both_years() -> None:
    rows = []
    for year in (2024, 2025):
        for i in range(30):
            rows.append({"date": f"{year}-{i // 5 + 1:02d}-08",
                         "code": f"sh.{year}{i:02d}",
                         "notice_date": f"{year}-{i // 5 + 1:02d}-07",
                         "pdf_url": f"pdf-{year}-{i}",
                         "provisional_purpose": "cancel",
                         "decision": "pure_cancel", "evidence": "原件全部注销"})
    review = pd.DataFrame(rows)
    assert validate_rows(review, review)["ready_for_matching"]
    review.loc[0, "decision"] = "pending"
    review.loc[0, "evidence"] = ""
    assert not validate_rows(review, pd.DataFrame(rows))["ready_for_matching"]
    with pytest.raises(ValueError, match="exact frozen originals"):
        validate_rows(review.iloc[1:], pd.DataFrame(rows))
    review.loc[0, "evidence"] = "not yet reviewed"
    with pytest.raises(ValueError, match="Pending original"):
        validate_rows(review, pd.DataFrame(rows))


def test_review_rejects_source_metadata_change() -> None:
    source = pd.DataFrame([{"date": "2024-01-03", "code": "sh.600001",
                            "notice_date": "2024-01-02", "pdf_url": "pdf-a",
                            "provisional_purpose": "cancel", "decision": "pending",
                            "evidence": ""}])
    review = source.copy()
    review.loc[0, "code"] = "sh.600002"
    with pytest.raises(ValueError, match="metadata changed"):
        validate_rows(review, source)
