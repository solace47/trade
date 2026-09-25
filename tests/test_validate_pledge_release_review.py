import pandas as pd
import pytest

from scripts.validate_pledge_release_review import COLUMNS, validate_rows


def _rows(decision: str = "pending") -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.DataFrame([{"notice_date": "2024-03-20",
                                "code": "sz.000001", "pdf_url": "source.pdf",
                                "screens": "after_only"}])
    review = pd.DataFrame([{"notice_date": "2024-03-20",
                            "code": "sz.000001", "pdf_url": "source.pdf",
                            "screens": "after_only", "decision": decision,
                            "actor": "", "registered_date": "",
                            "released": "", "held": "",
                            "after_pledged": "", "evidence": ""}],
                          columns=COLUMNS)
    return review, candidates


def test_pending_original_blocks_matching_and_missing_one_fails() -> None:
    review, candidates = _rows()
    assert not validate_rows(review, candidates)["ready_for_matching"]
    with pytest.raises(ValueError, match="exact source originals"):
        validate_rows(review.iloc[:0], candidates)


def test_verified_original_must_satisfy_frozen_counts() -> None:
    review, candidates = _rows("verified")
    review.loc[0, ["actor", "registered_date", "released", "held",
                   "after_pledged", "evidence"]] = [
        "直接控股股东", "2024-03-19", "20000000", "100000000",
        "30000000", "原件载明已登记且无新增质押"]
    assert validate_rows(review, candidates)["ready_for_matching"]
    review.loc[0, "after_pledged"] = "55000000"
    with pytest.raises(ValueError, match="frozen crossing rule"):
        validate_rows(review, candidates)
