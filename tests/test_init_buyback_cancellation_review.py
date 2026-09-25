from pathlib import Path

import pandas as pd
import pytest

from scripts.init_buyback_cancellation_review import initialize


def test_review_covers_both_frozen_years_and_cannot_reset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pairs = tmp_path / "pairs.parquet"
    events = pd.DataFrame([
        {"date": "2024-05-08", "code": "sh.600001",
         "notice_date": "2024-05-07", "pdf_url": "pdf-a",
         "candidate": "buyback_plan"},
        {"date": "2025-05-08", "code": "sz.000001",
         "notice_date": "2025-05-07", "pdf_url": "pdf-b",
         "candidate": "buyback_plan"},
    ])
    events.to_parquet(pairs, index=False)
    for year, code, url, purpose in (
        (2024, "sh.600001", "pdf-a", "全部用于注销并减少注册资本"),
        (2025, "sz.000001", "pdf-b", "用于员工持股计划"),
    ):
        pd.DataFrame([{"code": code, "notice_date": f"{year}-05-07",
                       "pdf_url": url, "status": "ok",
                       "text_first_three_pages": f"回购股份的用途：{purpose}"}]
                     ).to_json(source / f"pdf_audit_{year}.jsonl",
                               orient="records", lines=True, force_ascii=False)
    output = tmp_path / "review.csv"
    review = initialize(source, pairs, output)
    assert review.provisional_purpose.tolist() == ["cancel", "employee"]
    assert review.decision.eq("pending").all()
    with pytest.raises(FileExistsError):
        initialize(source, pairs, output)
