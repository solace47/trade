import json

import pandas as pd
import pytest

from trade_research.regulatory_investigation_source import (
    audit, initial_title, screen,
)


def test_first_notice_and_full_pdf_review_gate(tmp_path):
    assert initial_title("关于收到中国证监会立案告知书的公告")
    assert not initial_title("关于立案调查进展暨风险提示的公告")
    decisions = []
    for year in (2024, 2025):
        rows = [
            {"code": "sz.000001", "notice_date": f"{year}-03-20",
             "title": "关于收到中国证监会立案告知书的公告",
             "pdf_url": f"https://example.org/{year}/issuer.pdf",
             "announcement_time_ms": 1},
            {"code": "sz.000002", "notice_date": f"{year}-03-21",
             "title": "关于实际控制人收到立案告知书的公告",
             "pdf_url": f"https://example.org/{year}/person.pdf",
             "announcement_time_ms": 2},
            {"code": "sz.000001", "notice_date": f"{year}-04-20",
             "title": "关于立案调查进展暨风险提示的公告",
             "pdf_url": f"https://example.org/{year}/progress.pdf",
             "announcement_time_ms": 3},
        ]
        frame = pd.DataFrame(rows)
        for name in ("notice", "investigation", "broad"):
            frame.to_parquet(tmp_path / f"search_{name}_{year}.parquet")
        originals = [
            {**row, "status": "ok", "pages": 1,
             "text_full_pdf": ("证券代码" + row["code"][-6:]
                               + "。公司收到证监会立案告知书，"
                               "因公司涉嫌信息披露违法违规，决定对公司立案。")}
            for row in rows
        ]
        (tmp_path / f"pdf_audit_{year}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in originals), encoding="utf-8")
        decisions.extend([
            {"pdf_url": rows[0]["pdf_url"], "accepted": 1,
             "reason": "公司被调查"},
            {"pdf_url": rows[1]["pdf_url"], "accepted": 0,
             "reason": "仅实际控制人"},
        ])
    assert screen(tmp_path)["2024"]["unique_original_pdfs_to_audit"] == 3
    review = tmp_path / "reviews.csv"
    pd.DataFrame(decisions).to_csv(review, index=False)
    report = audit(tmp_path, review)
    assert report["2024"]["confirmed_issuer_disclosure_notices"] == 1
    assert len(pd.read_parquet(tmp_path / "confirmed_2025.parquet")) == 1
    pd.DataFrame(decisions[:-1]).to_csv(review, index=False)
    with pytest.raises(ValueError, match="lacks a review decision"):
        audit(tmp_path, review)


def test_broad_search_must_contain_narrow_pdf(tmp_path):
    for year in (2024, 2025):
        row = {"code": "sz.000001", "notice_date": f"{year}-03-20",
               "title": "收到立案告知书", "pdf_url": f"https://a/{year}/one.pdf",
               "announcement_time_ms": 1}
        pd.DataFrame([row]).to_parquet(tmp_path / f"search_broad_{year}.parquet")
        pd.DataFrame([{**row, "pdf_url": f"https://a/{year}/two.pdf"}]).to_parquet(
            tmp_path / f"search_notice_{year}.parquet")
        pd.DataFrame([row]).to_parquet(
            tmp_path / f"search_investigation_{year}.parquet")
    with pytest.raises(ValueError, match="Broad search missed notice"):
        screen(tmp_path)
