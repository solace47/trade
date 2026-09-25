import json

import pandas as pd
import pytest

from trade_research.insider_midpoint_source import (
    audit, midpoint_title, screen, zero_evidence,
)


def test_midpoint_requires_actor_and_original_zero_statement():
    assert midpoint_title("控股股东增持计划时间过半进展")
    assert not midpoint_title("董事增持计划时间过半进展")
    assert not midpoint_title("控股股东增持金额过半进展")
    assert zero_evidence(
        "股票代码000001。本次增持计划时间过半，控股股东尚未增持公司股份。",
        "sz.000001")
    assert zero_evidence(
        "股票代码000001。本次增持计划时间过半，计划尚未实施完毕。",
        "sz.000001") is None
    assert zero_evidence(
        "股票代码000002。本次增持计划时间过半，控股股东尚未增持公司股份。",
        "sz.000001") is None
    assert zero_evidence(
        "股票代码000001。增持计划时间过半，暂未通过交易所以集中竞价方式进行增持。",
        "sz.000001")


def test_original_review_ledger_is_complete_and_excludes_mixed_plans(tmp_path):
    reviews = []
    for year in (2024, 2025):
        records = []
        for code, mixed in (("sz.000001", False), ("sz.000002", True)):
            day = f"{year}-05-01"
            records.append({"code": code, "notice_date": day,
                            "title": "控股股东增持计划时间过半进展",
                            "pdf_url": f"https://example.org/{year}/{code}.pdf",
                            "announcement_time_ms": 1})
            reviews.append({"code": code, "notice_date": day,
                            "zero_confirmed": int(not mixed),
                            "reason": "mixed" if mixed else "zero"})
        source = pd.DataFrame(records)
        for name in ("primary", "implementation", "broad"):
            source.to_parquet(tmp_path / f"search_{name}_{year}.parquet")
        originals = [{**row, "status": "ok", "text_full_pdf":
                      f"股票代码{row['code'][-6:]}。控股股东增持计划时间过半，"
                      "控股股东尚未增持公司股份。"}
                     for row in records]
        (tmp_path / f"pdf_audit_{year}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in originals))
    review_path = tmp_path / "review.csv"
    pd.DataFrame(reviews).to_csv(review_path, index=False)
    assert screen(tmp_path, tmp_path)["2024"]["midpoint_actor_titles"] == 2
    result = audit(tmp_path, review_path)
    assert result["2024"]["confirmed_zero_pdfs"] == 1
    assert result["2025"]["rejected_or_ambiguous_pdfs"] == 1
    assert len(pd.read_parquet(tmp_path / "confirmed_2025.parquet")) == 1
    pd.DataFrame(reviews[:-1]).to_csv(review_path, index=False)
    with pytest.raises(ValueError, match="reviewed ledger differ"):
        audit(tmp_path, review_path)
