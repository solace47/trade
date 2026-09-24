import json

import pandas as pd
import pytest

from trade_research.insider_sell_completion_inputs import _events


def _source(tmp_path):
    for year, day, code in ((2024, "2024-06-03", "sz.300001"),
                            (2025, "2025-06-03", "sz.300002")):
        url = f"https://static.cninfo.com.cn/finalpage/{day}/{year}.PDF"
        row = {"code": code, "notice_date": day,
               "title": "关于持股5%以上股东减持计划完成的公告",
               "pdf_url": url}
        pd.DataFrame([row]).to_parquet(tmp_path / f"search_{year}.parquet")
        (tmp_path / f"pdf_audit_{year}.jsonl").write_text(
            json.dumps({**row, "status": "ok"}, ensure_ascii=False) + "\n")


def test_title_inputs_lag_notice_and_require_complete_pdf_audit(tmp_path) -> None:
    _source(tmp_path)
    calendar = ["2024-06-03", "2024-06-04", "2025-06-03", "2025-06-04"]
    events, report = _events(tmp_path, calendar)
    assert list(events.date) == ["2024-06-04", "2025-06-04"]
    assert report["mapped_title_events"] == 2
    audit = tmp_path / "pdf_audit_2024.jsonl"
    audit.write_text(audit.read_text().replace('"status": "ok"',
                                               '"status": "identity_unconfirmed"'))
    with pytest.raises(ValueError, match="unverified issuer"):
        _events(tmp_path, calendar)


def test_same_stock_same_notice_ambiguity_removes_both_pdfs(tmp_path) -> None:
    _source(tmp_path)
    index_path = tmp_path / "search_2024.parquet"
    index = pd.read_parquet(index_path)
    extra = index.iloc[0].copy()
    extra["pdf_url"] = extra["pdf_url"].replace("2024.PDF", "extra.PDF")
    pd.concat([index, extra.to_frame().T], ignore_index=True).to_parquet(
        index_path, index=False)
    with (tmp_path / "pdf_audit_2024.jsonl").open("a") as audit:
        audit.write(json.dumps({**extra.to_dict(), "status": "ok"},
                               ensure_ascii=False) + "\n")
    calendar = ["2024-06-03", "2024-06-04", "2025-06-03", "2025-06-04"]
    events, report = _events(tmp_path, calendar)
    assert list(events.code) == ["sz.300002"]
    assert report["duplicate_code_notice_rows_removed"] == 2
