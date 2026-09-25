from datetime import date

import pytest

from scripts import collect_sell_completion_index as collector
from scripts.audit_sell_completion_pdfs import (
    candidate_title, identity_status, pdf_code_matches,
)


def test_large_cninfo_query_is_split_before_later_pages(monkeypatch, tmp_path) -> None:
    requested = []

    def fake_fetch(start, end, page, cache, term):
        assert page == 1
        return {"totalAnnouncement": 210 if start == date(2025, 1, 1)
                and end == date(2025, 1, 4) else 105}

    def fake_range(start, end, cache, term):
        requested.append((start, end))
        return [{"range": str(start), "row": n} for n in range(105)]

    monkeypatch.setattr(collector, "_fetch", fake_fetch)
    monkeypatch.setattr(collector, "_range_rows", fake_range)
    rows = collector._bounded_rows(date(2025, 1, 1), date(2025, 1, 4),
                                   tmp_path, "减持计划实施完成")
    assert len(rows) == 210
    assert requested == [(date(2025, 1, 1), date(2025, 1, 2)),
                         (date(2025, 1, 3), date(2025, 1, 4))]


def test_single_day_above_safe_page_limit_is_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(collector, "_fetch", lambda *_: {
        "totalAnnouncement": collector.MAX_SAFE_PAGES * collector.PAGE_SIZE + 1})
    with pytest.raises(ValueError, match="exceeds"):
        collector._bounded_rows(date(2025, 1, 1), date(2025, 1, 1),
                                tmp_path, "减持计划实施完成")


def test_broad_query_keeps_only_explicit_actor_completion_titles() -> None:
    assert collector.TERMS == ("减持计划",)
    assert candidate_title("关于持股5%以上股东减持计划完成的公告")
    assert candidate_title("关于控股股东减持计划实施完毕的公告")
    assert not candidate_title("关于持股5%以上股东减持计划期限届满的公告")
    assert not candidate_title("关于董事减持计划实施完成的公告")
    assert not candidate_title("关于副董事长减持计划完成的公告")
    assert candidate_title("关于5%以上非第一大股东减持计划完成的公告")
    assert candidate_title("关于持股５％以上股东减持计划实施完毕的公告")


def test_pdf_identity_uses_header_code_not_later_body_mentions() -> None:
    text = "证券代码：002268 关于减持计划完成。其他公告股票代码002286。"
    assert pdf_code_matches(text, "sz.002268")
    assert not pdf_code_matches(text, "sz.002286")
    dual_listing = "证券代码：A股 600613 股票简称：神奇制药 B股 900904"
    assert pdf_code_matches(dual_listing, "sh.600613")
    assert not pdf_code_matches(dual_listing, "sh.900904")
    assert identity_status(text, {
        "code": "sz.002286",
        "pdf_url": "https://static.cninfo.com.cn/finalpage/2025-09-18/random.PDF",
    }) == "identity_unconfirmed"


def test_verified_issuer_header_typo_needs_exact_original_and_issuer() -> None:
    row = {
        "code": "sz.002286",
        "pdf_url": "https://static.cninfo.com.cn/finalpage/2025-09-18/1224667235.PDF",
    }
    body = "证券代码：002268 保龄宝生物股份有限公司关于减持计划实施完成"
    assert identity_status(body, row) == "verified_header_typo"
    assert identity_status(body.replace("保龄宝", "另一家"), row) == "identity_unconfirmed"
    assert not candidate_title("关于持股5%以上股东增持计划完成的公告")
