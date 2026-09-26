from datetime import date

import pandas as pd
import pytest

from trade_research.cash_dividend_notices import compare_catalog, notice_rows, range_rows


def notice(number=1, **changes):
    return dict({"secCode": "600001", "announcementId": str(number),
        "announcementTitle": "2023 年年度权益分派实施公告",
        "adjunctUrl": f"finalpage/2024-05-20/{number}.PDF",
        "announcementTime": int(pd.Timestamp("2024-05-20", tz="Asia/Shanghai").timestamp() * 1000)},
        **changes)


def test_distributions_are_distinct_from_adjustment_and_repurchase_announcements():
    data = [notice(), notice(), notice(2, secCode="000001", announcementTitle=
        "2023 年年度利润分配及资本公积金转增股本实施公告（修订后）"),
        notice(3, announcementTitle="关于权益分派实施后调整回购价格上限的公告"),
        notice(4, announcementTitle="分红派息实施公告", secCode="900001")]
    frame = notice_rows(data)
    assert len(frame) == 2
    assert frame.revision_flag.sum() == 1
    assert not frame.terms_verified.any()
    with pytest.raises(ValueError, match="conflicting metadata"):
        notice_rows([notice(), notice(announcementTitle="2023 年分红派息实施公告")])


@pytest.mark.parametrize("title", ["2024年度权益分派实施的公告", "2024年年度权益派发实施公告",
    "2024年末期利润分派A股实施公告", "2024年A股股息分派实施公告", "年度现金红利实施公告",
    "2023年年度权益分派公告", "2024年年度权益分派的公告", "权益分派实施公告2024-25",
    "600328_中盐化工_权益分派实施公告_2024-05-17", "权益分派实施公告.docx", "权益分派实施公告V4",
    "2024年年度权益分配实施公告", "关于权益分派实施方案的公告", "2024年年度权益分派实施",
    "2024年度权益分派的实施公告", "2024年度利润分配方案实施公告", "2024年权益分派（A股）实施公告",
    "2024年度分红派息、转增股本实施公告"])
def test_historical_title_variants_remain_searchable(title):
    assert len(notice_rows([notice(announcementTitle=title)])) == 1


@pytest.mark.parametrize("title", ["关于收到控股股东提议实施2024年度分红的公告",
    "关于2024年拟实施中期分红的公告", "关于参股公司实施现金分红的公告",
    "关于推迟2023年度利润分配实施的公告", "2024年度拟不实施利润分配的公告"])
def test_proposals_and_postponements_are_not_implementation_notices(title):
    assert notice_rows([notice(announcementTitle=title)]).empty


def test_final_partial_page_is_required_and_repeated_pages_are_rejected(tmp_path, monkeypatch):
    def fetch(start, end, keyword, page, cache):
        return {"totalAnnouncement": 31, "totalpages": 1,
                "announcements": [notice(n) for n in (range(30) if page == 1 else [30])]}
    monkeypatch.setattr("trade_research.cash_dividend_notices.fetch_page", fetch)
    assert len(range_rows(date(2024, 5, 1), date(2024, 5, 31), "k", tmp_path)) == 31

    def repeated(*args):
        return {"totalAnnouncement": 31, "announcements": [notice(n) for n in range(30)]}
    monkeypatch.setattr("trade_research.cash_dividend_notices.fetch_page", repeated)
    with pytest.raises(ValueError, match="repeated"):
        range_rows(date(2024, 5, 1), date(2024, 5, 31), "k", tmp_path)


def test_notice_date_is_local_and_future_year_is_rejected():
    assert notice_rows([notice()]).iloc[0].notice_date == "2024-05-20"
    with pytest.raises(ValueError, match="input years"):
        notice_rows([notice(announcementTime=int(pd.Timestamp("2026-01-01", tz="Asia/Shanghai").timestamp() * 1000))])


def test_changing_pagination_is_rebuilt_as_disjoint_smaller_ranges(tmp_path, monkeypatch):
    calls = []

    def fetch(start, end, keyword, page, cache):
        calls.append((start, end, page))
        if start != end:
            return {"totalAnnouncement": 31 if page == 1 else 30,
                    "announcements": [notice(n) for n in range(30)]}
        stamp = int(pd.Timestamp(str(start), tz="Asia/Shanghai").timestamp() * 1000)
        return {"totalAnnouncement": 1, "announcements": [notice(start.day, announcementTime=stamp)]}

    monkeypatch.setattr("trade_research.cash_dividend_notices.fetch_page", fetch)
    result = range_rows(date(2024, 5, 1), date(2024, 5, 2), "k", tmp_path)
    assert [r["announcementId"] for r in result] == ["1", "2"]
    assert (date(2024, 5, 1), date(2024, 5, 1), 1) in calls
    assert (date(2024, 5, 2), date(2024, 5, 2), 1) in calls


def test_two_way_comparison_keeps_missing_and_ambiguous_notices():
    events = pd.DataFrame([
        {"code": "sh.600001", "dividPlanDate": "2024-05-20", "dividOperateDate": "2024-05-30"},
        {"code": "sz.000001", "dividPlanDate": "", "dividOperateDate": "2024-05-30"}])
    notices = notice_rows([notice(), notice(2, announcementTitle="权益分派实施公告（修订后）"),
                          notice(3, secCode="600002", announcementTitle="A股派息实施公告")])
    linked, missing = compare_catalog(events, notices)
    assert linked.notice_candidates.tolist() == [2, 0]
    assert linked.notice_revision_candidates.tolist() == [1, 0]
    assert linked.notice_precedes_action.tolist() == [True, False]
    assert missing.code.tolist() == ["sh.600002"]
    assert not linked.terms_verified.any()
