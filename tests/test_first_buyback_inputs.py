import json

import pandas as pd

from trade_research.first_buyback_inputs import _events, _plan_gap


def test_first_trade_signal_strictly_follows_notice_and_removes_duplicates(tmp_path) -> None:
    title = "关于首次回购公司股份的公告"
    for year, rows in {
        2024: [("sh.600001", "2024-01-05", "a"),
               ("sh.600001", "2024-01-05", "b"),
               ("sh.600002", "2024-01-05", "c")],
        2025: [("sz.000001", "2025-01-02", "d")],
    }.items():
        frame = pd.DataFrame([
            {"code": code, "notice_date": date, "title": title,
             "pdf_url": f"https://example.org/{suffix}.pdf"}
            for code, date, suffix in rows
        ])
        frame.to_parquet(tmp_path / f"title_candidates_{year}.parquet")
        with (tmp_path / f"pdf_audit_{year}.jsonl").open("w") as output:
            for url in frame.pdf_url:
                output.write(json.dumps({"pdf_url": url, "status": "ok"}) + "\n")
    calendar = ["2024-01-05", "2024-01-08", "2025-01-02", "2025-01-03",
                "2026-01-05"]
    events, report = _events(tmp_path, calendar)
    assert report["duplicate_code_notice_rows_removed"] == 2
    assert set(zip(events.code, events.date)) == {
        ("sh.600002", "2024-01-08"),
        ("sz.000001", "2025-01-03"),
    }
    assert events.date.gt(events.notice_date).all()


def test_recent_plan_flag_uses_only_announcements_already_published(tmp_path) -> None:
    for year, notices in {
        2024: ["2024-01-02", "2024-01-09"],
        2025: [],
    }.items():
        with (tmp_path / f"pdf_audit_{year}.jsonl").open("w") as output:
            for notice in notices:
                output.write(json.dumps({"code": "sh.600001",
                                         "notice_date": notice,
                                         "status": "ok"}) + "\n")
        if not notices:
            # read_json needs a valid JSONL row even when no valid plan exists.
            (tmp_path / f"pdf_audit_{year}.jsonl").write_text(json.dumps({
                "code": "sz.000001", "notice_date": "2025-01-02",
                "status": "identity_unconfirmed"}) + "\n")
    events = pd.DataFrame([{"date": "2024-01-08", "code": "sh.600001",
                            "notice_date": "2024-01-05"}])
    calendar = ["2024-01-02", "2024-01-03", "2024-01-05",
                "2024-01-08", "2024-01-09", "2024-01-10"]
    gap = _plan_gap(events, calendar, tmp_path).iloc[0]
    assert gap.prior_plan_notice == "2024-01-02"
    assert gap.plan_gap_sessions == 2
    assert gap.recent_plan_10
