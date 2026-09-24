import json

import pandas as pd

from trade_research.buyback_inputs import _capacity_events, _events


def test_buyback_announcement_is_available_only_next_session(tmp_path) -> None:
    title = "关于回购公司股份方案的公告"
    for year, code, notice, url in (
            (2024, "sh.600001", "2024-04-26", "https://example.test/a.pdf"),
            (2025, "sz.000001", "2025-04-30", "https://example.test/b.pdf")):
        pd.DataFrame([{"code": code, "notice_date": notice,
                       "title": title, "pdf_url": url}]).to_parquet(
                           tmp_path / f"title_candidates_{year}.parquet")
        (tmp_path / f"pdf_audit_{year}.jsonl").write_text(
            json.dumps({"pdf_url": url, "status": "ok"}) + "\n")
    days = ["2024-04-26", "2024-04-29", "2024-04-30", "2025-04-30",
            "2025-05-06", "2025-05-07"]
    events, _ = _events(tmp_path, days)
    assert dict(zip(events.notice_date, events.date)) == {
        "2024-04-26": "2024-04-29",
        "2025-04-30": "2025-05-06",
    }


def test_buyback_capacity_and_cooldown_use_inputs_only() -> None:
    days = [f"2024-01-{day:02d}" for day in range(2, 16)]
    rows = [{"date": days[0], "code": f"sh.{n:06d}",
             "avg20_amount": n} for n in range(1, 8)]
    rows += [{"date": days[4], "code": "sh.000007", "avg20_amount": 100},
             {"date": days[11], "code": "sh.000007", "avg20_amount": 100}]
    chosen = _capacity_events(pd.DataFrame(rows), days)
    assert chosen.loc[chosen.date.eq(days[0]), "code"].tolist() == [
        "sh.000007", "sh.000006", "sh.000005", "sh.000004", "sh.000003"]
    assert not chosen.date.eq(days[4]).any()
    assert chosen.loc[chosen.date.eq(days[11]), "code"].tolist() == [
        "sh.000007"]
