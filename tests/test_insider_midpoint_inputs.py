import json

import pandas as pd

from trade_research.insider_midpoint_inputs import COOLDOWN, _events, _tag


def test_midpoint_signal_waits_until_next_session(tmp_path):
    counts = {}
    for year, code, notice in (
            (2024, "sh.600001", "2024-04-26"),
            (2025, "sz.000001", "2025-04-30")):
        counts[str(year)] = {"confirmed_zero_pdfs": 1}
        pd.DataFrame([{
            "code": code, "notice_date": notice,
            "pdf_url": f"https://example.org/{year}/{code}.pdf",
        }]).to_parquet(tmp_path / f"confirmed_{year}.parquet")
    (tmp_path / "source_audit.json").write_text(json.dumps(counts))
    days = ["2024-04-26", "2024-04-29", "2025-04-30", "2025-05-06"]
    events, report = _events(tmp_path, days)
    assert COOLDOWN == 120
    assert report["mapped_event_stock_days"] == 2
    assert dict(zip(events.notice_date, events.date)) == {
        "2024-04-26": "2024-04-29",
        "2025-04-30": "2025-05-06",
    }
    pairs = pd.DataFrame([{"date": events.iloc[0].date,
                           "pair_code": events.iloc[0].code,
                           "candidate": "insider_midpoint_zero"}])
    tagged = _tag(pairs, events)
    assert tagged.iloc[0].event_notice_date == "2024-04-26"
    assert tagged.iloc[0].event_exchange == "sh"
