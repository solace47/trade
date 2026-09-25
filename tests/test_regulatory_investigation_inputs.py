import json

import pandas as pd

from trade_research.regulatory_investigation_inputs import (
    _events, _five_peer_pairs,
)


def test_issuer_notice_maps_to_next_session_and_skips_2026(tmp_path):
    reports = {}
    for year, day, code in (
            (2024, "2024-04-26", "sh.600001"),
            (2025, "2025-12-31", "sz.000001")):
        reports[str(year)] = {"confirmed_issuer_disclosure_notices": 1}
        pd.DataFrame([{"code": code, "notice_date": day,
                       "pdf_url": f"https://a/{year}/one.pdf"}]).to_parquet(
                           tmp_path / f"confirmed_{year}.parquet")
    (tmp_path / "source_audit.json").write_text(json.dumps(reports))
    calendar = ["2024-04-26", "2024-04-29", "2025-12-31", "2026-01-05"]
    events, report = _events(tmp_path, calendar)
    assert report["confirmed_originals"] == 2
    assert len(events) == 1
    assert events.iloc[0].date == "2024-04-29"


def test_main_matching_needs_three_same_industry_peers():
    rows = []
    for code, industry, liquidity in [
        ("sh.600001", "C1", 100), ("sh.600002", "C1", 101),
        ("sh.600003", "C1", 102), ("sh.600004", "C1", 103),
        ("sh.600005", "C1", 104), ("sh.600006", "C1", 105),
        ("sh.600007", "C2", 100),
    ]:
        rows.append({"date": "2024-04-29", "trade_date": "2024-04-26",
                     "code": code, "board": "sh_main", "industry": industry,
                     "avg20_amount": liquidity, "float_mv": 1000,
                     "return20_prior_adjusted": .01, "return_1450": .01})
    universe = pd.DataFrame(rows)
    event = pd.DataFrame([{"date": "2024-04-29", "code": "sh.600001",
                           "notice_date": "2024-04-26", "pdf_url": "https://a/1.pdf"}])
    pairs, report = _five_peer_pairs(universe, event,
                                     ["2024-04-26", "2024-04-29"])
    assert report["matched_events"] == 1
    assert len(pairs) == 6
    assert "sh.600007" not in set(pairs.code)
