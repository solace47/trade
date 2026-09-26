import pandas as pd

from trade_research.limit_reopen_inputs import _touches


def test_limit_touch_requires_actual_high_and_reopened_cutoff(tmp_path):
    rows = []
    for code, high, cutoff in (
        ("sh.600001", 11.0, 10.8),
        ("sh.600002", 11.0, 11.0),
        ("sh.600003", 10.99, 10.8),
    ):
        rows.append({
            "date": "2024-06-03", "code": code, "price_1450": cutoff,
            "high_1450": high, "preclose": 10.0, "open_1450": 10.1,
            "isST": 0, "tradestatus": 1, "listing_age_sessions": 100,
            "reference_gap": False, "quote_outside_traded_range": False,
            "amount_1450": 50_000_000,
        })
    rows.append({**rows[0], "date": "2024-12-25", "code": "sh.600004"})
    rows.append({**rows[0], "date": "2025-12-25", "code": "sh.600005"})
    # 17.15 × 1.10 = 18.865 must round to 18.87 in decimal cents.
    rows.append({**rows[0], "code": "sh.600006", "preclose": 17.15,
                 "high_1450": 18.86, "price_1450": 18.29})
    rows.append({**rows[0], "code": "sh.600007", "preclose": 17.15,
                 "high_1450": 18.87, "price_1450": 18.29})
    pd.DataFrame(rows).to_parquet(tmp_path / "sample.parquet", index=False)
    touched = _touches(tmp_path)
    assert set(touched.code) == {"sh.600001", "sh.600002", "sh.600007"}
    assert touched.set_index("code").reopened.to_dict() == {
        "sh.600001": True, "sh.600002": False, "sh.600007": True,
    }
