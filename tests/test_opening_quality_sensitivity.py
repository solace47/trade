import pandas as pd

from scripts.evaluate_opening_quality_sensitivity import _released_days


def test_release_keeps_other_bad_day_excluded():
    classified = pd.DataFrame([
        {"date": "2024-01-02", "code": "sh.600001",
         "classification": "opening_only_volume_matched"},
        {"date": "2024-01-03", "code": "sh.600001",
         "classification": "opening_only_volume_matched"},
        {"date": "2024-01-04", "code": "sh.600001",
         "classification": "other_ohlc_or_turnover_issue"},
    ])
    issues = pd.DataFrame([
        {"date": "2024-01-02", "code": "sh.600001",
         "kind": "ohlc_disagreement"},
        {"date": "2024-01-03", "code": "sh.600001",
         "kind": "ohlc_disagreement"},
        {"date": "2024-01-03", "code": "sh.600001",
         "kind": "partial_minute_day"},
        {"date": "2024-01-04", "code": "sh.600001",
         "kind": "ohlc_disagreement"},
    ])
    released = _released_days(classified, issues)
    assert released.to_dict("records") == [
        {"date": "2024-01-02", "code": "sh.600001"}]
