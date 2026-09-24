import numpy as np
import pandas as pd

from trade_research.earnings_drift import event_pairs, match_controls, next_session


def test_announcements_wait_until_next_session_and_exclude_2023() -> None:
    events = pd.DataFrame({
        "code": ["sh.600000", "sh.600001", "sh.600002"],
        "published": ["2023-12-31", "2024-01-05", "2024-01-08"],
    })
    sessions = np.array(["2024-01-05", "2024-01-08", "2024-01-09"])
    mapped = next_session(events, "published", sessions)
    assert mapped[["code", "date"]].values.tolist() == [
        ["sh.600001", "2024-01-08"],
        ["sh.600002", "2024-01-09"],
    ]


def test_mixed_forecast_types_on_one_day_are_not_positive() -> None:
    forecast = pd.DataFrame({
        "code": ["sh.600000", "sh.600000", "sh.600001"],
        "profitForcastExpPubDate": ["2024-01-06"] * 3,
        "profitForcastType": ["预增", "预减", "扭亏"],
    })
    express = pd.DataFrame({
        "code": ["sh.600002"], "performanceExpUpdateDate": ["2024-01-06"],
    })
    positive, all_events = event_pairs(
        forecast, express, np.array(["2024-01-08", "2024-01-09"]))
    assert positive.code.tolist() == ["sh.600001"]
    assert set(all_events.code) == {"sh.600000", "sh.600001", "sh.600002"}


def test_controls_are_same_day_distinct_nonannouncements() -> None:
    pool = pd.DataFrame({
        "date": ["2024-01-08"] * 4,
        "code": ["sh.600000", "sh.600001", "sh.600002", "sh.600003"],
        "open_gap": [.02, .03, .021, .031],
        "return_1450": [.03, .04, .031, .041],
        "amount_1450": [1e8] * 4,
        "return20_prior_adjusted": [.01] * 4,
        "has_event": [True, True, False, False],
    })
    chosen = pool.iloc[:2].copy()
    matched = match_controls(chosen, pool)
    assert matched.code.tolist() == ["sh.600002", "sh.600003"]
    assert matched.pair_code.tolist() == ["sh.600000", "sh.600001"]
    assert matched.match_distance.max() < .2
