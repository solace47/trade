"""The opening-price exception must not admit other source defects."""

from types import SimpleNamespace

import pandas as pd

from scripts.audit_open_only_quality import _classify
from trade_research.hf_audit import EXPECTED_LABELS


def test_opening_only_requires_matching_later_prices_and_turnover() -> None:
    bars = pd.DataFrame({
        "label": EXPECTED_LABELS,
        "open": [10.1] + [10.0] * 240,
        "high": [10.1] + [10.0] * 240,
        "low": [10.0] * 241,
        "close": [10.0] * 241,
        "volume": [100] * 241,
        "turnover": [1000.0] * 241,
    })
    daily = SimpleNamespace(
        tradestatus=1, open=10.0, high=10.1, low=10.0, close=10.0,
        volume=24100, amount=241000.0,
    )
    assert _classify(bars, daily)["classification"] == (
        "opening_only_volume_matched")
    assert _classify(bars, SimpleNamespace(
        **{**vars(daily), "high": 10.2}))["classification"] == (
            "other_ohlc_or_turnover_issue")
    assert _classify(bars, SimpleNamespace(
        **{**vars(daily), "amount": 250000.0}))["classification"] == (
            "other_ohlc_or_turnover_issue")
    assert _classify(bars.iloc[:-1], daily)["classification"] == (
        "invalid_minute_grid")
