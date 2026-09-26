from itertools import product

import numpy as np
import pandas as pd
import pytest

from trade_research.entry_price_ranges import build_ranges, hypothetical_buy_cash, linear_interval


def test_signed_interval_matches_all_vertex_values():
    weights = np.array([.7, -.3, -.4, 0.])
    lower = np.array([-2., -4., 3., -100.])
    upper = np.array([3., 1., 9., 100.])
    vertices = [float(weights @ np.where(choice, upper, lower)) for choice in product([False, True], repeat=4)]
    lo, hi = linear_interval(weights, lower, upper)
    assert lo == pytest.approx(min(vertices))
    assert hi == pytest.approx(max(vertices))
    assert linear_interval(weights, lower, lower) == pytest.approx((weights@lower, weights@lower))


@pytest.mark.parametrize("weights,low,high", [([1],[2],[1]), ([1],[0,1],[1,2]), ([1],[0],[np.nan]), ([],[],[])])
def test_invalid_ranges_are_not_silently_zeroed(weights, low, high):
    with pytest.raises(ValueError):
        linear_interval(weights, low, high)


def test_cash_commission_floor_and_slippage_are_monotone():
    assert hypothetical_buy_cash(100, 10) == pytest.approx(1000.5 + 5 + .010005)
    assert hypothetical_buy_cash(2000, 10) == pytest.approx(20010 * 1.00031)
    assert hypothetical_buy_cash(2000, 10.02) > hypothetical_buy_cash(2000, 10)


def test_bad_vwap_is_retained_and_zero_volume_price_does_not_set_bounds():
    signals = pd.DataFrame({"date": ["2024-05-06"], "code": ["sh.600001"], "half": ["2024H1"],
        "side": ["above"], "quote_cents": [1001], "price_1449": [10.01], "contrast_weight": [.2]})
    entries = signals[["date", "code"]].assign(bars_valid=False, target_shares=1900)
    raw = pd.DataFrame({"timestamp": pd.date_range("2024-05-06 14:52", periods=4, freq="min"),
        "date": "2024-05-06", "code": "sh.600001", "low": [10.,10.,10.,9.],
        "high": [10.02,10.02,10.02,11.], "volume": [10000,10000,10000,0],
        "turnover": [100400,100400,100400,0]})
    rows = build_ranges(signals, entries, raw)
    assert len(rows) == 1 and rows.source_flagged.iloc[0]
    assert rows.lower_raw_price.iloc[0] == 10.
    assert rows.upper_raw_price.iloc[0] == 10.02
    assert rows.unverified_raw_vwap.iloc[0] == 10.04
    assert rows.unverified_vwap_outside_ohlc.iloc[0]
    assert rows.baseline_drift_bps.iloc[0] > rows.upper_drift_bps.iloc[0]
