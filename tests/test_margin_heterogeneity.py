"""Check chronological model math and the ten-session position cooldown."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_research.margin_heterogeneity import (
    TREATED, _hac_fit, _select_window,
)


def test_hac_fit_recovers_synthetic_slope() -> None:
    random = np.random.default_rng(42)
    x = random.normal(size=120)
    frame = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=120).strftime("%Y-%m-%d"),
        "interest": .1 + .01 * x,
        "return20_prior_adjusted": random.normal(0, .05, 120),
        "return_1450": random.normal(0, .01, 120),
        "relative_net": .02 * x + random.normal(0, .005, 120),
    })
    fitted = _hac_fit(frame, 80)
    assert fitted is not None
    assert fitted["beta"] > .01
    assert fitted["hac_t"] > 1.645
    frame["relative_net"] *= -1
    negative = _hac_fit(frame, 80)
    assert negative is not None and negative["hac_t"] < -1.645


def test_cooldown_prevents_overlapping_stock_entries() -> None:
    dates = pd.bdate_range("2024-07-01", periods=12).strftime("%Y-%m-%d")
    rows = []
    for day in dates:
        for code, interest in (("sz.000001", .12), ("sz.000002", .05)):
            rows.append({
                "date": day, "trade_date": "2024-06-28", "code": code,
                "board": "sz_main", "size_bucket": 3,
                "interest": interest, "avg20_amount": 150_000_000.0,
                "amount_1450": 150_000_000.0, "float_mv": 10_000_000_000.0,
                "return20_prior_adjusted": .01, "return_1450": 0.0,
                "open_gap": 0.0, "price_1450": 10.0,
            })
    model = pd.DataFrame([{
        "code": "sz.000001", "interest_mean": .1,
        "interest_std": .01, "beta": .01, "hac_t": 2.0,
    }])
    selected, report = _select_window(
        pd.DataFrame(rows), model, dates[0], dates[-1],
        {day: index for index, day in enumerate(dates)},
    )
    high = selected.loc[selected.candidate.eq(TREATED)]
    assert high.date.tolist() == [dates[0], dates[11]]
    assert report["matched_pairs"] == 2
