"""Protect the all-candidate upper bound used to stop before outcome reads."""

import hashlib
from pathlib import Path

import pandas as pd

from trade_research import late_semivariance_input


def test_input_upper_bound_checks_highs_beyond_daily_first_five(
    tmp_path: Path, monkeypatch,
) -> None:
    dates = ("2024-01-10", "2024-07-10", "2025-01-10", "2025-07-10")
    first_codes = [f"sh.{number:06d}" for number in range(600001, 600007)]
    ordered = sorted(first_codes, key=lambda code: (
        hashlib.md5(("semivar-v1" + dates[0] + code).encode()).hexdigest(),
        code,
    ))
    matchable_beyond_five = ordered[-1]
    rows = [(dates[0], code, code == matchable_beyond_five, True)
            for code in first_codes]
    rows.append((dates[0], "sz.000001", True, False))
    rows.extend((date, f"sh.{index + 600100:06d}", False, True)
                for index, date in enumerate(dates[1:]))
    snapshots = []
    intraday = []
    variance = []
    for date, code, negative_tail, high in rows:
        snapshots.append({
            "date": date, "code": code, "isST": 0,
            "listing_age_sessions": 100, "reference_gap": False,
            "quote_outside_traded_range": False, "price_1450": 10.0,
            "amount_1450": 200_000_000,
            "return20_prior_adjusted": 0.0, "return_1450": .01,
            "position_1450": .8,
        })
        intraday.append({
            "date": date, "code": code, "price_1450": 10.0,
            "return_last30": -.003 if negative_tail else .003,
            "volume_share_last30": .1,
        })
        variance.append({
            "date": date, "code": code, "price_1450": 10.0,
            "realized_variance_last30": 1e-6,
            "downside_variance_last30": 8e-7 if high else 2e-7,
            "upside_variance_last30": 2e-7 if high else 8e-7,
            "downside_share_last30": .8 if high else .2,
            "down_moves_last30": 3, "up_moves_last30": 3,
        })
    for directory, data in (
        ("market_snapshots_ci", snapshots),
        ("intraday_features", intraday),
        ("late_semivariance", variance),
    ):
        folder = tmp_path / directory
        folder.mkdir()
        pd.DataFrame(data).to_parquet(folder / "input.parquet", index=False)
    monkeypatch.setattr(late_semivariance_input, "ROOT", tmp_path)

    result = late_semivariance_input.audit(tmp_path / "audit.json")

    assert result["by_half_upper_bound"][0]["all_high"] == 6
    assert result["by_half_upper_bound"][0]["full_match"] == 1
    assert result["outcome_gate_possible"] is False
