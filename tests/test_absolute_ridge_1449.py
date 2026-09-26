"""Guard the conservative cutoff against later-bar inputs and joins."""

import duckdb
import pandas as pd
import pytest

from trade_research import absolute_ridge_1449


def _sources(root, later_price: float, extra_prefix: bool = False) -> None:
    prefix = pd.DataFrame([{
        "date": "2024-07-01", "code": "sh.600000",
        "price_1449": 10.0, "amount_1449": 200_000_000.0,
        "volume_1449": 20_000_000, "high_1449": 11.0,
        "low_1449": 9.0, "volume_last29": 2_000_000,
        "amount_last29": 20_000_000.0, "price_1420": 9.9,
        "quote_outside_traded_range": False,
    }])
    if extra_prefix:
        unmatched = prefix.copy()
        unmatched["code"] = "sh.600001"
        prefix = pd.concat([prefix, unmatched], ignore_index=True)
    snapshot = pd.DataFrame([{
        "date": "2024-07-01", "code": "sh.600000",
        "isST": 0, "listing_age_sessions": 100,
        "reference_gap": False, "preclose": 10.0,
        "open_1450": 10.0, "volume5_prior": 20_000_000,
        "return5_prior_adjusted": 0.01,
        "return20_prior_adjusted": 0.02,
        "ma20_prior_adjusted": 10.0,
        "price_1450": later_price,
        "volume_1450": 20_000_000 if later_price == 10 else 90_000_000,
    }])
    prefix_path = root / "minute_prefix_1449" / "2024"
    snapshot_path = root / "market_snapshots_ci"
    prefix_path.mkdir(parents=True, exist_ok=True)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    prefix.to_parquet(prefix_path / "part_000.parquet", index=False)
    snapshot.to_parquet(snapshot_path / "part_000.parquet", index=False)


def test_later_bar_values_cannot_change_1449_features(tmp_path, monkeypatch):
    monkeypatch.setattr(absolute_ridge_1449, "ROOT", tmp_path)
    _sources(tmp_path, later_price=10.0)
    first = absolute_ridge_1449.feature_frame(duckdb.connect())
    _sources(tmp_path, later_price=99.0)
    second = absolute_ridge_1449.feature_frame(duckdb.connect())
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 1
    assert first.iloc[0].price_signal == 10.0


def test_amount_eligible_prefix_cannot_depend_on_later_join(tmp_path, monkeypatch):
    monkeypatch.setattr(absolute_ridge_1449, "ROOT", tmp_path)
    _sources(tmp_path, later_price=10.0, extra_prefix=True)
    with pytest.raises(ValueError, match="14:49 amount-eligible"):
        absolute_ridge_1449.feature_frame(duckdb.connect())
