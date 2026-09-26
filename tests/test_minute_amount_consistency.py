import numpy as np
import pandas as pd
import pytest

from trade_research.minute_amount_consistency import amount_envelope, mark_consistency, shifted_ranges


def bars():
    return pd.DataFrame({"timestamp": pd.date_range("2024-05-06 14:52", periods=2, freq="min"),
        "date": "2024-05-06", "code": "sh.600001", "open": [10., 10.02], "close": [10., 10.02],
        "low": [10., 10.02], "high": [10., 10.02], "volume": [10000., 10000.],
        "turnover": [100200., 100000.], "half": "2024H1", "label": ["1452", "1453"],
        "affected": True, "reference": False})


def test_internal_amount_conflicts_can_cancel_without_proving_correctness():
    frame = bars()
    marked = mark_consistency(frame)
    assert marked.inconsistent.all()
    assert amount_envelope(frame)["within_weighted_price_envelope"]
    frame["turnover"] = 110000.
    audit = amount_envelope(frame)
    assert not audit["within_weighted_price_envelope"]
    assert audit["outside_amount_yuan"] > 0


def test_small_storage_rounding_cannot_explain_material_vwap_conflict():
    marked = mark_consistency(bars())
    assert (marked.outside_yuan > marked.float32_envelope_yuan).all()
    assert marked.float32_envelope_yuan.max() < .00001
    zero = bars()
    zero.loc[0, ["volume", "turnover"]] = 0
    marked = mark_consistency(zero)
    assert not marked.positive_finite.iloc[0] and not marked.inconsistent.iloc[0]
    assert np.isnan(marked.vwap.iloc[0])


def test_offset_direction_is_exact_clock_time_and_does_not_wrap_missing_minutes():
    checked = shifted_ranges(mark_consistency(bars()))
    original = checked.loc[checked.offset_minutes.eq(0)]
    assert not original.compatible.any()
    future = checked.loc[checked.offset_minutes.eq(1)].reset_index(drop=True)
    assert future.matched.tolist() == [True, False]
    assert future.compatible.tolist() == [True, False]
    past = checked.loc[checked.offset_minutes.eq(-1)].reset_index(drop=True)
    assert past.matched.tolist() == [False, True]
    assert past.compatible.tolist() == [False, True]


def test_duplicate_timestamp_is_not_silently_matched():
    marked = mark_consistency(bars())
    with pytest.raises(pd.errors.MergeError):
        shifted_ranges(pd.concat([marked, marked.iloc[:1]], ignore_index=True))
