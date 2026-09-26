import numpy as np
import pandas as pd
import pytest

from trade_research.reference_gain_pairs import comparable_pairs


def sample():
    return pd.DataFrame({"date": ["2024-06-01"] * 4,
        "code": ["sh.600001", "sh.600002", "sh.600003", "sh.600004"],
        "half": ["2024H1"] * 4, "day_return": [-.01] * 4,
        "return_last29": [0.] * 4, "prior20_return": [0, .01, .03, .05],
        "price_1449": [10.] * 4, "amount_1449": [2e8] * 4,
        "prior20_turnover_pct": [2.] * 4, "gain_seed_one": [.1, 0, .2, 0],
        "screen_gain_low": [.099, -.001, .199, -.001],
        "screen_gain_high": [.101, .001, .201, .001]})


def unordered_ends(frame):
    return {tuple(sorted([r.code, r.control_code])) for r in frame.itertuples()}


def test_controls_choose_pairs_before_reference_orientation():
    frame = sample()
    original = comparable_pairs(frame)
    assert unordered_ends(original) == {("sh.600001", "sh.600002"), ("sh.600003", "sh.600004")}
    changed = frame.copy()
    changed["gain_seed_one"] *= -1
    assert unordered_ends(comparable_pairs(changed)) == unordered_ends(original)
    assert comparable_pairs(changed).code.tolist() != original.code.tolist()
    pd.testing.assert_frame_equal(comparable_pairs(frame.sample(frac=1, random_state=42)), original)


def test_no_reuse_and_exchange_calipers():
    frame = sample()
    frame.loc[3, "code"] = "sz.000001"
    frame.loc[2, "prior20_return"] = .20
    out = comparable_pairs(frame)
    assert len(out) == 1
    assert out.iloc[0].scenario_gap == pytest.approx(.098)
    ends = out.code.tolist() + out.control_code.tolist()
    assert len(set(ends)) == len(ends)


def test_invalid_features_are_not_zero_filled():
    frame = sample()
    frame.loc[0, "prior20_turnover_pct"] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        comparable_pairs(frame)
