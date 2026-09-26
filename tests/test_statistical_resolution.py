import numpy as np
import pandas as pd
import pytest

from trade_research.statistical_resolution import block_table, noise_summary, paired_days, sample_block_means


def outcome(code, arm, day, pair, value, **kwargs):
    return {"code": code, "candidate": arm, "date": day, "pair_id": pair,
            "known_return5": value, "unknown_after_buy": False, "entry_status": "filled",
            "half": "2025H1", **kwargs}


def test_pair_and_day_denominators_match_original_pairing_not_all_model_rows():
    rows = [outcome("sh.600001", "absolute_model", "2025-01-02", "a", .02),
            outcome("sh.600002", "same_day_control", "2025-01-02", "a", 0),
            outcome("sh.600003", "absolute_model", "2025-01-02", "b", .04),
            outcome("sh.600004", "same_day_control", "2025-01-02", "b", 0),
            outcome("sh.600005", "absolute_model", "2025-01-03", "c", .09),
            outcome("sh.600006", "same_day_control", "2025-01-03", "c", 0),
            outcome("sh.600007", "absolute_model", "2025-01-03", "unmatched", -.9)]
    pairs, daily, counts = paired_days(pd.DataFrame(rows))
    assert len(pairs) == 3 and counts["unpaired_model_rows"] == 1
    assert daily.gap.mean() == pytest.approx(.06)
    assert daily.gap.mean() != pytest.approx(pairs.gap.mean())
    rows[0]["unknown_after_buy"] = True
    with pytest.raises(ValueError, match="unknown is not zero"):
        paired_days(pd.DataFrame(rows))


def test_calendar_weeks_preserve_holiday_lengths_and_removed_daily_mean():
    daily = pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06"], "gap": [1., 1., -2.]})
    blocks = block_table(daily, "calendar_weeks")
    assert blocks.days.tolist() == [2, 1]
    assert blocks.residual_sum.tolist() == [2., -2.]
    class FixedDraws:
        def integers(self, *_args, **_kwargs):
            return np.array([[0, 1], [0, 0], [1, 1]])
    assert sample_block_means(blocks, FixedDraws(), 3).tolist() == [0., 1., -2.]


def test_fortnight_blocks_follow_calendar_not_number_of_observed_weeks():
    daily = pd.DataFrame({"date": ["2025-01-06", "2025-01-13", "2025-01-27"], "gap": [1., 2., 3.]})
    blocks = block_table(daily, "calendar_fortnights")
    assert blocks.block.tolist() == ["0", "1"]
    assert blocks.days.tolist() == [2, 1]


def test_seed_reproducibility_and_artificial_shifts_change_no_observed_outcome():
    daily = pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06"], "gap": [.01, -.02, .03]})
    original = daily.copy(deep=True)
    blocks = block_table(daily, "independent_days")
    first = sample_block_means(blocks, np.random.default_rng(17), 2000)
    second = sample_block_means(blocks, np.random.default_rng(17), 2000)
    assert np.array_equal(first, second)
    summary = noise_summary(first)
    assert summary["shift_for_80pct_exceedance_bps"] == pytest.approx(
        (np.quantile(first, .975) - np.quantile(first, .20)) * 10000)
    probabilities = [r["fraction_above_zero_shift_q975"] for r in summary["shift_exceedance"]]
    assert probabilities == sorted(probabilities)
    pd.testing.assert_frame_equal(daily, original)
