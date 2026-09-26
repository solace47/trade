import numpy as np
import pandas as pd
import pytest

from trade_research.fixed_return_ranges import complete_summary, corner_return_range, economic_return, pair_ranges


def test_share_distribution_cash_and_commission_floor_enter_economic_return():
    expected = (1040 - 5 - .0104 - .52 + 50 - 10) / (1000 + 5 + .01) - 1
    assert economic_return(100, 130, 10, 8, 50, 10) == pytest.approx(expected)
    assert economic_return(100, 130, 10, 8, 50, 10) != pytest.approx(8/10-1)


def test_return_range_includes_interior_prices_and_fee_threshold():
    low, high = corner_return_range(1000, 1000, 16.6, 16.8, 16.5, 17., 0, 0)
    values = [economic_return(1000, 1000, buy, sell, 0, 0)
              for buy in np.linspace(16.6, 16.8, 9) for sell in np.linspace(16.5, 17., 9)]
    assert min(values) == pytest.approx(low) and max(values) == pytest.approx(high)
    point = economic_return(1000, 1000, 16.7, 16.9, 0, 0)
    assert corner_return_range(1000, 1000, 16.7, 16.7, 16.9, 16.9, 0, 0) == pytest.approx((point, point))


def test_unknown_terminal_row_never_shrinks_a_mean_denominator():
    frame = pd.DataFrame({"date": ["2024-07-01", "2024-07-01"], "unknown_terminal": [False, True],
        "baseline": [.1, np.nan], "lower": [.09, np.nan], "upper": [.11, np.nan]})
    summary = complete_summary(frame)
    assert summary['rows'] == 2 and summary['unknown_terminal_rows'] == 1
    assert all(summary[key] is None for key in ['baseline_bps', 'lower_bps', 'upper_bps'])
    frame.loc[1, "unknown_terminal"] = False
    with pytest.raises(ValueError, match="missing"):
        complete_summary(frame)


def test_daily_equal_weight_keeps_no_buy_zero_in_signal_denominator():
    frame = pd.DataFrame({"date": ['2024-07-01','2024-07-01','2024-07-02'],
        "unknown_terminal": False, "baseline": [.1,0,.2], "lower": [.1,0,.2], "upper": [.1,0,.2]})
    assert complete_summary(frame)['baseline_bps'] == pytest.approx(1250.)


def test_pair_bounds_use_opposite_control_endpoint_and_preserve_unmatched_model():
    frame = pd.DataFrame({"date": '2024-07-01', "code": ['sh.600001','sh.600002','sh.600003'],
        "candidate": ['absolute_model','same_day_control','absolute_model'], "pair_id": ['p1','p1','p2'],
        "unknown_terminal": False, "baseline": [.05,.02,.3], "lower": [.03,.01,.2], "upper": [.07,.04,.4]})
    paired = pair_ranges(frame)
    assert len(paired) == 1
    assert paired.lower.iloc[0] == pytest.approx(-.01)
    assert paired.upper.iloc[0] == pytest.approx(.06)
    assert len(frame) == 3
