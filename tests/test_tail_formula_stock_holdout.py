import numpy as np
import pandas as pd
import pytest

from trade_research import tail_formula_stock_holdout as study


def test_both_venues_follow_the_same_fixed_last_digit_groups():
    codes = pd.Series([venue + '.' + '60000' + str(i) for venue in ['sh', 'sz'] for i in range(10)])
    np.testing.assert_array_equal(study.group_ids(codes), ([0] * 5 + [1] * 5) * 2)
    with pytest.raises(AssertionError):
        study.group_ids(pd.Series(['sh.60000']))


def test_calibration_uses_opposite_stocks_and_keeps_unknown_outcomes():
    f = pd.DataFrame(dict(date=['2024-12-31'] * 3 + ['2025-01-01'],
        code=['sh.600001', 'sh.600005', 'sz.000006', 'sz.000007'],
        formula_input_valid=[True] * 4, score=[100., 1., 3., 500.], known15=[True, False, True, True]))
    cut, frame = study.visible_group_cut(f, '2024-01-01', '2025-01-01', 0)
    assert frame.code.tolist() == ['sh.600005', 'sz.000006']
    assert cut == np.quantile([1., 3.], .995)
    f.known15 = ~f.known15
    assert study.visible_group_cut(f, '2024-01-01', '2025-01-01', 0)[0] == cut


def test_opposite_score_routing_and_invalid_input_are_not_reversed():
    f = pd.DataFrame(dict(code=['sh.600001', 'sz.000006', 'sh.600002'], formula_input_valid=[True, True, False]))
    # Model 0 selects only group 1; model 1 selects only group 0.
    score0, score1, cuts = [0., 3., 3.], [4., 0., 4.], [2., 2.]
    np.testing.assert_array_equal(study.route_flags(f, score0, score1, cuts, True), [True, True, False])
    np.testing.assert_array_equal(study.route_flags(f, score0, score1, cuts, False), [False, False, False])


def test_nonfinite_valid_opposite_score_fails_instead_of_changing_population():
    f = pd.DataFrame(dict(date=['2024-12-31'], code=['sh.600005'], formula_input_valid=[True], score=[np.nan]))
    with pytest.raises(AssertionError):
        study.visible_group_cut(f, '2024-01-01', '2025-01-01', 0)
