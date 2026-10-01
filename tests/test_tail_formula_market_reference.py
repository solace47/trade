import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_tail_formula_market_reference import calendar_difference, target_groups, visible_index_groups


def test_reference_target_includes_input_invalid_but_mature_known_only():
    d = pd.DataFrame(dict(date=['2023-12-27']*4 + ['2023-12-29'],
        code=['sh.600001', 'sh.600002', 'sz.000001', 'sz.000002', 'sh.600003'],
        next_date=['2023-12-28']*4 + ['2024-01-02'], known15=[True, True, True, False, True],
        known_no_trade=[False]*5, mark_0959_return15=[.01, .03, -.01, .4, .9],
        formula_input_valid=[True, False, True, True, True]))
    r = target_groups(d, '2023-01-01', '2024-01-01', '2024-01-01')
    assert len(r) == 2 and r.reference_rows.tolist() == [2, 1]
    np.testing.assert_allclose(r.target, [2, -1], rtol=0, atol=1e-12)
    assert r.last_observation.max() == '2023-12-28'


def test_index_projection_ignores_future_unknown_and_fill_fields():
    d = pd.DataFrame(dict(date=['2024-01-02']*3, exchange=['sh', 'sh', 'sz'],
        formula_input_valid=[True, False, True], J02=[1., 99., 2.], J03=[3., 99., 4.], J04=[5., 99., 6.],
        known15=[True, True, False], known_no_trade=[False, False, True]))
    a = visible_index_groups(d); d.known15 = ~d.known15; d.known_no_trade = ~d.known_no_trade
    pd.testing.assert_frame_equal(a, visible_index_groups(d), check_exact=True)
    assert a.J02.tolist() == [1., 2.]


def test_calendar_keeps_empty_bootstrap_draws_and_exact_identical_zero():
    dates = ['2024-01-02', '2024-01-09']
    old = pd.DataFrame(dict(date=dates, rows=[2, 2], rate=[.4, .6]))
    same = calendar_difference(dates, old, old, 'rate', 'rate')
    assert same['difference'] == 0 and same['ci'] == [0., 0.]
    narrow = calendar_difference(dates, old.iloc[:1], old, 'rate', 'rate')
    assert narrow['ci'] is None and narrow['zero_denominator_bootstrap_draws'] > 0
