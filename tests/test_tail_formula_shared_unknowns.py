import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_tail_formula_shared_unknowns import daily_bounds


def rows(left, right, known, opportunity, no_trade=None):
    n = len(left)
    return pd.DataFrame(dict(date=['2025-01-02']*n, half=['2025H1']*n, code=[str(i) for i in range(n)],
        left=left, right=right, known=known, opportunity=opportunity, no_trade=no_trade or [False]*n))


def test_identical_lists_cancel_even_with_all_unknown():
    d = daily_bounds(rows([True, True], [True, True], [False, False], [np.nan, np.nan]))
    assert d.lower.iloc[0] == d.upper.iloc[0] == 0
    assert d.shared_unknown.iloc[0] == 2


def test_shared_unknown_with_unequal_weights_only_partly_cancels():
    d = daily_bounds(rows([True, True], [True, False], [False, True], [np.nan, 1]))
    assert d.lower.iloc[0] == 0 and d.upper.iloc[0] == .5
    assert d.unknown_weight.iloc[0] == .5


def test_unshared_unknown_and_no_trade_keep_full_denominators():
    d = daily_bounds(rows([True, True, False], [False, False, True],
        [False, False, True], [np.nan, np.nan, 1], [False, True, False]))
    assert d.lower.iloc[0] == -1 and d.upper.iloc[0] == -.5
    assert d.left_rows.iloc[0] == 2 and d.right_rows.iloc[0] == 1
