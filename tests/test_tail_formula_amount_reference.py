import numpy as np
import pandas as pd

from trade_research.tail_formula_amount_reference import reference


def members():
    return pd.DataFrame(dict(date=['2025-01-02']*2, code=['a', 'b'], p49=[110., 90.], pc=[100., 100.],
        p20=[100., 100.], amount_1449=[300., 100.], amount_last29=[0., 100.]))


def test_separate_amount_weights_and_zero_tail_member_retained():
    m = members(); r = reference(m, minimum_members=2).iloc[0]
    assert r.aw_members == 2 and r.amount_reference_valid
    np.testing.assert_allclose([r.aw_day_mean, r.aw_tail_mean], [5., -10.])
    # Amount units do not change the weighted price benchmark.
    m[['amount_1449', 'amount_last29']] *= 100
    s = reference(m, minimum_members=2).iloc[0]
    np.testing.assert_allclose([s.aw_day_mean, s.aw_tail_mean], [5., -10.])


def test_invalid_member_or_zero_denominator_invalidates_whole_date():
    m = members(); m.loc[0, 'amount_last29'] = 301
    r = reference(m, minimum_members=2).iloc[0]
    assert r.aw_members == 2 and r.aw_bad_members == 1 and not r.amount_reference_valid
    assert np.isnan(r.aw_day_mean) and np.isnan(r.aw_tail_mean)
    m = members(); m['amount_last29'] = 0
    r = reference(m, minimum_members=2).iloc[0]
    assert not r.amount_reference_valid and np.isnan(r.aw_tail_mean)
