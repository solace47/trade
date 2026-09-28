import numpy as np
import pandas as pd

from trade_research import tail_formula_amount_concentration as study


def members(day, tail):
    return pd.DataFrame(dict(date='2024-01-02', code=[str(i) for i in range(len(day))],
        amount_1449=day, amount_last29=tail))


def test_uniform_and_concentrated_amounts_use_all_members_including_zero_tail():
    m = members([100., 100., 100., 100.], [100., 0., 0., 0.])
    r = study.reference(m, minimum_members=4).iloc[0]
    assert r.ac_members == 4 and r.amount_concentration_valid
    np.testing.assert_allclose([r.ac_day_concentration, r.ac_tail_concentration], [0., 3.], atol=1e-12)
    np.testing.assert_allclose(study.native_values(m.amount_1449, m.amount_last29), [0., 3.], atol=1e-12)


def test_scale_invariance_and_variance_of_relative_amounts():
    a = np.array([1., 2., 4., 8.])
    m = members(100*a, 10*a)
    r = study.reference(m, minimum_members=4).iloc[0]
    expected = np.var(a / a.mean())
    np.testing.assert_allclose([r.ac_day_concentration, r.ac_tail_concentration], expected, atol=1e-12)
    np.testing.assert_allclose(study.native_values(m.amount_1449, m.amount_last29), [expected, expected], atol=1e-12)


def test_bad_member_or_zero_total_invalidates_whole_date_instead_of_dropping_it():
    for m in [members([100., 100., np.nan], [10., 20., 30.]),
              members([100., 100., 100.], [0., 0., 0.]),
              members([100., 100., 100.], [10., 101., 0.])]:
        r = study.reference(m, minimum_members=3).iloc[0]
        assert r.ac_members == 3 and not r.amount_concentration_valid
        assert np.isnan(r.ac_day_concentration) and np.isnan(r.ac_tail_concentration)
