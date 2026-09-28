import numpy as np
import pandas as pd

from trade_research.tail_formula_cross_rotation import mapped_inputs, native_values, reference


def members(x, y):
    pc = np.full(len(x), 10000.)
    p20 = pc * (1 + np.asarray(x)/100)
    return pd.DataFrame(dict(date='2024-01-02', code=[str(i) for i in range(len(x))],
        pc=pc, p20=p20, p49=p20*(1+np.asarray(y)/100)))


def test_market_continuation_and_rotation_are_removed_from_individual_residual():
    for slope in [1, -1]:
        m = members([-2, 0, 2], [-2*slope, 0, 2*slope])
        atoms, stats = reference(m, minimum_members=3)
        result = mapped_inputs(m[['date', 'code']], atoms, stats)
        np.testing.assert_allclose(result.XR01, slope*100, atol=1e-10)
        np.testing.assert_allclose(result.XR02, 0, atol=1e-10)
        np.testing.assert_allclose(native_values(m, m.iloc[0]), [slope*100, 0], atol=1e-10)


def test_flat_early_prices_and_single_stock_tail_departure():
    m = members([0, 0, 0], [-1, 0, 1])
    atoms, stats = reference(m, minimum_members=3)
    result = mapped_inputs(m[['date', 'code']], atoms, stats)
    np.testing.assert_allclose(result.XR01, 0)
    assert result.XR02.iloc[0] < 0 and result.XR02.iloc[2] > 0
    np.testing.assert_allclose(np.mean(result.XR02**2), 1)
    flat, flat_stats = reference(members([0, 0, 0], [0, 0, 0]), minimum_members=3)
    np.testing.assert_allclose(mapped_inputs(m[['date', 'code']], flat, flat_stats)[['XR01', 'XR02']], 0)


def test_bad_member_invalidates_whole_date_and_extra_future_fields_do_not_enter():
    m = members([-2, 0, 2], [1, -1, 0])
    baseline = reference(m, minimum_members=3)
    changed = m.assign(next_day_return=1000, price_1500=-999)
    for got, wanted in zip(reference(changed, minimum_members=3), baseline):
        pd.testing.assert_frame_equal(got, wanted, check_exact=True)
    m.loc[1, 'p49'] = np.nan
    _, stats = reference(m, minimum_members=3)
    assert stats.xr_members.iloc[0] == 3 and stats.xr_bad_members.iloc[0] == 1
    assert not stats.xr_valid.iloc[0] and np.isnan(stats.xr_correlation.iloc[0])
