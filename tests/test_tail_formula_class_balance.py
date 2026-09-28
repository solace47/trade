import numpy as np
import pandas as pd

from trade_research.tail_formula_class_balance import balance


def test_extreme_class_imbalance_still_gives_each_class_half_date_mass():
    frame = pd.DataFrame(dict(date=['a']*101, opportunity15=[0]*100+[1]))
    result = balance(frame)
    np.testing.assert_allclose(result.groupby('opportunity15').w.sum(), [.5, .5], atol=2e-12)
    assert result.w.iloc[-1] == .5
    np.testing.assert_allclose((result.target*result.w).sum(), 0., atol=2e-12)


def test_single_class_dates_remain_with_zero_target_and_unit_mass():
    result = balance(pd.DataFrame(dict(date=['a']*3+['b']*2, opportunity15=[1]*3+[0]*2)))
    assert len(result) == 5 and result.target.eq(0).all()
    np.testing.assert_allclose(result.groupby('date').w.sum(), [1., 1.], atol=2e-12)


def test_duplicating_one_class_preserves_class_mass_and_other_dates():
    frame = pd.DataFrame(dict(date=['a', 'a', 'b', 'b'], opportunity15=[0, 1, 0, 1]))
    original = balance(frame)
    changed = balance(pd.concat([frame, frame.iloc[[0]*6]], ignore_index=True))
    pd.testing.assert_frame_equal(original.loc[original.date.eq('b')], changed.loc[changed.date.eq('b')], check_exact=True)
    np.testing.assert_allclose(changed.loc[changed.date.eq('a')].groupby('opportunity15').w.sum(), [.5, .5], atol=2e-12)
