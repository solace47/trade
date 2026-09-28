import numpy as np
import pandas as pd

from trade_research.tail_formula_score_center import center


def test_visible_mean_keeps_unknown_peers_and_ignores_invalid():
    f = pd.DataFrame(dict(date=['a']*4+['b']*2, code=['1','2','3','4','1','2'],
        score=[1., 3., 5., np.nan, 8., np.nan], formula_input_valid=[True,True,True,False,True,False],
        known15=[True,False,True,True,False,True]))
    got, daily = center(f)
    np.testing.assert_allclose(got.centered_score, [-2,0,2,np.nan,0,np.nan], equal_nan=True)
    assert daily.members.tolist() == [3,1]
    assert daily.mean_score.tolist() == [3,8]
    pd.testing.assert_frame_equal(got[f.columns], f)


def test_date_offsets_removed_without_mixing_dates_and_constant_zero():
    f = pd.DataFrame(dict(date=['a']*3+['b']*3+['c']*2, code=['1','2','3']*2+['1','2'],
        score=[1.,2.,4.,101.,102.,104.,7.,7.], formula_input_valid=True))
    got, _ = center(f)
    np.testing.assert_allclose(got.centered_score.iloc[:3], got.centered_score.iloc[3:6], atol=2e-14)
    np.testing.assert_array_equal(got.centered_score.iloc[6:], [0,0])
    assert not got.centered_score.iloc[6:].gt(0).any()
